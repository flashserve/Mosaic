from __future__ import annotations

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from configuration_llada import (
    ModelConfig,
    BlockType,
)

from .llada_components import (
    init_weights,
    ensure_finite_,
    BufferCache,
    ModuleType,
)
from .llada_norms_activations import (
    Dropout,
    LayerNormBase,
    Activation,
)
from .llada_rotary_bias import RotaryEmbedding

# vLLM paged attention minimal依赖
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.attention.utils.fa_utils import (
    reshape_and_cache_flash,
    flash_attn_varlen_func,
    get_flash_attn_version,
)


from flash_attn.flash_attn_varlen_inplace import flash_attn_varlen_inplace

import math
from vllm import _custom_ops as ops
from vllm.model_executor.layers.layernorm import fused_add_rms_norm
from vllm.logger import init_logger

from vllm_add_llada.cuda_kernels import fused_ops
import os

logger = init_logger(__name__)

class LLaDABlock(nn.Module):
    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )
        self.__cache = cache
        assert config.d_model % config.n_heads == 0

        # Inference-only: no activation checkpointing

        self.dropout = Dropout(config.residual_dropout)

        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        if config.attention_layer_norm:
            self.k_norm = LayerNormBase.build(
                config,
                size=(config.d_model // config.n_heads) * config.effective_n_kv_heads,
                elementwise_affine=config.attention_layer_norm_with_affine,
            )
            self.q_norm = LayerNormBase.build(config, elementwise_affine=config.attention_layer_norm_with_affine)

        self.act = Activation.build(config)
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0

        self.attn_out = nn.Linear(
            config.d_model, config.d_model, bias=config.include_bias, device=config.init_device
        )

        self.ff_out = nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.ff_out._is_residual = True  # type: ignore

        if self.config.rope:
            self.rotary_emb = RotaryEmbedding(config, self.__cache)
            # 优先使用 vLLM 原生 CUDA RoPE（更快），利用 BufferCache 跨层共享（节省 ~64MB）
            self._rotary_cuda = self.__cache.get("shared_rotary_cuda")
            if self._rotary_cuda is None:
                # 第一个 layer 创建共享实例
                try:
                    from vllm.model_executor.layers.rotary_embedding import (
                        RotaryEmbedding as VLLMRotaryEmbedding,
                    )
                    head_dim = config.d_model // config.n_heads
                    # 采用 NeoX 风格（LLaMA 系列），rotary_dim == head_dim
                    self._rotary_cuda = VLLMRotaryEmbedding(
                        head_size=head_dim,
                        rotary_dim=head_dim,
                        max_position_embeddings=config.max_sequence_length,
                        base=config.rope_theta,
                        is_neox_style=True,
                        dtype=torch.float32,
                    )
                    # 强制使用 CUDA kernel，避免 forward_native 的内存开销
                    self._rotary_cuda._forward_method = self._rotary_cuda.forward_cuda
                    # 存入 cache 供后续 layer 复用
                    self.__cache["shared_rotary_cuda"] = self._rotary_cuda
                except Exception:
                    self._rotary_cuda = None
            # else: 后续 layer 直接使用已缓存的实例

        self.flash_attn_func = None
        if config.flash_attention:
            try:
                from flash_attn import flash_attn_func  # type: ignore

                self.flash_attn_func = flash_attn_func
            except ModuleNotFoundError:
                pass

    def reset_parameters(self):
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        init_weights(
            self.config,
            self.attn_out,
            d=self.config.d_model,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )
        init_weights(
            self.config,
            self.ff_out,
            d=self.ff_out.in_features,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    # Inference-only: remove activation checkpointing API
    # def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
    #     pass

    @classmethod
    def _cast_attn_bias(cls, bias: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
        target_dtype = input_dtype
        if bias.device.type == "cuda" and torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif bias.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            target_dtype = torch.get_autocast_cpu_dtype()
        if bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
            ensure_finite_(bias, check_neg_inf=True, check_pos_inf=False)
        return bias

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if self.flash_attn_func is not None and attn_mask is None:
            r = self.flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p, causal=False
            )
            return r.transpose(1, 2)
        else:
            assert k.size(1) == v.size(1)
            num_kv_heads = k.size(1)
            num_q_heads = q.size(1)
            if num_q_heads != num_kv_heads:
                assert num_q_heads % num_kv_heads == 0
                k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
                v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)

            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=False,
            )











    # def get_activation_pool(self):
    #     """获取activation memory pool引用"""
    #     try:
    #         from vllm.forward_context import get_forward_context
    #         fctx = get_forward_context()
    #         return getattr(fctx, 'activation_memory_pool', None)
    #     except:
    #         return None




    # def allocate_activation_tensors(self, num_tokens: int):
    #     """为当前层分配所有需要的activation张量"""
    #     activation_pool = self.get_activation_pool()
    #     if activation_pool is None or not activation_pool._initialized:
    #         return None  # 回退到普通分配
            
    #     try:
    #         # 分配整个层需要的显存
    #         allocated_buffer = activation_pool.allocate_for_tokens(num_tokens)
            
    #         # 计算各个张量的偏移和大小
    #         d_model = self.config.d_model
    #         n_kv_heads = self.config.effective_n_kv_heads
    #         head_dim = d_model // self.config.n_heads
            
    #         dtype = torch.bfloat16  # 根据实际情况调整
    #         element_size = torch.tensor([], dtype=dtype).element_size()
            
    #         tensors = {}
    #         offset = 0
            

    #         tensors['x'] = activation_pool.get_tensor_view(
    #             allocated_buffer, (num_tokens, d_model), dtype, offset)
    #         offset += num_tokens * d_model * element_size

    #         # 1. RMSNorm输出
    #         tensors['x_normed'] = activation_pool.get_tensor_view(
    #             allocated_buffer, (num_tokens, d_model), dtype, offset)
    #         offset += num_tokens * d_model * element_size
            
    #         # 2. Q张量
    #         tensors['q'] = activation_pool.get_tensor_view(
    #             allocated_buffer, (num_tokens, d_model), dtype, offset)
    #         offset += num_tokens * d_model * element_size
            
    #         # 3. K张量
    #         k_size = n_kv_heads * head_dim
    #         tensors['k'] = activation_pool.get_tensor_view(
    #             allocated_buffer, (num_tokens, k_size), dtype, offset)
    #         offset += num_tokens * k_size * element_size
            
    #         # 4. V张量
    #         tensors['v'] = activation_pool.get_tensor_view(
    #             allocated_buffer, (num_tokens, k_size), dtype, offset)
    #         offset += num_tokens * k_size * element_size
            
    #         # 5. Attention输出
    #         tensors['att_out'] = activation_pool.get_tensor_view(
    #             allocated_buffer, (num_tokens, self.config.n_heads,head_dim), dtype, offset)
    #         offset += num_tokens * self.config.n_heads * head_dim * element_size
            
    #         # 6. MLP中间张量
    #         tensors['mlp_x'] = activation_pool.get_tensor_view(
    #             allocated_buffer, (num_tokens, self.hidden_size), dtype, offset)
    #         offset += num_tokens * self.hidden_size * element_size
            
    #         tensors['mlp_x_up'] = activation_pool.get_tensor_view(
    #             allocated_buffer, (num_tokens, self.hidden_size), dtype, offset)
    #         offset += num_tokens * self.hidden_size * element_size
            
    #         # 7. MLP激活后的张量
    #         tensors['mlp_activated'] = activation_pool.get_tensor_view(
    #             allocated_buffer, (num_tokens, self.hidden_size), dtype, offset)
    #         offset += num_tokens * self.hidden_size * element_size
            
    #         # 8. MLP element-wise乘法结果
    #         tensors['mlp_gated'] = activation_pool.get_tensor_view(
    #             allocated_buffer, (num_tokens, self.hidden_size), dtype, offset)
    #         offset += num_tokens * self.hidden_size * element_size
            
    #         return tensors
            
    #     except Exception as e:
    #         import logging
    #         logger = logging.getLogger(__name__)
    #         logger.warning(f"[ActivationPool] 分配失败，回退到普通分配: {e}")
    #         return None






    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 仅保留 varlen 路径：注意力输入为拍扁 [N, C]
        dtype = k.dtype
        # if self.q_norm is not None and self.k_norm is not None:
        #     q = self.q_norm(q).to(dtype=dtype)
        #     k = self.k_norm(k).to(dtype=dtype)

        N, C = q.size()
        num_heads = self.config.n_heads
        num_kv_heads = self.config.effective_n_kv_heads
        head_size = C // num_heads

        # 读取 varlen 元数据
        from vllm.forward_context import get_forward_context
        fctx = get_forward_context()
        attn_meta = getattr(fctx, "attn_metadata", None)
        if isinstance(attn_meta, dict):
            meta = attn_meta.get(getattr(self, "layer_name", None), attn_meta)
        else:
            meta = attn_meta
        assert meta is not None, "Missing varlen attention metadata in forward context"
        query_start_loc = meta.query_start_loc  # int32 [B+1]
        seq_lens = meta.seq_lens                # int32 [B]
        slot_mapping = meta.slot_mapping.long() # int64 [N]
        block_table = meta.block_table_tensor   # int32 [B, max_blocks_per_seq]
        B = int(seq_lens.numel())
        max_seq_len = int(seq_lens.max().item()) if B > 0 else 0
        max_query_len = max_seq_len

        # [N, C] → [N, H, Dh]
        device = q.device
        q_var = q.view(N, num_heads, head_size)
        k_var = k.view(N, num_kv_heads, head_size)
        v_var = v.view(N, num_kv_heads, head_size)

        # RoPE：优先使用 vLLM CUDA 内核一次性应用；不可用时回退到逐序列 PyTorch 实现
        if self.config.rope and B > 0 and N > 0:
            if getattr(self, "_rotary_cuda", None) is not None and q.is_cuda:
                # 优先复用元数据里的 positions；缺失则向量化构造
                meta_positions = getattr(meta, "positions", None)
                if meta_positions is not None:
                    positions = meta_positions
                    if positions.device != device:
                        positions = positions.to(device)
                    if positions.dtype != torch.int64:
                        positions = positions.to(torch.int64)
                else:
                   exit(0)
                # vLLM 自定义算子会就地更新 q_var/k_var，并返回相同视图
                q_var, k_var = self._rotary_cuda(positions, q_var, k_var)
            else:
                exit(0)
        # 分配 paged KV 缓冲（统一使用 fp16 存储，kernel 不支持 bf16 KVCache）
        block_size = 16  # 最小兼容值，后续可从配置暴露
        num_blocks = (N + block_size - 1) // block_size
        kv_shape = FlashAttentionBackend.get_kv_cache_shape(
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )
        stride_order = FlashAttentionBackend.get_kv_cache_stride_order()
        raw = torch.empty(int(torch.tensor(kv_shape).prod().item()), dtype=torch.bfloat16, device=device)
        kv_cache = raw.view(kv_shape).permute(*[stride_order.index(i) for i in range(len(stride_order))]).contiguous()
        key_cache, value_cache = kv_cache.unbind(0)

        # 写入 paged KV
        kv_cache_dtype_str = "auto"
        one_scale = torch.tensor(1.0, device=device)
        reshape_and_cache_flash(
            # (k_var if k_var.dtype == torch.float16 else k_var.to(torch.float16)),
            # (v_var if v_var.dtype == torch.float16 else v_var.to(torch.float16)),
            k_var, v_var,
            key_cache, value_cache,
            slot_mapping,
            kv_cache_dtype_str,
            one_scale, one_scale,
        )

        # 写入 paged KV（scatter）——KVCache 使用 fp16，K/V/Q 也统一为 fp16
        # kv_cache_dtype_str = "auto"
        # one_scale = torch.tensor(1.0, device=device)
        # reshape_and_cache_flash(
        #     # (k_var if k_var.dtype == torch.float16 else k_var.to(torch.float16)),
        #     # (v_var if v_var.dtype == torch.float16 else v_var.to(torch.float16)),
        #     k_var, v_var,
        #     key_cache, value_cache,
        #     slot_mapping,
        #     kv_cache_dtype_str,
        #     one_scale, one_scale,
        # )

        # varlen kernel 调用
        # q_in = q_var if q_var.dtype == torch.float16 else q_var.to(torch.float16)
        att.set_(att.view(N, num_heads , head_size))
        flash_attn_varlen_func(
            q=q_var,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=query_start_loc,
            max_seqlen_q=max_query_len,
            seqused_k=seq_lens,
            max_seqlen_k=max_seq_len,
            softmax_scale=1.0 / math.sqrt(head_size),
            causal=False,  # DLLM 双向注意力
            window_size=None,
            alibi_slopes=None,
            block_table=block_table,
            softcap=0.0,
            out=att,
            fa_version=get_flash_attn_version(),
            q_descale=None, k_descale=None, v_descale=None,
        )

        # [N, H, Dh] → [N, C]
        att.set_(att.view(N, num_heads * head_size))
        #target_dtype = self.attn_out.weight.dtype
        # print(f'target_dtype = {target_dtype}')
        # if att.dtype != target_dtype:
        #     att = att.to(target_dtype)

        # torch.mm(att, self.attn_out.weight.t(), out=att)
        att.copy_(self.attn_out(att))
        # return self.attn_out(att)






    def no_paged_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_tmp: torch.Tensor,
        att: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 仅保留 varlen 路径：注意力输入为拍扁 [N, C]
        dtype = k.dtype
        # if self.q_norm is not None and self.k_norm is not None:
        #     q = self.q_norm(q).to(dtype=dtype)
        #     k = self.k_norm(k).to(dtype=dtype)

        N, C = q.size()
        num_heads = self.config.n_heads
        num_kv_heads = self.config.effective_n_kv_heads
        head_size = C // num_heads

        # 读取 varlen 元数据
        from vllm.forward_context import get_forward_context
        fctx = get_forward_context()
        attn_meta = getattr(fctx, "attn_metadata", None)
        if isinstance(attn_meta, dict):
            meta = attn_meta.get(getattr(self, "layer_name", None), attn_meta)
        else:
            meta = attn_meta
        assert meta is not None, "Missing varlen attention metadata in forward context"
        query_start_loc = meta.query_start_loc  # int32 [B+1]
        seq_lens = meta.seq_lens                # int32 [B]
        
        B = int(seq_lens.numel())
        max_seq_len = int(seq_lens.max().item()) if B > 0 else 0
        max_query_len = max_seq_len

        # [N, C] → [N, H, Dh]
        device = q.device
        q_var = q.view(N, num_heads, head_size)
        k_var = k.view(N, num_kv_heads, head_size)
        v_var = v.view(N, num_kv_heads, head_size)

        # RoPE：优先使用 vLLM CUDA 内核一次性应用；不可用时回退到逐序列 PyTorch 实现
        if self.config.rope and B > 0 and N > 0:
            if getattr(self, "_rotary_cuda", None) is not None and q.is_cuda:
                # 优先复用元数据里的 positions；缺失则向量化构造
                meta_positions = getattr(meta, "positions", None)
                if meta_positions is not None:
                    positions = meta_positions
                    if positions.device != device:
                        positions = positions.to(device)
                    if positions.dtype != torch.int64:
                        positions = positions.to(torch.int64)
                else:
                   exit(0)
                # vLLM 自定义算子会就地更新 q_var/k_var，并返回相同视图
                q_var, k_var = self._rotary_cuda(positions, q_var, k_var)
            else:
                exit(0)
      
        att_tmp.set_(att_tmp.view(N, num_heads , head_size))
        flash_attn_varlen_inplace(
            q=q_var,
            k=k_var,
            v=v_var,
            cu_seqlens_q=query_start_loc,
            max_seqlen_q=max_query_len,
            cu_seqlens_k=query_start_loc,
            max_seqlen_k=max_seq_len,
            softmax_scale=1.0 / math.sqrt(head_size),
            causal=False,  # DLLM 双向注意力
            alibi_slopes=None,
            softcap=0.0,
            out=att_tmp
        )

        # [N, H, Dh] → [N, C]
        att_tmp.set_(att_tmp.view(N, num_heads * head_size))
        #target_dtype = self.attn_out.weight.dtype
        # print(f'target_dtype = {target_dtype}')
        # if att.dtype != target_dtype:
        #     att = att.to(target_dtype)

        # torch.mm(att, self.attn_out.weight.t(), out=att)
        # att.copy_(self.attn_out(att))
        # return self.attn_out(att)
        torch.mm(att_tmp, self.attn_out.weight.t(), out=att)






    # def attention(
    #     self,
    #     q: torch.Tensor,
    #     k: torch.Tensor,
    #     v: torch.Tensor,
    #     attention_bias: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     # 仅保留 varlen 路径：注意力输入为拍扁 [N, C]
    #     dtype = k.dtype
    #     if self.q_norm is not None and self.k_norm is not None:
    #         q = self.q_norm(q).to(dtype=dtype)
    #         k = self.k_norm(k).to(dtype=dtype)

    #     N, C = q.size()
    #     num_heads = self.config.n_heads
    #     num_kv_heads = self.config.effective_n_kv_heads
    #     head_size = C // num_heads

    #     # 读取 varlen 元数据
    #     from vllm.forward_context import get_forward_context
    #     fctx = get_forward_context()
    #     attn_meta = getattr(fctx, "attn_metadata", None)
    #     if isinstance(attn_meta, dict):
    #         meta = attn_meta.get(getattr(self, "layer_name", None), attn_meta)
    #     else:
    #         meta = attn_meta
    #     assert meta is not None, "Missing varlen attention metadata in forward context"
    #     query_start_loc = meta.query_start_loc  # int32 [B+1]
    #     seq_lens = meta.seq_lens                # int32 [B]
    #     slot_mapping = meta.slot_mapping.long() # int64 [N]
    #     block_table = meta.block_table_tensor   # int32 [B, max_blocks_per_seq]
    #     B = int(seq_lens.numel())
    #     max_seq_len = int(seq_lens.max().item()) if B > 0 else 0
    #     max_query_len = max_seq_len

    #     # [N, C] → [N, H, Dh]
    #     device = q.device
    #     q_var = q.view(N, num_heads, head_size)
    #     k_var = k.view(N, num_kv_heads, head_size)
    #     v_var = v.view(N, num_kv_heads, head_size)

    #     # RoPE：优先使用 vLLM CUDA 内核一次性应用；不可用时回退到逐序列 PyTorch 实现
    #     if self.config.rope and B > 0 and N > 0:
    #         if getattr(self, "_rotary_cuda", None) is not None and q.is_cuda:
    #             # 优先复用元数据里的 positions；缺失则向量化构造
    #             meta_positions = getattr(meta, "positions", None)
    #             if meta_positions is not None:
    #                 positions = meta_positions
    #                 if positions.device != device:
    #                     positions = positions.to(device)
    #                 if positions.dtype != torch.int64:
    #                     positions = positions.to(torch.int64)
    #             else:
    #                exit(0)
    #             # vLLM 自定义算子会就地更新 q_var/k_var，并返回相同视图
    #             q_var, k_var = self._rotary_cuda(positions, q_var, k_var)
    #         else:
    #             exit(0)
    #             # for b in range(B):
    #             #     s = int(query_start_loc[b].item())
    #             #     t = int(seq_lens[b].item())
    #             #     if t == 0:
    #             #         continue
    #             #     e = s + t
    #             #     qb = q_var[s:e].transpose(0, 1).unsqueeze(0).contiguous()  # [1, H, t, Dh]
    #             #     kb = k_var[s:e].transpose(0, 1).unsqueeze(0).contiguous()
    #             #     qb, kb = self.rotary_emb(qb, kb)
    #             #     q_var[s:e] = qb.squeeze(0).transpose(0, 1).contiguous()
    #             #     k_var[s:e] = kb.squeeze(0).transpose(0, 1).contiguous()

    #     # 分配 paged KV 缓冲（统一使用 fp16 存储，kernel 不支持 bf16 KVCache）
    #     block_size = 16  # 最小兼容值，后续可从配置暴露
    #     num_blocks = (N + block_size - 1) // block_size
    #     kv_shape = FlashAttentionBackend.get_kv_cache_shape(
    #         num_blocks=num_blocks,
    #         block_size=block_size,
    #         num_kv_heads=num_kv_heads,
    #         head_size=head_size,
    #     )
    #     stride_order = FlashAttentionBackend.get_kv_cache_stride_order()
    #     raw = torch.empty(int(torch.tensor(kv_shape).prod().item()), dtype=torch.float16, device=device)
    #     kv_cache = raw.view(kv_shape).permute(*[stride_order.index(i) for i in range(len(stride_order))]).contiguous()
    #     key_cache, value_cache = kv_cache.unbind(0)

    #     # 写入 paged KV
    #     kv_cache_dtype_str = "auto"
    #     one_scale = torch.tensor(1.0, device=device)
    #     reshape_and_cache_flash(
    #         (k_var if k_var.dtype == torch.float16 else k_var.to(torch.float16)),
    #         (v_var if v_var.dtype == torch.float16 else v_var.to(torch.float16)),
    #         key_cache, value_cache,
    #         slot_mapping,
    #         kv_cache_dtype_str,
    #         one_scale, one_scale,
    #     )

    #     # 写入 paged KV（scatter）——KVCache 使用 fp16，K/V/Q 也统一为 fp16
    #     kv_cache_dtype_str = "auto"
    #     one_scale = torch.tensor(1.0, device=device)
    #     reshape_and_cache_flash(
    #         (k_var if k_var.dtype == torch.float16 else k_var.to(torch.float16)),
    #         (v_var if v_var.dtype == torch.float16 else v_var.to(torch.float16)),
    #         key_cache, value_cache,
    #         slot_mapping,
    #         kv_cache_dtype_str,
    #         one_scale, one_scale,
    #     )

    #     # varlen kernel 调用
    #     q_in = q_var if q_var.dtype == torch.float16 else q_var.to(torch.float16)
    #     out_var = torch.empty_like(q_in)
    #     flash_attn_varlen_func(
    #         q=q_in,
    #         k=key_cache,
    #         v=value_cache,
    #         cu_seqlens_q=query_start_loc,
    #         max_seqlen_q=max_query_len,
    #         seqused_k=seq_lens,
    #         max_seqlen_k=max_seq_len,
    #         softmax_scale=1.0 / math.sqrt(head_size),
    #         causal=False,  # DLLM 双向注意力
    #         window_size=None,
    #         alibi_slopes=None,
    #         block_table=block_table,
    #         softcap=0.0,
    #         out=out_var,
    #         fa_version=get_flash_attn_version(),
    #         q_descale=None, k_descale=None, v_descale=None,
    #     )

    #     # [N, H, Dh] → [N, C]
    #     out_nc = out_var.reshape(N, num_heads * head_size)
    #     target_dtype = self.attn_out.weight.dtype
    #     if out_nc.dtype != target_dtype:
    #         out_nc = out_nc.to(target_dtype)
    #     return self.attn_out(out_nc)














    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:  # abstract
        raise NotImplementedError

    @classmethod
    def build(cls, layer_id: int, config: ModelConfig, cache: BufferCache) -> "LLaDABlock":
        if config.block_type == BlockType.sequential:
            return LLaDASequentialBlock(layer_id, config, cache)
        elif config.block_type == BlockType.llama:
            return LLaDALlamaBlock(layer_id, config, cache)
        else:
            raise NotImplementedError(f"Unknown block type: '{config.block_type}'")


class LLaDASequentialBlock(LLaDABlock):
    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        from .llada_norms_activations import LayerNorm

        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        head_dim = config.d_model // config.n_heads
        self.fused_dims = (
            config.d_model,
            config.effective_n_kv_heads * head_dim,
            config.effective_n_kv_heads * head_dim,
        )
        self.att_proj = nn.Linear(
            config.d_model, sum(self.fused_dims), bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )
        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        init_weights(
            self.config, self.att_proj, d=self.config.d_model, layer_id=None, type_of_module=ModuleType.in_module
        )
        init_weights(
            self.config, self.ff_proj, d=self.config.d_model, layer_id=None, type_of_module=ModuleType.in_module
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, k, v = self.att_proj(self.attn_norm(x)).split(self.fused_dims, dim=-1)
        att = self.attention(q, k, v, attention_bias)

        x = x + self.dropout(att)

        og_x = x
        x = self.ff_norm(x)
        x = self.ff_proj(x)
        x = self.act(x)
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

        return x


class LLaDALlamaBlock(LLaDABlock):
    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        from .llada_norms_activations import LayerNorm

        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache

        head_dim = config.d_model // config.n_heads
        q_proj_out_dim = config.d_model
        k_proj_out_dim = config.effective_n_kv_heads * head_dim
        v_proj_out_dim = config.effective_n_kv_heads * head_dim
        self.q_proj = nn.Linear(
            config.d_model, q_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )
        self.k_proj = nn.Linear(
            config.d_model, k_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )
        self.v_proj = nn.Linear(
            config.d_model, v_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )
        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )
        self.up_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        init_weights(self.config, self.q_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.k_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.v_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.up_proj, d=self.config.d_model, layer_id=None)

    # def forward(
    #     self,
    #     x: torch.Tensor,
    #     attention_bias: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     # 导入vLLM优化的RMSNorm算子
     
    
    #     x_normed = torch.empty_like(x)
    #     ops.rms_norm(x_normed, x, self.attn_norm.weight, self.attn_norm.eps)
  
       
        
    #     # 2. QKV投影
    #     q = self.q_proj(x_normed)
    #     k = self.k_proj(x_normed)
    #     v = self.v_proj(x_normed)
    #     # print(f'enter LLaDALlamaBlock')
    #     att = self.attention(q, k, v, attention_bias)

     
    #     x, og_x = fused_add_rms_norm(x, att, self.ff_norm.weight, self.ff_norm.eps)
 
    #     x, x_up = self.ff_proj(x), self.up_proj(x)
    #     x = self.act(x)
    #     x = x * x_up
    #     x = self.ff_out(x)
   
    #     x = og_x + x
    #     return x










    # def forward(
    #     self,
    #     activation_tensors,
    #     attention_bias: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     # 导入vLLM优化的RMSNorm算子
     
    #     x = activation_tensors['x']
    #     x_normed = activation_tensors['x_normed']
    #     # x_normed = torch.empty_like(x)
    #     ops.rms_norm(x_normed, x, self.attn_norm.weight, self.attn_norm.eps)
  
       
        
    #     # 2. QKV投影
    #     # q = self.q_proj(x_normed)
    #     # k = self.k_proj(x_normed)
    #     # v = self.v_proj(x_normed)
    #     # print(f'enter LLaDALlamaBlock')


    #     q = activation_tensors['q']
    #     k = activation_tensors['k'] 
    #     v = activation_tensors['v']
    #     torch.mm(x_normed, self.q_proj.weight.t(), out=q)
    #     torch.mm(x_normed, self.k_proj.weight.t(), out=k)
    #     torch.mm(x_normed, self.v_proj.weight.t(), out=v)
    #     att = activation_tensors['att_out']
    #     self.attention(q, k, v, att, attention_bias)

     
    #     x, og_x = fused_add_rms_norm(x, att, self.ff_norm.weight, self.ff_norm.eps)
 
    #     x, x_up = self.ff_proj(x), self.up_proj(x)
    #     x = self.act(x)
    #     x = x * x_up
    #     x = self.ff_out(x)
   
    #     x = og_x + x
    #     return x














#-----------------------naive use plan----------------------------

    # def forward(
    #     self,
    #     activation_tensors,
    #     attention_bias: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     # 导入vLLM优化的RMSNorm算子
     
    #     # N, d_model = x.shape
    
    #     # 尝试使用activation pool分配张量
    #     # activation_tensors = self.allocate_activation_tensors(N)

    #     # assert activation_tensors is not None, "Failed to allocate activation tensors"


    #     # x_normed = torch.empty_like(x)
    #     #print(f'x_normed shape: {activation_tensors["x_normed"].shape} and x shape = {x.shape}')
    #     x_normed = activation_tensors['x_normed']
    #     x = activation_tensors['x']
    #     ops.rms_norm(x_normed, x, self.attn_norm.weight, self.attn_norm.eps)
  
       
        
    #     # 2. QKV投影
    #     # q = self.q_proj(x_normed)
    #     # k = self.k_proj(x_normed)
    #     # v = self.v_proj(x_normed)
    #     q = activation_tensors['q']
    #     k = activation_tensors['k'] 
    #     v = activation_tensors['v']
    #     # print(f'self.q_proj.bias = {self.q_proj.bias}')
    #     # print(f'self.k_proj.bias = {self.k_proj.bias}')
    #     # print(f'self.v_proj.bias = {self.v_proj.bias}')
    #     torch.mm(x_normed, self.q_proj.weight.t(), out=q)
    #     torch.mm(x_normed, self.k_proj.weight.t(), out=k)
    #     torch.mm(x_normed, self.v_proj.weight.t(), out=v)
    #     # print(f'enter LLaDALlamaBlock')
    #     att_tmp = activation_tensors['att_tmp']
    #     att = activation_tensors['att_out']
    #     self.no_paged_attention(q, k, v, att_tmp, att, attention_bias)
    #     # print(f'att.dtype = {att.dtype}')
    #     # print(f'att.shape = {att.shape}')
     
    #     # x, og_x = fused_add_rms_norm(x, att, self.ff_norm.weight, self.ff_norm.eps)
    #     # ops.fused_add_rms_norm(x, att, self.ff_norm.weight, self.ff_norm.eps)
    #     ops.fused_add_rms_norm_out(x, att, x, att, self.ff_norm.weight, self.ff_norm.eps)
    #     #x是原本的x+attn norm之后的，attn是原本的x+attn

    #     mlp_x = activation_tensors['mlp_x']
    #     mlp_x_up = activation_tensors['mlp_x_up']
    #     mlp_gated = activation_tensors['mlp_gated']
    #     # x, x_up = self.ff_proj(x), self.up_proj(x)
    #     torch.mm(x, self.ff_proj.weight.t(), out=mlp_x)
    #     torch.mm(x, self.up_proj.weight.t(), out=mlp_x_up)
    #     # x = self.act(x)
    #     torch.nn.functional.silu(mlp_x, inplace=True)
    #     # x = x * x_up
    #     torch.mul(mlp_x, mlp_x_up, out=mlp_gated)
    #     #x = self.ff_out(x)
    #     # print(f'mlp_gated.shape = {mlp_gated.shape}')
    #     # print(f'self.ff_out.weight.shape = {self.ff_out.weight.shape}')
    #     torch.mm(mlp_gated, self.ff_out.weight.t(), out=x)
   
    #     # x = att + x
    #     x.add_(att)
    #     # return x


#-----------------------naive use plan----------------------------

    def forward(
        self,
        activation_tensors,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 导入vLLM优化的RMSNorm算子
        use_chunkwise = os.getenv("VLLM_USE_CHUNKWISE_GRAPH", "0") == "1"
        # N, d_model = x.shape
    
        # 尝试使用activation pool分配张量
        # activation_tensors = self.allocate_activation_tensors(N)

        # assert activation_tensors is not None, "Failed to allocate activation tensors"


        # x_normed = torch.empty_like(x)
        #print(f'x_normed shape: {activation_tensors["x_normed"].shape} and x shape = {x.shape}')
        x_normed = activation_tensors['x_normed']
        x = activation_tensors['x']
        ops.rms_norm(x_normed, x, self.attn_norm.weight, self.attn_norm.eps)
  
       
        
        # 2. QKV投影
        # q = self.q_proj(x_normed)
        # k = self.k_proj(x_normed)
        # v = self.v_proj(x_normed)
        q = activation_tensors['q']
        k = activation_tensors['k'] 
        v = activation_tensors['v']
        # print(f'self.q_proj.bias = {self.q_proj.bias}')
        # print(f'self.k_proj.bias = {self.k_proj.bias}')
        # print(f'self.v_proj.bias = {self.v_proj.bias}')
        torch.mm(x_normed, self.q_proj.weight.t(), out=q)
        torch.mm(x_normed, self.k_proj.weight.t(), out=k)
        torch.mm(x_normed, self.v_proj.weight.t(), out=v)
        # print(f'enter LLaDALlamaBlock')
        att_tmp = activation_tensors['att_tmp']
        att_out = activation_tensors['att_out']
        self.no_paged_attention(q, k, v, att_tmp, att_out, attention_bias)
        # print(f'att.dtype = {att.dtype}')
        # print(f'att.shape = {att.shape}')
     
        # x, og_x = fused_add_rms_norm(x, att, self.ff_norm.weight, self.ff_norm.eps)
        # ops.fused_add_rms_norm(x, att, self.ff_norm.weight, self.ff_norm.eps)
        residual = activation_tensors['residual']
        x_normed_2 = activation_tensors['x_normed_2']
        ops.fused_add_rms_norm_out(x_normed_2, residual, x, att_out, self.ff_norm.weight, self.ff_norm.eps)
        #x是原本的x+attn norm之后的，attn是原本的x+attn

        mlp_x = activation_tensors['mlp_x']
        mlp_x_up = activation_tensors['mlp_x_up']
        mlp_gated = activation_tensors['mlp_gated']
        mlp_output = activation_tensors['mlp_output']

        #--------------------------------------cuda event计时  -----------------------
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        
        # # 记录开始时间
        # start_event.record()

        #--------------------------------------cuda event计时  -----------------------
        #--------------------------------------full-mlp--------------------------------
        # 获取 chunk 策略（由 model 设置）
        strategy = activation_tensors.get('_chunk_strategy')
        use_chunkwise = strategy and strategy.get('chunk_mlp', False)
        
        if use_chunkwise:
            N = x_normed_2.shape[0]
            num_chunks = strategy.get('num_chunks_mlp', 5) if strategy else 5  # 从策略读取，或默认5
            tile_size = (N + num_chunks - 1) // num_chunks
            
            for i in range(0, N, tile_size):
                end_i = min(i + tile_size, N)
                t_i=end_i-i
                # print(f'mlp t_i = {t_i}')
                # print(f'mlp i = {i}')
                # print(f'mlp end_i = {end_i}')
                torch.mm(x_normed_2[i:end_i], self.ff_proj.weight.t(), out=mlp_x[0:t_i])
                torch.mm(x_normed_2[i:end_i], self.up_proj.weight.t(), out=mlp_x_up[0:t_i])

                # mlp_x[0:t_i] = torch.mm(x_normed_2[i:end_i], self.ff_proj.weight.t())
                # mlp_x_up[0:t_i] = torch.mm(x_normed_2[i:end_i], self.up_proj.weight.t())

                torch.nn.functional.silu(mlp_x[0:t_i], inplace=True)
                # mlp_x[0:t_i] =  torch.nn.functional.silu(mlp_x[0:t_i])
                torch.mul(mlp_x[0:t_i], mlp_x_up[0:t_i], out=mlp_gated[0:t_i])
                # mlp_x[0:t_i] = torch.nn.functional.silu(mlp_x[0:t_i])
                # mlp_gated[0:t_i] = torch.mul(mlp_x[0:t_i], mlp_x_up[0:t_i])
                torch.mm(mlp_gated[0:t_i], self.ff_out.weight.t(), out=mlp_output[i:end_i])
                # mlp_output[i:end_i] = torch.mm(mlp_gated[0:t_i], self.ff_out.weight.t())
        else:
            torch.mm(x_normed_2, self.ff_proj.weight.t(), out=mlp_x)
            torch.mm(x_normed_2, self.up_proj.weight.t(), out=mlp_x_up)
        
            torch.nn.functional.silu(mlp_x, inplace=True)
        
            torch.mul(mlp_x, mlp_x_up, out=mlp_gated)
            
            
            torch.mm(mlp_gated, self.ff_out.weight.t(), out=mlp_output)
        #---------------------------------------full-mlp----------------------------

        #--------------------------------------chunk-mlp------------------------------

        # end_event.record()
        
        # # 等待所有操作完成
        # torch.cuda.synchronize()
        
        # # 计算时间差
        # elapsed_time_ms = start_event.elapsed_time(end_event)
        # print(f"No chunk MLP CUDA Event 计时结果: {elapsed_time_ms:.3f} ms")


        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        
        # # 记录开始时间
        # start_event.record()

        # N=x_normed_2.shape[0]
        # print(f'N = {N}')
        # num_chunks=5
        # # tile_size=N//5
        # tile_size = (N + num_chunks - 1) // num_chunks

        # for i in range(0, N, tile_size):
        #     end_i = min(i + tile_size, N)
        #     print(f'x_normed_2[i:end_i].is_contiguous = {x_normed_2[i:end_i].is_contiguous()}')
        #     print(f'mlp_x[i:end_i].is_contiguous = {mlp_x[i:end_i].is_contiguous()}')
        #     print(f'mlp_x_up[i:end_i].is_contiguous = {mlp_x_up[i:end_i].is_contiguous()}')
        #     print(f'mlp_gated[i:end_i].is_contiguous = {mlp_gated[i:end_i].is_contiguous()}')
        #     print(f'mlp_output[i:end_i].is_contiguous = {mlp_output[i:end_i].is_contiguous()}')
            
        #     torch.mm(x_normed_2[i:end_i], self.ff_proj.weight.t(), out=mlp_x[i:end_i])
        #     torch.mm(x_normed_2[i:end_i], self.up_proj.weight.t(), out=mlp_x_up[i:end_i])
        #     torch.nn.functional.silu(mlp_x[i:end_i], inplace=True)
        #     # fused_ops.silu_inplace(mlp_x[i:end_i])
        #     #mlp_x[i:end_i] = torch.nn.functional.silu(mlp_x[i:end_i])
        #     torch.mul(mlp_x[i:end_i], mlp_x_up[i:end_i], out=mlp_gated[i:end_i])
        #     torch.mm(mlp_gated[i:end_i], self.ff_out.weight.t(), out=mlp_output[i:end_i])


        #------------------------------------chunk-mlp------------------------------------

        #--------------------------------------cuda event计时  -----------------------

        # end_event.record()
        
        # # 等待所有操作完成
        # torch.cuda.synchronize()
        
        # # 计算时间差
        # elapsed_time_ms = start_event.elapsed_time(end_event)
        # print(f"MLP CUDA Event 计时结果: {elapsed_time_ms:.3f} ms")

        #--------------------------------------cuda event计时  -----------------------
        
        final_output = activation_tensors['final_output']
        torch.add(residual, mlp_output, out=final_output)
       









class LLaDABlockGroup(nn.ModuleList):
    def __init__(self, config: ModelConfig, layer_offset: int, modules: Optional[List[nn.Module]] = None):
        super().__init__(modules)
        self.config = config
        self.layer_offset = layer_offset
        # Inference-only: no activation checkpointing

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        for block_idx, block in enumerate(self):
            block_idx += self.layer_offset
            x = block(x, attention_bias=attention_bias)
        return x

    def reset_parameters(self):
        for block in self:
            block.reset_parameters()

    # Inference-only: remove activation checkpointing API
    # def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
    #     pass


"""
Transformer blocks: MLP, Attention, DecoderLayer
"""
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.utils import logging

from configuration_dream import DreamConfig
from .dream_norms_activations import DreamRMSNorm
from typing import Dict
# vLLM varlen attention 依赖
from flash_attn import flash_attn_varlen_inplace

logger = logging.get_logger(__name__)


class DreamMLP(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, device=device)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, device=device)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, device=device)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        x_normed_2: torch.Tensor,
        mlp_output: torch.Tensor,
        activation_tensors: Dict[str, torch.Tensor],
    ):
        """MLP forward：从 activation_tensors 获取中间张量，输出写到 mlp_output"""
        # 从 activation_tensors 获取预分配的中间张量
        x_gate = activation_tensors['x_gate']
        x_up = activation_tensors['x_up']
        x_mlp = activation_tensors['x_mlp']
        
        # 获取 chunk 策略（由 model 设置）
        strategy = activation_tensors.get('_chunk_strategy')
        use_chunkwise = strategy and strategy.get('chunk_mlp', False)
        
        if use_chunkwise:
            # ============ Chunkwise 版本：分块计算 MLP ============
            N = x_normed_2.shape[0]
            num_chunks = strategy.get('num_chunks_mlp', 5) if strategy else 5  # 从策略读取，或默认5
            tile_size = (N + num_chunks - 1) // num_chunks
            
            for i in range(0, N, tile_size):
                end_i = min(i + tile_size, N)
                t_i = end_i - i
                
                # Gate and Up projections（重用 buffer 前 t_i 行）
                torch.mm(x_normed_2[i:end_i], self.gate_proj.weight.t(), out=x_gate[0:t_i])
                torch.mm(x_normed_2[i:end_i], self.up_proj.weight.t(), out=x_up[0:t_i])
                
                # SiLU activation (in-place)
                torch.nn.functional.silu(x_gate[0:t_i], inplace=True)
                
                # Element-wise multiply (gating)
                torch.mul(x_gate[0:t_i], x_up[0:t_i], out=x_mlp[0:t_i])
                
                # Down projection（写回完整的 mlp_output 对应位置）
                torch.mm(x_mlp[0:t_i], self.down_proj.weight.t(), out=mlp_output[i:end_i])
        else:
            # ============ 非 Chunkwise 版本：一次性计算 ============
            # Gate and Up projections (写入预分配张量)
            torch.mm(x_normed_2, self.gate_proj.weight.t(), out=x_gate)
            torch.mm(x_normed_2, self.up_proj.weight.t(), out=x_up)
            
            # SiLU activation (in-place)
            torch.nn.functional.silu(x_gate, inplace=True)
            
            # Element-wise multiply (gating)
            torch.mul(x_gate, x_up, out=x_mlp)
            
            # Down projection (写入 mlp_output)
            torch.mm(x_mlp, self.down_proj.weight.t(), out=mlp_output)


class DreamSdpaAttention(nn.Module):
    """
    Dream attention module - 仅支持 Varlen FlashAttention（照搬 llada 风格）
    """

    def __init__(self, config: DreamConfig, layer_idx: Optional[int] = None, device=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = False
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True, device=device)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True, device=device)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True, device=device)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False, device=device)

        # vLLM CUDA RoPE kernel（varlen 专用），跨层共享 buffer（节省显存）
        if not hasattr(config, '_shared_rope_cache'):
            config._shared_rope_cache = {}
        
        cache_key = "shared_rotary_cuda"
        if cache_key in config._shared_rope_cache:
            self._rotary_cuda = config._shared_rope_cache[cache_key]
        else:
            from vllm_add_dream.dream_vllm_rope import DreamVLLMRotaryEmbedding
            self._rotary_cuda = DreamVLLMRotaryEmbedding(
                head_size=self.head_dim,
                rotary_dim=self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
                is_neox_style=True,
                dtype=torch.bfloat16,
            )
            config._shared_rope_cache[cache_key] = self._rotary_cuda

    def forward(
        self,
        x_normed: torch.Tensor,
        att_out: torch.Tensor,
        activation_tensors: Dict[str, torch.Tensor],
    ):
        """Varlen forward: 从 activation_tensors 获取 QKV 张量，输出写到 att_out"""
        return self._forward_varlen(x_normed, att_out, activation_tensors)
    
    def _forward_varlen(
        self,
        x_normed: torch.Tensor,
        att_out: torch.Tensor,
        activation_tensors: Dict[str, torch.Tensor],
    ):
        """Varlen 路径：使用 activation_tensors 进行显存复用"""
        
        N, C = x_normed.size()
        num_heads = self.num_heads
        num_kv_heads = self.num_key_value_heads
        head_size = self.head_dim
        
        # 从 activation_tensors 获取预分配的 QKV 张量
        q = activation_tensors['q']
        k = activation_tensors['k']
        v = activation_tensors['v']
        
        # QKV projection (写入到预分配的张量)
        torch.mm(x_normed, self.q_proj.weight.t(), out=q)
        torch.mm(x_normed, self.k_proj.weight.t(), out=k)
        torch.mm(x_normed, self.v_proj.weight.t(), out=v)
        
        # 如果有 bias，需要加上（Dream 的 QKV 投影有 bias）
        if self.q_proj.bias is not None:
            q.add_(self.q_proj.bias)
        if self.k_proj.bias is not None:
            k.add_(self.k_proj.bias)
        if self.v_proj.bias is not None:
            v.add_(self.v_proj.bias)
        
        # 从 forward_context 获取 metadata
        from vllm.forward_context import get_forward_context
        fctx = get_forward_context()
        attn_meta = getattr(fctx, "attn_metadata", None)
        if isinstance(attn_meta, dict):
            meta = attn_meta.get(getattr(self, "layer_name", None), attn_meta)
        else:
            meta = attn_meta
        assert meta is not None, "Missing varlen attention metadata in forward context"
        
        query_start_loc = meta.query_start_loc  # int32 [B+1]
        seq_lens = meta.seq_lens  # int32 [B]
        B = int(seq_lens.numel())
        max_seq_len = int(seq_lens.max().item()) if B > 0 else 0
        max_query_len = max_seq_len
        
        # [N, C] → [N, H, Dh]
        device = x_normed.device
        q_var = q.view(N, num_heads, head_size)
        k_var = k.view(N, num_kv_heads, head_size)
        v_var = v.view(N, num_kv_heads, head_size)
        
        # RoPE：应用 vLLM CUDA RoPE（in-place 修改 q_var 和 k_var）
        if B > 0 and N > 0:
            # 从 metadata 获取 positions
            meta_positions = getattr(meta, "positions", None)
            assert meta_positions is not None, "Missing positions in varlen metadata"
            positions = meta_positions
            if positions.device != device:
                positions = positions.to(device)
            if positions.dtype != torch.int64:
                positions = positions.to(torch.int64)
            
            q_var, k_var = self._rotary_cuda.forward(positions, q_var, k_var)
        
        # 从 activation_tensors 获取 att_tmp 中间张量
        att_tmp = activation_tensors['att_out_var']
        
        # FlashAttention varlen inplace 接口 (与 LLaDA 一致)
        # 使用 set_ 改变 att_tmp 的形状视图为 [N, H, Dh]
        att_tmp.set_(att_tmp.view(N, num_heads, head_size))
        
        # 计算 softmax_scale (与 LLaDA 一致)
        softmax_scale = 1.0 / math.sqrt(head_size)
        
        # FlashAttention 原生支持 GQA，不需要手动 repeat_kv
        flash_attn_varlen_inplace(
            q=q_var,
            k=k_var,
            v=v_var,
            cu_seqlens_q=query_start_loc,
            max_seqlen_q=max_query_len,
            cu_seqlens_k=query_start_loc,
            max_seqlen_k=max_seq_len,
            softmax_scale=softmax_scale,
            causal=False,  # Dream 使用双向注意力
            alibi_slopes=None,
            softcap=0.0,
            out=att_tmp  # in-place 写入到 att_tmp
        )
        
        # [N, H, Dh] → [N, C]
        att_tmp.set_(att_tmp.view(N, num_heads * head_size))
        
        # O projection (写入到预分配的 att_out 张量)
        torch.mm(att_tmp, self.o_proj.weight.t(), out=att_out)


class DreamDecoderLayer(nn.Module):
    def __init__(self, config: DreamConfig, layer_idx: int, device=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = DreamSdpaAttention(config, layer_idx, device=device)
        self.mlp = DreamMLP(config, device=device)
        self.input_layernorm = DreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device)
        self.post_attention_layernorm = DreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device)

    def forward(
        self,
        activation_tensors: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Varlen forward: 使用 activation_tensors 和优化的 ops"""
        from vllm import _custom_ops as ops
        
        # 从 activation_tensors 取出所有需要的张量
        x = activation_tensors['x']
        x_normed = activation_tensors['x_normed']
        
        # 1. Input RMS Norm (使用 vLLM CUDA 优化的 rms_norm)
        ops.rms_norm(x_normed, x, self.input_layernorm.weight, self.input_layernorm.variance_epsilon)
        
        # 2. Self Attention (从 activation_tensors 获取输出张量)
        att_out = activation_tensors['att_out']
        self.self_attn(x_normed, att_out, activation_tensors)
        
        # 3. 第一个残差连接 + Post Attention RMS Norm (fused 操作)
        residual = activation_tensors['residual']
        x_normed_2 = activation_tensors['x_normed_2']
        ops.fused_add_rms_norm_out(
            x_normed_2,                                    # 输出1: norm 后的结果 (给 MLP 用)
            residual,                                      # 输出2: x + att_out (保存给第二个残差连接)
            x,                                             # 输入1: 原始 x
            att_out,                                       # 输入2: attention 输出
            self.post_attention_layernorm.weight,          # norm 权重
            self.post_attention_layernorm.variance_epsilon # norm eps
        )
        
        # 4. MLP (从 activation_tensors 获取输出张量)
        mlp_output = activation_tensors['mlp_output']
        self.mlp(x_normed_2, mlp_output, activation_tensors)
        
        # 5. 第二个残差连接
        final_output = activation_tensors['final_output']
        torch.add(residual, mlp_output, out=final_output)
        
        return final_output


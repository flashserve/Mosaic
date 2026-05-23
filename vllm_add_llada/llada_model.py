from __future__ import annotations

import logging
import math
from typing import Optional, List, Tuple, Sequence, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from configuration_llada import (
    LLaDAConfig,
    ModelConfig,
)

from .llada_components import (
    BufferCache,
    _non_meta_init_device,
    init_weights,
    ModuleType,
)
from .llada_norms_activations import Dropout, LayerNormBase
from .llada_rotary_bias import (
    get_causal_attention_bias,
    alibi_attention_bias,
)
from .llada_blocks import (
    LLaDABlock,
    LLaDABlockGroup,
)

from vllm.v1.worker.graph_builder import define_llada_block_graph,define_llada_block_graph_chunkwise, MemoryAnalyzer

from vllm import _custom_ops as ops

from vllm.v1.diffusion.utils import _add_gumbel_noise

log = logging.getLogger(__name__)
import time
import os

class LLaDAOutput(NamedTuple):
    # logits: torch.FloatTensor  # 已移除，节省显存（约33GB for N=131500）
    attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    hidden_states: Optional[Tuple[torch.Tensor]]
    index:Optional[torch.Tensor]
    confidence:Optional[torch.Tensor]


class LLaDAModel(nn.Module):
    def __init__(self, config: ModelConfig, init_params: bool = True):
        super().__init__()
        self.config = config
        self.__cache = BufferCache()

        if self.config.alibi and self.config.flash_attention:
            raise Exception("ALiBi is currently not supported with FlashAttention")
        if self.config.alibi and self.config.rope:
            raise Exception("ALiBi and RoPE are mutually exclusive")
        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise Exception("embedding size should be at least as big as vocab size")
            elif self.config.embedding_size % 128 != 0:
                import warnings
                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.",
                    UserWarning,
                )

        # Inference-only: no activation checkpointing

        if not (
            0 < self.config.block_group_size <= self.config.n_layers
            and self.config.n_layers % self.config.block_group_size == 0
        ):
            raise Exception("n layers must be divisible by block group size")

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.embedding_size or config.vocab_size, config.d_model, device=config.init_device
                ),
                emb_drop=Dropout(config.embedding_dropout),
                ln_f=LayerNormBase.build(config),
            )
        )

        blocks = [LLaDABlock.build(i, config, self.__cache) for i in range(config.n_layers)]
        if self.config.block_group_size > 1:
            block_groups = [
                LLaDABlockGroup(config, i, blocks[i : i + config.block_group_size])
                for i in range(0, config.n_layers, config.block_group_size)
            ]
            self.transformer.update({"block_groups": nn.ModuleList(block_groups)})
        else:
            self.transformer.update({"blocks": nn.ModuleList(blocks)})

        if not (self.config.alibi or self.config.rope):
            self.transformer.update(
                {"wpe": nn.Embedding(config.max_sequence_length, config.d_model, device=config.init_device)}
            )
        if not config.weight_tying:
            self.transformer.update(
                {
                    "ff_out": nn.Linear(
                        config.d_model,
                        config.embedding_size or config.vocab_size,
                        bias=config.include_bias,
                        device=config.init_device,
                    )
                }
            )
        if init_params and self.config.init_device != "meta":
            self.reset_parameters()
        self.__num_fwd_flops: Optional[int] = None

        if self.config.alibi:
            get_causal_attention_bias(self.__cache, config.max_sequence_length, _non_meta_init_device(config))
            self.get_alibi_attention_bias(config.max_sequence_length, _non_meta_init_device(config))

    # Inference-only: remove activation checkpointing API
    # def set_activation_checkpointing(self, strategy: Optional["ActivationCheckpointingStrategy"]):
    #     pass






    
    def get_activation_pool(self):
        """获取activation memory pool引用"""
        try:
            from vllm.forward_context import get_forward_context
            fctx = get_forward_context()
            return getattr(fctx, 'activation_memory_pool', None)
        except:
            return None





    def get_optimized_reuse_plan(self, N: int, M: int) -> Dict[str, Dict[str, Any]]:
        """基于graph_builder生成优化的reuse_plan"""
        import os
        num_tokens = N
        use_chunkwise = os.getenv("VLLM_USE_CHUNKWISE_GRAPH", "0") == "1"
        
        if not use_chunkwise:
            # 路径1: 不 chunk
            graph_blueprint = define_llada_block_graph(N, M)
            self._current_chunk_strategy = None
        else:
            # 路径2: 在线搜索最优 chunk 配置（Bottleneck-Oriented）
            from diffusion_tools import search_optimal_config_online
            
            # 获取 activation pool 大小
            activation_pool = self.get_activation_pool()
            if activation_pool is None or not activation_pool._initialized:
                raise RuntimeError("ActivationPool 未初始化，无法进行在线搜索")
            
            pool_size = activation_pool.pool_size
            
            # 在线搜索最优配置
            import logging
            logging.info(f"[LLaDAModel] 开始在线搜索最优 chunk 配置: N={N}, M={M}")
            
            strategy = search_optimal_config_online(
                P=N,
                O=M,
                pool_size=pool_size,
                model_name='llada'
            )
            
            logging.info(
                f"[LLaDAModel] 在线搜索完成: "
                f"chunk_logits={strategy['chunk_logits']}({strategy['num_chunks_logits']}), "
                f"chunk_mlp={strategy['chunk_mlp']}({strategy['num_chunks_mlp']}), "
                f"需要 {strategy['total_bytes']/1024**3:.3f}GB"
            )
            
            # 生成计算图
            graph_blueprint = define_llada_block_graph_chunkwise(
                N, M,
                chunk_logits=strategy['chunk_logits'],
                chunk_mlp=strategy['chunk_mlp'],
                num_chunks_logits=strategy['num_chunks_logits'],
                num_chunks_mlp=strategy['num_chunks_mlp'],
            )
            
            self._current_chunk_strategy = strategy
        
        # 实例化分析器并生成优化的reuse_plan
        analyzer = MemoryAnalyzer(graph_blueprint)
        analysis_result = analyzer.analyze()
        reuse_plan = analysis_result['reuse_plan']
        
        return reuse_plan




    def get_naive_reuse_plan(self, num_tokens: int) -> Dict[str, Dict[str, Any]]:
        """生成naive版本的reuse_plan（简单偏移累加，不做复用）"""
        d_model = self.config.d_model
        n_kv_heads = self.config.effective_n_kv_heads
        head_dim = d_model // self.config.n_heads
        self.hidden_size = self.config.mlp_hidden_size
        dtype = torch.bfloat16
        element_size = torch.tensor([], dtype=dtype).element_size()

        # 定义张量顺序（按照计算图的执行顺序）
        tensor_specs = {
            'x': {'shape': (num_tokens, d_model), 'dtype': dtype},
            'x_normed': {'shape': (num_tokens, d_model), 'dtype': dtype},
            'q': {'shape': (num_tokens, d_model), 'dtype': dtype},
            'k': {'shape': (num_tokens, n_kv_heads * head_dim), 'dtype': dtype},
            'v': {'shape': (num_tokens, n_kv_heads * head_dim), 'dtype': dtype},
            'att_tmp': {'shape': (num_tokens, self.config.n_heads, head_dim), 'dtype': dtype},
            'att_out': {'shape': (num_tokens, d_model), 'dtype': dtype},
            'mlp_x': {'shape': (num_tokens, self.hidden_size), 'dtype': dtype},
            'mlp_x_up': {'shape': (num_tokens, self.hidden_size), 'dtype': dtype},
            'mlp_activated': {'shape': (num_tokens, self.hidden_size), 'dtype': dtype},
            'mlp_gated': {'shape': (num_tokens, self.hidden_size), 'dtype': dtype},
        }

        # naive版本：简单累加偏移，不做复用
        reuse_plan = {}
        offset = 0

        for tensor_name, spec in tensor_specs.items():
            tensor_size = torch.prod(torch.tensor(spec['shape'])).item() * element_size
            reuse_plan[tensor_name] = {
                'offset': offset,
                'size': tensor_size,
                'shape': spec['shape'],
                'dtype': spec['dtype']
            }
            offset += tensor_size

        reuse_plan['_total_buffer_size_bytes'] = offset
        return reuse_plan

    def allocate_activation_tensors_with_reuse_plan(self, N: int, M: int):
        """使用reuse_plan接口分配activation张量（新接口）"""
        activation_pool = self.get_activation_pool()
        if activation_pool is None or not activation_pool._initialized:
            return None  # 回退到普通分配

        try:
            num_tokens = N
            # 获取naive版本的reuse_plan
            # reuse_plan = self.get_naive_reuse_plan(num_tokens)
            reuse_plan = self.get_optimized_reuse_plan(N, M)

            # 分配整个层需要的显存
            total_size = reuse_plan['_total_buffer_size_bytes']
            allocated_buffer = activation_pool.allocate_for_tokens_with_size(num_tokens, total_size)

            tensors = {}

            # 根据reuse_plan创建张量视图
            for tensor_name, plan in reuse_plan.items():
                if tensor_name.startswith('_'):  # 跳过元数据
                    continue
                tensors[tensor_name] = activation_pool.get_tensor_view(
                    allocated_buffer,
                    plan['shape'],
                    plan['dtype'],
                    plan['offset']
                )

            print(f'[NewInterface] 使用reuse_plan分配，总大小: {total_size}字节')
            return tensors

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"[NewInterface] 分配失败，回退到普通分配: {e}")
            return None

    # def allocate_activation_tensors(self, num_tokens: int):
    #     """为当前层分配所有需要的activation张量（原接口，已注释）"""
    #     activation_pool = self.get_activation_pool()
    #     if activation_pool is None or not activation_pool._initialized:
    #         return None  # 回退到普通分配
    #
    #     try:
    #         # 分配整个层需要的显存
    #
    #         allocated_buffer = activation_pool.allocate_for_tokens(num_tokens)
    #
    #         # 计算各个张量的偏移和大小
    #         d_model = self.config.d_model
    #         n_kv_heads = self.config.effective_n_kv_heads
    #         head_dim = d_model // self.config.n_heads
    #         self.hidden_size = self.config.mlp_hidden_size
    #         dtype = torch.bfloat16  # 根据实际情况调整
    #         element_size = torch.tensor([], dtype=dtype).element_size()
    #
    #         tensors = {}
    #         offset = 0
    #
    #
    #         tensors['x'] = activation_pool.get_tensor_view(
    #             allocated_buffer, (num_tokens, d_model), dtype, offset)
    #         offset += num_tokens * d_model * element_size
    #
    #         # 1. RMSNorm输出
    #         tensors['x_normed'] = activation_pool.get_tensor_view(
    #             allocated_buffer, (num_tokens, d_model), dtype, offset)
    #         offset += num_tokens * d_model * element_size
    #
    #         # 2. Q张量
    #         tensors['q'] = activation_pool.get_tensor_view(
    #             allocated_buffer, (num_tokens, d_model), dtype, offset)
    #         offset += num_tokens * d_model * element_size
    #
    #         # 3. K张量
    #         k_size = n_kv_heads * head_dim
    #         tensors['k'] = activation_pool.get_tensor_view(
    #             allocated_buffer, (num_tokens, k_size), dtype, offset)
    #         offset += num_tokens * k_size * element_size
    #
    #         # 4. V张量
    #         tensors['v'] = activation_pool.get_tensor_view(
    #             allocated_buffer, (num_tokens, k_size), dtype, offset)
    #         offset += num_tokens * k_size * element_size
    #
    #         # 5. Attention输出
    #         tensors['att_tmp'] = activation_pool.get_tensor_view(
    #             allocated_buffer, (num_tokens, self.config.n_heads,head_dim), dtype, offset)
    #         offset += num_tokens * self.config.n_heads * head_dim * element_size
    #
    #         tensors['att_out'] = activation_pool.get_tensor_view(
    #             allocated_buffer, (num_tokens, d_model), dtype, offset)
    #         offset += num_tokens * d_model * element_size
    #
    #         # 6. MLP中间张量
    #         tensors['mlp_x'] = activation_pool.get_tensor_view(
    #             allocated_buffer, (num_tokens, self.hidden_size), dtype, offset)
    #         offset += num_tokens * self.hidden_size * element_size
    #
    #         tensors['mlp_x_up'] = activation_pool.get_tensor_view(
    #             allocated_buffer, (num_tokens, self.hidden_size), dtype, offset)
    #         offset += num_tokens * self.hidden_size * element_size
    #
    #         # 7. MLP激活后的张量
    #         tensors['mlp_activated'] = activation_pool.get_tensor_view(
    #             allocated_buffer, (num_tokens, self.hidden_size), dtype, offset)
    #         offset += num_tokens * self.hidden_size * element_size
    #
    #         # 8. MLP element-wise乘法结果
    #         tensors['mlp_gated'] = activation_pool.get_tensor_view(
    #             allocated_buffer, (num_tokens, self.hidden_size), dtype, offset)
    #         offset += num_tokens * self.hidden_size * element_size
    #         print(f'offset = {offset}')
    #         return tensors
    #
    #     except Exception as e:
    #         import logging
    #         logger = logging.getLogger(__name__)
    #         logger.warning(f"[ActivationPool] 分配失败，回退到普通分配: {e}")
    #         return None








    @property
    def device(self) -> torch.device:
        device: torch.device = self.transformer.wte.weight.device  # type: ignore
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        else:
            return device

    def reset_parameters(self):
        log.info("Initializing model parameters...")
        init_weights(
            self.config,
            self.transformer.wte,  # type: ignore
            std_factor=(0.5 * math.sqrt(self.config.d_model)) if self.config.scale_logits else 1.0,
            type_of_module=ModuleType.emb,
        )
        if hasattr(self.transformer, "wpe"):
            init_weights(self.config, self.transformer.wpe, type_of_module=ModuleType.emb)  # type: ignore
        self.transformer.ln_f.reset_parameters()  # type: ignore
        if hasattr(self.transformer, "ff_out"):
            init_weights(self.config, self.transformer.ff_out, type_of_module=ModuleType.final_out)  # type: ignore
        if self.config.block_group_size == 1:
            for block in self.transformer.blocks:  # type: ignore[attr-defined]
                block.reset_parameters()
        else:
            for block_group in self.transformer.block_groups:  # type: ignore[attr-defined]
                block_group.reset_parameters()

    def get_alibi_attention_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        alibi_bias = self.__cache.get("alibi_attention_bias")
        if alibi_bias is not None and alibi_bias.shape[-1] >= seq_len:
            if alibi_bias.device != device:
                alibi_bias = alibi_bias.to(device)
                self.__cache["alibi_attention_bias"] = alibi_bias
            return alibi_bias
        with torch.autocast(device.type, enabled=False):
            alibi_bias = alibi_attention_bias(seq_len, self.config, device)
        self.__cache["alibi_attention_bias"] = alibi_bias
        return alibi_bias

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
    ) -> LLaDAOutput:
        assert not self.config.alibi, "Alibi length extrapolation is not supported for MDM."
        assert self.config.rope, "Rope must be used in Llama-Encoder for MDM."
        assert (past_key_values is None and not use_cache), "The kvcache is not supported for MDM."
        
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        output_hidden_states = False
        print(f'output_hidden_states = {output_hidden_states}')
        N = input_ids.shape[0]

        # 从 ForwardContext 获取 P、O 和 mask_indices
        from vllm.forward_context import get_forward_context
        fctx = get_forward_context()
        P = fctx.prompt_length
        O = fctx.output_length
        mask_indices = getattr(fctx, 'mask_indices', None)
        
        # N 是总长度，M 是实际需要计算 logits 的 tokens 数量
        N = P + O
        M = len(mask_indices)

        # 使用新的reuse_plan接口，传入 N（total_length）和 M
        t_act_pool_start = time.perf_counter()
        tensors = self.allocate_activation_tensors_with_reuse_plan(N, M)
        t_act_pool_end = time.perf_counter()
        print(f'[ActivationPool] allocate_activation_tensors_with_reuse_plan time: {(t_act_pool_end-t_act_pool_start)*1000:.3f}ms, N={N}, M={M}')

    
      

        x = tensors['x']

        # temp_x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings  # type: ignore
        # x.copy_(temp_x)

        torch.index_select(self.transformer.wte.weight, 0, input_ids, out=x)

        # 将 chunk 策略传递给 layers（如果存在）
        if hasattr(self, '_current_chunk_strategy'):
            tensors['_chunk_strategy'] = self._current_chunk_strategy

        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        all_hidden_states: List[torch.Tensor] = []
        
        # st=time.time()

        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
            
        # # 记录开始时间
        # start_event.record()


        for block_idx, block in enumerate(self.transformer.blocks):  # type: ignore[attr-defined]
            # if output_hidden_states:
            #     all_hidden_states.append(x)
            # x = block(tensors, attention_bias=attention_bias)
            # tensors['x'] = x
            block(tensors, attention_bias=attention_bias)
            
            # x = block(x, attention_bias=attention_bias)
       
        x = tensors['x']
        # print(f'forward time = {time.time() - st}')
        # end_event.record()
            
        # # 等待所有操作完成
        # torch.cuda.synchronize()
        
        # # 计算时间差
        # elapsed_time_ms = start_event.elapsed_time(end_event)
        # print(f"Logits计算 CUDA Event 计时结果: {elapsed_time_ms:.3f} ms")




        # 仅当维度为 [B, T, C] 时才抽取最后一步；拍扁 [N, C] 下跳过
        if last_logits_only and x.dim() == 3:
            x = x[:, -1, :].unsqueeze(1)

        #x = self.transformer.ln_f(x)  # type: ignore
        ops.rms_norm(x, x, self.transformer.ln_f.weight, self.transformer.ln_f.eps)

        # print(f'self.transformer.ln_f = {self.transformer.ln_f}')
        if output_hidden_states:
            all_hidden_states.append(x)

        if self.config.weight_tying:
            # logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore
            pass
        else:
            logits = tensors['logits']
            print(f'x.shape = {x.shape}')
            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)
            
            # 记录开始时间
            # start_event.record()


            # logits = self.transformer.ff_out(x)  # type: ignore
            
            #torch.mm(x, self.transformer.ff_out.weight.t(), out=logits)
            
            #print(f'self.transformer.ff_out.weight.t().shape = {self.transformer.ff_out.weight.t().shape}')

            #   # 记录结束时间
            # end_event.record()
            
            # # 等待所有操作完成
            # torch.cuda.synchronize()
            
            # # 计算时间差
            # elapsed_time_ms = start_event.elapsed_time(end_event)
            # print(f"Logits计算 CUDA Event 计时结果: {elapsed_time_ms:.3f} ms")

        strategy = getattr(self, '_current_chunk_strategy', None)
        import flash_sample
        from gather_gemm.gather_gemm_kernel import gather_gemm
        
        x0_2d=[]
        x0_p_2d=[]
        
        # 获取 mask_indices（长度 M，值范围 0 到 O-1）
        mask_indices = getattr(fctx, 'mask_indices', None)
    
        # 只对后 O 个 tokens (output tokens) 计算 logits
        # x 的形状是 [N, D_MODEL]，其中 N = P + O
        # 我们取 x[-O:] 即后 O 个 tokens
        x_output = x[-O:] if O > 0 else x  # [O, D_MODEL]
        
        # M 是实际需要计算的 tokens 数量（mask_indices 的长度）
        M = len(mask_indices)
        
        # # 创建 CUDA events 用于计时
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        
        # # 记录开始时间
        # start_event.record()
        
        if strategy and strategy.get('chunk_logits', False):
            # Chunk-wise logits computation (只对 M 个 tokens)
            logits = tensors['logits']  # [max_chunk_size_logits, VOCAB_SIZE]，由 graph_builder 分配
            num_chunks = strategy.get('num_chunks_logits', 7)
            tile_size = (M + num_chunks - 1) // num_chunks
            
            for i in range(0, M, tile_size):
                end_i = min(i + tile_size, M)
                t_i = end_i - i
                logits_tile = logits[0:t_i]
                tile_indices = mask_indices[i:end_i]
                
                gather_gemm(x_output, tile_indices, self.transformer.ff_out.weight.t(), logits_tile)
                
                fused_indices, fused_probs = flash_sample.low_confidence(logits_tile.contiguous())
                x0_2d.append(fused_indices)
                x0_p_2d.append(fused_probs)

            x0_2d = torch.cat(x0_2d, dim=0)
            x0_p_2d = torch.cat(x0_p_2d, dim=0)
        else:
            # 非 chunk 模式：直接对 M 个 tokens 计算 logits
            logits = tensors['logits']  # [M, VOCAB_SIZE]，由 graph_builder 分配
            
            gather_gemm(x_output, mask_indices, self.transformer.ff_out.weight.t(), logits)
            
            x0_2d, x0_p_2d = flash_sample.low_confidence(logits.contiguous())
        
        # # 记录结束时间
        # end_event.record()
        
        # # 等待所有 CUDA 操作完成
        # torch.cuda.synchronize()
        
        # # 计算时间差
        # elapsed_time_ms = start_event.elapsed_time(end_event)
        # print(f"[Logits计算] CUDA Event 计时结果: {elapsed_time_ms:.3f} ms (M={M}, O={O}, chunk={strategy.get('chunk_logits', False) if strategy else False})")


            #--------------------------------------------chunk logits-----------------------------------
            # logits = tensors['logits']
            # N=x.shape[0]
            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)
            
            # # 记录开始时间
            # start_event.record()


            # # logits = self.transformer.ff_out(x)  # type: ignore
            # TILE_SIZE=N//5
            # for i in range(0,N,TILE_SIZE):
            #     end_i = min(i+TILE_SIZE,N)
            #     x_tile=x[i:end_i]
            #     logits_tile=logits[i:end_i]
            #     torch.mm(x_tile, self.transformer.ff_out.weight.t(), out=logits_tile)

            # print(f'ff_out.weight.shape = {self.transformer.ff_out.weight.shape}')
            # # torch.mm(x, self.transformer.ff_out.weight.t(), out=logits)
            # #print(f'self.transformer.ff_out.weight.t().shape = {self.transformer.ff_out.weight.t().shape}')

            #   # 记录结束时间
            # end_event.record()
            
            # # 等待所有操作完成
            # torch.cuda.synchronize()
            
            # # 计算时间差
            # elapsed_time_ms = start_event.elapsed_time(end_event)
            # print(f"Chunk Logits计算 CUDA Event 计时结果: {elapsed_time_ms:.3f} ms")









        # if self.config.scale_logits:
        #     logits.mul_(1 / math.sqrt(self.config.d_model))



        

        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
            
        # # 记录开始时间
        # start_event.record()

        # logits_2d_with_noise = _add_gumbel_noise(logits, temperature=0)
        # x0_2d = torch.argmax(logits_2d_with_noise, dim=-1)  # [N]  #选出来本次解码后得到的所有token_id
        # remasking = "low_confidence"
        # if remasking == "low_confidence":
        #     p_2d = F.softmax(logits, dim=-1)  # [N, V]
        #     x0_p_2d = torch.gather(p_2d, dim=-1, index=x0_2d.unsqueeze(-1)).squeeze(-1)  # [N]
        # elif remasking == "random":
        #     x0_p_2d = torch.rand(x0_2d.shape[0], device=x0_2d.device)  # [N]
        # else:
        #     raise NotImplementedError(remasking)






        # end_event.record()
            
        # # 等待所有操作完成
        # torch.cuda.synchronize()
            
        # # 计算时间差
        # elapsed_time_ms = start_event.elapsed_time(end_event)
        # print(f"从Logits到Index和Confidence torch native CUDA Event 计时结果: {elapsed_time_ms:.3f} ms")


        # from vllm_add_llada.cuda_kernels import fused_ops
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        # start_event.record()
        # fused_indices, fused_probs = fused_ops.forward(logits.contiguous())
        # end_event.record()
        # torch.cuda.synchronize()
        # elapsed_time_ms = start_event.elapsed_time(end_event)
        # print(f"从Logits到Index和Confidence fused CUDA Event 计时结果: {elapsed_time_ms:.3f} ms")



        return LLaDAOutput(
            # logits=logits,  # 已移除，不再返回以节省显存
            attn_key_values=attn_key_values,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            index=x0_2d,
            confidence=x0_p_2d
        )


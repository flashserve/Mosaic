"""
LLaDA MoE Model for vLLM inference.
Adapted from modeling_lladamoe.py with vLLM optimizations.
"""

from __future__ import annotations
import logging
import math
import time
from typing import Optional, List, Tuple, NamedTuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .llada_moe_blocks import (
    LLaDAMoERMSNorm,
    LLaDAMoEDecoderLayer,
    LLaDAMoEBlockGroup,
)

from vllm.v1.diffusion.utils import _add_gumbel_noise
from vllm.v1.worker.graph_builder import (
    define_llada_moe_block_graph, 
    define_llada_moe_block_graph_chunkwise,
    MemoryAnalyzer
)
from vllm import _custom_ops as ops

log = logging.getLogger(__name__)


class LLaDAMoEOutput(NamedTuple):
    """Model output"""
    logits: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.Tensor]]
    router_logits: Optional[Tuple[torch.Tensor]]
    index: Optional[torch.Tensor]
    confidence: Optional[torch.Tensor]


class LLaDAMoEModel(nn.Module):
    """
    LLaDA MoE Transformer Model for vLLM inference.
    Optimized for varlen attention and inference-only.
    """
    def __init__(self, config, init_params: bool = True):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Embedding layer
        self.embed_tokens = nn.Embedding(
            config.vocab_size, 
            config.hidden_size, 
            self.padding_idx
        )
        
        # Decoder layers
        self.layers = nn.ModuleList([
            LLaDAMoEDecoderLayer(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = LLaDAMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # LM head (output projection) - 在model内部，但权重文件中是顶层
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights if configured
        if getattr(config, 'tie_word_embeddings', False):
            self.lm_head.weight = self.embed_tokens.weight
        
        # No gradient checkpointing for inference
        self.gradient_checkpointing = False
        
        if init_params and getattr(config, 'init_device', None) != "meta":
            self.post_init()

    def post_init(self):
        """Initialize weights"""
        pass

    @property
    def device(self) -> torch.device:
        return self.embed_tokens.weight.device

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_activation_pool(self):
        """获取activation memory pool引用"""
        try:
            from vllm.forward_context import get_forward_context
            fctx = get_forward_context()
            return getattr(fctx, 'activation_memory_pool', None)
        except:
            return None

    def get_optimized_reuse_plan(self, N: int, M: int) -> Dict[str, Dict[str, Any]]:
        """基于graph_builder生成优化的reuse_plan
        
        Args:
            N: 总 token 数 (P + O)
            M: 需要计算 logits 的 mask tokens 数量
        """
        import os
        use_chunkwise = os.getenv("VLLM_USE_CHUNKWISE_GRAPH", "0") == "1"
        
        if not use_chunkwise:
            # 路径1: 不 chunk
            graph_blueprint = define_llada_moe_block_graph(N, M)
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
            log.info(f"[LLaDAMoEModel] 开始在线搜索最优 chunk 配置: N={N}, M={M}")
            
            strategy = search_optimal_config_online(
                P=N,  # 使用 N 作为总 token 数
                O=M,  # 使用 M 作为需要计算 logits 的数量
                pool_size=pool_size,
                model_name='llada-moe'
            )
            
            log.info(
                f"[LLaDAMoEModel] 在线搜索完成: "
                f"chunk_logits={strategy['chunk_logits']}({strategy['num_chunks_logits']}), "
                f"chunk_moe={strategy['chunk_mlp']}({strategy['num_chunks_mlp']}), "
                f"需要 {strategy['total_bytes']/1024**3:.3f}GB"
            )
            
            # 生成计算图
            graph_blueprint = define_llada_moe_block_graph_chunkwise(
                N, M,
                chunk_logits=strategy['chunk_logits'],
                chunk_moe=strategy['chunk_mlp'],
                num_chunks_logits=strategy['num_chunks_logits'],
                num_chunks_moe=strategy['num_chunks_mlp'],
            )
            
            self._current_chunk_strategy = strategy
        
        # 实例化分析器并生成优化的reuse_plan
        analyzer = MemoryAnalyzer(graph_blueprint)
        analysis_result = analyzer.analyze()
        reuse_plan = analysis_result['reuse_plan']
        
        return reuse_plan

    def allocate_activation_tensors_with_reuse_plan(self, N: int, M: int):
        """使用reuse_plan接口分配activation张量
        
        Args:
            N: 总 token 数 (P + O)
            M: 需要计算 logits 的 mask tokens 数量
        """
        activation_pool = self.get_activation_pool()
        if activation_pool is None or not activation_pool._initialized:
            return None  # 回退到普通分配
        
        try:
            # 获取优化后的reuse_plan
            reuse_plan = self.get_optimized_reuse_plan(N, M)
            
            # 分配整个层需要的显存
            total_size = reuse_plan['_total_buffer_size_bytes']
            allocated_buffer = activation_pool.allocate_for_tokens_with_size(N, total_size)
            
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
            
            print(f'[LLaDAMoE-ActivationPool] 使用reuse_plan分配，总大小: {total_size} 字节')
            return tensors
        
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"[LLaDAMoE-ActivationPool] 分配失败，回退到普通分配: {e}")
            return None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        last_logits_only: bool = False,
    ) -> LLaDAMoEOutput:
        """Forward pass optimized for vLLM varlen attention."""
        
        output_hidden_states = False
        output_router_logits = False
        
        # 从 ForwardContext 获取 P、O 和 mask_indices
        from vllm.forward_context import get_forward_context
        fctx = get_forward_context()
        P = fctx.prompt_length
        O = fctx.output_length
        mask_indices = getattr(fctx, 'mask_indices', None)
        
        # N 是总长度，M 是实际需要计算 logits 的 tokens 数量
        N = P + O
        M = len(mask_indices) if mask_indices is not None else O
        
        # 使用 activation pool 分配张量
        t_act_pool_start = time.perf_counter()
        tensors = self.allocate_activation_tensors_with_reuse_plan(N, M)
        t_act_pool_end = time.perf_counter()
        print(f'[LLaDAMoE-ActivationPool] allocate time: {(t_act_pool_end-t_act_pool_start)*1000:.3f}ms, N={N}, M={M}')
        
        # 使用 tensors['x'] 作为 embedding 的输出
        x = tensors['x']
        torch.index_select(self.embed_tokens.weight, 0, input_ids, out=x)
        
        # 将 chunk 策略传递给 layers（如果存在）
        if hasattr(self, '_current_chunk_strategy'):
            tensors['_chunk_strategy'] = self._current_chunk_strategy
        
        # Pass through decoder layers (传入 tensors 字典)
        for decoder_layer in self.layers:
            decoder_layer(tensors)
        
        # Final layer norm (原地操作)
        x = tensors['x']
        ops.rms_norm(x, x, self.norm.weight, self.norm.variance_epsilon)
        
        # ===== Mask-Only Logits 计算：只对 M 个 mask tokens 计算 =====
        # 从 ForwardContext 获取 mask_indices（长度 M，值范围 0 到 O-1）
        mask_indices = getattr(fctx, 'mask_indices', None)
        
        # 只对后 O 个 tokens (output tokens) 计算 logits
        x_output = x[-O:] if O > 0 else x  # [O, D_MODEL]
        
        # M 是实际需要计算的 tokens 数量（mask_indices 的长度）
        M = len(mask_indices) if mask_indices is not None else O
        
        strategy = getattr(self, '_current_chunk_strategy', None)
        import flash_sample
        from gather_gemm.gather_gemm_kernel import gather_gemm
        
        x0_2d_masked = []  # M 个 mask tokens 的采样结果
        x0_p_2d_masked = []  # M 个 mask tokens 的置信度
        
        if strategy and strategy.get('chunk_logits', False):
            # Chunk-wise logits 计算和采样（只对 M 个 mask tokens）
            logits_buffer = tensors['logits']  # 小的 chunk buffer
            num_chunks = strategy.get('num_chunks_logits', 7)  # 从策略读取
            tile_size = (M + num_chunks - 1) // num_chunks
            
            for i in range(0, M, tile_size):
                end_i = min(i + tile_size, M)
                t_i = end_i - i
                logits_tile = logits_buffer[0:t_i]  # 复用小 buffer
                tile_indices = mask_indices[i:end_i]  # 当前 chunk 的 mask indices
                
                # 使用 gather_gemm 只对 tile_indices 对应的 tokens 计算
                gather_gemm(x_output, tile_indices, self.lm_head.weight.t(), logits_tile)
                
                # 立即采样（释放 logits 内存）
                fused_indices, fused_probs = flash_sample.low_confidence(logits_tile.contiguous())
                x0_2d_masked.append(fused_indices)
                x0_p_2d_masked.append(fused_probs)
            
            x0_2d = torch.cat(x0_2d_masked, dim=0)  # [M]
            x0_p_2d = torch.cat(x0_p_2d_masked, dim=0)  # [M]
            
            # Chunk 模式下不返回完整 logits
            output_logits = None
            
        else:
            # 非 chunk：一次性计算（只对 M 个 mask tokens）
            logits_masked = tensors['logits']  # [M, VOCAB_SIZE]
            
            # 使用 gather_gemm 只对 mask_indices 对应的 tokens 计算
            gather_gemm(x_output, mask_indices, self.lm_head.weight.t(), logits_masked)
            
            # ===== 使用融合算子（避免完整 softmax 张量，节省 ~4GB 显存）=====
            x0_2d, x0_p_2d = flash_sample.low_confidence(logits_masked.contiguous())
        
        # ============ 直接返回 M 维结果，映射逻辑在 gpu_model_runner 中处理 ============
            
            # ===== 原始实现（已注释，会创建 [N, V] 的 softmax 张量）=====
            # # 采样：加噪声并取 argmax
            # logits_2d_with_noise = _add_gumbel_noise(logits, temperature=0)
            # x0_2d = torch.argmax(logits_2d_with_noise, dim=-1)  # [N]
            # 
            # # 计算 confidence
            # remasking = "low_confidence"
            # if remasking == "low_confidence":
            #     p_2d = F.softmax(logits, dim=-1)  # [N, V] - 这里会分配 4.4GB！
            #     x0_p_2d = torch.gather(p_2d, dim=-1, index=x0_2d.unsqueeze(-1)).squeeze(-1)  # [N]
            # elif remasking == "random":
            #     x0_p_2d = torch.rand(x0_2d.shape[0], device=x0_2d.device)  # [N]
            # else:
            #     raise NotImplementedError(remasking)
            
            output_logits = logits_masked if 'logits_masked' in locals() else None
        
        return LLaDAMoEOutput(
            logits=output_logits,
            hidden_states=None,
            router_logits=None,
            index=x0_2d,
            confidence=x0_p_2d,
        )


def create_model_config_from_pretrained_config(config):
    """Create model config from HF config."""
    return config

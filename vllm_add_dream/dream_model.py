"""
DreamModel 主模型实现（纯 nn.Module，扁平结构，镜像 LLaDAModel）
"""
from typing import Any, Dict, List, Optional, Tuple, NamedTuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from configuration_dream import DreamConfig
from .dream_norms_activations import DreamRMSNorm
from .dream_rotary_bias import DreamRotaryEmbedding
from .dream_blocks import DreamDecoderLayer

from vllm.v1.worker.graph_builder import define_dream_block_graph, MemoryAnalyzer


class DreamOutput(NamedTuple):
    """模型输出"""
    logits: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    index: Optional[torch.Tensor] = None  # 采样得到的 token indices [N]
    confidence: Optional[torch.Tensor] = None  # 对应的置信度 [N]


class DreamModel(nn.Module):
    """Dream Model with LM head (for generation) - 扁平结构，类似 LLaDAModel"""

    def __init__(self, config: DreamConfig, init_params: bool = True, init_device: str = "cpu"):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # 直接在目标设备上创建所有组件（像 llada 一样）
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx, device=init_device)
        self.layers = nn.ModuleList(
            [DreamDecoderLayer(config, layer_idx, device=init_device) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=init_device)
        # 3D 路径需要 rotary_emb 预计算 position_embeddings
        self.rotary_emb = DreamRotaryEmbedding(config=config, device=init_device)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device=init_device)

        self.gradient_checkpointing = False
        
        # 只在 init_params=True 且不是 meta device 时才初始化（和 llada 一样）
        if init_params and init_device != "meta":
            self._init_weights()

    def _init_weights(self):
        """简单的权重初始化（但 vLLM 插件一般用 init_params=False，由权重加载覆盖）"""
        std = self.config.initializer_range
        if hasattr(self.embed_tokens, 'weight'):
            nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=std)
            if self.padding_idx is not None:
                self.embed_tokens.weight.data[self.padding_idx].zero_()
        if hasattr(self.lm_head, 'weight'):
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)

    def get_activation_pool(self):
        """获取 activation memory pool 引用"""
        try:
            from vllm.forward_context import get_forward_context
            fctx = get_forward_context()
            return getattr(fctx, 'activation_memory_pool', None)
        except:
            return None

    def get_optimized_reuse_plan(self, N: int, M: int) -> Dict[str, Dict[str, Any]]:
        """基于 graph_builder 生成优化的 reuse_plan
        
        Args:
            N: 总 token 数 (P + O)
            M: 需要计算 logits 的 mask tokens 数量
        """
        import os
        use_chunkwise = os.getenv("VLLM_USE_CHUNKWISE_GRAPH", "0") == "1"
        
        if not use_chunkwise:
            # 路径1: 不 chunk
            from vllm.v1.worker.graph_builder import define_dream_block_graph
            graph_blueprint = define_dream_block_graph(N, M)
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
            logger = logging.getLogger(__name__)
            logger.info(f"[DreamModel] 开始在线搜索最优 chunk 配置: N={N}, M={M}")
            
            strategy = search_optimal_config_online(
                P=N,  # 使用 N 作为总 token 数
                O=M,  # 使用 M 作为需要计算 logits 的数量
                pool_size=pool_size,
                model_name='dream'
            )
            
            logger.info(
                f"[DreamModel] 在线搜索完成: "
                f"chunk_logits={strategy['chunk_logits']}({strategy['num_chunks_logits']}), "
                f"chunk_mlp={strategy['chunk_mlp']}({strategy['num_chunks_mlp']}), "
                f"需要 {strategy['total_bytes']/1024**3:.3f}GB"
            )
            
            # 生成计算图
            from vllm.v1.worker.graph_builder import define_dream_block_graph_chunkwise
            graph_blueprint = define_dream_block_graph_chunkwise(
                N, M,
                chunk_logits=strategy['chunk_logits'],
                chunk_mlp=strategy['chunk_mlp'],
                num_chunks_logits=strategy['num_chunks_logits'],
                num_chunks_mlp=strategy['num_chunks_mlp'],
            )
            
            self._current_chunk_strategy = strategy
        
        # 实例化分析器并生成优化的 reuse_plan
        analyzer = MemoryAnalyzer(graph_blueprint)
        analysis_result = analyzer.analyze()
        reuse_plan = analysis_result['reuse_plan']
        
        return reuse_plan

    def allocate_activation_tensors_with_reuse_plan(self, N: int, M: int):
        """使用 reuse_plan 接口分配 activation 张量
        
        Args:
            N: 总 token 数 (P + O)
            M: 需要计算 logits 的 mask tokens 数量
        """
        activation_pool = self.get_activation_pool()
        if activation_pool is None or not activation_pool._initialized:
            return None  # 回退到普通分配
        
        try:
            # 获取优化的 reuse_plan
            reuse_plan = self.get_optimized_reuse_plan(N, M)
            
            # 分配整个层需要的显存
            total_size = reuse_plan['_total_buffer_size_bytes']
            allocated_buffer = activation_pool.allocate_for_tokens_with_size(N, total_size)
            
            tensors = {}
            
            # 根据 reuse_plan 创建张量视图
            for tensor_name, plan in reuse_plan.items():
                if tensor_name.startswith('_'):  # 跳过元数据
                    continue
                tensors[tensor_name] = activation_pool.get_tensor_view(
                    allocated_buffer,
                    plan['shape'],
                    plan['dtype'],
                    plan['offset']
                )
            
            print(f'[DreamActivationPool] 使用 reuse_plan 分配，总大小: {total_size} 字节')
            return tensors
        
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"[DreamActivationPool] 分配失败: {e}")
            return None

    @property
    def device(self) -> torch.device:
        return self.embed_tokens.weight.device

    def forward(
        self,
        input_ids: torch.LongTensor,
        # Dream diffusion 特有参数
        temperature: Optional[float] = None,
        remasking: Optional[str] = None,
    ) -> DreamOutput:
        # 从 ForwardContext 获取 P、O 和 mask_indices
        from vllm.forward_context import get_forward_context
        fctx = get_forward_context()
        P = fctx.prompt_length
        O = fctx.output_length
        mask_indices = getattr(fctx, 'mask_indices', None)
        
        # N 是总长度，M 是实际需要计算 logits 的 tokens 数量
        N = P + O
        M = len(mask_indices) if mask_indices is not None else O
        
        # 尝试使用 activation pool 分配张量
        import time
        t_act_pool_start = time.perf_counter()
        tensors = self.allocate_activation_tensors_with_reuse_plan(N, M)
        t_act_pool_end = time.perf_counter()
        print(f'[DreamActivationPool] allocate_activation_tensors_with_reuse_plan time: {(t_act_pool_end-t_act_pool_start)*1000:.3f}ms, N={N}, M={M}')
        
        # 使用 activation pool 中的张量（x 作为 hidden_states）
        x = tensors['x']
        torch.index_select(self.embed_tokens.weight, 0, input_ids, out=x)

        # 将 chunk 策略传递给 layers（如果存在）
        if hasattr(self, '_current_chunk_strategy'):
            tensors['_chunk_strategy'] = self._current_chunk_strategy

        # decoder layers
        for decoder_layer in self.layers:
            decoder_layer(tensors)

        # Final RMS Norm (in-place, 与 LLaDA 一致)
        x = tensors['x']
        from vllm import _custom_ops as ops
        ops.rms_norm(x, x, self.norm.weight, self.norm.variance_epsilon)

        # 初始化 logits（从 tensors 获取预分配的 buffer）
        logits = tensors['logits']
        
        # Dream diffusion 特有后处理：如果提供了 temperature 和 remasking 参数
        x0_index = None
        x0_confidence = None
        
        if temperature is not None and remasking is not None:
            # 从 forward_context 中获取 seq_lens 和 query_start_loc
            from vllm.forward_context import get_forward_context
            fctx = get_forward_context()
            attn_metadata = fctx.attn_metadata
            seq_lens = attn_metadata.seq_lens
            query_start_loc = attn_metadata.query_start_loc
            
            # ===== Mask-Only Logits 计算：只对 M 个 mask tokens 计算 =====
            # 从 ForwardContext 获取 mask_indices（长度 M，值范围 0 到 O-1）
            mask_indices = getattr(fctx, 'mask_indices', None)
            
            # 只对后 O 个 tokens (output tokens) 计算 logits
            # x 的形状是 [N, D_MODEL]，其中 N = P + O
            # 我们取 x[-O:] 即后 O 个 tokens
            # x_output = x[-O:] if O > 0 else x  # [O, D_MODEL]
            x_output = x[-O-1:-1] if O > 0 else x  # [O, D_MODEL]
            # M 是实际需要计算的 tokens 数量（mask_indices 的长度）
            mask_indices = torch.nonzero(input_ids[-O:] == 151666, as_tuple=False).squeeze(1)
            M = len(mask_indices) if mask_indices is not None else O
            
            strategy = getattr(self, '_current_chunk_strategy', None)
            import flash_sample
            from gather_gemm.gather_gemm_kernel import gather_gemm
            
            x0_index_masked = []  # M 个 mask tokens 的采样结果
            x0_confidence_masked = []  # M 个 mask tokens 的置信度
            
            if strategy and strategy.get('chunk_logits', False):
                # ============ Chunk-wise 版本：分块计算 logits + 采样（只对 M 个 mask tokens）============
                num_chunks = strategy.get('num_chunks_logits', 7)  # 从策略读取
                tile_size = (M + num_chunks - 1) // num_chunks
                
                for i in range(0, M, tile_size):
                    end_i = min(i + tile_size, M)
                    t_i = end_i - i
                    logits_tile = logits[0:t_i]  # [t_i, vocab_size]
                    tile_indices = mask_indices[i:end_i]  # 当前 chunk 的 mask indices
                    
                    # 使用 gather_gemm 只对 tile_indices 对应的 tokens 计算
                    gather_gemm(x_output, tile_indices, self.lm_head.weight.t(), logits_tile)
                    
                    # 立即采样
                    idx, conf = flash_sample.entropy_witht(logits_tile.contiguous(), temperature if temperature is not None else 0.0)
                    x0_index_masked.append(idx)
                    x0_confidence_masked.append(conf)
                
                # 拼接所有 chunks
                x0_index_masked = torch.cat(x0_index_masked, dim=0)  # [M]
                x0_confidence_masked = torch.cat(x0_confidence_masked, dim=0)  # [M]
                
                # 映射回 O 维：非 mask 位置保持原来的 token 值
                # x_output 对应的是 input_ids[-O:]（output 段）
                # x0_index = input_ids[-O:].clone()  # [O] - 从 input_ids 的 output 段初始化
                # # 非 mask 位置的 confidence：用 -inf 表示不参与 topk 竞争（与原生 Dream 一致）
                # x0_confidence = torch.full((O,), float("-inf"), dtype=x0_confidence_masked.dtype, device=x0_confidence_masked.device)
                # x0_index[mask_indices.long()] = x0_index_masked
                # x0_confidence[mask_indices.long()] = x0_confidence_masked
                
            else:
                # ============ 非 Chunk 版本：一次性计算（只对 M 个 mask tokens）============
                logits_masked = tensors['logits']  # [M, VOCAB_SIZE]
                
                # 使用 gather_gemm 只对 mask_indices 对应的 tokens 计算
                gather_gemm(x_output, mask_indices, self.lm_head.weight.t(), logits_masked)
                
                # 调用 flash_sample entropy 算子
                x0_index_masked, x0_confidence_masked = flash_sample.entropy_witht(logits_masked.contiguous(), temperature if temperature is not None else 0.0)
                
                # 映射回 O 维：非 mask 位置保持原来的 token 值
             
            
            # # ============ Dream 特有逻辑：对 index 和 confidence 做位移 ============
            # # 注意：x0_index 和 x0_confidence 现在是 O 维 [O_total]
            # # 直接在 O 空间操作：shift 时第一个位置用 prompt[-1] 填充
            # x0_index = input_ids[-O:].clone()  # [O] - 从 input_ids 的 output 段初始化
            # # 非 mask 位置的 confidence：用 -inf 表示不参与 topk 竞争（与原生 Dream 一致）
            # x0_confidence = torch.full((O,), float("-inf"), dtype=x0_confidence_masked.dtype, device=x0_confidence_masked.device)
            # x0_index[mask_indices.long()] = x0_index_masked
            # x0_confidence[mask_indices.long()] = x0_confidence_masked
            # B = seq_lens.numel()
            # output_offset = 0
            
            # for i in range(B):
            #     s = int(query_start_loc[i].item())
            #     t = int(seq_lens[i].item())
            #     prompt_len_seq = P  # 假设所有序列 P 相同
            #     output_len_seq = t - prompt_len_seq
                
            #     if output_len_seq > 1:
            #         # 提取当前序列的 output tokens（在 O 空间）
            #         seq_index = x0_index[output_offset:output_offset + output_len_seq]
            #         seq_confidence = x0_confidence[output_offset:output_offset + output_len_seq]
                    
            #         # Shift: [0] <- prompt[-1], [1] <- [0], [2] <- [1], ...
            #         if prompt_len_seq > 0:
            #             last_prompt_token = input_ids[s + prompt_len_seq - 1:s + prompt_len_seq]
            #             seq_index_shifted = torch.cat([last_prompt_token, seq_index[:-1]], dim=0)
            #         else:
            #             # 如果没有 prompt，则 [0] 不变
            #             seq_index_shifted = torch.cat([seq_index[:1], seq_index[:-1]], dim=0)
                    
            #         # confidence 的 shift：[0] 不变，[1] <- [0], [2] <- [1], ...
            #         # seq_confidence_shifted = torch.cat([seq_confidence[:1], seq_confidence[:-1]], dim=0)
            #         # 第0位置(last_prompt)设为-inf（不参与更新），其他位置左移
            #         seq_confidence_shifted = torch.cat([torch.tensor([float("-inf")], device=seq_confidence.device, dtype=seq_confidence.dtype), seq_confidence[:-1]], dim=0)
                    
            #         # 写回到 O 空间
            #         x0_index[output_offset:output_offset + output_len_seq] = seq_index_shifted
            #         x0_confidence[output_offset:output_offset + output_len_seq] = seq_confidence_shifted
                
            #     output_offset += output_len_seq
            
            # # ============ x_shift 完成后，提取 M 维结果返回 ============
            # # 从 O 维中提取出 mask positions 的结果
            # x0_index = x0_index[mask_indices.long()]  # [M]
            # x0_confidence = x0_confidence[mask_indices.long()]  # [M]

        return DreamOutput(
            logits=logits,
            hidden_states=None,
            attentions=None,
            index=x0_index_masked,
            confidence=x0_confidence_masked,
        )

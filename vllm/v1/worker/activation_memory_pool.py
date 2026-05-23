"""
Activation Memory Pool for DLLM models
用于管理DLLM模型前向传播过程中的activation显存分配
"""

import logging
import torch
from typing import Optional, Tuple, Dict, Any
import gc
import os


logger = logging.getLogger(__name__)


class ActivationMemoryPool:
    """
    Activation显存池，用于管理DLLM模型的中间激活张量显存
    由于DLLM没有KV cache，可以为每个batch分配一层的显存，层之间复用
    
    支持两种模式：
    1. torch.empty 模式（默认）：立即分配全部物理显存
    2. VMM 模式：预留虚拟地址，按需映射物理显存（零浪费）
    """
    
    def __init__(self, device: torch.device, use_vmm: Optional[bool] = None):
        """
        初始化 Activation Memory Pool
        
        Args:
            device: CUDA 设备
            use_vmm: 是否使用 CUDA VMM（Virtual Memory Management）模式
                    - None（默认）：从环境变量 VLLM_USE_VMM 读取（0=禁用，1=启用）
                    - False：强制使用 torch.empty，立即分配物理显存
                    - True：强制使用 VMM，按需映射物理显存（需要编译 vmm_allocator）
        """
        self.device = device
        self.pool_buffer: Optional[torch.Tensor] = None
        self.pool_size: int = 0
        self.allocated_size: int = 0
        self.memory_per_token: int = 0
        self._initialized = False
        
        # VMM 相关状态
        # 如果 use_vmm 未指定，从环境变量读取
        if use_vmm is None:
            use_vmm = os.environ.get('VLLM_USE_VMM', '0') == '1'
        self.use_vmm = use_vmm
        self._vmm_allocator = None
        self._current_mapped_size: int = 0
        
        # VMM chunk size 也可以从环境变量配置
        default_chunk_size = 512 * 1024**2  # 512MB
        self._vmm_chunk_size: int = int(os.environ.get('VLLM_VMM_CHUNK_SIZE', default_chunk_size))
        
        # 如果启用 VMM，尝试导入 vmm_allocator
        if self.use_vmm:
            try:
                import sys
                sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'vmm_allocator'))
                import vmm_allocator
                self._vmm_module = vmm_allocator
                logger.info("[ActivationPool] VMM mode enabled")
            except ImportError as e:
                logger.warning(f"[ActivationPool] Failed to import vmm_allocator: {e}")
                logger.warning("[ActivationPool] Falling back to torch.empty mode")
                self.use_vmm = False
        
    # def initialize_pool(self, model_config: Any) -> None:
    #     """
    #     初始化显存池：检测可用显存并预分配
    #     """
    #     if self._initialized:
    #         return
            
    #     # 清理显存碎片
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     torch.cuda.synchronize()
        
    #     # 获取当前显存状态
    #     free_memory, total_memory = torch.cuda.mem_get_info()
    #     used_memory = total_memory - free_memory
        
    #     logger.info(f"[ActivationPool] GPU显存状态: 总计={total_memory/1024**3:.2f}GB, "
    #                f"已用={used_memory/1024**3:.2f}GB, 空闲={free_memory/1024**3:.2f}GB")
        
    #     # 保留一些安全裕量，使用80%的可用显存作为activation pool
    #     safety_margin = 0.1
    #     self.pool_size = int(free_memory * (1.0 - safety_margin))
        
    #     # 分配显存池
    #     try:
    #         self.pool_buffer = torch.empty(self.pool_size, dtype=torch.uint8, device=self.device)
    #         logger.info(f"[ActivationPool] 成功分配显存池: {self.pool_size/1024**3:.2f}GB")
    #     except torch.cuda.OutOfMemoryError:
    #         # 如果分配失败，尝试更小的大小
    #         self.pool_size = int(free_memory * 0.5)
    #         self.pool_buffer = torch.empty(self.pool_size, dtype=torch.uint8, device=self.device)
    #         logger.warning(f"[ActivationPool] 降级分配显存池: {self.pool_size/1024**3:.2f}GB")
            
    #     self.allocated_size = 0
    #     self._initialized = True
        
    #     # 估算每个token的显存占用
    #     self._estimate_memory_per_token(model_config)


    def initialize_pool(self, model_config: Any, total_bytes: Optional[int] = None) -> None:
        """
        初始化显存池：使用graph_builder提供的总字节数或检测可用显存并预分配

        Args:
            model_config: 模型配置
            total_bytes: graph_builder计算的总字节数，如果为None则使用传统方法
        """
        if self._initialized:
            return

        # 清理显存碎片
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if total_bytes is not None:
            # 使用graph_builder提供的精确字节数
            self.pool_size = total_bytes
            logger.info(f"[ActivationPool] 使用graph_builder提供的显存大小: {self.pool_size/1024**3:.2f}GB")

        else:
            # 传统方法：检测可用显存并预分配
            free_memory, total_memory = torch.cuda.mem_get_info()
            used_memory = total_memory - free_memory

            logger.info(f"[ActivationPool] GPU显存状态: 总计={total_memory/1024**3:.2f}GB, "
                       f"已用={used_memory/1024**3:.2f}GB, 空闲={free_memory/1024**3:.2f}GB")

            # 保留一些安全裕量，使用80%的可用显存作为activation pool
            safety_margin = 0.1
            self.pool_size = int(free_memory * (1.0 - safety_margin))

        # 分配显存池
        try:
            if self.use_vmm:
                # VMM 模式：预留虚拟地址，不分配物理内存
                self._allocate_vmm_pool()
            else:
                # 传统模式：立即分配物理内存
                self.pool_buffer = torch.empty(self.pool_size, dtype=torch.uint8, device=self.device)
                logger.info(f"[ActivationPool] 成功分配显存池（torch.empty）: {self.pool_size/1024**3:.2f}GB")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if total_bytes is not None:
                # 如果graph_builder提供的字节数导致OOM，直接抛出错误
                logger.error(f"[ActivationPool] graph_builder提供的显存大小({self.pool_size/1024**3:.2f}GB)过大: {e}")
                raise
            else:
                # 如果分配失败，尝试更小的大小
                free_memory, _ = torch.cuda.mem_get_info()
                self.pool_size = int(free_memory * 0.5)
                if self.use_vmm:
                    self._allocate_vmm_pool()
                else:
                    self.pool_buffer = torch.empty(self.pool_size, dtype=torch.uint8, device=self.device)
                logger.warning(f"[ActivationPool] 降级分配显存池: {self.pool_size/1024**3:.2f}GB")

        self.allocated_size = 0
        self._initialized = True

        # 估算每个token的显存占用（用于兼容性）
        # NOTE: 已注释掉，因为现在 P/O 分离后无法简单估算 memory_per_token
        # self._estimate_memory_per_token(model_config)
    
    def _allocate_vmm_pool(self) -> None:
        """使用 VMM 分配显存池（预留虚拟地址，不分配物理内存）"""
        if not self.use_vmm or self._vmm_module is None:
            raise RuntimeError("VMM mode not enabled or vmm_allocator not available")
        
        # 初始化 CUDA 上下文（VMM 需要）
        _ = torch.zeros(1, device=self.device)
        torch.cuda.synchronize()
        
        # 创建 VMM 分配器
        self._vmm_allocator = self._vmm_module.VMMAllocator()
        
        # 预留虚拟地址空间（不分配物理内存）
        self._vmm_allocator.reserve(self.pool_size)
        
        # 包装为 torch.Tensor（零拷贝）
        self.pool_buffer = self._vmm_allocator.wrap_as_tensor(self.pool_size)
        
        # 初始映射大小为 0
        self._current_mapped_size = 0
        
        logger.info(f"[ActivationPool] 成功预留虚拟地址空间（VMM）: {self.pool_size/1024**3:.2f}GB")
        logger.info(f"[ActivationPool] 物理显存占用: 0 GB（按需映射）")
    
    def _ensure_mapped(self, required_bytes: int) -> None:
        """
        确保至少 required_bytes 的物理内存已被映射
        
        Args:
            required_bytes: 需要的字节数
        """
        if not self.use_vmm or self._vmm_allocator is None:
            return  # 非 VMM 模式，无需操作
        
        if required_bytes <= self._current_mapped_size:
            return  # 已经映射够了
        
        # 按 chunk 对齐新的映射大小
        new_mapped_size = ((required_bytes + self._vmm_chunk_size - 1) // self._vmm_chunk_size) * self._vmm_chunk_size
        new_mapped_size = min(new_mapped_size, self.pool_size)
        
        if new_mapped_size <= self._current_mapped_size:
            return
        
        # 映射新增的物理内存
        logger.debug(f"[ActivationPool-VMM] 动态映射显存: "
                    f"{self._current_mapped_size/1024**3:.3f}GB → {new_mapped_size/1024**3:.3f}GB "
                    f"(增加 {(new_mapped_size - self._current_mapped_size)/1024**2:.1f}MB)")
        
        self._vmm_allocator.map_physical(0, new_mapped_size)
        self._current_mapped_size = new_mapped_size
        
        # 获取统计信息
        stats = self._vmm_allocator.get_stats()
        logger.info(f"[ActivationPool-VMM] 当前状态: "
                   f"预留={stats['reserved_size_mb']:.0f}MB, "
                   f"映射={stats['mapped_size_mb']:.0f}MB, "
                   f"利用率={stats['utilization']*100:.1f}%")
    
    def __del__(self):
        """清理资源"""
        if self.use_vmm and self._vmm_allocator is not None:
            try:
                self._vmm_allocator.free()
                logger.debug("[ActivationPool] VMM resources freed")
            except Exception as e:
                logger.warning(f"[ActivationPool] Error freeing VMM resources: {e}")






    # def _estimate_memory_per_token(self, model_runner_or_config: Any) -> None:
    #     """
    #     估算每个token在前向传播中的显存占用量
    #     包括attention和MLP层的中间激活，以及paged KV buffer
    #     """

    #     try:
    #         # 如果传入的是GPUModelRunner实例，直接使用已经计算好的参数
    #         if hasattr(model_runner_or_config, 'hidden_size') and hasattr(model_runner_or_config, 'num_query_heads'):
    #             # GPUModelRunner实例
    #             d_model = model_runner_or_config.hidden_size
    #             n_heads = model_runner_or_config.num_query_heads
                
    #             # 获取KV heads数量
    #             model_config = model_runner_or_config.model_config
    #             hf_config = getattr(model_config, 'hf_text_config', None)
    #             if hf_config is not None:
    #                 n_kv_heads = getattr(hf_config, 'num_key_value_heads', n_heads)
    #                 hidden_size = getattr(hf_config, 'intermediate_size', d_model * 4)
    #             else:
    #                 n_kv_heads = n_heads  # 假设GQA ratio为1
    #                 hidden_size = d_model * 4  # 标准的MLP ratio
                    
    #             logger.info(f"[ActivationPool] 从ModelRunner获取参数: d_model={d_model}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, hidden_size={hidden_size}")
                
    #         elif hasattr(model_runner_or_config, 'd_model'):
    #             # LLaDA ModelConfig
    #             d_model = model_runner_or_config.d_model
    #             n_heads = model_runner_or_config.n_heads
    #             n_kv_heads = getattr(model_runner_or_config, 'effective_n_kv_heads', n_heads)
    #             hidden_size = getattr(model_runner_or_config, 'mlp_hidden_size', None)
    #             if hidden_size is None:
    #                 mlp_ratio = getattr(model_runner_or_config, 'mlp_ratio', 4)
    #                 hidden_size = int(mlp_ratio * d_model)
                    
    #             logger.info(f"[ActivationPool] 从LLaDA配置获取参数: d_model={d_model}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, hidden_size={hidden_size}")
                
    #         else:
    #             # 尝试从hf_text_config获取
    #             hf_config = getattr(model_runner_or_config, 'hf_text_config', None)
    #             if hf_config is not None:
    #                 d_model = getattr(hf_config, 'hidden_size', 4096)
    #                 n_heads = getattr(hf_config, 'num_attention_heads', 32)
    #                 n_kv_heads = getattr(hf_config, 'num_key_value_heads', n_heads)
    #                 hidden_size = getattr(hf_config, 'intermediate_size', d_model * 4)
                    
    #                 logger.info(f"[ActivationPool] 从HF配置获取参数: d_model={d_model}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, hidden_size={hidden_size}")
    #             else:
    #                 raise ValueError("无法从输入中获取模型参数")
            
    #         head_dim = d_model // n_heads
            
    #         # 估算每个token的显存占用 (以字节为单位，假设使用fp16)
    #         bytes_per_element = 2  # fp16
            
    #         # 1. Attention部分的激活
    #         # - Q投影后的张量: n_heads * head_dim = d_model
    #         # - K投影后的张量: n_kv_heads * head_dim  
    #         # - V投影后的张量: n_kv_heads * head_dim
    #         # - Attention scores: n_heads * head_dim (after attention computation)
    #         # - Attention输出: d_model (after o_proj)
    #         q_memory = d_model * bytes_per_element  # Q: [seq_len, d_model]
    #         k_memory = n_kv_heads * head_dim * bytes_per_element  # K: [seq_len, n_kv_heads * head_dim]
    #         v_memory = n_kv_heads * head_dim * bytes_per_element  # V: [seq_len, n_kv_heads * head_dim]
    #         attn_output_memory = d_model * bytes_per_element  # attention output: [seq_len, d_model]
    #         attention_memory = q_memory + k_memory + v_memory + attn_output_memory
            
    #         # 2. MLP部分的激活  
    #         # - ff_proj输出: hidden_size
    #         # - up_proj输出: hidden_size
    #         # - 激活函数后: hidden_size
    #         # - ff_out输入: hidden_size
    #         mlp_memory = (hidden_size * 4) * bytes_per_element
            
    #         # 3. Paged KV buffer (每个token需要存储K和V)
    #         # 使用fp16存储，block_size=16
    #         block_size = 16
    #         kv_memory_per_token = (n_kv_heads * head_dim * 2) * bytes_per_element  # K + V
    #         # 考虑paging开销，向上取整到block边界
    #         blocks_needed = (1 + block_size - 1) // block_size
    #         paged_kv_memory = blocks_needed * block_size * kv_memory_per_token
            
    #         # 4. 其他中间张量 (normalization等)
    #         other_memory = d_model * bytes_per_element * 2  # 预留一些空间
            
    #         self.memory_per_token = attention_memory + mlp_memory + paged_kv_memory + other_memory
            
    #         logger.info(f"[ActivationPool] 估算每token显存占用: {self.memory_per_token}字节")
    #         logger.info(f"  - Attention激活: {attention_memory}字节 "
    #                    f"(Q={q_memory}, K={k_memory}, V={v_memory}, output={attn_output_memory})")
    #         logger.info(f"  - MLP激活: {mlp_memory}字节 (hidden_size={hidden_size})")
    #         logger.info(f"  - Paged KV buffer: {paged_kv_memory}字节 "
    #                    f"(n_kv_heads={n_kv_heads}, head_dim={head_dim})")
    #         logger.info(f"  - 其他张量: {other_memory}字节")
                       
    #     except Exception as e:
    #         # 如果估算失败，使用保守的默认值
    #         logger.warning(f"[ActivationPool] 显存估算失败，使用默认值: {e}")
    #         self.memory_per_token = 8192  # 8KB per token as fallback
    

    def _estimate_memory_per_token(self, model_runner_or_config: Any) -> None:
        """
        基于 graph_builder 的复用计划估算每个 token 的显存占用量。
        采用 N=1 生成复用计划，内含精确 dtype/shape 与别名复用关系。
        """
        from vllm.v1.worker.graph_builder import (
            define_llada_block_graph,  # type: ignore
            define_llada_block_graph_chunkwise,  # type: ignore
            MemoryAnalyzer,            # type: ignore
        )

        N = 100000
        # graph_blueprint = define_llada_block_graph(N)
        use_chunkwise = os.getenv("VLLM_USE_CHUNKWISE_GRAPH", "0") == "1"
        if use_chunkwise:
            graph_blueprint = define_llada_block_graph_chunkwise(N)
        else:
            graph_blueprint = define_llada_block_graph(N)
        analyzer = MemoryAnalyzer(graph_blueprint)
        analysis = analyzer.analyze()
        reuse_plan = analysis["reuse_plan"]
        total_bytes = int(reuse_plan["_total_buffer_size_bytes"]) # 峰值所需总字节

        # 对 DLLM（无 KV cache）而言，这里直接按复用计划计算的总大小除以 N 即可
        self.memory_per_token = max(1, total_bytes // N)

        logger.info(
            f"[ActivationPool] 基于graph_builder估算每token显存占用: "
            f"{self.memory_per_token} 字节 (N={N}, total={total_bytes} 字节)"
        )
    


    def calculate_max_num_tokens(self) -> int:
        """
        根据显存池大小计算最大token数量（适配varlen输入）
        """
        if not self._initialized or self.memory_per_token == 0:
            return 1
            
        max_num_tokens = max(1, self.pool_size // self.memory_per_token)
        
        logger.info(f"[ActivationPool] 最大token数量={max_num_tokens}")
        return max_num_tokens
    
    def allocate_for_tokens(self, num_tokens: int) -> torch.Tensor:
        """
        为指定数量的tokens分配显存（适配varlen输入）
        返回分配的显存张量，可以通过as_strided等方法重新解释为不同形状
        """
        if not self._initialized:
            raise RuntimeError("ActivationPool未初始化")

        required_memory = num_tokens * self.memory_per_token

        if required_memory > self.pool_size:
            raise RuntimeError(f"请求的显存({required_memory/1024**2:.2f}MB) "
                             f"超过池大小({self.pool_size/1024**2:.2f}MB)")

        # 分配从pool_buffer的开头开始的一段内存
        # self.allocated_size = required_memory
        allocated_buffer = self.pool_buffer[:required_memory]
        allocated_buffer = self.pool_buffer

        logger.debug(f"[ActivationPool] 为tokens分配显存: "
                    f"{required_memory/1024**2:.2f}MB "
                    f"(num_tokens={num_tokens})")

        return allocated_buffer

    def allocate_for_tokens_with_size(self, num_tokens: int, required_memory: int) -> torch.Tensor:
        """
        为指定数量的tokens分配指定大小的显存（新接口，支持reuse_plan）
        
        如果使用 VMM 模式，会自动按需映射物理内存
        """
        if not self._initialized:
            raise RuntimeError("ActivationPool未初始化")

        if required_memory > self.pool_size:
            raise RuntimeError(f"请求的显存({required_memory/1024**2:.2f}MB) "
                             f"超过池大小({self.pool_size/1024**2:.2f}MB)")

        # VMM 模式：确保需要的物理内存已映射
        if self.use_vmm:
            self._ensure_mapped(required_memory)

        # 分配从pool_buffer的开头开始的一段内存
        allocated_buffer = self.pool_buffer[:required_memory]

        logger.debug(f"[ActivationPool] 为tokens分配指定大小显存: "
                    f"{required_memory/1024**2:.2f}MB "
                    f"(num_tokens={num_tokens})")

        return allocated_buffer
    
    def get_tensor_view(self, allocated_buffer: torch.Tensor, 
                       shape: Tuple[int, ...], dtype: torch.dtype,
                       offset_bytes: int = 0) -> torch.Tensor:
        """
        从分配的buffer中创建指定形状和类型的tensor视图
        """
        element_size = torch.tensor([], dtype=dtype).element_size()
        total_elements = torch.prod(torch.tensor(shape)).item()
        required_bytes = total_elements * element_size
        
        if offset_bytes + required_bytes > len(allocated_buffer):
            raise RuntimeError(f"Tensor视图超出分配的显存范围")
        
        # 创建视图
        buffer_view = allocated_buffer[offset_bytes:offset_bytes + required_bytes]
        return buffer_view.view(dtype).reshape(shape)
    
    def deallocate(self) -> None:
        """
        释放当前分配的显存
        """
        self.allocated_size = 0
        # pool_buffer保持不变，可以继续复用
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取显存池统计信息
        """
        stats = {
            "pool_size_mb": self.pool_size / 1024**2,
            "allocated_size_mb": self.allocated_size / 1024**2,
            "memory_per_token": self.memory_per_token,
            "utilization": self.allocated_size / self.pool_size if self.pool_size > 0 else 0,
            "initialized": self._initialized,
            "use_vmm": self.use_vmm,
        }
        
        # VMM 特有的统计信息
        if self.use_vmm and self._vmm_allocator is not None:
            vmm_stats = self._vmm_allocator.get_stats()
            stats.update({
                "vmm_reserved_mb": vmm_stats['reserved_size_mb'],
                "vmm_mapped_mb": vmm_stats['mapped_size_mb'],
                "vmm_utilization": vmm_stats['utilization'],
                "vmm_saved_mb": vmm_stats['reserved_size_mb'] - vmm_stats['mapped_size_mb'],
            })
        
        return stats

"""
修改版 vLLM RoPE - 使用 bfloat16 精度与 3D 路径保持一致
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple


class DreamVLLMRotaryEmbedding(nn.Module):
    """基于 vLLM RotaryEmbedding，直接使用 bfloat16 精度与 3D 路径对齐"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        # print(f'DreamVLLMRotaryEmbedding dtype: {dtype}, device: {device}')
        cache = self._compute_cos_sin_cache()
        # ✅ 修复：直接使用传入的 dtype (bfloat16)，与 3D 路径保持一致
        # 避免 float32 → bfloat16 的转换损失
        if device is not None:
            cache = cache.to(device=device, dtype=dtype)
        else:
            cache = cache.to(dtype=dtype)
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        """Compute the inverse frequency."""
        inv_freq = 1.0 / (base**(torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播 - 直接使用 bfloat16 精度计算 RoPE，与 3D 路径保持一致
        
        Args:
            positions: [N] position indices
            query: [N, num_heads, head_size]
            key: [N, num_kv_heads, head_size]
        """
        from vllm import _custom_ops as ops

        # # 打印调试信息（仅第一次）
        # if not hasattr(self, '_debug_printed'):
        #     print(f"[DEBUG DreamVLLMRoPE] cos_sin_cache.dtype = {self.cos_sin_cache.dtype}")
        #     print(f"[DEBUG DreamVLLMRoPE] query.dtype = {query.dtype}")
        #     print(f"[DEBUG DreamVLLMRoPE] cos_sin_cache.device = {self.cos_sin_cache.device}")
        #     self._debug_printed = True

        # ✅ 修复：确保 cache 在正确设备上，但不需要转换 dtype
        # cache 已经在 __init__ 时转换为与 query 相同的 dtype (bfloat16)
        if self.cos_sin_cache.device != query.device:
            self.cos_sin_cache.data = self.cos_sin_cache.to(device=query.device)

        # ✅ 直接使用 bfloat16 cache，避免 float32 → bfloat16 的转换损失
        # ops.rotary_embedding() 是 in-place 操作
        ops.rotary_embedding(positions, query, key, self.head_size,
                           self.cos_sin_cache, self.is_neox_style)
        return query, key


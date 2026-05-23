"""
LLaDA MoE Decoder Layer and Attention for vLLM.
Adapted from modeling_lladamoe.py with vLLM varlen attention support.
"""

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .llada_moe_mlp import LLaDAMoEMLP, LLaDAMoESparseMoeBlock

# vLLM varlen attention dependencies
from flash_attn import flash_attn_varlen_inplace


class LLaDAMoERMSNorm(nn.Module):
    """RMS Layer Normalization"""
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LLaDAMoERotaryEmbedding(nn.Module):
    """Rotary Position Embedding - placeholder, using vLLM's implementation"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        # We'll use vLLM's CUDA rotary embedding in practice


class LLaDAMoEAttention(nn.Module):
    """
    Multi-headed attention with vLLM varlen support.
    Adapted for bidirectional attention (is_causal=False) for diffusion models.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # Diffusion language model uses bidirectional attention
        self.is_causal = False
        
        # QKV projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        
        # Optional QK LayerNorm
        if config.qk_layernorm:
            self.q_norm = LLaDAMoERMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = LLaDAMoERMSNorm(self.head_dim, eps=config.rms_norm_eps)
        
        # vLLM RoPE（跨层共享 buffer，节省显存）
        if not hasattr(config, '_shared_rope_cache'):
            config._shared_rope_cache = {}
        
        cache_key = "shared_rotary_cuda"
        if cache_key in config._shared_rope_cache:
            self._rotary_cuda = config._shared_rope_cache[cache_key]
        else:
            try:
                from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding as VLLMRotaryEmbedding
                
                # [Fix] 强制扩大 RoPE 表大小以支持长序列测试 (原为 config.max_position_embeddings)
                max_pos = getattr(config, "max_position_embeddings", 81920)
                max_pos = max(max_pos, 262144)  # 扩大到 256k

                self._rotary_cuda = VLLMRotaryEmbedding(
                    head_size=self.head_dim,
                    rotary_dim=self.head_dim,
                    max_position_embeddings=max_pos,
                    base=config.rope_theta,
                    is_neox_style=True,
                    dtype=torch.float32,
                )
                # 强制使用 CUDA kernel，避免 forward_native 的内存开销
                self._rotary_cuda._forward_method = self._rotary_cuda.forward_cuda
                config._shared_rope_cache[cache_key] = self._rotary_cuda
            except Exception:
                self._rotary_cuda = None

    def forward(self, tensors: dict):
        """
        Args:
            tensors: 张量字典
        """
        # 从 tensors 中读取
        x_normed = tensors['x_normed']
        q = tensors['q']
        k = tensors['k']
        v = tensors['v']
        att_out_var = tensors['att_out_var']
        att_out = tensors['att_out']
        
        # QKV projections (原地写入)
        torch.mm(x_normed, self.q_proj.weight.t(), out=q)
        torch.mm(x_normed, self.k_proj.weight.t(), out=k)
        torch.mm(x_normed, self.v_proj.weight.t(), out=v)
        
        # Apply QK LayerNorm if configured
        if hasattr(self, 'q_norm') and hasattr(self, 'k_norm'):
            # 使用 vLLM 融合算子原地操作（避免 RMSNorm 内部的 float32 临时张量）
            from vllm import _custom_ops as ops
    
            # 关键：用 view 创建 [-1, head_dim] 的视图，然后原地 norm
            # view 不会 copy 数据，修改 view 会直接影响原始的 q/k
            q_view = q.view(-1, self.head_dim)
            k_view = k.view(-1, self.head_dim)
            
            # 原地 RMSNorm（修改 q_view/k_view，从而修改底层的 q/k）
            ops.rms_norm(q_view, q_view, self.q_norm.weight, self.q_norm.variance_epsilon)
            ops.rms_norm(k_view, k_view, self.k_norm.weight, self.k_norm.variance_epsilon)
            
            # 注意：q 和 k 的底层数据已经被修改，但它们的 shape 还是 [N, C]
            # 不需要再 reshape 回去，因为我们操作的是 view
            
            # # 原始实现（会在 RMSNorm 内部产生 float32 临时张量）
            # q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(q.size(0), -1)
            # k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(q.size(0), -1)
        
        N, C = q.size()
        num_heads = self.num_heads
        num_kv_heads = self.num_key_value_heads
        head_size = C // num_heads
        
        # [N, C] → [N, H, Dh]
        q_var = q.view(N, num_heads, head_size)
        k_var = k.view(N, num_kv_heads, head_size)
        v_var = v.view(N, num_kv_heads, head_size)
        
        # 读取 varlen 元数据
        from vllm.forward_context import get_forward_context
        fctx = get_forward_context()
        attn_meta = getattr(fctx, "attn_metadata", None)
        if isinstance(attn_meta, dict):
            meta = attn_meta.get(getattr(self, "layer_name", None), attn_meta)
        else:
            meta = attn_meta
        
        assert meta is not None, "Missing varlen attention metadata"
        query_start_loc = meta.query_start_loc
        seq_lens = meta.seq_lens
        B = int(seq_lens.numel())
        max_seq_len = int(seq_lens.max().item()) if B > 0 else 0
        
        # RoPE
        if self._rotary_cuda is not None and B > 0 and N > 0:
            meta_positions = getattr(meta, "positions", None)
            if meta_positions is not None:
                positions = meta_positions
                if positions.device != q.device:
                    positions = positions.to(q.device)
                if positions.dtype != torch.int64:
                    positions = positions.to(torch.int64)
                q_var, k_var = self._rotary_cuda(positions, q_var, k_var)
        
        # Flash attention (原地写入 att_out_var)
        att_out_var.set_(att_out_var.view(N, num_heads, head_size))
        flash_attn_varlen_inplace(
            q=q_var,
            k=k_var,
            v=v_var,
            cu_seqlens_q=query_start_loc,
            max_seqlen_q=max_seq_len,
            cu_seqlens_k=query_start_loc,
            max_seqlen_k=max_seq_len,
            softmax_scale=1.0 / math.sqrt(head_size),
            causal=False,
            alibi_slopes=None,
            softcap=0.0,
            out=att_out_var
        )
        
        # [N, H, Dh] → [N, C]
        att_out_var.set_(att_out_var.view(N, num_heads * head_size))
        
        # O projection (原地写入 att_out)
        torch.mm(att_out_var, self.o_proj.weight.t(), out=att_out)
        
        # ========== SDPA VERSION (COMMENTED OUT) ==========
        # # Convert 2D varlen [N, C] to 3D [B, T, C] for SDPA
        # from vllm.forward_context import get_forward_context
        # fctx = get_forward_context()
        # attn_meta = getattr(fctx, "attn_metadata", None)
        # if isinstance(attn_meta, dict):
        #     meta = attn_meta.get(getattr(self, "layer_name", None), attn_meta)
        # else:
        #     meta = attn_meta
        # 
        # assert meta is not None, "Missing attention metadata"
        # seq_lens = meta.seq_lens  # [B]
        # query_start_loc = meta.query_start_loc  # [B+1]
        # positions = getattr(meta, "positions", None)  # [N]
        # 
        # B = int(seq_lens.numel())
        # q_len = int(seq_lens[0].item()) if B > 0 else 0  # Assume all sequences have same length for now
        # 
        # # Reshape [N, C] -> [B, q_len, C]
        # hidden_states_3d = hidden_states.view(B, q_len, -1)
        # bsz, q_len, _ = hidden_states_3d.size()
        # 
        # # Project to Q, K, V
        # query_states = self.q_proj(hidden_states_3d)
        # key_states = self.k_proj(hidden_states_3d)
        # 
        # # Apply QK LayerNorm if configured
        # if hasattr(self, 'q_norm') and hasattr(self, 'k_norm'):
        #     query_states = self.q_norm(query_states.reshape(-1, self.head_dim)).reshape(bsz, q_len, -1)
        #     key_states = self.k_norm(key_states.reshape(-1, self.head_dim)).reshape(bsz, q_len, -1)
        # 
        # value_states = self.v_proj(hidden_states_3d)
        # 
        # # Clip QKV if configured
        # if self.config.clip_qkv is not None:
        #     query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
        #     key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
        #     value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
        # 
        # # Reshape: [B, T, C] -> [B, H, T, Dh]
        # query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # 
        # # Apply RoPE (using simple CPU version for now)
        # if positions is not None:
        #     # Compute cos/sin for RoPE
        #     device_type = hidden_states.device.type
        #     device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        #     
        #     # Simple RoPE implementation
        #     position_ids_3d = positions.view(bsz, q_len)
        #     
        #     # Create frequency tensor
        #     dim = self.head_dim
        #     inv_freq = 1.0 / (self.config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=hidden_states.device) / dim))
        #     
        #     # Compute embeddings
        #     with torch.autocast(device_type=device_type, enabled=False):
        #         inv_freq_expanded = inv_freq[None, :, None].float().expand(bsz, -1, 1)
        #         position_ids_expanded = position_ids_3d[:, None, :].float()
        #         freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        #         emb = torch.cat((freqs, freqs), dim=-1)
        #         cos = emb.cos()
        #         sin = emb.sin()
        #     
        #     cos = cos.to(dtype=query_states.dtype)
        #     sin = sin.to(dtype=query_states.dtype)
        #     
        #     # Apply rotary embedding
        #     def rotate_half(x):
        #         x1 = x[..., : x.shape[-1] // 2]
        #         x2 = x[..., x.shape[-1] // 2 :]
        #         return torch.cat((-x2, x1), dim=-1)
        #     
        #     def apply_rotary_pos_emb(q, k, cos, sin):
        #         cos = cos.unsqueeze(1)  # [B, 1, T, D]
        #         sin = sin.unsqueeze(1)
        #         q_embed = (q * cos) + (rotate_half(q) * sin)
        #         k_embed = (k * cos) + (rotate_half(k) * sin)
        #         return q_embed, k_embed
        #     
        #     query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # 
        # # Repeat KV for GQA
        # if self.num_key_value_groups > 1:
        #     key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        #     value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        # 
        # # SDPA with bidirectional attention (is_causal=False for diffusion models)
        # attn_output = torch.nn.functional.scaled_dot_product_attention(
        #     query_states,
        #     key_states,
        #     value_states,
        #     attn_mask=None,
        #     dropout_p=0.0,
        #     is_causal=False,
        # )
        # 
        # # Reshape back: [B, H, T, Dh] -> [B, T, C]
        # attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        # 
        # # Output projection
        # attn_output = self.o_proj(attn_output)
        # 
        # # Flatten back to [N, C]
        # attn_output = attn_output.view(-1, self.hidden_size)
        # 
        # return attn_output


class LLaDAMoEDecoderLayer(nn.Module):
    """
    Single decoder layer with attention and MoE/Dense FFN.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        # Determine if this layer uses MoE or dense FFN
        self.mlp_type = 'dense' if config.moe_layer_freq[layer_idx] == 0 else 'moe'
        
        # Attention
        self.self_attn = LLaDAMoEAttention(config, layer_idx=layer_idx)
        
        # FFN: either MoE or Dense
        if self.mlp_type == 'moe':
            self.mlp = LLaDAMoESparseMoeBlock(config)
        else:
            self.mlp = LLaDAMoEMLP(config, 'dense')
        
        # Layer norms
        self.input_layernorm = LLaDAMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LLaDAMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Optional shared expert
        if config.shared_expert_intermediate_size is not None and self.mlp_type == 'moe':
            self.shared_expert = LLaDAMoEMLP(config, 'shared_expert')

    def forward(self, tensors: dict):
        """
        Args:
            tensors: 张量字典，包含 'x' 等 key
        """
        from vllm import _custom_ops as ops
        
        # 传递 chunk 策略（如果存在）
        # 这样 MLP/MoE 层可以访问策略信息
        
        # 1. Input RMS norm
        x = tensors['x']
        x_normed = tensors['x_normed']
        ops.rms_norm(x_normed, x, self.input_layernorm.weight, self.input_layernorm.variance_epsilon)
        
        # 2. Attention (写入 att_out)
        self.self_attn(tensors)
        
        # 3. 第一个残差连接 + post_attention_layernorm (fused)
        att_out = tensors['att_out']
        x_normed_2 = tensors['x_normed_2']
        residual = tensors['residual']
        ops.fused_add_rms_norm_out(
            x_normed_2,
            residual,
            x,
            att_out,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.variance_epsilon
        )
        
        # 4. MoE (写入 moe_output)
        self.mlp(tensors)
        
        # 5. 第二个残差连接
        moe_output = tensors['moe_output']
        final_output = tensors['final_output']
        torch.add(residual, moe_output, out=final_output)
        
        # 6. 复用：final_output -> x (下一层的输入)
        tensors['x'] = final_output


class LLaDAMoEBlockGroup(nn.ModuleList):
    """Group of decoder layers for block-wise processing"""
    def __init__(self, config, layer_offset: int, modules: Optional[list] = None):
        super().__init__(modules)
        self.config = config
        self.layer_offset = layer_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block_idx, block in enumerate(self):
            x = block(x)
        return x

    def reset_parameters(self):
        for block in self:
            # Reset parameters if needed
            pass


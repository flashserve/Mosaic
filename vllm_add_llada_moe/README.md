# vLLM Add LLaDA-MoE

LLaDA-MoE模型的vLLM优化实现，支持varlen attention和高效推理。

## 📁 文件结构

```
vllm_add_llada_moe/
├── __init__.py                      # 模块入口
├── llada_moe_mlp.py                 # MLP和MoE核心实现
│   ├── LLaDAMoEMLP                  # 单个MLP（支持dense/expert/shared）
│   ├── LLaDAMoESparseMoeBlock       # Sparse MoE模块
│   └── load_balancing_loss_func     # 负载均衡损失
├── llada_moe_blocks.py              # Decoder Layer和Attention
│   ├── LLaDAMoERMSNorm              # RMS归一化
│   ├── LLaDAMoEAttention            # 注意力层（varlen优化）
│   ├── LLaDAMoEDecoderLayer         # 单层Decoder
│   └── LLaDAMoEBlockGroup           # Decoder层组
├── llada_moe_model.py               # 主模型
│   ├── LLaDAMoEModel                # 核心Transformer模型
│   └── LLaDAMoEOutput               # 输出数据结构
├── llada_moe_vllm.py                # HF兼容包装
│   └── LLaDAMoEModelLM              # PreTrainedModel包装
└── llada_moe_components.py          # 工具函数（复用llada）
```

## 🔑 核心特性

### 1. Sparse MoE架构
- 64个专家，每个token激活8个专家
- Expert intermediate_size: 1024
- Dense intermediate_size: 8192
- 支持shared expert（可选）

### 2. vLLM优化
- ✅ Varlen attention支持（flash attention）
- ✅ 推理优化（无gradient checkpointing）
- ✅ CUDA RoPE (vLLM原生实现)
- ✅ 双向注意力（diffusion models）

### 3. 与LLaDA的主要差异
- **FFN层**: MoE替代标准FFN
- **Router**: 动态专家选择
- **Aux Loss**: 负载均衡损失
- **更统一**: 只有一个DecoderLayer实现（vs LLaDA的两个Block类型）

## 🚀 使用方法

### 基础推理

```python
from transformers import AutoConfig, AutoTokenizer
from vllm_add_llada_moe import LLaDAMoEModelLM

# 加载模型
config = AutoConfig.from_pretrained("llada-moe-7b-a1b", trust_remote_code=True)
model = LLaDAMoEModelLM(config)
model.load_weights("llada-moe-7b-a1b")

# 推理
tokenizer = AutoTokenizer.from_pretrained("llada-moe-7b-a1b")
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)
```

### 计算Aux Loss

```python
# 训练时启用router logits输出
outputs = model(**inputs, output_router_logits=True)
aux_loss = outputs.aux_loss  # 负载均衡损失
```

## 📊 模型配置

LLaDA-MoE-7B-A1B关键参数：
```python
{
    "hidden_size": 2048,
    "num_hidden_layers": 16,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    
    # MoE配置
    "num_experts": 64,
    "num_experts_per_tok": 8,
    "expert_intermediate_size": 1024,
    "dense_intermediate_size": 8192,
    "moe_layer_freq": [1, 1, ..., 1],  # 每层都用MoE
    "router_aux_loss_coef": 0.01,
    
    # RoPE配置
    "rope_theta": 50000,
    "max_position_embeddings": 8192,
    
    # 其他
    "qk_layernorm": true,
    "hidden_act": "silu",
}
```

## 🔧 相比vllm_add_llada的变化

### 新增文件
- `llada_moe_mlp.py` - MoE专家和Router实现（~250行）
- 核心MoE逻辑在 `LLaDAMoESparseMoeBlock`（~60行）

### 修改文件
- `llada_moe_blocks.py` - 统一的DecoderLayer（更简洁）
- `llada_moe_model.py` - 支持router_logits输出
- `llada_moe_vllm.py` - 添加aux_loss计算

### 复用文件
- 可以复用 `llada_components.py`
- 可以复用 `llada_norms_activations.py`（或使用MoE自带的RMSNorm）
- 可以复用 `llada_rotary_bias.py`（如果需要）

## ⚠️ 注意事项

1. **双向注意力**: LLaDA-MoE是diffusion language model，使用双向注意力（is_causal=False）
2. **无KV Cache**: 不支持past_key_values
3. **Router Logits**: 训练时需要启用output_router_logits=True来计算aux loss
4. **内存占用**: 64个专家会占用较多显存，但推理时只激活8个

## 🧪 测试

先按仓库根目录 `README.md` 编译自定义算子，然后启动 LLaDA-MoE 服务并发送
一次 smoke request：

```bash
cd "$MOSAIC_HOME"
CUDA_VISIBLE_DEVICES=0 scripts/run_llada_moe_server.sh
```

另开一个终端：

```bash
python scripts/test_api_request.py \
    --model-family llada-moe \
    --host http://127.0.0.1:10001 \
    --steps 8 \
    --gen-length 32 \
    --block-length 32
```

## 📝 已知限制

- Activation memory manager、paged attention、expert parallelism 仍是后续优化项。
- 当前 release 目标是推理 smoke path 和 benchmark entry point 可复现。
- 如果需要训练或 router auxiliary loss，请先补齐相应验证用例。


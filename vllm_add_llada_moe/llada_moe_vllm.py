import os
import sys
import importlib
import importlib.util
import types
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Optional
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.model_executor.model_loader.weight_utils import (
    safetensors_weights_iterator,
    pt_weights_iterator,
)

# 确保能在独立子进程中导入到 llada-moe 的实现（兄弟目录）
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
_MODEL_ROOT = os.environ.get("MOSAIC_MODEL_ROOT",
                             os.path.join(_REPO_ROOT, "models"))
_LLADA_MOE_DIR = os.environ.get(
    "LLADA_MOE_MODEL_DIR",
    os.path.join(_MODEL_ROOT, "llada-moe-7b-a1b"),
)
if _LLADA_MOE_DIR not in sys.path and os.path.isdir(_LLADA_MOE_DIR):
    sys.path.insert(0, _LLADA_MOE_DIR)


def _load_model_module(package_name: str, module_name: str, model_dir: str):
    module_path = os.path.join(model_dir, f"{module_name}.py")
    if not os.path.exists(module_path):
        return importlib.import_module(module_name)

    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [model_dir]
        sys.modules[package_name] = package

    full_name = f"{package_name}.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    spec = importlib.util.spec_from_file_location(full_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


_llada_moe_config_module = _load_model_module(
    "mosaic_llada_moe_hf", "configuration_lladamoe", _LLADA_MOE_DIR)
LLaDAConfig = _llada_moe_config_module.LLaDAConfig
from vllm_add_llada_moe.llada_moe_model import LLaDAMoEModel, create_model_config_from_pretrained_config


class LLaDAMoEModelLM(nn.Module):
    """
    vLLM 插件模型类：名称与 HF architectures 对齐，便于 vLLM 解析。
    仅做整段前向（prefill），不使用 KVCache/Decode。
    支持 MoE 架构的 LLaDA 模型。
    """

    # 可选：HF -> 本模型参数名的映射器，若无需映射则保持为 None
    # HF 权重常见以 "model." 为顶层前缀，而内部 self.model 通常无该前缀
    hf_to_vllm_mapper: Optional[WeightsMapper] = None

    def named_parameters(self, *args, **kwargs):
        # 让 DefaultModelLoader 在顶层模块上看到所有参数名
        # 对于lm_head，HF权重是顶层的，不带model.前缀
        for n, p in self.model.named_parameters(*args, **kwargs):
            if n.startswith('lm_head'):
                yield (n, p)  # lm_head.* 保持顶层
            else:
                yield (f"model.{n}", p)  # 其他参数加model.前缀

    def parameters(self, *args, **kwargs):
        yield from self.model.parameters(*args, **kwargs)

    def __init__(self, vllm_config, prefix: str = ""):
        super().__init__()
        # vLLM 会将 --model 路径注入到此配置
        self.model_path = vllm_config.model_config.model

        # 读取 HF 配置，构建内部 LLaDA-MoE 模型（保持 init_params=False 的懒初始化策略）
        hf_cfg = LLaDAConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        
        # 保存配置用于后续使用
        self.config = hf_cfg
        
        # 直接在当前 CUDA 设备上初始化参数存储，避免 meta tensor
        # 确定目标设备
        try:
            device = f"cuda:{torch.cuda.current_device()}"
        except Exception:
            device = "cuda"
        
        # 在GPU上直接创建模型，避免CPU->GPU的昂贵拷贝
        with torch.device(device):
            self.model = LLaDAMoEModel(hf_cfg, init_params=False)
        
        # 显式添加lm_head引用，使AutoWeightsLoader能找到它
        # 这样loader在加载"lm_head.weight"时能通过getattr(self, 'lm_head')找到模块
        self.lm_head = self.model.lm_head

        # 懒加载权重
        self._loaded = False
        # 标记是否由 vLLM 默认权重加载器加载过（Engine 路径）。
        self._loaded_by_vllm = False
        self.dtype = (
            torch.bfloat16 if str(getattr(hf_cfg, "torch_dtype", "")).endswith("bfloat16") 
            else torch.float16
        )

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """
        生成专家权重映射，用于权重加载
        
        返回格式: [(param_name, weight_name, expert_id, shard_id), ...]
        """
        mappings = []
        num_layers = len(self.model.layers)
        num_experts = self.config.num_experts
        
        for layer_idx in range(num_layers):
            for expert_id in range(num_experts):
                # gate_proj -> w13_weight (前半部分)
                mappings.append((
                    f"model.layers.{layer_idx}.mlp.experts.w13_",
                    f"model.layers.{layer_idx}.mlp.experts.{expert_id}.gate_proj.",
                    expert_id,
                    "w1"
                ))
                # up_proj -> w13_weight (后半部分)
                mappings.append((
                    f"model.layers.{layer_idx}.mlp.experts.w13_",
                    f"model.layers.{layer_idx}.mlp.experts.{expert_id}.up_proj.",
                    expert_id,
                    "w3"
                ))
                # down_proj -> w2_weight
                mappings.append((
                    f"model.layers.{layer_idx}.mlp.experts.w2_",
                    f"model.layers.{layer_idx}.mlp.experts.{expert_id}.down_proj.",
                    expert_id,
                    "w2"
                ))
        
        return mappings
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """使用 vLLM 原生加载器将权重灌入到 self.model。"""
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        
        for name, loaded_weight in weights:
            # 检查是否是专家权重
            expert_found = False
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                
                # 找到匹配的专家权重
                # 根据 shard_id 确定是 w13_weight 还是 w2_weight
                if shard_id in ["w1", "w3"]:
                    full_param_name = f"{param_name}weight"
                else:  # w2
                    full_param_name = f"{param_name}weight"
                
                if full_param_name not in params_dict:
                    continue
                
                param = params_dict[full_param_name]
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is not None:
                    weight_loader(param, loaded_weight, name)
                    # 关键：加载多个 checkpoint 权重到同一个参数，只记录一次参数名
                    loaded_params.add(full_param_name)
                    expert_found = True
                    break
            
            if expert_found:
                continue
            
            # 非专家权重，使用默认加载
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", lambda p, w: p.data.copy_(w))
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
        
        self.model.eval()
        self._loaded_by_vllm = True
        self._loaded = True
        return loaded_params

    def _lazy_load_weights(self) -> None:
        """懒加载模型权重（当直接使用时，非 vLLM Engine 路径）"""
        if self._loaded:
            return
        # 使用 vLLM 原生迭代器 + AutoWeightsLoader 灌权重
        import glob

        hf_folder = self.model_path
        st_files = sorted(glob.glob(os.path.join(hf_folder, "*.safetensors")))
        pt_files = (
            sorted(glob.glob(os.path.join(hf_folder, "*.bin")))
            + sorted(glob.glob(os.path.join(hf_folder, "*.pt")))
        )

        # 直接加载到GPU，避免CPU->GPU拷贝
        device = torch.device("cuda", torch.cuda.current_device())
        
        if st_files:
            weights_iter = safetensors_weights_iterator(
                st_files, use_tqdm_on_load=True
            )
        elif pt_files:
            # pt文件直接加载到GPU
            weights_iter = pt_weights_iterator(
                pt_files, use_tqdm_on_load=True, pt_load_map_location=device
            )
        else:
            raise FileNotFoundError(
                f"No weights found in: {hf_folder}. Expect *.safetensors or *.bin/*.pt"
            )

        # 预取一个权重用于对齐 dtype，并构造链式迭代器
        weights_iter = iter(weights_iter)
        try:
            first_name, first_tensor = next(weights_iter)
        except StopIteration:
            raise ValueError("Empty weights iterator.")

        # 模型已经在GPU上初始化了，无需再次移动
        # 只需对齐 dtype 到首个权重的 dtype
        try:
            self.model.to(dtype=first_tensor.dtype)
        except Exception:
            pass

        def _chained_iter():
            yield (first_name, first_tensor)
            for item in weights_iter:
                yield item

        loaded_names = self.load_weights(_chained_iter()) or set()

        # 覆盖率校验：确认所有参数均已被加载（将返回集合去掉 'model.' 前缀再比较）
        def _strip_prefix(n: str) -> str:
            return n[6:] if n.startswith("model.") else n

        expected_names = {name for name, _ in self.model.named_parameters()}
        normalized_loaded = {_strip_prefix(n) for n in loaded_names}
        missing = expected_names - normalized_loaded
        if missing:
            sample = sorted(list(missing))[:20]
            raise ValueError(
                f"缺失权重 {len(missing)}/{len(expected_names)}，示例: {sample}"
            )

        # 将最终 dtype 统一到 self.dtype（若与权重 dtype 不同）
        if first_tensor.dtype != self.dtype:
            try:
                self.model.to(dtype=self.dtype)
            except Exception:
                pass

        self.model.eval()
        self._loaded = True

    def forward(
        self,
        input_ids=None,
        positions=None,  # 忽略
        kv_caches=None,  # 忽略
        attn_metadata=None,  # 忽略
        inputs_embeds=None,
        attention_mask=None,
        **kwargs,
    ):
        """
        vLLM forward 接口。
        
        Args:
            input_ids: 输入token IDs，可能是1D或2D
            positions: 位置索引（忽略，RoPE在attention中处理）
            kv_caches: KV缓存（忽略，diffusion模型不支持）
            attn_metadata: 注意力元数据（由forward_context传递）
            inputs_embeds: 可选的输入embeddings
            attention_mask: 注意力mask（忽略，双向注意力）
        
        Returns:
            模型输出，包含logits或hidden states
        """
        # 由 v1 默认加载器负责灌权重；这里不做懒加载，避免重复加载与覆盖率校验冲突。
        
        out = self.model.forward(
            input_ids=input_ids,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=positions,
            output_hidden_states=True,
            output_router_logits=False,  # 推理时不需要router logits
            last_logits_only=False,
        )

        return out

    def compute_logits(self, hidden_states, sampling_metadata):
        """
        将 hidden states 映射到 logits。
        为满足 vLLM 的"生成型模型"判定（interfaces_base.VllmModelForTextGeneration），
        提供 compute_logits 签名。
        
        Args:
            hidden_states: 模型输出的hidden states [N, hidden_size]
            sampling_metadata: 采样元数据
        
        Returns:
            logits: [N, vocab_size]
        """
        # Map hidden states to logits using model's lm_head
        logits = self.model.lm_head(hidden_states)
        
        # Apply scaling if configured
        if getattr(self.config, "scale_logits", False):
            logits = logits * (1 / math.sqrt(self.config.hidden_size))
        
        return logits

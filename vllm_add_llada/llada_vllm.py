import os
import sys
import importlib
import importlib.util
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModel
from typing import Iterable, Optional
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.model_executor.model_loader.weight_utils import (
    safetensors_weights_iterator,
    pt_weights_iterator,
)

# 确保能在独立子进程中导入到 llada 的实现（兄弟目录）
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
_MODEL_ROOT = os.environ.get("MOSAIC_MODEL_ROOT",
                             os.path.join(_REPO_ROOT, "models"))
_LLADA_DIR = os.environ.get(
    "LLADA_MODEL_DIR",
    os.path.join(_MODEL_ROOT, "llada-8b-instruct"),
)
if _LLADA_DIR not in sys.path and os.path.isdir(_LLADA_DIR):
    sys.path.insert(0, _LLADA_DIR)


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


_llada_config_module = _load_model_module(
    "mosaic_llada_hf", "configuration_llada", _LLADA_DIR)
_llada_modeling_module = _load_model_module(
    "mosaic_llada_hf", "modeling_llada", _LLADA_DIR)
LLaDAConfig = _llada_config_module.LLaDAConfig
create_model_config_from_pretrained_config = (
    _llada_modeling_module.create_model_config_from_pretrained_config
)
from vllm_add_llada.llada_model import LLaDAModel














class LLaDAModelLM(nn.Module):
    def named_parameters(self, *args, **kwargs):
        # 让 DefaultModelLoader 在顶层模块上看到所有参数名
        yield from ((f"model.{n}", p) for n, p in self.model.named_parameters(*args, **kwargs))

    def parameters(self, *args, **kwargs):
        yield from self.model.parameters(*args, **kwargs)
    """
    vLLM 插件模型类：名称与 HF architectures 对齐，便于 vLLM 解析。
    仅做整段前向（prefill），不使用 KVCache/Decode。
    """

    # 可选：HF -> 本模型参数名的映射器，若无需映射则保持为 None
    # HF 权重常见以 "model." 为顶层前缀，而内部 self.model 通常无该前缀
    hf_to_vllm_mapper: Optional[WeightsMapper] = None

    def __init__(self, vllm_config, prefix: str = ""):
        super().__init__()
        # vLLM 会将 --model 路径注入到此配置
        self.model_path = vllm_config.model_config.model

        # 读取 HF 配置，构建内部 LLaDA 模型（保持 init_params=False 的懒初始化策略）
        hf_cfg = LLaDAConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model_cfg = create_model_config_from_pretrained_config(hf_cfg)
        requested_max_len = int(
            getattr(vllm_config.model_config, "max_model_len", 0) or 0)
        if requested_max_len > getattr(model_cfg, "max_sequence_length", 0):
            model_cfg.max_sequence_length = requested_max_len
        # 直接在当前 CUDA 设备上初始化参数存储，避免 meta tensor
        try:
            model_cfg.init_device = f"cuda:{torch.cuda.current_device()}"
        except Exception:
            model_cfg.init_device = "cuda"
        # 初始化真实参数（设 init_params=False，避免 reset_parameters 覆盖）
        self.model = LLaDAModel(model_cfg, init_params=False)

        # 懒加载权重
        self._loaded = False
        # 标记是否由 vLLM 默认权重加载器加载过（Engine 路径）。
        self._loaded_by_vllm = False
        self.dtype = (
            torch.bfloat16 if str(getattr(hf_cfg, "precision", "")).endswith("bf16") else torch.float16
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """使用 vLLM 原生加载器将权重灌入到 self.model。

        参数
        - weights: 形如 (name, tensor) 的可迭代对象，可来自 safetensors/pt 等迭代器。

        返回
        - 已成功加载的参数全名集合（set[str]）。
        """
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        self.model.eval()
        return loaded

    def _lazy_load_weights(self) -> None:
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

        if st_files:
            weights_iter = safetensors_weights_iterator(
                st_files, use_tqdm_on_load=True
            )
        elif pt_files:
            weights_iter = pt_weights_iterator(
                pt_files, use_tqdm_on_load=True, pt_load_map_location="cpu"
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

        # 确保模型已在当前 CUDA 设备
        device = torch.device("cuda", torch.cuda.current_device())
        self.model.to(device=device)
        # 对齐参数 dtype 到首个权重的 dtype，避免 copy_ 因 dtype 不一致报错
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
        # 由 v1 默认加载器负责灌权重；这里不做懒加载，避免重复加载与覆盖率校验冲突。
        # 形状/类型适配：vLLM 在 profile/graph 捕获与常规 prefill 路径下可能传入 1D tokens。
        # 统一补 batch 维，并确保 embedding 索引为 int64。
        # if input_ids is not None and isinstance(input_ids, torch.Tensor):
        #     if input_ids.dim() == 1:
        #         input_ids = input_ids.unsqueeze(0)
        #     if input_ids.dtype != torch.long:
        #         input_ids = input_ids.to(dtype=torch.long)
        # if inputs_embeds is not None and isinstance(inputs_embeds, torch.Tensor) and inputs_embeds.dim() == 2:
        #     inputs_embeds = inputs_embeds.unsqueeze(0)
        # if attention_mask is not None and isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 1:
        #     attention_mask = attention_mask.unsqueeze(0)
        out = self.model.forward(
            input_ids=input_ids,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=None,
            past_key_values=None,
            use_cache=False,
            last_logits_only=False,
            output_hidden_states=True,
        )
        # v1 runner expects 2D hidden states [num_tokens, hidden_size]
        # hidden_states = out.hidden_states[-1]
        # hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        # return hidden_states
        return out

    # 为满足 vLLM 的“生成型模型”判定（interfaces_base.VllmModelForTextGeneration），
    # 提供 compute_logits 签名。扩散模式不会调用此方法，返回 None 即可。
    def compute_logits(self, hidden_states, sampling_metadata):
        # Map hidden states to logits following model config
        if getattr(self.model.config, "weight_tying", False):
            logits = F.linear(hidden_states, self.model.transformer.wte.weight, None)
        else:
            logits = self.model.transformer.ff_out(hidden_states)
        if getattr(self.model.config, "scale_logits", False):
            logits = logits * (1 / math.sqrt(self.model.config.d_model))
        return logits



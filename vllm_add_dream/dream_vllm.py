"""
vLLM 薄封装层 - 权重加载与前向传递
"""
import os
import sys
import importlib
import importlib.util
import types
from typing import Iterable, Optional
import glob

import torch
import torch.nn as nn
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.model_executor.model_loader.weight_utils import (
    safetensors_weights_iterator,
    pt_weights_iterator,
)

# 确保能在独立子进程中导入到 dream 的实现（兄弟目录）
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
_MODEL_ROOT = os.environ.get("MOSAIC_MODEL_ROOT",
                             os.path.join(_REPO_ROOT, "models"))
_DREAM_DIR = os.environ.get(
    "DREAM_MODEL_DIR",
    os.path.join(_MODEL_ROOT, "dream-v0-instruct-7b"),
)
if _DREAM_DIR not in sys.path and os.path.isdir(_DREAM_DIR):
    sys.path.insert(0, _DREAM_DIR)


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


_dream_config_module = _load_model_module(
    "mosaic_dream_hf", "configuration_dream", _DREAM_DIR)
DreamConfig = _dream_config_module.DreamConfig
from vllm_add_dream.dream_model import DreamModel as DreamModelInternal


class DreamModel(nn.Module):
    """
    vLLM 插件模型类：名称与 HF architectures 对齐（DreamModel）。
    """
    
    # 明确指定这不是 pooling 模型，防止被 vLLM 自动转换
    is_pooling_model = False

    def named_parameters(self, *args, **kwargs):
        # 把所有参数都加上 "model." 前缀，然后用 mapper 处理 HF 权重名映射
        yield from ((f"model.{n}", p) for n, p in self.model.named_parameters(*args, **kwargs))

    def parameters(self, *args, **kwargs):
        yield from self.model.parameters(*args, **kwargs)

    # WeightsMapper: HF 的 lm_head 权重不带 model. 前缀，但其他都带
    # 我们统一加了 model. 前缀，所以需要把 HF 的 lm_head 也加上 model. 前缀来匹配
    hf_to_vllm_mapper: Optional[WeightsMapper] = WeightsMapper(
        orig_to_new_prefix={
            "lm_head.": "model.lm_head.",  # lm_head.weight -> model.lm_head.weight
        }
    )

    def __init__(self, vllm_config, prefix: str = ""):
        super().__init__()
        # vLLM 会将 --model 路径注入到此配置
        self.model_path = vllm_config.model_config.model

        # 读取 HF 配置，构建内部 Dream 模型
        hf_cfg = DreamConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        # 补丁：如果有 vLLM config 里的 max_model_len 覆盖，且比默认配置大，则强制更新 hf_cfg
        # 否则 RoPE cache 只按 config.json 初始化 (307200)，遇到 >300k 的请求会越界崩溃
        if hasattr(vllm_config, 'model_config') and vllm_config.model_config.max_model_len is not None:
            if vllm_config.model_config.max_model_len > hf_cfg.max_position_embeddings:
                print(f"[DreamModel] Overriding max_position_embeddings: {hf_cfg.max_position_embeddings} -> {vllm_config.model_config.max_model_len}")
                hf_cfg.max_position_embeddings = vllm_config.model_config.max_model_len
        
        # 直接在当前 CUDA 设备上初始化参数存储，避免 meta tensor（和 llada 一样）
        try:
            device_str = f"cuda:{torch.cuda.current_device()}"
        except Exception:
            device_str = "cuda"

        # 初始化内部模型（init_params=False，权重由后续加载；直接在目标设备创建）
        self.model = DreamModelInternal(hf_cfg, init_params=False, init_device=device_str)

        # 懒加载权重
        self._loaded = False
        # 判断 dtype
        if hasattr(hf_cfg, 'torch_dtype'):
            self.dtype = torch.bfloat16 if str(hf_cfg.torch_dtype) == "torch.bfloat16" else torch.float16
        else:
            self.dtype = torch.bfloat16  # 默认使用 bf16

    @torch._dynamo.disable
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """使用 vLLM 原生加载器将权重灌入到 self（顶层模块）。
        
        参数
        - weights: 形如 (name, tensor) 的可迭代对象，可来自 safetensors/pt 等迭代器。
        
        返回
        - 已成功加载的参数全名集合（set[str]）。
        """
        # # 检查并打印 rotary_emb inv_freq 的初始 dtype（用于 debug）
        # print(f"[DEBUG load_weights] 调用前 Layer 0 inv_freq.dtype = {self.model.layers[0].self_attn.rotary_emb.inv_freq.dtype}")
        
        # 加载权重
        loader = AutoWeightsLoader(self)
        loaded = set(loader.load_weights(weights, mapper=self.hf_to_vllm_mapper))
        
        # print(f"[DEBUG load_weights] 调用后 Layer 0 inv_freq.dtype = {self.model.layers[0].self_attn.rotary_emb.inv_freq.dtype}")
        
        self.model.eval()
        return loaded

    @torch._dynamo.disable
    def _lazy_load_weights(self) -> None:
        if self._loaded:
            return
        # 使用 vLLM 原生迭代器 + AutoWeightsLoader 灌权重
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

        # 重要：在 .to(dtype) 前先保存 RoPE 的 inv_freq，避免精度损失
        # 因为 .to(dtype) 会把 buffer 也转成 bf16/fp16，导致 inv_freq 精度丢失
        saved_inv_freqs = []
        if hasattr(self.model, 'rotary_emb') and hasattr(self.model.rotary_emb, 'inv_freq'):
            saved_inv_freqs.append(('model', self.model.rotary_emb.inv_freq.clone()))
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'rotary_emb'):
                if hasattr(layer.self_attn.rotary_emb, 'inv_freq'):
                    saved_inv_freqs.append((i, layer.self_attn.rotary_emb.inv_freq.clone()))
        
        # 对齐参数 dtype 到首个权重的 dtype，避免 copy_ 因 dtype 不一致报错
        # 模型已经在目标设备上了，不需要再 .to(device)
        try:
            self.model.to(dtype=first_tensor.dtype)
        except Exception:
            pass
        
        # 恢复 inv_freq 的原始 float32 值（避免 bf16 精度损失）
        for key, inv_freq_val in saved_inv_freqs:
            if key == 'model':
                self.model.rotary_emb.inv_freq = inv_freq_val
                self.model.rotary_emb.original_inv_freq = inv_freq_val
            else:
                self.model.layers[key].self_attn.rotary_emb.inv_freq = inv_freq_val
                self.model.layers[key].self_attn.rotary_emb.original_inv_freq = inv_freq_val

        def _chained_iter():
            yield (first_name, first_tensor)
            for item in weights_iter:
                yield item

        loaded_names = self.load_weights(_chained_iter()) or set()

        # 覆盖率校验：确认所有参数均已被加载
        expected_names = {name for name, _ in self.named_parameters()}
        missing = expected_names - loaded_names
        if missing:
            sample = sorted(list(missing))[:20]
            raise ValueError(
                f"缺失权重 {len(missing)}/{len(expected_names)}，示例: {sample}"
            )

        # 将最终 dtype 统一到 self.dtype（若与权重 dtype 不同）
        if first_tensor.dtype != self.dtype:
            # 再次保存 inv_freq（防止第二次 .to(dtype) 也损失精度）
            saved_inv_freqs2 = []
            if hasattr(self.model, 'rotary_emb') and hasattr(self.model.rotary_emb, 'inv_freq'):
                saved_inv_freqs2.append(('model', self.model.rotary_emb.inv_freq.clone()))
            for i, layer in enumerate(self.model.layers):
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'rotary_emb'):
                    if hasattr(layer.self_attn.rotary_emb, 'inv_freq'):
                        saved_inv_freqs2.append((i, layer.self_attn.rotary_emb.inv_freq.clone()))
            
            try:
                self.model.to(dtype=self.dtype)
            except Exception:
                pass
            
            # 再次恢复 inv_freq
            for key, inv_freq_val in saved_inv_freqs2:
                if key == 'model':
                    self.model.rotary_emb.inv_freq = inv_freq_val
                    self.model.rotary_emb.original_inv_freq = inv_freq_val
                else:
                    self.model.layers[key].self_attn.rotary_emb.inv_freq = inv_freq_val
                    self.model.layers[key].self_attn.rotary_emb.original_inv_freq = inv_freq_val

        self.model.eval()
        self._loaded = True

    def forward(
        self,
        input_ids=None,
        positions=None,  # vLLM 可能传入，但我们用 position_ids
        kv_caches=None,  # 忽略（diffusion 模式不用 KVCache）
        attn_metadata=None,  # 忽略
        inputs_embeds=None,
        attention_mask=None,
        **kwargs,
    ):
        # 由 v1 默认加载器负责灌权重；这里不做懒加载，避免重复加载与覆盖率校验冲突。
        # self._lazy_load_weights()
        
        # 提取 diffusion 相关参数并透传给内部模型
        temperature = kwargs.get('temperature', None)
        remasking = kwargs.get('remasking', None)

        # 调用内部模型的 forward（精简版）
        out = self.model.forward(
            input_ids=input_ids,
            temperature=temperature,
            remasking=remasking,
        )
        return out



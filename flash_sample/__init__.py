# Flash Sample - Fused CUDA kernels for efficient sampling in diffusion models
# Shared by vllm_add_llada and vllm_add_dream

__version__ = "0.1.0"

# Import CUDA extensions
import torch  # 必须先导入 torch
import sys
import os

# 添加当前目录到 sys.path，确保能找到编译的 .so 文件
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    import flash_sample_low_confidence as _low_conf_module
    low_confidence = _low_conf_module.low_confidence
    silu_inplace = _low_conf_module.silu_inplace
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import flash_sample_low_confidence: {e}")
    low_confidence = None
    silu_inplace = None

try:
    import flash_sample_entropy as _entropy_module
    entropy = _entropy_module.entropy
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import flash_sample_entropy: {e}")
    entropy = None

try:
    import flash_sample_entropy_witht as _entropy_witht_module
    entropy_witht = _entropy_witht_module.entropy_witht
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import flash_sample_entropy_witht: {e}")
    entropy_witht = None

__all__ = ['low_confidence', 'entropy', 'entropy_witht', 'silu_inplace']

#python setup.py build_ext --inplace

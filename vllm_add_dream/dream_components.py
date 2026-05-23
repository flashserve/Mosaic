"""
通用组件和工具函数
"""
import torch


def _non_meta_init_device(device_str: str = None) -> torch.device:
    """返回非meta的初始化设备"""
    if device_str is not None and device_str != "meta":
        return torch.device(device_str)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


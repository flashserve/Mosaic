from __future__ import annotations

from typing import Optional, Dict, Any

_baseline: Optional[Dict[str, Any]] = None


def set_memory_baseline(cuda_bytes: int,
                        torch_reserved_bytes: int,
                        torch_peak_bytes: int,
                        step_id: Optional[int] = None) -> None:
    global _baseline
    _baseline = {
        "cuda_bytes": int(cuda_bytes),
        "torch_reserved_bytes": int(torch_reserved_bytes),
        "torch_peak_bytes": int(torch_peak_bytes),
        "step_id": step_id,
    }


def get_memory_baseline() -> Optional[Dict[str, Any]]:
    return _baseline


def clear_memory_baseline() -> None:
    global _baseline
    _baseline = None



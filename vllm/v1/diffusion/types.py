# vllm/v1/diffusion/types.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class DiffusionParams:
    steps: int = 128
    gen_length: int = 128
    block_length: int = 32
    temperature: float = 0.0
    cfg_scale: float = 0.0
    remasking: str = "low_confidence"
    mask_id: int = 126336
    # For P/O split in graph builder
    prompt_length: Optional[int] = None  # Number of prompt tokens (set after tokenization)
    output_length: Optional[int] = None  # Number of output tokens (defaults to gen_length)

@dataclass
class DiffusionRequest:
    req_id: str
    input_token_ids: List[int]          # 仅 prompt（不含生成区）
    params: DiffusionParams
    extra: Optional[Dict[str, Any]] = None

@dataclass
class DiffusionBatch:
    requests: List[DiffusionRequest]


# ---------------- 迭代级调度新增类型 ----------------

@dataclass
class DiffusionRuntimeState:
    """单个请求在扩散过程中的运行时状态（由调度器维护）。

    - x: 当前工作序列（长度 = prompt_len + gen_length）。
         其中前段为 prompt，生成区初始为 mask_id，并在迭代中被逐步覆盖。
    - block_idx / step_idx_in_block: 当前所在的 block 与该 block 内的步数下标。
    """
    req_id: str
    x: List[int]
    prompt_len: int
    total_len: int
    params: DiffusionParams
    block_idx: int = 0
    step_idx_in_block: int = 0


@dataclass
class DiffusionStepRequest:
    """发往模型的一步请求（由调度器打包）。"""
    req_id: str
    x: List[int]
    prompt_len: int
    total_len: int
    params: DiffusionParams
    block_idx: int
    step_idx_in_block: int


@dataclass
class DiffusionStepBatch:
    """一个迭代步的批次。"""
    requests: List[DiffusionStepRequest]


@dataclass
class DiffusionStepUpdate:
    """模型返回的一步更新结果。

    首版实现简单起见返回整段 new_x，后续可优化为稀疏增量（位置 + token）。
    """
    req_id: str
    new_x: List[int]
    block_idx: int
    step_idx_in_block: int


@dataclass
class FinishedDiffusionItem:
    """用于 EngineCore 打包输出的完成项。"""
    req_id: str
    tail_token_ids: List[int]
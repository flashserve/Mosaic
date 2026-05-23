import logging
import math
import sys
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from configuration_llada import (
    ModelConfig,
    ActivationCheckpointingStrategy,
    StrEnum,
    InitFnType,
)


log = logging.getLogger(__name__)


if sys.version_info.minor > 8:
    from collections.abc import MutableMapping
elif sys.version_info.minor == 8:
    from typing import MutableMapping
else:
    raise SystemExit("This script supports Python 3.8 or higher")


class ModuleType(StrEnum):
    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"


def init_weights(
    config: ModelConfig,
    module: nn.Module,
    d: Optional[int] = None,
    layer_id: Optional[int] = None,
    std_factor: float = 1.0,
    type_of_module: Optional[ModuleType] = None,
) -> None:
    d = d if d is not None else config.d_model
    if isinstance(module, (nn.Linear, nn.Embedding)):
        if config.init_fn == InitFnType.normal:
            std = config.init_std * std_factor
            if config.init_cutoff_factor is not None:
                cutoff_value = config.init_cutoff_factor * std
                nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=std)
        elif config.init_fn == InitFnType.mitchell:
            std = std_factor / math.sqrt(d)
            if layer_id is not None:
                std = std / math.sqrt(2 * (layer_id + 1))
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        elif config.init_fn == InitFnType.kaiming_normal:
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        elif config.init_fn == InitFnType.fan_in:
            std = std_factor / math.sqrt(d)
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif config.init_fn == InitFnType.full_megatron:
            if type_of_module is None:
                raise RuntimeError(
                    f"When using the {InitFnType.full_megatron} init, every module must have a type."
                )

            cutoff_factor = config.init_cutoff_factor
            if cutoff_factor is None:
                cutoff_factor = 3

            if type_of_module == ModuleType.in_module:
                std = config.init_std
            elif type_of_module == ModuleType.out_module:
                std = config.init_std / math.sqrt(2.0 * config.n_layers)
            elif type_of_module == ModuleType.emb:
                std = config.init_std
            elif type_of_module == ModuleType.final_out:
                std = config.d_model ** -0.5
            else:
                raise RuntimeError(f"Unknown module type '{type_of_module}'")
            nn.init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=std,
                a=-cutoff_factor * std,
                b=cutoff_factor * std,
            )
        else:
            raise NotImplementedError(config.init_fn)

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

            if config.init_fn == InitFnType.normal and getattr(module, "_is_residual", False):
                with torch.no_grad():
                    module.weight.div_(math.sqrt(2 * config.n_layers))


def ensure_finite_(x: torch.Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False):
    if check_neg_inf:
        x.masked_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.masked_fill_(x == float("inf"), torch.finfo(x.dtype).max)


def activation_checkpoint_function(cfg: ModelConfig):
    preserve_rng_state = (
        (cfg.attention_dropout == 0.0) and (cfg.embedding_dropout == 0.0) and (cfg.residual_dropout == 0.0)
    )
    from torch.utils.checkpoint import checkpoint

    return partial(
        checkpoint,
        preserve_rng_state=preserve_rng_state,
        use_reentrant=False,
    )


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    """
    Cache for attention biases and other tensors that would normally be stored as buffers.
    We avoid using buffers due to issues with FSDP synchronization and -inf values.
    """

    pass


def _non_meta_init_device(config: ModelConfig) -> torch.device:
    if config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


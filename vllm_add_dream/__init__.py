from vllm import ModelRegistry

def register() -> None:
    ModelRegistry.register_model(
        "DreamModel",  # 必须与 config.json architectures[0] 一致
        "vllm_add_dream.dream_vllm:DreamModel",
    )

__all__ = ["register"]
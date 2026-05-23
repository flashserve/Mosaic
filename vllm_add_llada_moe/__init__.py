from vllm import ModelRegistry


def register() -> None:
    """注册 LLaDA-MoE 到 vLLM 的模型注册表。

    需与 HF `config.json` 中的 `architectures[0]` 完全一致。
    """
    ModelRegistry.register_model(
        "LLaDAMoEModel",
        "vllm_add_llada_moe.llada_moe_vllm:LLaDAMoEModelLM",
    )


__all__ = ["register"]


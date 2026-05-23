from vllm import ModelRegistry


def register() -> None:
    """注册 LLaDA 到 vLLM 的模型注册表。

    需与 HF `config.json` 中的 `architectures[0]` 完全一致。
    """
    ModelRegistry.register_model(
        "LLaDAModelLM",
        "vllm_add_llada.llada_vllm:LLaDAModelLM",
    )


__all__ = ["register"]


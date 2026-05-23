# Auto-register LLaDA plugin for any Python process that has this repo on PYTHONPATH.
try:
    from vllm_add_llada import register as _register_llada
    _register_llada()
except Exception as _e:  # pragma: no cover
    # Be silent if plugin not available; server may still start for other models.
    pass

# Auto-register Dream plugin for any Python process that has this repo on PYTHONPATH.
try:
    from vllm_add_dream import register as _register_dream
    _register_dream()
except Exception as _e:  # pragma: no cover
    # Be silent if plugin not available; server may still start for other models.
    pass

# Auto-register LLaDA-MoE plugin for any Python process that has this repo on PYTHONPATH.
try:
    from vllm_add_llada_moe import register as _register_llada_moe
    _register_llada_moe()
except Exception as _e:  # pragma: no cover
    # Be silent if plugin not available; server may still start for other models.
    pass


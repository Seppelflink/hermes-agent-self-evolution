"""Shared DSPy language-model construction."""

import os
from typing import Optional

import dspy


def create_dspy_lm(
    model: str,
    api_base: Optional[str] = None,
    api_key_env: Optional[str] = "OPENAI_API_KEY",
) -> dspy.LM:
    """Create a DSPy LM, optionally targeting an OpenAI-compatible endpoint.

    Credentials are read at call time from the named environment variable.
    When it is unset, DSPy/LiteLLM retains its normal environment lookup.
    Local OpenAI-compatible servers commonly ignore a required placeholder
    bearer key, so ``EMPTY`` is supplied only when a custom endpoint is set
    and the configured credential is absent.
    """
    kwargs = {}
    if api_base is None:
        api_base = os.environ.get("DSPY_API_BASE")
    if api_base:
        kwargs["api_base"] = api_base

    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if api_key:
            kwargs["api_key"] = api_key
    if api_base and "api_key" not in kwargs:
        kwargs["api_key"] = "EMPTY"

    return dspy.LM(model, **kwargs)

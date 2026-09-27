"""Tests for shared DSPy LM configuration without network calls."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from evolution.core.config import EvolutionConfig
from evolution.core.dataset_builder import SyntheticDatasetBuilder
from evolution.core.fitness import LLMJudge
from evolution.core.lm import create_dspy_lm
from evolution.skills.evolve_skill import _compile_with_optimizer


def test_create_dspy_lm_uses_optional_endpoint_and_environment_key(monkeypatch):
    monkeypatch.setenv("LOCAL_INFERENCE_TOKEN", "test-token")

    with patch("evolution.core.lm.dspy.LM") as lm_constructor:
        result = create_dspy_lm(
            "openai/mistralai/Mamba-Codestral-7B-v0.1",
            api_base="http://127.0.0.1:8000/v1",
            api_key_env="LOCAL_INFERENCE_TOKEN",
        )

    assert result is lm_constructor.return_value
    lm_constructor.assert_called_once_with(
        "openai/mistralai/Mamba-Codestral-7B-v0.1",
        api_base="http://127.0.0.1:8000/v1",
        api_key="test-token",
    )


def test_create_dspy_lm_preserves_default_hosted_model_configuration(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DSPY_API_BASE", raising=False)

    with patch("evolution.core.lm.dspy.LM") as lm_constructor:
        create_dspy_lm("openai/gpt-4.1")

    lm_constructor.assert_called_once_with("openai/gpt-4.1")


def test_create_dspy_lm_reads_endpoint_from_environment(monkeypatch):
    monkeypatch.setenv("DSPY_API_BASE", "http://localhost:8000/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with patch("evolution.core.lm.dspy.LM") as lm_constructor:
        create_dspy_lm("openai/local-model")

    lm_constructor.assert_called_once_with(
        "openai/local-model",
        api_base="http://localhost:8000/v1",
        api_key="EMPTY",
    )


def test_create_dspy_lm_uses_placeholder_for_explicit_local_endpoint(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DSPY_API_BASE", raising=False)

    with patch("evolution.core.lm.dspy.LM") as lm_constructor:
        create_dspy_lm(
            "openai/local-model",
            api_base="http://localhost:8000/v1",
        )

    lm_constructor.assert_called_once_with(
        "openai/local-model",
        api_base="http://localhost:8000/v1",
        api_key="EMPTY",
    )


def test_dataset_generation_uses_configured_lm_settings():
    config = EvolutionConfig(
        judge_model="openai/local-judge",
        api_base="http://localhost:8000/v1",
        api_key_env="LOCAL_INFERENCE_TOKEN",
    )
    builder = SyntheticDatasetBuilder(config)
    builder.generator = MagicMock(
        return_value=SimpleNamespace(
            test_cases='[{"task_input":"test request","expected_behavior":"test outcome"}]'
        )
    )
    lm = object()

    with patch("evolution.core.dataset_builder.create_dspy_lm", return_value=lm) as make_lm, \
         patch("evolution.core.dataset_builder.dspy.context") as context:
        context.return_value.__enter__ = MagicMock(return_value=None)
        context.return_value.__exit__ = MagicMock(return_value=False)
        dataset = builder.generate("Test artifact", num_cases=1)

    assert dataset.train[0].task_input == "test request"
    make_lm.assert_called_once_with(
        "openai/local-judge",
        api_base="http://localhost:8000/v1",
        api_key_env="LOCAL_INFERENCE_TOKEN",
    )


def test_fitness_judge_uses_configured_lm_settings():
    config = EvolutionConfig(
        eval_model="openai/local-evaluator",
        api_base="http://localhost:8000/v1",
        api_key_env="LOCAL_INFERENCE_TOKEN",
    )
    judge = LLMJudge.__new__(LLMJudge)
    judge.config = config
    judge.judge = MagicMock(return_value=SimpleNamespace(
        correctness=0.8,
        procedure_following=0.7,
        conciseness=0.9,
        feedback="Improve the steps.",
    ))
    lm = object()

    with patch("evolution.core.fitness.create_dspy_lm", return_value=lm) as make_lm, \
         patch("evolution.core.fitness.dspy.context") as context:
        context.return_value.__enter__ = MagicMock(return_value=None)
        context.return_value.__exit__ = MagicMock(return_value=False)
        score = judge.score("task", "expected", "output", "skill")

    assert score.correctness == 0.8
    make_lm.assert_called_once_with(
        "openai/local-evaluator",
        api_base="http://localhost:8000/v1",
        api_key_env="LOCAL_INFERENCE_TOKEN",
    )


def test_optimizer_uses_configured_reflection_lm():
    baseline = object()
    trainset = [object()]
    valset = [object()]
    reflection_lm = object()
    optimized = object()

    with patch("evolution.skills.evolve_skill.dspy.GEPA") as gepa_constructor:
        gepa_constructor.return_value.compile.return_value = optimized
        result = _compile_with_optimizer(
            baseline,
            trainset,
            valset,
            iterations=4,
            optimizer_lm=reflection_lm,
        )

    assert result is optimized
    gepa_constructor.assert_called_once()
    assert gepa_constructor.call_args.kwargs["reflection_lm"] is reflection_lm
    gepa_constructor.return_value.compile.assert_called_once_with(
        baseline,
        trainset=trainset,
        valset=valset,
    )

"""Tests for candidate rejection in the evolution orchestration."""

import inspect
from types import SimpleNamespace

import click
import pytest

import evolution.skills.evolve_skill as evolve_skill
from evolution.core.constraints import ConstraintResult


def test_failed_test_suite_rejects_candidate_before_holdout(
    tmp_path,
    monkeypatch,
):
    hermes_repo = tmp_path / "hermes-agent"
    skill_path = hermes_repo / "skills" / "demo" / "SKILL.md"
    skill = {
        "path": skill_path,
        "name": "demo",
        "description": "A demonstration skill",
        "raw": "---\nname: demo\ndescription: A demonstration skill\n---\n\n# Baseline\n",
        "frontmatter": "name: demo\ndescription: A demonstration skill",
        "body": "# Baseline",
    }

    class Dataset:
        train = ["train"]
        val = ["validation"]
        holdout = ["holdout"]
        all_examples = ["train", "validation", "holdout"]

        def __init__(self):
            self.requested_splits = []

        def to_dspy_examples(self, split):
            self.requested_splits.append(split)
            return [split]

    dataset = Dataset()

    class Validator:
        def __init__(self):
            self.validated_artifacts = []
            self.test_suite_called = False

        def validate_all(self, artifact, artifact_type, baseline_text=None):
            self.validated_artifacts.append((artifact, artifact_type, baseline_text))
            return [
                ConstraintResult(
                    passed=True,
                    constraint_name="all_constraints",
                    message="passed",
                )
            ]

        def run_test_suite(self, path):
            self.test_suite_called = True
            assert path == hermes_repo
            return ConstraintResult(
                passed=False,
                constraint_name="test_suite",
                message="Test suite failed (exit code 1)",
                details="1 failed",
            )

    validator = Validator()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(evolve_skill, "resolve_hermes_agent_path", lambda path: hermes_repo)
    monkeypatch.setattr(evolve_skill, "find_skill", lambda name, repo: skill_path)
    monkeypatch.setattr(evolve_skill, "load_skill", lambda path: skill)
    monkeypatch.setattr(
        evolve_skill.GoldenDatasetLoader,
        "load",
        lambda path: dataset,
    )
    monkeypatch.setattr(evolve_skill, "ConstraintValidator", lambda config: validator)
    monkeypatch.setattr(evolve_skill, "create_dspy_lm", lambda *args, **kwargs: object())
    monkeypatch.setattr(evolve_skill.dspy, "configure", lambda **kwargs: None)
    monkeypatch.setattr(
        evolve_skill,
        "SkillModule",
        lambda body: SimpleNamespace(skill_text=body),
    )
    monkeypatch.setattr(
        evolve_skill,
        "_compile_with_optimizer",
        lambda *args, **kwargs: SimpleNamespace(skill_text="# Evolved"),
    )

    with pytest.raises(click.ClickException, match="test suite failed"):
        evolve_skill.evolve(
            skill_name="demo",
            eval_source="golden",
            dataset_path="golden.jsonl",
            hermes_repo=str(hermes_repo),
        )

    assert validator.test_suite_called
    assert dataset.requested_splits == ["train", "val"]
    assert validator.validated_artifacts[0][0] == skill["raw"]
    assert validator.validated_artifacts[1][0].startswith("---\n")
    assert (tmp_path / "output" / "demo" / "evolved_FAILED.md").exists()


def test_full_test_gate_is_enabled_by_default():
    assert inspect.signature(evolve_skill.evolve).parameters["run_tests"].default is True


def test_cli_documents_full_test_gate_default():
    from click.testing import CliRunner

    result = CliRunner().invoke(evolve_skill.main, ["--help"])

    assert result.exit_code == 0
    assert "--run-tests / --skip-tests" in result.output
    assert "default: run" in result.output

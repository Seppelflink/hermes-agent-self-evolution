"""Tests for constraint validators."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
from evolution.core.constraints import ConstraintValidator
from evolution.core.config import EvolutionConfig


@pytest.fixture
def validator():
    config = EvolutionConfig()
    return ConstraintValidator(config)


class TestSizeConstraints:
    def test_skill_under_limit(self, validator):
        result = validator._check_size("x" * 1000, "skill")
        assert result.passed

    def test_skill_over_limit(self, validator):
        result = validator._check_size("x" * 20_000, "skill")
        assert not result.passed
        assert "exceeded" in result.message

    def test_tool_description_under_limit(self, validator):
        result = validator._check_size("Search files by content", "tool_description")
        assert result.passed

    def test_tool_description_over_limit(self, validator):
        result = validator._check_size("x" * 600, "tool_description")
        assert not result.passed

    @pytest.mark.parametrize(
        ("artifact_type", "limit"),
        [
            ("skill", 15_000),
            ("tool_description", 500),
            ("param_description", 200),
        ],
    )
    def test_size_boundary_is_inclusive(self, validator, artifact_type, limit):
        assert validator._check_size("x" * limit, artifact_type).passed
        assert not validator._check_size("x" * (limit + 1), artifact_type).passed

    def test_parameter_description_limit(self, validator):
        assert validator._check_size("x" * 200, "param_description").passed
        assert not validator._check_size("x" * 201, "param_description").passed


class TestGrowthConstraints:
    def test_acceptable_growth(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 1100  # 10% growth
        result = validator._check_growth(evolved, baseline, "skill")
        assert result.passed

    def test_excessive_growth(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 1300  # 30% growth
        result = validator._check_growth(evolved, baseline, "skill")
        assert not result.passed

    def test_shrinkage_is_ok(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 800  # 20% smaller
        result = validator._check_growth(evolved, baseline, "skill")
        assert result.passed

    def test_empty_baseline_cannot_be_used_to_measure_growth(self, validator):
        result = validator._check_growth("new content", "", "skill")
        assert not result.passed
        assert "empty baseline" in result.message

    def test_whitespace_baseline_cannot_be_used_to_measure_growth(self, validator):
        result = validator._check_growth("new content", " \n ", "skill")
        assert not result.passed
        assert "empty baseline" in result.message


class TestNonEmpty:
    def test_non_empty_passes(self, validator):
        result = validator._check_non_empty("some content")
        assert result.passed

    def test_empty_fails(self, validator):
        result = validator._check_non_empty("")
        assert not result.passed

    def test_whitespace_only_fails(self, validator):
        result = validator._check_non_empty("   \n  ")
        assert not result.passed


class TestSkillStructure:
    def test_valid_skill(self, validator):
        skill = (
            "---\nname: test-skill\n"
            'description: "A test: skill"\n---\n\n# Test\nContent here'
        )
        result = validator._check_skill_structure(skill)
        assert result.passed

    def test_validates_frontmatter_as_yaml_not_substrings(self, validator):
        skill = (
            "---\nname: test-skill\n"
            'notes: "description: this is not the description field"\n'
            "---\n\n# Test\nContent here"
        )
        result = validator._check_skill_structure(skill)
        assert not result.passed
        assert "description" in result.message

    def test_missing_frontmatter(self, validator):
        skill = "# Test\nContent without frontmatter"
        result = validator._check_skill_structure(skill)
        assert not result.passed

    def test_malformed_yaml_fails(self, validator):
        skill = "---\nname: [unterminated\ndescription: broken\n---\n\n# Test"
        result = validator._check_skill_structure(skill)
        assert not result.passed
        assert "invalid YAML" in result.message

    def test_missing_name(self, validator):
        skill = "---\ndescription: A test skill\n---\n\n# Test"
        result = validator._check_skill_structure(skill)
        assert not result.passed

    def test_missing_description(self, validator):
        skill = "---\nname: test-skill\n---\n\n# Test"
        result = validator._check_skill_structure(skill)
        assert not result.passed

    def test_empty_markdown_body_fails(self, validator):
        result = validator._check_skill_structure("---\nname: x\ndescription: y\n---\n")
        assert not result.passed
        assert "body" in result.message


class TestValidateAll:
    def test_valid_skill_passes_all(self, validator):
        skill = "---\nname: test\ndescription: Test skill\n---\n\n# Procedure\n1. Do thing"
        results = validator.validate_all(skill, "skill")
        assert all(r.passed for r in results)

    def test_empty_skill_fails(self, validator):
        results = validator.validate_all("", "skill")
        failed = [r for r in results if not r.passed]
        assert len(failed) > 0

    def test_checks_full_skill_file_and_baseline_growth(self, validator):
        baseline = "---\nname: test\ndescription: Test\n---\n\n# A"
        candidate = "---\nname: test\ndescription: Test\n---\n\n# B"
        results = validator.validate_all(candidate, "skill", baseline_text=baseline)
        assert all(result.passed for result in results)
        assert [result.constraint_name for result in results] == [
            "size_limit",
            "growth_limit",
            "non_empty",
            "skill_structure",
        ]


class TestRunTestSuite:
    def test_uses_active_python_and_includes_failure_output(self, validator, tmp_path, monkeypatch):
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="1 failed\n",
            stderr="AssertionError: failed check\n",
        )
        calls = []

        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            return completed

        monkeypatch.setattr("evolution.core.constraints.subprocess.run", fake_run)
        result = validator.run_test_suite(tmp_path)

        assert not result.passed
        assert result.message == "Test suite failed (exit code 1)"
        assert "1 failed" in result.details
        assert "AssertionError" in result.details
        assert calls[0][0] == [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"]
        assert calls[0][1]["cwd"] == str(tmp_path)

    def test_passes_with_successful_test_summary(self, validator, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "evolution.core.constraints.subprocess.run",
            lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="8 passed\n", stderr=""),
        )

        result = validator.run_test_suite(tmp_path)

        assert result.passed
        assert result.details == "8 passed"

    def test_reports_timeout_and_partial_output(self, validator, tmp_path, monkeypatch):
        def timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(
                cmd="pytest",
                timeout=300,
                output=b"collecting tests",
                stderr=b"still running",
            )

        monkeypatch.setattr("evolution.core.constraints.subprocess.run", timeout)
        result = validator.run_test_suite(tmp_path)

        assert not result.passed
        assert result.message == "Test suite timed out (300s)"
        assert "collecting tests" in result.details
        assert "still running" in result.details

    def test_reports_missing_test_runner(self, validator, tmp_path, monkeypatch):
        def missing_runner(*args, **kwargs):
            raise FileNotFoundError("python not found")

        monkeypatch.setattr("evolution.core.constraints.subprocess.run", missing_runner)
        result = validator.run_test_suite(tmp_path)

        assert not result.passed
        assert "Python executable not found" in result.message
        assert "python not found" in result.message

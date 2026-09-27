"""Constraint validators for evolved artifacts.

Every candidate variant must pass ALL constraints before it can be
considered valid. Failed constraints = immediate rejection.
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from evolution.core.config import EvolutionConfig


@dataclass
class ConstraintResult:
    """Result of constraint validation."""
    passed: bool
    constraint_name: str
    message: str
    details: Optional[str] = None


class ConstraintValidator:
    """Validates evolved artifacts against hard constraints."""

    def __init__(self, config: EvolutionConfig):
        self.config = config

    def validate_all(
        self,
        artifact_text: str,
        artifact_type: str,
        baseline_text: Optional[str] = None,
    ) -> list[ConstraintResult]:
        """Run all applicable constraints. Returns list of results."""
        results = []

        # 1. Size limits
        results.append(self._check_size(artifact_text, artifact_type))

        # 2. Growth limit (if baseline provided)
        if baseline_text is not None:
            results.append(self._check_growth(artifact_text, baseline_text, artifact_type))

        # 3. Non-empty
        results.append(self._check_non_empty(artifact_text))

        # 4. Structural integrity
        if artifact_type == "skill":
            results.append(self._check_skill_structure(artifact_text))

        return results

    def run_test_suite(self, hermes_repo: Path) -> ConstraintResult:
        """Run the full hermes-agent test suite. Must pass 100%."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(hermes_repo),
            )

            output = self._format_process_output(result.stdout, result.stderr)
            if result.returncode == 0:
                return ConstraintResult(
                    passed=True,
                    constraint_name="test_suite",
                    message="All tests passed",
                    details=output.splitlines()[-1] if output else "",
                )
            return ConstraintResult(
                passed=False,
                constraint_name="test_suite",
                message=f"Test suite failed (exit code {result.returncode})",
                details=self._last_output_lines(output),
            )
        except subprocess.TimeoutExpired as e:
            output = self._format_process_output(e.stdout, e.stderr)
            return ConstraintResult(
                passed=False,
                constraint_name="test_suite",
                message="Test suite timed out (300s)",
                details=self._last_output_lines(output),
            )
        except FileNotFoundError as e:
            return ConstraintResult(
                passed=False,
                constraint_name="test_suite",
                message=f"Python executable not found: {e}",
            )
        except (OSError, subprocess.SubprocessError) as e:
            return ConstraintResult(
                passed=False,
                constraint_name="test_suite",
                message=f"Failed to run tests: {e}",
            )

    @staticmethod
    def _format_process_output(stdout, stderr) -> str:
        """Join captured process output, including byte output from timeouts."""
        parts = []
        for output in (stdout, stderr):
            if output:
                if isinstance(output, bytes):
                    output = output.decode(errors="replace")
                parts.append(output.strip())
        return "\n".join(parts)

    @staticmethod
    def _last_output_lines(output: str, count: int = 10) -> str:
        return "\n".join(output.splitlines()[-count:])

    def _check_size(self, text: str, artifact_type: str) -> ConstraintResult:
        size = len(text)
        if artifact_type == "skill":
            limit = self.config.max_skill_size
        elif artifact_type == "tool_description":
            limit = self.config.max_tool_desc_size
        elif artifact_type == "param_description":
            limit = self.config.max_param_desc_size
        else:
            limit = self.config.max_skill_size  # Default

        if size <= limit:
            return ConstraintResult(
                passed=True,
                constraint_name="size_limit",
                message=f"Size OK: {size}/{limit} chars",
            )
        else:
            return ConstraintResult(
                passed=False,
                constraint_name="size_limit",
                message=f"Size exceeded: {size}/{limit} chars ({size - limit} over)",
            )

    def _check_growth(self, text: str, baseline: str, artifact_type: str) -> ConstraintResult:
        if not baseline.strip():
            if not text.strip():
                return ConstraintResult(
                    passed=True,
                    constraint_name="growth_limit",
                    message="Growth OK: baseline and candidate are both empty",
                )
            return ConstraintResult(
                passed=False,
                constraint_name="growth_limit",
                message="Growth cannot be measured against an empty baseline",
            )

        growth = (len(text) - len(baseline)) / max(1, len(baseline))
        max_growth = self.config.max_prompt_growth

        if growth <= max_growth:
            return ConstraintResult(
                passed=True,
                constraint_name="growth_limit",
                message=f"Growth OK: {growth:+.1%} (max {max_growth:+.1%})",
            )
        else:
            return ConstraintResult(
                passed=False,
                constraint_name="growth_limit",
                message=f"Growth exceeded: {growth:+.1%} (max {max_growth:+.1%})",
            )

    def _check_non_empty(self, text: str) -> ConstraintResult:
        if text.strip():
            return ConstraintResult(
                passed=True,
                constraint_name="non_empty",
                message="Artifact is non-empty",
            )
        else:
            return ConstraintResult(
                passed=False,
                constraint_name="non_empty",
                message="Artifact is empty",
            )

    def _check_skill_structure(self, text: str) -> ConstraintResult:
        """Check that a skill file has valid YAML frontmatter and markdown body."""
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return ConstraintResult(
                passed=False,
                constraint_name="skill_structure",
                message="Skill must start with YAML frontmatter (---)",
            )

        closing_delimiter = next(
            (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
            None,
        )
        if closing_delimiter is None:
            return ConstraintResult(
                passed=False,
                constraint_name="skill_structure",
                message="Skill frontmatter is missing its closing delimiter (---)",
            )

        try:
            frontmatter = yaml.safe_load("\n".join(lines[1:closing_delimiter]))
        except yaml.YAMLError as e:
            return ConstraintResult(
                passed=False,
                constraint_name="skill_structure",
                message=f"Skill frontmatter is invalid YAML: {e}",
            )

        if not isinstance(frontmatter, dict):
            return ConstraintResult(
                passed=False,
                constraint_name="skill_structure",
                message="Skill frontmatter must be a YAML mapping",
            )

        missing = [
            field
            for field in ("name", "description")
            if not isinstance(frontmatter.get(field), str) or not frontmatter[field].strip()
        ]
        if missing:
            return ConstraintResult(
                passed=False,
                constraint_name="skill_structure",
                message=f"Skill frontmatter requires non-empty: {', '.join(missing)}",
            )

        if not "\n".join(lines[closing_delimiter + 1:]).strip():
            return ConstraintResult(
                passed=False,
                constraint_name="skill_structure",
                message="Skill markdown body must not be empty",
            )

        return ConstraintResult(
            passed=True,
            constraint_name="skill_structure",
            message="Skill has valid YAML frontmatter and a non-empty markdown body",
        )

"""Tests for the pull-request validation entry point and workflow contract."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from tools import validate_pr


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "pull-request-validation.yml"


class ChangedPythonFilesTests(unittest.TestCase):
    def test_base_ref_combines_committed_uncommitted_and_untracked_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative_path in ("committed.py", "working.py", "staged.py", "new.py"):
                (root / relative_path).touch()

            outputs = {
                ("diff", "--name-only", "--diff-filter=ACMR", "base-sha", "HEAD"): [
                    "committed.py",
                    "notes.md",
                ],
                ("diff", "--name-only", "--diff-filter=ACMR"): ["working.py"],
                ("diff", "--cached", "--name-only", "--diff-filter=ACMR"): ["staged.py"],
                ("ls-files", "--others", "--exclude-standard"): ["new.py"],
            }

            with (
                mock.patch.object(validate_pr, "ROOT", root),
                mock.patch.dict(os.environ, {"VALIDATION_BASE_REF": "base-sha"}),
                mock.patch.object(
                    validate_pr,
                    "git_output",
                    side_effect=lambda *arguments: outputs[arguments],
                ),
            ):
                self.assertEqual(
                    validate_pr.changed_python_files(),
                    ["committed.py", "new.py", "staged.py", "working.py"],
                )

    def test_missing_main_uses_previous_commit_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fallback.py").touch()

            def git_output(*arguments: str) -> list[str]:
                if arguments[:2] == ("merge-base", "HEAD"):
                    raise subprocess.CalledProcessError(1, ["git", *arguments])
                if arguments == (
                    "diff",
                    "--name-only",
                    "--diff-filter=ACMR",
                    "HEAD^",
                    "HEAD",
                ):
                    return ["fallback.py"]
                return []

            with (
                mock.patch.object(validate_pr, "ROOT", root),
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(validate_pr, "git_output", side_effect=git_output),
            ):
                self.assertEqual(validate_pr.changed_python_files(), ["fallback.py"])


class CommandFailureTests(unittest.TestCase):
    def test_run_propagates_a_failed_check(self) -> None:
        failure = subprocess.CalledProcessError(1, ["python", "-m", "ruff"])
        with mock.patch.object(validate_pr.subprocess, "run", side_effect=failure):
            with self.assertRaises(subprocess.CalledProcessError):
                validate_pr.run(["python", "-m", "ruff"])


class WorkflowContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_pull_request_events_cover_open_and_update_paths(self) -> None:
        self.assertIn(
            "types: [opened, synchronize, reopened, ready_for_review, edited]",
            self.workflow,
        )

    def test_required_jobs_and_least_privilege_are_declared(self) -> None:
        self.assertIn("permissions:\n  contents: read", self.workflow)
        for name in ("Python syntax", "Lint and formatting", "Complete test suite"):
            with self.subTest(name=name):
                self.assertIn(f"name: {name}", self.workflow)

    def test_each_validation_command_can_fail_its_job(self) -> None:
        expected_commands = (
            "python tools/validate_pr.py --syntax-only",
            "python -m ruff check .",
            "python tools/validate_pr.py --format-only",
            "python -m unittest discover -s tests -v",
        )
        for command in expected_commands:
            with self.subTest(command=command):
                self.assertIn(f"run: {command}", self.workflow)
        self.assertNotIn("continue-on-error: true", self.workflow)


if __name__ == "__main__":
    unittest.main()

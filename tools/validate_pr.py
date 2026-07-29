"""Run the deterministic checks used by pull-request validation."""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "venv",
}


def python_files() -> list[Path]:
    return sorted(
        path for path in ROOT.rglob("*.py") if not EXCLUDED_PARTS.intersection(path.parts)
    )


def run(command: list[str]) -> None:
    print(f"+ {' '.join(command)}", flush=True)
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run(command, cwd=ROOT, check=True, env=environment)


def check_syntax() -> None:
    files = python_files()
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"Parsed {len(files)} Python files successfully.")


def git_output(*arguments: str) -> list[str]:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def changed_python_files() -> list[str]:
    base_ref = os.environ.get("VALIDATION_BASE_REF")
    changed: set[str] = set()
    if base_ref and set(base_ref) != {"0"}:
        changed.update(git_output("diff", "--name-only", "--diff-filter=ACMR", base_ref, "HEAD"))
    else:
        try:
            merge_base = git_output("merge-base", "HEAD", "main")[0]
            changed.update(
                git_output("diff", "--name-only", "--diff-filter=ACMR", merge_base, "HEAD")
            )
        except (subprocess.CalledProcessError, IndexError):
            changed.update(git_output("diff", "--name-only", "--diff-filter=ACMR", "HEAD^", "HEAD"))

    changed.update(git_output("diff", "--name-only", "--diff-filter=ACMR"))
    changed.update(git_output("diff", "--cached", "--name-only", "--diff-filter=ACMR"))
    changed.update(git_output("ls-files", "--others", "--exclude-standard"))
    return sorted(path for path in changed if path.endswith(".py") and (ROOT / path).is_file())


def check_format() -> None:
    files = changed_python_files()
    if not files:
        print("No changed Python files require a formatting check.")
        return
    run([sys.executable, "-m", "ruff", "format", "--check", *files])


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--syntax-only", action="store_true")
    mode.add_argument("--format-only", action="store_true")
    args = parser.parse_args()

    if args.syntax_only:
        check_syntax()
        return 0
    if args.format_only:
        check_format()
        return 0

    check_syntax()
    run([sys.executable, "-m", "ruff", "check", "."])
    check_format()
    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

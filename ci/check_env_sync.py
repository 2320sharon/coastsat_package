"""Fail if ci/environment.yml has drifted from [tool.pixi.dependencies] in pyproject.toml.

    python ci/check_env_sync.py

The CI matrix cannot use pixi.lock -- a lockfile pins exactly one interpreter, and the point
of the matrix is to span four. So ci/environment.yml restates the same conda-forge stack with
the `python` entry left out for the workflow to inject. Two hand-maintained copies of one
dependency list drift the moment somebody adds a package to only one of them, and the failure
mode is quiet: CI keeps passing against an environment that no longer resembles the one
developers use.

This compares the package NAMES on both sides. It deliberately does not compare version
constraints -- keeping those in lockstep is a judgement call (ci/environment.yml has reason to
stay looser), whereas a package present in one file and absent from the other is always a bug.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Injected per matrix leg by the workflow rather than declared, so its absence from
# ci/environment.yml is the whole design and not drift.
INJECTED = {"python"}

# Needed to install the checkout itself; conda-only, with no counterpart in pyproject.
CI_ONLY = {"pip"}


def spec_name(spec: object) -> str:
    """Reduce a conda match-spec such as 'onnxruntime=*=*cpu*' to its package name."""
    return re.split(r"[<>=!\s]", str(spec), maxsplit=1)[0].strip()


def main() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pixi = pyproject["tool"]["pixi"]

    expected = set(pixi["dependencies"]) | set(pixi["feature"]["test"]["dependencies"])
    expected -= INJECTED

    environment = yaml.safe_load((ROOT / "ci" / "environment.yml").read_text(encoding="utf-8"))
    actual = {spec_name(d) for d in environment["dependencies"]} - CI_ONLY

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    if missing or extra:
        print("FAIL: ci/environment.yml is out of sync with pyproject.toml", file=sys.stderr)
        if missing:
            print(
                "  in [tool.pixi.dependencies] but missing from ci/environment.yml:\n    "
                + "\n    ".join(missing),
                file=sys.stderr,
            )
        if extra:
            print(
                "  in ci/environment.yml but not declared for pixi:\n    " + "\n    ".join(extra),
                file=sys.stderr,
            )
        sys.exit(1)

    print(f"ci/environment.yml matches pyproject.toml ({len(expected)} packages)")


if __name__ == "__main__":
    main()

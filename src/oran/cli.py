"""Unified command-line entry point for the reproducible O-RAN study.

The scientific stages remain implemented in small, independently testable
modules.  This dispatcher makes the ``oran-study`` console script declared in
``pyproject.toml`` usable without importing every optional reporting dependency
at startup.
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence


COMMAND_MODULES: dict[str, str] = {
    "data": "oran.data",
    "manifest": "oran.manifest",
    "prepare": "oran.experiment",
    "capture-audit": "oran.capture_audit",
    "broad-search": "oran.policy_search",
    "matched-search": "oran.matched_search",
    "strict-selection": "oran.strict_selection",
    "confirmatory": "oran.confirmatory",
    "inference": "oran.inference",
    "repro-audit": "oran.repro_audit",
    "lease-sensitivity": "oran.sensitivity",
    "benchmark": "oran.benchmark",
    "report": "oran.reporting",
}


def cli(argv: Sequence[str] | None = None) -> int:
    """Dispatch one study stage while forwarding its remaining arguments.

    ``prepare`` forwards to :mod:`oran.experiment`; callers must include that
    module's own ``prepare`` or ``fit`` stage, e.g. ``oran-study prepare
    prepare --source ...``.  The slight repetition keeps backward-compatible,
    directly executable module CLIs and avoids a second argument schema.
    """

    parser = argparse.ArgumentParser(
        prog="oran-study",
        description="Reproducible friction-budgeted O-RAN containment study",
    )
    parser.add_argument("command", choices=tuple(COMMAND_MODULES))
    parser.add_argument(
        "arguments",
        nargs=argparse.REMAINDER,
        help="arguments forwarded verbatim to the selected stage",
    )
    parsed = parser.parse_args(argv)
    module_name = COMMAND_MODULES[parsed.command]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # Reporting/benchmark modules are optional during minimal installs, but
        # do not hide a missing transitive dependency inside an existing module.
        if exc.name == module_name:
            parser.error(f"stage is not installed: {parsed.command}")
        raise
    stage_cli = getattr(module, "cli", None)
    if stage_cli is None:
        # Reporting and benchmarking expose an argparse-compatible ``main``
        # directly. Accept that conventional shape while retaining ``cli`` as
        # the preferred interface for the scientific stages.
        stage_cli = getattr(module, "main", None)
    if stage_cli is None:
        parser.error(f"stage has no command-line interface: {parsed.command}")
    result = stage_cli(parsed.arguments)
    return 0 if result is None else int(result)


def main() -> None:
    raise SystemExit(cli())


if __name__ == "__main__":
    main()

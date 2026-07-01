"""Command-line entry point for the Rapthor pipeline."""

from __future__ import annotations

import logging
import optparse
from collections.abc import Sequence

from rapthor import modifystate
from rapthor._version import __version__ as version


def _build_parser() -> optparse.OptionParser:
    parser = optparse.OptionParser(
        prog="rapthor",
        usage="%prog <parset>",
        version="%%prog v%s" % (version),
    )
    parser.add_option("-q", help="enable quiet mode", action="store_true", default=False)
    parser.add_option("-v", help="enable verbose mode", action="store_true", default=False)
    parser.add_option("-r", help="reset one or more operations", action="store_true", default=False)
    return parser


def _logging_level(options: optparse.Values) -> str:
    if options.q:
        return "warning"
    if options.v:
        return "debug"
    return "info"


def _run_pipeline(parset_file: str, *, logging_level: str) -> None:
    from rapthor.execution.pipeline.flow import pipeline_flow

    pipeline_flow(parset_file, logging_level=logging_level)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Rapthor command-line interface."""

    parser = _build_parser()
    options, args = parser.parse_args(list(argv) if argv is not None else None)

    if len(args) != 1:
        parser.print_help()
        return 0

    parset_file = args[0]

    if options.r:
        modifystate.run(parset_file)
        return 0

    try:
        _run_pipeline(parset_file, logging_level=_logging_level(options))
    except Exception as exc:
        logging.getLogger("rapthor").exception(exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

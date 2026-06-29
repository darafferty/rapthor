"""Command-line adapter for calibration solution plotting."""

from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from typing import Optional, Sequence

from rapthor.execution.calibrate.plotting import plot_solutions


def parse_args(argv: Optional[Sequence[str]] = None) -> Namespace:
    """Parse command-line arguments for calibration solution plotting."""
    parser = ArgumentParser(
        description="Plot solutions.\n",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("h5file", help="Name of solution h5parm file")
    parser.add_argument("soltype", help="Type of solution to plot: 'phase' or 'amplitude'")
    parser.add_argument("--root", help="Root name for output plots", default=None)
    parser.add_argument("--refstat", help="Name of reference station", default=None)
    parser.add_argument("--soltab", help="Name of solution table", default=None)
    parser.add_argument("--dir", dest="direction", help="Name of direction", default=None)
    parser.add_argument(
        "--first-dir",
        help="Plot only the first direction if the solution table has direction labels",
        action="store_true",
    )
    return parser.parse_args(argv)


def run(argv: Optional[Sequence[str]] = None) -> None:
    """Run the calibration solution plotting CLI adapter."""
    args = parse_args(argv)
    plot_solutions(
        args.h5file,
        args.soltype,
        root=args.root,
        refstat=args.refstat,
        soltab=args.soltab,
        direction=args.direction,
        first_dir=args.first_dir,
    )


if __name__ == "__main__":
    run()

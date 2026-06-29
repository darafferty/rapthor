"""Calibration solution plotting helpers."""

import logging
from typing import Optional

log = logging.getLogger("rapthor:calibrate:plotting")


def _load_plotting_backend():
    import matplotlib as mpl

    mpl.use("Agg")
    from losoto.h5parm import h5parm
    from losoto.operations import plot

    return h5parm, plot


def plot_solutions(
    h5file: str,
    soltype: str,
    root: Optional[str] = None,
    refstat: Optional[str] = None,
    soltab: Optional[str] = None,
    direction: Optional[str] = None,
    first_dir: bool = False,
    h5parm_factory=None,
    plot_runner=None,
) -> None:
    """
    Plot calibration solutions from an h5parm file.

    Parameters mirror the historical plotting command, but the implementation
    is importable so flows can call or wrap it directly.
    """
    if h5parm_factory is None or plot_runner is None:
        default_h5parm_factory, default_plot_runner = _load_plotting_backend()
        h5parm_factory = h5parm_factory or default_h5parm_factory
        plot_runner = plot_runner or default_plot_runner

    with h5parm_factory(h5file) as h5parm_file:
        solset = h5parm_file.getSolset("sol000")
        if soltype == "phase":
            solution_table = solset.getSoltab(soltab or "phase000")
            reference_station = solution_table.ant[0]
            minmax = [-3.2, 3.2]
        elif soltype == "amplitude":
            solution_table = solset.getSoltab(soltab or "amplitude000")
            reference_station = ""
            minmax = [0, 0]
        else:
            raise ValueError(
                f'Solution type "{soltype}" not understood. Must be one of "phase" or "amplitude"'
            )

        if root is None:
            root = f"{'scalar' if 'pol' not in solution_table.axesNames else ''}{soltype}_"
        if refstat is not None:
            reference_station = refstat

        log.info("Plotting %s solutions...", soltype)
        selected_direction = direction
        if (
            selected_direction is None
            and first_dir
            and hasattr(solution_table, "dir")
            and len(solution_table.dir) > 0
        ):
            selected_direction = solution_table.dir[0]
        if selected_direction is not None:
            solution_table.setSelection(dir=selected_direction)

        axes = _plot_axes(solution_table)
        if axes is None:
            log.warning("Solution table contains only a single time and frequency. No plots made.")
            return

        plot_runner.run(
            solution_table,
            axes,
            axisInTable="ant",
            axisInCol="pol" if "pol" in solution_table.axesNames else "",
            NColFig=0,
            refAnt=reference_station,
            prefix=root,
            minmax=minmax,
            plotFlag=True,
            markerSize=4,
        )


def _plot_axes(solution_table) -> Optional[list[str]]:
    """Return the LoSoTo plotting axes for the solution table shape."""
    has_time_axis = len(solution_table.time) > 1
    has_frequency_axis = len(solution_table.freq) > 1
    if has_time_axis and has_frequency_axis:
        return ["time", "freq"]
    if has_time_axis:
        return ["time"]
    if has_frequency_axis:
        return ["freq"]
    return None

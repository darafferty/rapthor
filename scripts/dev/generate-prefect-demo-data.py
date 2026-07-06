#!/usr/bin/env python3
"""Generate a richer synthetic Measurement Set for the Prefect demo."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import casacore.tables as pt
import numpy as np
from astropy.coordinates import SkyCoord

DEFAULT_OUTPUT_DIR = Path("examples/generated/prefect_demo_rich")
DEFAULT_TEMPLATE_MS = Path("tests/resources/test.ms")
PHASE_CENTRE = SkyCoord("1h37m41.299s", "+33d09m35.132s", frame="icrs")
REFERENCE_FREQUENCY_HZ = 134375000.0
CHANNEL_WIDTH_HZ = 24414.0625
BENCHMARK_PARSET_FILENAME = "prefect_demo_benchmark.parset"
BENCHMARK_STRATEGY_FILENAME = "prefect_demo_benchmark_strategy.py"
BENCHMARK_STRATEGY_TEXT = """\
\"\"\"Shared demo and CI benchmark strategy for the generated dataset.\"\"\"

COMMON_SETTINGS = {
    "do_calibrate": True,
    "do_image": True,
    "do_normalize": False,
    "do_check": False,
    "peel_outliers": False,
    "peel_bright_sources": False,
    "fast_timestep_sec": 20.0,
    "medium_timestep_sec": 40.0,
    "slow_timestep_sec": 80.0,
    "fulljones_timestep_sec": 80.0,
    "max_normalization_delta": 0.3,
    "scale_normalization_delta": True,
    "solve_min_uv_lambda": 80,
    "target_flux": 0.6,
    "max_directions": 5,
    "max_distance": None,
    "regroup_model": False,
    "auto_mask": 5.0,
    "auto_mask_nmiter": 1,
    "channel_width_hz": 48828.125,
    "threshisl": 3.0,
    "threshpix": 5.0,
    "max_nmiter": 6,
}


def _step(calibration_strategy, **overrides):
    step = {
        **COMMON_SETTINGS,
        "calibration_strategy": calibration_strategy,
    }
    step.update(overrides)
    return step


strategy_steps = [
    _step(
        {"di": ["fast_phase", "medium_phase"]},
        max_nmiter=6,
    ),
    _step(
        {"di": [], "dd": ["fast_phase", "medium_phase"]},
        max_nmiter=6,
        regroup_model=True,
    ),
    _step(
        {"di": [], "dd": ["fast_phase", "medium_phase", "slow_gains", "medium_phase"]},
        max_nmiter=6,
        regroup_model=True,
    ),
    _step(
        {"di": ["full_jones"], "dd": []},
        max_nmiter=6,
        regroup_model=False,
    ),
]
"""


@dataclass(frozen=True)
class PatchSpec:
    name: str
    delta_ra_deg: float
    delta_dec_deg: float


@dataclass(frozen=True)
class SourceSpec:
    name: str
    patch: str
    delta_ra_deg: float
    delta_dec_deg: float
    true_flux_jy: float
    apparent_flux_jy: float
    spectral_index: float


PATCHES = [
    PatchSpec("Patch_rich_centre", 0.00, 0.00),
    PatchSpec("Patch_rich_east", 0.40, 0.05),
    PatchSpec("Patch_rich_west", -0.42, -0.04),
    PatchSpec("Patch_rich_north", 0.06, 0.38),
    PatchSpec("Patch_rich_south", -0.10, -0.40),
]

SOURCES = [
    SourceSpec("rich_centre_a", "Patch_rich_centre", 0.000, 0.000, 3.8, 3.25, -0.75),
    SourceSpec("rich_centre_b", "Patch_rich_centre", 0.025, -0.018, 0.65, 0.54, -0.65),
    SourceSpec("rich_east_a", "Patch_rich_east", 0.400, 0.050, 3.1, 2.55, -0.82),
    SourceSpec("rich_east_b", "Patch_rich_east", 0.435, 0.074, 0.45, 0.36, -0.70),
    SourceSpec("rich_west_a", "Patch_rich_west", -0.420, -0.040, 2.8, 2.30, -0.78),
    SourceSpec("rich_west_b", "Patch_rich_west", -0.455, -0.062, 0.40, 0.31, -0.62),
    SourceSpec("rich_north_a", "Patch_rich_north", 0.060, 0.380, 2.4, 1.95, -0.88),
    SourceSpec("rich_north_b", "Patch_rich_north", 0.032, 0.405, 0.35, 0.27, -0.72),
    SourceSpec("rich_south_a", "Patch_rich_south", -0.100, -0.400, 2.2, 1.75, -0.84),
    SourceSpec("rich_south_b", "Patch_rich_south", -0.132, -0.424, 0.30, 0.23, -0.68),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a richer local Prefect demo dataset with multiple bright sources, "
            "time/frequency solution bins, and at least two calibration chunks."
        )
    )
    parser.add_argument(
        "--template-ms",
        type=Path,
        default=DEFAULT_TEMPLATE_MS,
        help="Small Measurement Set to clone and extend.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to create or replace with generated demo inputs.",
    )
    parser.add_argument(
        "--strategy",
        type=Path,
        default=None,
        help=(
            "Strategy file to reference from the generated demo parset. Defaults "
            "to the generated benchmark strategy next to the parset."
        ),
    )
    parser.add_argument(
        "--n-time-slots",
        type=int,
        default=48,
        help="Requested number of time slots. Rounded up to a template-MS multiple.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for thermal noise.",
    )
    parser.add_argument(
        "--noise-jy",
        type=float,
        default=0.035,
        help="Complex Gaussian noise scale added to the predicted visibilities.",
    )
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Skip DP3 prediction and leave the MS with only synthetic noise/corruptions.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def offset_coord(delta_ra_deg: float, delta_dec_deg: float) -> SkyCoord:
    ra = PHASE_CENTRE.ra + (delta_ra_deg / np.cos(PHASE_CENTRE.dec.radian)) * u.deg
    dec = PHASE_CENTRE.dec + delta_dec_deg * u.deg
    return SkyCoord(ra=ra, dec=dec, frame="icrs")


def format_ra(coord: SkyCoord) -> str:
    return coord.ra.to_string(unit=u.hourangle, sep=":", precision=3, pad=True)


def format_dec(coord: SkyCoord) -> str:
    return coord.dec.to_string(unit=u.deg, sep=".", precision=3, pad=True, alwayssign=False)


def path_for_parset(path: Path, repo_root: Path) -> str:
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def write_sky_model(path: Path, apparent: bool) -> None:
    lines = [
        (
            "FORMAT = Name, Type, Patch, Ra, Dec, I, SpectralIndex='[]', "
            "LogarithmicSI, ReferenceFrequency='134375000.0', MajorAxis, "
            "MinorAxis, Orientation"
        ),
        "",
    ]
    for patch in PATCHES:
        coord = offset_coord(patch.delta_ra_deg, patch.delta_dec_deg)
        lines.append(f" , , {patch.name}, {format_ra(coord)}, {format_dec(coord)}")
    for source in SOURCES:
        coord = offset_coord(source.delta_ra_deg, source.delta_dec_deg)
        flux = source.apparent_flux_jy if apparent else source.true_flux_jy
        lines.append(
            f"{source.name}, POINT, {source.patch}, {format_ra(coord)}, {format_dec(coord)}, "
            f"{flux:.6g}, [{source.spectral_index:.2f}], false, "
            f"{REFERENCE_FREQUENCY_HZ:.1f}, 0, 0, 0"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_benchmark_strategy(path: Path) -> None:
    path.write_text(BENCHMARK_STRATEGY_TEXT, encoding="utf-8")


def extend_ms_times(ms_path: Path, n_time_slots: int) -> int:
    with pt.table(str(ms_path), readonly=False, ack=False) as table:
        base_nrows = table.nrows()
        base_times = table.getcol("TIME")
        base_time_centroid = (
            table.getcol("TIME_CENTROID") if "TIME_CENTROID" in table.colnames() else None
        )
        unique_times = np.unique(base_times)
        slots_per_template = len(unique_times)
        if n_time_slots < slots_per_template:
            raise ValueError(f"--n-time-slots must be at least {slots_per_template} for {ms_path}")

        repeats = int(math.ceil(n_time_slots / slots_per_template))
        total_time_slots = repeats * slots_per_template
        for _ in range(repeats - 1):
            table.copyrows(table, startrowin=0, startrowout=-1, nrow=base_nrows)

        time_step = float(np.median(np.diff(unique_times)))
        block_duration = slots_per_template * time_step
        for repeat in range(repeats):
            offset = repeat * block_duration
            startrow = repeat * base_nrows
            table.putcol("TIME", base_times + offset, startrow=startrow, nrow=base_nrows)
            if base_time_centroid is not None:
                table.putcol(
                    "TIME_CENTROID",
                    base_time_centroid + offset,
                    startrow=startrow,
                    nrow=base_nrows,
                )

        for column in ("DATA", "MODEL_DATA", "CORRECTED_DATA"):
            if column in table.colnames():
                values = table.getcol(column)
                values[...] = 0.0j
                table.putcol(column, values)
        if "FLAG" in table.colnames():
            flags = table.getcol("FLAG")
            flags[...] = False
            table.putcol("FLAG", flags)

    update_observation_time_range(ms_path)
    return total_time_slots


def update_observation_time_range(ms_path: Path) -> None:
    with pt.table(str(ms_path), ack=False) as table:
        start_time = float(np.min(table.getcol("TIME")))
        end_time = float(np.max(table.getcol("TIME")))
    with pt.table(f"{ms_path}::OBSERVATION", readonly=False, ack=False) as observation:
        if "TIME_RANGE" not in observation.colnames():
            return
        time_range = observation.getcol("TIME_RANGE")
        time_range[:, 0] = start_time
        time_range[:, 1] = end_time
        observation.putcol("TIME_RANGE", time_range)


def set_synthetic_uvw_geometry(ms_path: Path) -> None:
    with pt.table(f"{ms_path}::SPECTRAL_WINDOW", ack=False) as spectral_window:
        reference_frequency_hz = float(spectral_window.getcol("REF_FREQUENCY")[0])
    ref_wavelength_m = 299792458.0 / reference_frequency_hz

    with pt.table(str(ms_path), readonly=False, ack=False) as table:
        uvw = table.getcol("UVW")
        ant1 = table.getcol("ANTENNA1")
        ant2 = table.getcol("ANTENNA2")
        times = table.getcol("TIME")

        unique_times = np.unique(times)
        antennas = np.unique(np.concatenate([ant1, ant2]))
        base_radius_lambda = np.linspace(0.0, 2500.0, len(antennas))
        antenna_index = {antenna: index for index, antenna in enumerate(antennas)}
        time_index = {time_value: index for index, time_value in enumerate(unique_times)}

        positions = {}
        for time_value in unique_times:
            t_index = time_index[time_value]
            for antenna in antennas:
                a_index = antenna_index[antenna]
                theta = (2.0 * np.pi * a_index / len(antennas)) + 0.18 * t_index
                radius_lambda = base_radius_lambda[a_index]
                positions[(time_value, antenna)] = np.array(
                    [
                        np.cos(theta) * radius_lambda * ref_wavelength_m,
                        np.sin(theta) * radius_lambda * ref_wavelength_m,
                        0.04 * np.sin(theta + 0.1 * t_index) * radius_lambda * ref_wavelength_m,
                    ]
                )

        for row_index in range(len(uvw)):
            first_position = positions[(times[row_index], ant1[row_index])]
            second_position = positions[(times[row_index], ant2[row_index])]
            uvw[row_index] = second_position - first_position

        table.putcol("UVW", uvw)


def run_dp3_predict(ms_path: Path, skymodel_path: Path) -> None:
    predicted_ms = ms_path.with_name(f"{ms_path.stem}_predicted.ms")
    if predicted_ms.exists():
        shutil.rmtree(predicted_ms)

    command = [
        "DP3",
        f"msin={ms_path}",
        "steps=[predict]",
        "predict.usebeammodel=True",
        "predict.beam_interval=120",
        "predict.beammode=array_factor",
        f"predict.sourcedb={skymodel_path}",
        f"msout={predicted_ms}",
    ]
    subprocess.run(command, check=True)
    shutil.rmtree(ms_path)
    predicted_ms.rename(ms_path)


def corrupt_data(ms_path: Path, seed: int, noise_jy: float) -> None:
    rng = np.random.default_rng(seed)
    with pt.table(f"{ms_path}::SPECTRAL_WINDOW", ack=False) as spectral_window:
        frequencies_hz = spectral_window.getcol("CHAN_FREQ")[0]
        reference_frequency_hz = float(spectral_window.getcol("REF_FREQUENCY")[0])

    with pt.table(str(ms_path), readonly=False, ack=False) as table:
        data = table.getcol("DATA")
        ant1 = table.getcol("ANTENNA1")
        ant2 = table.getcol("ANTENNA2")
        times = table.getcol("TIME")

        unique_times = np.unique(times)
        antennas = np.unique(np.concatenate([ant1, ant2]))
        time_index = {time_value: index for index, time_value in enumerate(unique_times)}
        antenna_index = {antenna: index for index, antenna in enumerate(antennas)}

        time_grid = np.arange(len(unique_times), dtype=float)[:, None, None]
        freq_grid = (reference_frequency_hz / frequencies_hz)[None, :, None]
        ant_grid = np.arange(len(antennas), dtype=float)[None, None, :]
        ant_scale = (ant_grid - np.mean(ant_grid)) / max(1.0, len(antennas) - 1.0)

        slow_phase = 0.45 * ant_scale * np.sin(2.0 * np.pi * time_grid / len(unique_times))
        dispersive_phase = (
            0.25
            * ant_scale
            * freq_grid
            * np.cos(4.0 * np.pi * time_grid / len(unique_times) + ant_grid * 0.7)
        )
        antenna_phase = slow_phase + dispersive_phase

        row_time_index = np.array([time_index[time_value] for time_value in times])
        row_ant1_index = np.array([antenna_index[antenna] for antenna in ant1])
        row_ant2_index = np.array([antenna_index[antenna] for antenna in ant2])
        baseline_phase = (
            antenna_phase[row_time_index, :, row_ant1_index]
            - antenna_phase[row_time_index, :, row_ant2_index]
        )
        data *= np.exp(1j * baseline_phase)[:, :, None].astype(data.dtype)

        noise = (
            rng.normal(scale=noise_jy, size=data.shape)
            + 1j * rng.normal(scale=noise_jy, size=data.shape)
        ).astype(data.dtype)
        table.putcol("DATA", data + noise)


def write_parset(
    output_dir: Path,
    path: Path,
    repo_root: Path,
    strategy_path: Path,
    *,
    title: str = "rich demo",
    work_dir_name: str = "work",
    grid_width_deg: float = 1.25,
    local_dask_workers: int = 2,
    cpus_per_task: int = 4,
    max_cores: int = 4,
    max_threads: int = 4,
    deconvolution_threads: int = 2,
    parallel_gridding_tasks: int = 1,
    publish_fits_previews: bool = True,
    publish_postage_stamp_previews: bool = True,
    postage_stamp_preview_count: int = 5,
    postage_stamp_preview_size_px: int = 96,
    fits_preview_clip_percentile: float = 99.9,
) -> None:
    ms_path = output_dir / "prefect_demo_rich.ms"
    true_sky_path = output_dir / "prefect_demo_rich_true_sky.txt"
    apparent_sky_path = output_dir / "prefect_demo_rich_apparent_sky.txt"
    work_dir = output_dir / work_dir_name

    path.write_text(
        textwrap.dedent(
            f"""\
            # Generated {title} parset for manually running the Prefect process flow.
            #
            # Regenerate with:
            #
            #   scripts/dev/generate-prefect-demo-data.py --force
            #
            # Run from the repository root:
            #
            #   rapthor {path_for_parset(path, repo_root)}
            #
            # Or use the helper when you want a persistent Prefect/Dask dashboard:
            #
            #   scripts/dev/run-rapthor-prefect-demo.py {path_for_parset(path, repo_root)}

            [global]
            dir_working = {path_for_parset(work_dir, repo_root)}
            input_ms = {path_for_parset(ms_path, repo_root)}
            data_colname = DATA
            generate_initial_skymodel = False
            download_initial_skymodel = False
            download_overwrite_skymodel = True
            input_skymodel = {path_for_parset(true_sky_path, repo_root)}
            apparent_skymodel = {path_for_parset(apparent_sky_path, repo_root)}
            regroup_input_skymodel = False
            strategy = {path_for_parset(strategy_path, repo_root)}
            selfcal_data_fraction = 1.0
            final_data_fraction = 1.0
            input_h5parm = None
            input_fulljones_h5parm = None
            facet_layout = None
            dde_mode = faceting

            [calibration]
            fast_freqstep_hz = {CHANNEL_WIDTH_HZ * 2}
            medium_freqstep_hz = {CHANNEL_WIDTH_HZ * 4}
            slow_freqstep_hz = {CHANNEL_WIDTH_HZ * 8}
            fulljones_freqstep_hz = {CHANNEL_WIDTH_HZ * 8}
            fast_smoothnessconstraint = 6.0e6
            medium_smoothnessconstraint = 5.0e6
            slow_smoothnessconstraint = 4.0e6
            dd_interval_factor = 1
            propagatesolutions = True
            maxiter = 50
            stepsize = 0.2
            tolerance = 0.001

            [imaging]
            grid_width_ra_deg = {grid_width_deg}
            grid_width_dec_deg = {grid_width_deg}
            grid_center_ra = 1h37m41.299s
            grid_center_dec = +33d09m35.132s
            grid_nsectors_ra = 1
            idg_mode = cpu
            reweight = False
            average_visibilities = False
            save_image_cube = False
            image_cube_stokes_list = [I]
            make_quv_images = False
            disable_iquv_clean = False
            photometry_skymodel =
            astrometry_skymodel =
            shared_facet_rw = False
            normalization_skymodels =
            normalization_reference_frequencies =

            [cluster]
            batch_system = single_machine
            max_nodes = 1
            local_dask_workers = {local_dask_workers}
            cpus_per_task = {cpus_per_task}
            mem_per_node_gb = 0
            max_cores = {max_cores}
            max_threads = {max_threads}
            deconvolution_threads = {deconvolution_threads}
            parallel_gridding_tasks = {parallel_gridding_tasks}
            local_scratch_dir =
            global_scratch_dir =
            use_container = False
            container_type = docker
            prefect_task_runner = local_dask
            dask_dashboard_address = :8787
            prefect_stream_output = True
            prefect_retries = 0
            prefect_log_commands = True
            prefect_command_profile = time
            prefect_publish_fits_previews = {publish_fits_previews}
            prefect_publish_postage_stamp_previews = {publish_postage_stamp_previews}
            prefect_postage_stamp_preview_count = {postage_stamp_preview_count}
            prefect_postage_stamp_preview_size_px = {postage_stamp_preview_size_px}
            prefect_fits_preview_clip_percentile = {fits_preview_clip_percentile}
            debug_workflow = False
            keep_temporary_files = False
            allow_internet_access = False
            """
        ),
        encoding="utf-8",
    )


def write_benchmark_parset(
    output_dir: Path, path: Path, repo_root: Path, strategy_path: Path
) -> None:
    write_parset(
        output_dir,
        path,
        repo_root,
        strategy_path,
        title="CI benchmark",
        work_dir_name="benchmark-work",
        grid_width_deg=1.0,
        local_dask_workers=2,
        cpus_per_task=0,
        max_cores=0,
        max_threads=0,
        deconvolution_threads=0,
        parallel_gridding_tasks=0,
        publish_fits_previews=False,
        publish_postage_stamp_previews=False,
    )


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    output_dir = args.output_dir
    if output_dir.exists():
        if not args.force:
            raise SystemExit(f"{output_dir} already exists; pass --force to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    if not args.template_ms.exists():
        raise SystemExit(f"Template Measurement Set does not exist: {args.template_ms}")
    ms_path = output_dir / "prefect_demo_rich.ms"
    true_sky_path = output_dir / "prefect_demo_rich_true_sky.txt"
    apparent_sky_path = output_dir / "prefect_demo_rich_apparent_sky.txt"
    parset_path = output_dir / "prefect_demo_rich.parset"
    benchmark_parset_path = output_dir / BENCHMARK_PARSET_FILENAME
    benchmark_strategy_path = output_dir / BENCHMARK_STRATEGY_FILENAME
    demo_strategy_path = args.strategy or benchmark_strategy_path
    if args.strategy is not None and not args.strategy.exists():
        raise SystemExit(f"Strategy file does not exist: {args.strategy}")

    print(f"Copying template MS to {ms_path}")
    shutil.copytree(args.template_ms, ms_path)
    n_time_slots = extend_ms_times(ms_path, args.n_time_slots)
    set_synthetic_uvw_geometry(ms_path)

    print(f"Writing sky models with {len(PATCHES)} bright patches")
    write_sky_model(true_sky_path, apparent=False)
    write_sky_model(apparent_sky_path, apparent=True)
    print(f"Writing demo/benchmark strategy to {benchmark_strategy_path}")
    write_benchmark_strategy(benchmark_strategy_path)

    if args.skip_predict:
        print("Skipping DP3 prediction; generated MS will contain synthetic noise only")
    else:
        print("Running DP3 predict to populate model visibilities")
        run_dp3_predict(ms_path, true_sky_path)

    print("Adding synthetic time/frequency antenna phases and thermal noise")
    corrupt_data(ms_path, seed=args.seed, noise_jy=args.noise_jy)
    write_parset(output_dir, parset_path, repo_root, demo_strategy_path)
    write_benchmark_parset(output_dir, benchmark_parset_path, repo_root, benchmark_strategy_path)

    print()
    print(f"Generated {n_time_slots} time slots in {ms_path}")
    print(f"Demo parset: {parset_path}")
    print(f"Demo strategy: {demo_strategy_path}")
    print(f"Benchmark parset: {benchmark_parset_path}")
    print(f"Benchmark strategy: {benchmark_strategy_path}")
    print("Run with:")
    print(f"  scripts/dev/run-rapthor-prefect-demo.py {path_for_parset(parset_path, repo_root)}")
    print(
        f"  scripts/dev/run-rapthor-prefect-demo.py {path_for_parset(benchmark_parset_path, repo_root)}"
    )


if __name__ == "__main__":
    main()

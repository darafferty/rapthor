"""Small representative execution payloads for focused tests."""

import os


def _work_path(work_dir: object, filename: str) -> str:
    return os.path.join(str(work_dir), filename)


def representative_image_payload(work_dir: object = "/work/image_1") -> dict:
    """Return a minimal image payload that passes flow-level validation."""
    work_dir = str(work_dir)
    return {
        "mode": "no_dde_stokes_i",
        "use_mpi": False,
        "pipeline_working_dir": work_dir,
        "sectors": [
            {
                "image_name": "sector_1",
                "prepare_tasks": [
                    {
                        "msin": "/data/obs.ms",
                        "msout": "obs.prep.ms",
                        "msout_path": _work_path(work_dir, "obs.prep.ms"),
                        "starttime": "59000.0",
                        "ntimes": 10,
                        "freqstep": 2,
                        "timestep": 3,
                        "maxinterval": None,
                        "minchannels": 1,
                    }
                ],
                "image_cube_specs": [
                    {
                        "pol": "I",
                        "filename": "sector_1_I_freq_cube.fits",
                        "path": _work_path(work_dir, "sector_1_I_freq_cube.fits"),
                    }
                ],
                "wsclean_imsize": [1024, 1024],
                "dd_psf_grid": [4, 4],
                "obs_original_paths": ["/data/obs.ms"],
                "obs_starttime": ["59000.0"],
                "obs_ntimes": [10],
            }
        ],
    }


def representative_calibrate_payload(work_dir: object = "/work/calibrate_1") -> dict:
    """Return a minimal calibration payload with one DD solve chunk."""
    work_dir = str(work_dir)
    return {
        "mode": "dd",
        "calibration_kind": "dd_calibration",
        "pipeline_working_dir": work_dir,
        "image_based_predict": False,
        "chunks": [
            {
                "msin": "/data/obs.ms",
                "starttime": "59000.0",
                "ntimes": 10,
                "output_h5parm": "solve1.h5",
                "output_h5parm_path": _work_path(work_dir, "solve1.h5"),
                "solve_slots": [
                    {
                        "slot": 1,
                        "solve_type": "fast_phase",
                        "solution_label": "fast",
                        "h5parm": "solve1.h5",
                        "h5parm_path": _work_path(work_dir, "solve1.h5"),
                        "solint": 1,
                        "mode": "scalarphase",
                        "nchan": 2,
                        "solutions_per_direction": [1],
                        "smoothness_dd_factors": [1.0],
                    }
                ],
            }
        ],
    }


def representative_predict_payload(
    work_dir: object = "/work/predict_1",
    *,
    mode: str = "di",
) -> dict:
    """Return a minimal predict payload for DI or DD flow wiring tests."""
    work_dir = str(work_dir)
    postprocess_task = {
        "msobs": "/data/obs_0.ms",
        "data_colname": "DATA",
        "obs_starttime": "50000.0",
        "infix": ".selfcal",
    }
    if mode == "dd":
        postprocess_task.update(
            {
                "solint_sec": 20.0,
                "solint_hz": 1000.0,
                "min_uv_lambda": 80.0,
                "max_uv_lambda": 1000000.0,
                "nr_outliers": 1,
                "peel_outliers": True,
                "nr_bright": 0,
                "peel_bright": False,
                "reweight": True,
            }
        )
    return {
        "mode": mode,
        "pipeline_working_dir": work_dir,
        "predict_tasks": [
            {
                "msin": "/data/obs_0.ms",
                "data_colname": "DATA",
                "msout": "obs_0.ms.sector_1_modeldata",
                "msout_path": _work_path(work_dir, "obs_0.ms.sector_1_modeldata"),
                "starttime": "50000.0",
                "ntimes": 10,
                "onebeamperpatch": True,
                "correctfreqsmearing": False,
                "correcttimesmearing": False,
                "sagecalpredict": False,
                "sourcedb": "/data/sector_1.skymodel",
                "directions": ["patch1", "patch2"],
                "numthreads": 4,
                "h5parm": None,
                "applycal_steps": None,
                "normalize_h5parm": None,
            }
        ],
        "postprocess_tasks": [postprocess_task],
    }


def representative_mosaic_payload(
    work_dir: object = "/work/mosaic_1",
    *,
    compress_images: bool = False,
) -> dict:
    """Return a minimal mosaic payload with one product."""
    work_dir = str(work_dir)
    return {
        "pipeline_working_dir": work_dir,
        "compress_images": compress_images,
        "skip_processing": False,
        "mosaic_products": [
            {
                "sector_image_filenames": [
                    "sector_1-I-image.fits",
                    "sector_2-I-image.fits",
                ],
                "sector_vertices_filenames": ["sector_1.vertices", "sector_2.vertices"],
                "template_image_filename": "mosaic_1_template.fits",
                "template_image_path": _work_path(work_dir, "mosaic_1_template.fits"),
                "regridded_image_filenames": [
                    "sector_1-I-image.fits.regridded",
                    "sector_2-I-image.fits.regridded",
                ],
                "mosaic_filename": "mosaic_1-I-image.fits",
                "mosaic_path": _work_path(work_dir, "mosaic_1-I-image.fits"),
            }
        ],
    }


def representative_concatenate_payload(work_dir: object = "/work/concatenate_1") -> dict:
    """Return a minimal concatenate payload with one epoch."""
    work_dir = str(work_dir)
    return {
        "pipeline_working_dir": work_dir,
        "data_colname": "DATA",
        "epochs": [
            {
                "input_filenames": ["/data/obs_0.ms", "/data/obs_1.ms"],
                "output_filename": "epoch_0.ms",
                "output_path": _work_path(work_dir, "epoch_0.ms"),
            }
        ],
    }

"""Predict command builders."""

from typing import Optional

from rapthor.execution.commands import bool_token, bracketed_list_token, comma_join


def _predict_type(sagecalpredict: bool, h5parm: Optional[str]) -> str:
    if sagecalpredict:
        return "sagecalpredict"
    if h5parm is None:
        return "predict"
    return "h5parmpredict"


def build_predict_model_data_command(
    msin: str,
    data_colname: str,
    msout: str,
    starttime: str,
    ntimes: int,
    onebeamperpatch: bool,
    correctfreqsmearing: bool,
    correcttimesmearing: bool,
    sagecalpredict: bool,
    sourcedb: str,
    directions: list[str],
    numthreads: int,
    h5parm: Optional[str] = None,
    applycal_steps: Optional[str] = None,
    normalize_h5parm: Optional[str] = None,
) -> list[str]:
    """Build the DP3 prediction command for one sector/observation pair."""
    command = [
        "DP3",
        "msout.overwrite=True",
        "steps=[predict]",
        "predict.operation=replace",
    ]
    if h5parm is not None:
        command.extend(
            [
                "predict.applycal.correction=phase000",
                "predict.applycal.fastphase.correction=phase000",
                "predict.applycal.fastphase.solset=sol000",
                "predict.applycal.slowgain.correction=amplitude000",
                "predict.applycal.slowgain.solset=sol000",
                "predict.applycal.normalization.correction=amplitude000",
                "predict.applycal.normalization.solset=sol000",
            ]
        )
    command.extend(
        [
            "predict.usebeammodel=True",
            "predict.beam_interval=120",
            "predict.beammode=array_factor",
            "msout.storagemanager=Dysco",
            "msout.storagemanager.databitrate=0",
            "msout.antennacompression=False",
            f"msin={msin}",
            f"msin.datacolumn={data_colname}",
            f"msout={msout}",
            f"msin.starttime={starttime}",
        ]
    )
    if ntimes > 0:
        command.append(f"msin.ntimes={ntimes}")
    command.extend(
        [
            f"predict.onebeamperpatch={bool_token(onebeamperpatch)}",
            f"predict.correctfreqsmearing={bool_token(correctfreqsmearing)}",
            f"predict.correcttimesmearing={bool_token(correcttimesmearing)}",
            f"predict.type={_predict_type(sagecalpredict, h5parm)}",
        ]
    )
    if applycal_steps is not None:
        command.append(f"predict.applycal.steps={applycal_steps}")
    if h5parm is not None:
        command.append(f"predict.applycal.parmdb={h5parm}")
    if normalize_h5parm is not None:
        command.append(f"predict.applycal.normalization.parmdb={normalize_h5parm}")
    command.extend(
        [
            f"predict.sourcedb={sourcedb}",
            f"predict.directions={bracketed_list_token(directions)}",
            f"numthreads={numthreads}",
        ]
    )
    return command


def build_add_sector_models_command(
    msobs: str,
    msmods: list[str],
    data_colname: str,
    obs_starttime: str,
    infix: str,
) -> list[str]:
    """Build the `add_sector_models.py` command for one observation."""
    return [
        "add_sector_models.py",
        msobs,
        comma_join(msmods),
        f"--msin_column={data_colname}",
        f"--starttime={obs_starttime}",
        f"--infix={infix}",
    ]


def build_subtract_sector_models_command(
    msobs: str,
    msmods: list[str],
    data_colname: str,
    obs_starttime: str,
    solint_sec: float,
    solint_hz: float,
    infix: str,
    min_uv_lambda: float,
    max_uv_lambda: float,
    nr_outliers: int,
    peel_outliers: bool,
    nr_bright: int,
    peel_bright: bool,
    reweight: bool,
) -> list[str]:
    """Build the `subtract_sector_models.py` command for one observation."""
    return [
        "subtract_sector_models.py",
        "--weights_colname=WEIGHT_SPECTRUM",
        "--phaseonly=True",
        msobs,
        comma_join(msmods),
        f"--msin_column={data_colname}",
        f"--starttime={obs_starttime}",
        f"--solint_sec={solint_sec}",
        f"--solint_hz={solint_hz}",
        f"--infix={infix}",
        f"--uvcut_min={min_uv_lambda}",
        f"--uvcut_max={max_uv_lambda}",
        f"--nr_outliers={nr_outliers}",
        f"--peel_outliers={bool_token(peel_outliers)}",
        f"--nr_bright={nr_bright}",
        f"--peel_bright={bool_token(peel_bright)}",
        f"--reweight={bool_token(reweight)}",
    ]

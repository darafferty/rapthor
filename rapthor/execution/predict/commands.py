"""Predict command builders."""

from typing import Optional

from rapthor.execution.commands import bool_token, bracketed_list_token


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
    applycal_step_names = _applycal_step_names(applycal_steps)
    predict_applycal_steps = [step for step in applycal_step_names if step != "normalization"]
    dp3_steps = ["predict"]
    if normalize_h5parm is not None:
        dp3_steps.append("applycal")
    command = [
        "DP3",
        "msout.overwrite=True",
        f"steps={bracketed_list_token(dp3_steps)}",
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
            ]
        )
    if normalize_h5parm is not None:
        command.extend(
            [
                "applycal.type=applycal",
                "applycal.steps=[normalization]",
                "applycal.normalization.correction=amplitude000",
                "applycal.normalization.solset=sol000",
                "applycal.normalization.usemodeldata=True",
                "applycal.normalization.invert=False",
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
    if predict_applycal_steps:
        command.append(f"predict.applycal.steps={bracketed_list_token(predict_applycal_steps)}")
    if h5parm is not None:
        command.append(f"predict.applycal.parmdb={h5parm}")
    if normalize_h5parm is not None:
        command.append(f"applycal.normalization.parmdb={normalize_h5parm}")
    command.extend(
        [
            f"predict.sourcedb={sourcedb}",
            f"predict.directions={bracketed_list_token(directions)}",
            f"numthreads={numthreads}",
        ]
    )
    return command


def _predict_type(sagecalpredict: bool, h5parm: Optional[str]) -> str:
    if sagecalpredict:
        return "sagecalpredict"
    if h5parm is None:
        return "predict"
    return "h5parmpredict"


def _applycal_step_names(applycal_steps: Optional[str]) -> list[str]:
    """Return DP3 applycal step names from a bracketed parset token."""
    if applycal_steps is None:
        return []
    steps = applycal_steps.strip()
    if steps.startswith("[") and steps.endswith("]"):
        steps = steps[1:-1]
    return [step.strip() for step in steps.split(",") if step.strip()]

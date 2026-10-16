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
    use_array_beam: bool,
    numthreads: int,
    h5parm: Optional[str] = None,
    applycal_steps: Optional[str] = None,
    normalize_h5parm: Optional[str] = None,
) -> list[str]:
    """Build the DP3 prediction command for one sector/observation pair.

    ``applycal_steps`` is an ordered list of correction step names serialized
    as a bracketed, comma-separated string, e.g.
    ``"[fastphase,slowgain,normalization]"``. The predict operation creates this
    string before building the execution payload. For SAGECal, this builder
    orders the fast-phase/slow-gain pair as amplitude then phase, as required
    by DP3's two-soltab reader. ``None`` omits ``predict.applycal.steps``.

    DP3's SAGECal predictor accepts at most two applycal substeps and reads their
    soltabs from ``predict.applycal.parmdb``. It ignores the per-step file
    ``predict.applycal.normalization.parmdb``: even with amplitude-first ordering,
    a phase/normalization pair reads the amplitude from the main solutions file
    instead. Adding normalization to the phase/slow-gain pair exceeds the two-step
    limit and is rejected. The ordering workaround does not enable normalization;
    these limitations are covered by strict xfails in ``test_predict_dp3.py``.

    With the default DP3 predictors, all corrections, including flux-scale
    normalization, are applied by the predict step's own applycal substeps.
    With ``predict.operation=replace``
    the prediction is written to the data buffer, so a separate ``applycal``
    step with ``usemodeldata=True`` would act on an empty model buffer and
    leave the written visibilities unchanged. The predict-internal applycal
    corrupts the model with the non-inverted solutions, which maps the model
    from the normalized flux scale back onto the scale of the observed data.
    """
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
            ]
        )
    if normalize_h5parm is not None:
        command.extend(
            [
                "predict.applycal.normalization.correction=amplitude000",
                "predict.applycal.normalization.solset=sol000",
            ]
        )
    # If calibration with the array beam is requested, make sure to only predict the array beam.
    # If the full beam is requested (i.e. use_array_beam = False), we do not have to add anything,
    # because using the full beam is the default.
    if use_array_beam:
        command.append("predict.beammode=array_factor")
    command.extend(
        [
            "predict.usebeammodel=True",
            "predict.beam_interval=120",
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
        if sagecalpredict:
            steps = [step.strip() for step in applycal_steps.strip().strip("[]").split(",")]
            # DP3's SagecalPredict reads two soltabs positionally as amplitude
            # and phase without checking their types. Reorder this gain pair
            # only; separate normalization files remain unsupported.
            if steps == ["fastphase", "slowgain"]:
                applycal_steps = "[slowgain,fastphase]"
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


def _predict_type(sagecalpredict: bool, h5parm: Optional[str]) -> str:
    if sagecalpredict:
        return "sagecalpredict"
    if h5parm is None:
        return "predict"
    return "h5parmpredict"

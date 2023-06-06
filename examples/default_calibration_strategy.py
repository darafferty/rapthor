"""
Script that defines the default user processing strategy. Specifying this file
as the calibration strategy in the Rapthor parset causes default Rapthor to use
the default calibration behaviour, which is equal to specifying no specific
calibration strategy.

This file is provided to base custom strategies from. See the documentation for
detailed information on each parameter.
"""
strategy_steps = []
min_selfcal_loops = 4
max_selfcal_loops = 8
for i in range(max_selfcal_loops):
    strategy_steps.append({})

    # Here we set the calibration strategy so that two cycles of phase-only
    # calibration are done (outliers -- sources that lie outside of imaged
    # regions -- are peeled in the first cycle). Starting with the third cycle,
    # slow-gain calibration is also done. The minimum uv distance used in the
    # solves is set to 350 lambda, except for the first slow-gain cycle where it
    # is often beneficial to exclude short baselines. Lastly, the maximum
    # allowed difference from unity in the normalized amplitude solutions (per
    # station) is set to 0.3, to allow for small adjustments to the station
    # calibration (done in LINC).
    strategy_steps[i]['do_calibrate'] = True
    if i == 0:
        strategy_steps[i]['do_slowgain_solve'] = False
        strategy_steps[i]['peel_outliers'] = True
    elif i == 1:
        strategy_steps[i]['do_slowgain_solve'] = False
        strategy_steps[i]['peel_outliers'] = False
    else:
        strategy_steps[i]['do_slowgain_solve'] = True
        strategy_steps[i]['peel_outliers'] = False
    if i == 2:
        strategy_steps[i]['solve_min_uv_lambda'] = 2000
    else:
        strategy_steps[i]['solve_min_uv_lambda'] = 350
    strategy_steps[i]['peel_bright_sources'] = False
    strategy_steps[i]['max_normalization_delta'] = 0.3
    strategy_steps[i]['scale_normalization_delta'] = True

    # Here we set the imaging strategy, lowering the masking thresholds as
    # selfcal proceeds to ensure all emission is properly cleaned and artifacts,
    # if any, are excluded from the resulting sky models
    strategy_steps[i]['do_image'] = True
    if i < 2:
        strategy_steps[i]['auto_mask'] = 5.0
        strategy_steps[i]['threshisl'] = 4.0
        strategy_steps[i]['threshpix'] = 5.0
    elif i == 2:
        strategy_steps[i]['auto_mask'] = 4.0
        strategy_steps[i]['threshisl'] = 3.0
        strategy_steps[i]['threshpix'] = 5.0
    else:
        strategy_steps[i]['auto_mask'] = 3.0
        strategy_steps[i]['threshisl'] = 3.0
        strategy_steps[i]['threshpix'] = 5.0

    # Here we set the calibrator selection strategy, decreasing the target
    # minimum flux density for sources to be used as calibrators as selfcal
    # proceeds. For each cycle, we set a maximum allowed number of calibrators
    # (directions) to ensure that the resource requirements and runtimes don't
    # grow unnecessarily large
    if i == 0:
        strategy_steps[i]['target_flux'] = 0.6
        strategy_steps[i]['max_nmiter'] = 8
        strategy_steps[i]['max_directions'] = 20
    elif i == 1:
        strategy_steps[i]['target_flux'] = 0.4
        strategy_steps[i]['max_nmiter'] = 9
        strategy_steps[i]['max_directions'] = 30
    elif i == 2:
        strategy_steps[i]['target_flux'] = 0.3
        strategy_steps[i]['max_nmiter'] = 10
        strategy_steps[i]['max_directions'] = 40
    else:
        strategy_steps[i]['target_flux'] = 0.25
        strategy_steps[i]['max_nmiter'] = 12
        strategy_steps[i]['max_directions'] = 50
    strategy_steps[i]['regroup_model'] = True

    # Here we specify that the convergence/divergence checks are done only when
    # needed, to prevent the selfcal from stopping early (before
    # min_selfcal_loops)
    if i < min_selfcal_loops - 1 or i == max_selfcal_loops - 1:
        strategy_steps[i]['do_check'] = False
    else:
        strategy_steps[i]['do_check'] = True
        strategy_steps[i]['convergence_ratio'] = 0.95
        strategy_steps[i]['divergence_ratio'] = 1.1

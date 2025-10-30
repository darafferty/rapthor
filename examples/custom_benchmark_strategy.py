"""
Script that defines the default user processing strategy for HBA data when the
initial sky model is generated from the input data. Specifying this file as the
strategy in the Rapthor parset causes Rapthor to use the default self-
calibration behaviour, which is equal to specifying no specific calibration
strategy.
 
This file is provided to base custom strategies from. See the documentation for
detailed information on each parameter.
"""

strategy_steps = []
min_selfcal_loops = 4
max_selfcal_loops = 8

for i in range(max_selfcal_loops):
    strategy_steps.append(step := {})

    # Here we set the calibration strategy. We specify direction-dependent corrections
    # without peeling outlier sources that lie outside of imaged regions, and slow-gain
    # calibration is done in every cycle. The minimum uv distance used in the solves is
    # set to 0 lambda. Lastly, the maximum allowed difference from unity in the normalized
    # amplitude solutions (per station) is set to 0.3, to allow for small adjustments to
    # the station calibration (done in LINC).
    step['do_calibrate'] = True
    step['do_slowgain_solve'] = True
    step['peel_outliers'] = False
    step['solve_min_uv_lambda'] = 0.0
    step['peel_bright_sources'] = False
    step['do_fulljones_solve'] = False
    step['max_normalization_delta'] = 0.3
    step['scale_normalization_delta'] = True

    # Here we use the same solution intervals for every cycle, but it is possible to
    # adjust the intervals so that, for example, the early cycles have longer intervals to
    # compensate for the generally lower signal-to-noise ratios of the solves
    step['fast_timestep_sec'] = 16.0
    step['slow_timestep_sec'] = 600.0

    # We do not do any flux-scale normalization
    step['do_normalize'] = False

    # Here we set the imaging strategy, lowering the masking thresholds as
    # selfcal proceeds to ensure all emission is properly cleaned and artifacts,
    # if any, are excluded from the resulting sky models. Conversely, the
    # maximum number of major iterations allowed during imaging is raised to
    # allow deeper cleaning in the later cycles. Lastly, the maximum number of
    # WSClean major iterations allowed after the automasking threshold is reached
    # is set to 2, which has been found to be a sufficient number of iterations in
    # most cases
    step['do_image'] = True
    step['auto_mask'] = 3.0
    step['threshisl'] = 3.0
    step['threshpix'] = 3.0
    step['max_nmiter'] = 20 if i == 0 else 25
    step['auto_mask_nmiter'] = 5

    # Here we set the calibrator selection strategy, decreasing the target
    # minimum flux density for sources to be used as calibrators as selfcal
    # proceeds. For each cycle, we set a maximum allowed number of calibrators
    # (directions) to ensure that the resource requirements and runtimes don't
    # grow unnecessarily large. We also limit the distance from the phase center
    # that calibrators can have to exclude distant sources
    if i == 0:
        step['target_flux'] = 0.3
        step['max_directions'] = 20
        step['max_distance'] = 3.0
    elif i == 1:
        step['target_flux'] = 0.25
        step['max_directions'] = 40
        step['max_distance'] = 3.5
    else:
        step['target_flux'] = 0.25
        step['max_directions'] = 50
        step['max_distance'] = 4.0
    step['regroup_model'] = True

    # Here we specify that the convergence/divergence checks are done only when
    # needed, to prevent the selfcal from stopping early (before
    # min_selfcal_loops)
    if i < min_selfcal_loops - 1:
        step['do_check'] = False
    else:
        step['do_check'] = True
        step['convergence_ratio'] = 0.95
        step['divergence_ratio'] = 1.1
        step['failure_ratio'] = 10.0

# Set the parameters for the final pass as duplicates of the last selfcal step
strategy_steps.append(strategy_steps[-1])
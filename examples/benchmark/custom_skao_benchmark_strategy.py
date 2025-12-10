"""
Strategy for simulated AA2 benchmark dataset. Further details on this dataset
and strategy can be found on confluence:
https://confluence.skatelescope.org/display/SE/%5BSimulations%5D+--+New+data+for+benchmarking
"""

strategy_steps = []
do_phase_only_solves = True
min_selfcal_loops = 4 if do_phase_only_solves else 2
max_selfcal_loops = 8 if do_phase_only_solves else 6

for i in range(max_selfcal_loops):
    strategy_steps.append(step := {})

    # Here we set the calibration strategy. We specify direction-dependent corrections
    # without peeling outlier sources that lie outside of imaged regions, and slow-gain
    # calibration is done in every cycle. The minimum uv distance used in the solves is
    # set to 0 lambda. Lastly, the maximum allowed difference from unity in the normalized
    # amplitude solutions (per station) is set to 0.3, to allow for small adjustments to
    # the station calibration (done in LINC).
    step['do_calibrate'] = True
    step['peel_outliers'] = False
    step['do_slowgain_solve'] = not do_phase_only_solves if i < 2 else True
    step['solve_min_uv_lambda'] = 2000 if i == 2 and do_phase_only_solves else 750
    step['do_fulljones_solve'] = False
    step['peel_bright_sources'] = False
    step['max_normalization_delta'] = 0.3
    step['scale_normalization_delta'] = True
    step['fast_timestep_sec'] = 32.0
    step['medium_timestep_sec'] = 120.0
    step['slow_timestep_sec'] = 600.0

    # Here we use the same solution intervals for every cycle, but it is possible to
    # adjust the intervals so that, for example, the early cycles have longer intervals to
    # compensate for the generally lower signal-to-noise ratios of the solves
    step['fast_timestep_sec'] = 32.0
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
    if i < 2 and do_phase_only_solves:
        step['auto_mask'] = 5.0
        step['threshisl'] = 4.0
        step['threshpix'] = 5.0
        step['max_nmiter'] = 8
    elif (i == 2 and do_phase_only_solves) or (i == 0 and not do_phase_only_solves):
        step['auto_mask'] = 4.0
        step['threshisl'] = 3.0
        step['threshpix'] = 5.0
        step['max_nmiter'] = 10
    else:
        step['auto_mask'] = 3.0
        step['threshisl'] = 3.0
        step['threshpix'] = 5.0
        step['max_nmiter'] = 12
    step['auto_mask_nmiter'] = 2
    step['channel_width_hz'] = 4e6

    # Here we set the calibrator selection strategy, decreasing the target
    # minimum flux density for sources to be used as calibrators as selfcal
    # proceeds. For each cycle, we set a maximum allowed number of calibrators
    # (directions) to ensure that the resource requirements and runtimes don't
    # grow unnecessarily large. We also limit the distance from the phase center
    # that calibrators can have to exclude distant sources
    step['regroup_model'] = True
    step['max_distance'] = 3.0     # overwritten below for cycle <2
    step['target_flux'] = 0.3      # overwritten below for cycle 0, 1
    step['max_directions'] = 40    # overwritten below for cycle 0, 1
    if do_phase_only_solves:
        if i == 0:
            step['target_flux'] = 0.6
            step['max_directions'] = 20
        elif i == 1:
            step['target_flux'] = 0.4
            step['max_directions'] = 30
        elif i == 2:
            step['max_distance'] = 3.5
        else:
            step['max_distance'] = 4.0

    # do_phase_only_solves = False
    elif i == 1:
        step['max_distance'] = 3.5
    else:
        step['max_distance'] = 4.0

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
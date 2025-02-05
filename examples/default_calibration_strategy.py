"""
Script that defines the default user processing strategy for HBA data when the
initial sky model is generated from the input data. Specifying this file as the
calibration strategy in the Rapthor parset causes default Rapthor to use the default
calibration behaviour, which is equal to specifying no specific calibration strategy.

This file is provided to base custom strategies from. See the documentation for
detailed information on each parameter.
"""
strategy_steps = []
min_selfcal_loops = 4
max_selfcal_loops = 8
for i in range(max_selfcal_loops):
    strategy_steps.append({})

    # Activate normalization of the flux scale (adjusts the amplitudes to
    # achieve obs_flux / true_flux = 1)
    strategy_steps[i]['do_normalize'] = True

    # Here we set the calibration strategy. We specify that outliers -- sources
    # that lie outside of imaged regions -- are peeled in the first cycle only
    # and that slow-gain calibration is done in every cycle. The minimum uv
    # distance used in the solves is set to 150 lambda. Lastly, the maximum
    # allowed difference from unity in the normalized amplitude solutions (per
    # station) is set to 0.3, to allow for small adjustments to the station
    # calibration (done in LINC).
    strategy_steps[i]['do_calibrate'] = True
    if i == 0:
        strategy_steps[i]['do_slowgain_solve'] = True
        strategy_steps[i]['peel_outliers'] = True
    else:
        strategy_steps[i]['do_slowgain_solve'] = True
        strategy_steps[i]['peel_outliers'] = False
    strategy_steps[i]['solve_min_uv_lambda'] = 150
    strategy_steps[i]['peel_bright_sources'] = False
    strategy_steps[i]['do_fulljones_solve'] = False
    strategy_steps[i]['max_normalization_delta'] = 0.3
    strategy_steps[i]['scale_normalization_delta'] = True

    # For HBA data, we use the same solution intervals for every cycle, but if
    # LBA data were to be used, the intervals for the early cycles can be longer to
    # compensate for the generally lower signal-to-noise ratios of the solves
    strategy_steps[i]['fast_timestep_sec'] = 8.0
    strategy_steps[i]['slow_timestep_joint_sec'] = 0.0
    strategy_steps[i]['slow_timestep_separate_sec'] = 600.0

    # Here we set the imaging strategy, lowering the masking thresholds as
    # selfcal proceeds to ensure all emission is properly cleaned and artifacts,
    # if any, are excluded from the resulting sky models. Conversely, the number
    # of major iterations allowed during imaging is raised to allow deeper
    # cleaning in the later cycles
    strategy_steps[i]['do_image'] = True
    if i == 0:
        strategy_steps[i]['auto_mask'] = 4.0
        strategy_steps[i]['threshisl'] = 3.0
        strategy_steps[i]['threshpix'] = 5.0
        strategy_steps[i]['max_nmiter'] = 10
    else:
        strategy_steps[i]['auto_mask'] = 3.0
        strategy_steps[i]['threshisl'] = 3.0
        strategy_steps[i]['threshpix'] = 5.0
        strategy_steps[i]['max_nmiter'] = 12

    # Here we set the calibrator selection strategy, decreasing the target
    # minimum flux density for sources to be used as calibrators as selfcal
    # proceeds. For each cycle, we set a maximum allowed number of calibrators
    # (directions) to ensure that the resource requirements and runtimes don't
    # grow unnecessarily large. We also limit the distance from the phase center
    # that calibrators can have to exclude distant sources
    if i == 0:
        strategy_steps[i]['target_flux'] = 0.3
        strategy_steps[i]['max_directions'] = 40
        strategy_steps[i]['max_distance'] = 3.0
    elif i == 1:
        strategy_steps[i]['target_flux'] = 0.25
        strategy_steps[i]['max_directions'] = 40
        strategy_steps[i]['max_distance'] = 3.5
    else:
        strategy_steps[i]['target_flux'] = 0.25
        strategy_steps[i]['max_directions'] = 50
        strategy_steps[i]['max_distance'] = 4.0
    strategy_steps[i]['regroup_model'] = True

    # Here we specify that the convergence/divergence checks are done only when
    # needed, to prevent the selfcal from stopping early (before
    # min_selfcal_loops)
    if i < min_selfcal_loops - 1:
        strategy_steps[i]['do_check'] = False
    else:
        strategy_steps[i]['do_check'] = True
        strategy_steps[i]['convergence_ratio'] = 0.95
        strategy_steps[i]['divergence_ratio'] = 1.1
        strategy_steps[i]['failure_ratio'] = 10.0

# Set the parameters for the final pass as duplicates of the last selfcal step
strategy_steps.append(strategy_steps[-1])

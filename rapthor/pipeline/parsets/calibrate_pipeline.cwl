cwlVersion: v1.2
class: Workflow
label: Rapthor calibration workflow
doc: |
  This workflow performs direction-dependent calibration. In general,
  calibration is done in three steps: (1) a fast phase-only calibration (with
  core stations constrianed to have the same solutions) to correct for
  ionospheric effects, (2) a joint slow amplitude calibration (with all stations
  constrained to have the same solutions) to correct for beam errors, and (3) a
  further unconstrained slow gain calibration to correct for station-to-station
  differences. Steps (2) and (3) are skipped if the calibration is phase-only.
  This calibration scheme works for both HBA and LBA data. The final products of
  this workflow are solution tables (h5parm files), plots, and a-term screens (FITS
  files).

requirements:
  ScatterFeatureRequirement: {}
  StepInputExpressionRequirement: {}

{% if max_cores is not none %}
hints:
  ResourceRequirement:
    coresMin: {{ max_cores }}
    coresMax: {{ max_cores }}
{% endif %}

inputs:
  - id: timechunk_filename
    label: Filename of input MS (time)
    doc: |
      The filenames of input MS files for which calibration will be done (length =
      n_obs * n_time_chunks).
    type: Directory[]

  - id: starttime
    label: Start time of each chunk
    doc: |
      The start time (in casacore MVTime) for each time chunk used in the fast-phase
      calibration (length = n_obs * n_time_chunks).
    type: string[]

  - id: ntimes
    label: Number of times of each chunk
    doc: |
      The number of timeslots for each time chunk used in the fast-phase calibration
      (length = n_obs * n_time_chunks).
    type: int[]

  - id: solint_fast_timestep
    label: Fast solution interval in time
    doc: |
      The solution interval in number of timeslots for the fast phase solve (length =
      n_obs * n_time_chunks).
    type: int[]

  - id: solint_fast_freqstep
    label: Fast solution interval in frequency
    doc: |
      The solution interval in number of frequency channels for the fast phase solve
      (length = n_obs * n_time_chunks).
    type: int[]

  - id: calibrator_patch_names
    label: Names of calibrator patches
    doc: |
      The names of the patches used in calibration (length = n_calibrators).
    type: string[]

  - id: calibrator_fluxes
    label: Values of calibrator flux densities
    doc: |
      The total flux densities in Jy of the patches used in calibration (length =
      n_calibrators).
    type: float[]

  - id: output_fast_h5parm
    label: Fast output solution table
    doc: |
      The filename of the output h5parm solution table for the fast phase solve (length
      = n_obs * n_time_chunks).
    type: string[]

  - id: combined_fast_h5parm
    label: Combined fast output solution table
    doc: |
      The filename of the output combined h5parm solution table for the fast phase solve
      (length = 1).
    type: string

  - id: calibration_skymodel_file
    label: Filename of sky model
    doc: |
      The filename of the input sky model text file (length = 1).
    type: File

  - id: fast_smoothnessconstraint
    label: Fast smoothnessconstraint
    doc: |
      The smoothnessconstraint kernel size in Hz for the fast phase solve (length = 1).
    type: float

  - id: fast_smoothnessreffrequency
    label: Fast smoothnessreffrequency
    doc: |
      The smoothnessreffrequency Hz for the fast phase solve (length = n_obs *
      n_time_chunks).
    type: float[]

  - id: fast_smoothnessrefdistance
    label: Fast smoothnessrefdistance
    doc: |
      The smoothnessrefdistance in m for the fast phase solve (length = 1).
    type: float

  - id: fast_antennaconstraint
    label: Fast antenna constraint
    doc: |
      The antenna constraint for the fast phase solve (length = 1).
    type: string

  - id: maxiter
    label: Maximum iterations
    doc: |
      The maximum number of iterations in the solves (length = 1).
    type: int

  - id: llssolver
    label: Linear least-squares solver
    doc: |
      The linear least-squares solver to use (length = 1).
    type: string

  - id: propagatesolutions
    label: Propagate solutions
    doc: |
      Flag that determines whether solutions are propagated as initial start values
      for the next solution interval (length = 1).
    type: boolean

  - id: solveralgorithm
    label: Solver algorithm
    doc: |
      The algorithm used for solving (length = 1).
    type: string

  - id: solverlbfgs_dof
    label: LBFGS degrees of freedom
    doc: |
      The degrees of freedom in LBFGS solver (length = 1).
    type: float

  - id: solverlbfgs_iter
    label: LBFGS iterations per minibatch
    doc: |
      The number of iterations per minibatch in LBFGS solver (length = 1).
    type: int

  - id: solverlbfgs_minibatches
    label: LBFGS minibatches
    doc: |
      The number of minibatches in LBFGS solver (length = 1).
    type: int

  - id: onebeamperpatch
    doc: |
      Flag that determines whether to apply the beam once per patch or per each
      source (length = 1).
    type: boolean

  - id: parallelbaselines
    doc: |
      Flag that enables parallelization of model computation over baselines.
    type: boolean

  - id: sagecalpredict
    doc: |
      Flag that enables model computation using SAGECal.
    type: boolean

  - id: stepsize
    label: Solver step size
    doc: |
      The solver step size used between iterations (length = 1).
    type: float

  - id: tolerance
    label: Solver tolerance
    doc: |
      The solver tolerance used to define convergence (length = 1).
    type: float

  - id: uvlambdamin
    label: Minimum uv distance
    doc: |
      The minimum uv distance in lambda used during the solve (length = 1).
    type: float

  - id: sector_bounds_deg
    label: Sector boundary
    doc: |
      The boundary of all imaging sectors in degrees (length = 1).
    type: string

  - id: sector_bounds_mid_deg
    label: Sector boundary
    doc: |
      The mid point of the boundary of all imaging sectors in degrees (length = 1).
    type: string

  - id: split_outh5parm
    label: Output split solution tables
    doc: |
      The filenames of the output split h5parm solution tables (length = n_obs * n_split).
    type: string[]

  - id: output_aterms_root
    label: Output root for a-terms
    doc: |
      The root names of the output a-term images (length = n_obs * n_split).
    type: string[]

  - id: screen_type
    label: Type of screen for a-terms
    doc: |
      The screen type to use to derive the a-term images (length = 1).
    type: string

  - id: max_threads
    label: Max number of threads
    doc: |
      The maximum number of threads to use for a job (length = 1).
    type: int

{% if do_slowgain_solve %}
# start do_slowgain_solve
  - id: freqchunk_filename_joint
    label: Filename of input MS for joint solve (frequency)
    doc: |
      The filenames of input MS files for which the first (joint) slow-gain calibration
      will be done (length = n_obs * n_freq_chunks).
    type: Directory[]

  - id: freqchunk_filename_separate
    label: Filename of input MS for separate solve (frequency)
    doc: |
      The filenames of input MS files for which the second (separate) slow-gain
      calibration will be done (length = n_obs * n_freq_chunks).
    type: Directory[]

  - id: slow_starttime_joint
    label: Start time of each chunk for joint solve
    doc: |
      The start time (in casacore MVTime) for each time chunk used in the first (joint)
      slow-gain calibration (length = n_obs * n_freq_chunks).
    type: string[]

  - id: slow_starttime_separate
    label: Start time of each chunk for separate solve
    doc: |
      The start time (in casacore MVTime) for each time chunk used in the second
      (separate) slow-gain calibration (length = n_obs * n_freq_chunks).
    type: string[]

  - id: slow_ntimes_joint
    label: Number of times of each chunk for joint solve
    doc: |
      The number of timeslots for each time chunk used in the first (joint) slow-gain
      calibration (length = n_obs * n_freq_chunks).
    type: int[]

  - id: slow_ntimes_separate
    label: Number of times of each chunk for separate solve
    doc: |
      The number of timeslots for each time chunk used in the second (separate) slow-
      gain calibration (length = n_obs * n_freq_chunks).
    type: int[]

  - id: startchan_joint
    label: Start channel of each chunk for joint solve
    doc: |
      The start channel for each frequency chunk used in the first (joint) slow-gain
      calibration (length = n_obs * n_freq_chunks).
    type: int[]

  - id: startchan_separate
    label: Start channel of each chunk for separate solve
    doc: |
      The start channel for each frequency chunk used in the second (separate) slow-gain
      calibration (length = n_obs * n_freq_chunks).
    type: int[]

  - id: nchan_joint
    label: Number of channels of each chunk for joint solve
    doc: |
      The number of channels for each frequency chunk used in the first (joint) slow-
      gain calibration (length = n_obs * n_freq_chunks).
    type: int[]

  - id: nchan_separate
    label: Number of channels of each chunk for separate solve
    doc: |
      The number of channels for each frequency chunk used in the second (separate)
      slow-gain calibration (length = n_obs * n_freq_chunks).
    type: int[]

  - id: solint_slow_timestep_joint
    label: Joint slow solution interval in time
    doc: |
      The solution interval in number of timeslots for the first (joint) slow-gain
      solve (length = n_obs * n_freq_chunks).
    type: int[]

  - id: solint_slow_timestep_separate
    label: Separate slow solution interval in time
    doc: |
      The solution interval in number of timeslots for the second (separate) slow-gain
      solve (length = n_obs * n_freq_chunks).
    type: int[]

  - id: solint_slow_freqstep_joint
    label: Joint slow solution interval in frequency
    doc: |
      The solution interval in number of frequency channels for the first (joint) slow-
      gain solve (length = n_obs * n_freq_chunks).
    type: int[]

  - id: solint_slow_freqstep_separate
    label: Separate slow solution interval in frequency
    doc: |
      The solution interval in number of frequency channels for the second (separate)
      slow-gain solve (length = n_obs * n_freq_chunks).
    type: int[]

  - id: slow_smoothnessconstraint_joint
    label: Joint slow smoothnessconstraint
    doc: |
      The smoothnessconstraint kernel size in Hz for the first (joint) slow-gain solve
      (length = 1).
    type: float

  - id: slow_smoothnessconstraint_separate
    label: Separate slow smoothnessconstraint
    doc: |
      The smoothnessconstraint kernel size in Hz for the second (separate) slow-gain
      solve (length = 1).
    type: float

  - id: slow_antennaconstraint
    label: Slow antenna constraint
    doc: |
      The antenna constraint for the first (joint) slow-gain solve (length = 1).
    type: string

  - id: max_normalization_delta
    label: Maximum normalization delta
    doc: |
      The maximum allowed difference in the median of the amplitudes from unity, per
      station (length = 1).
    type: float

  - id: scale_normalization_delta
    label: Scale normalization delta flag
    doc: |
      Flag that enables scaling (with distance from the phase center) of the
      maximum allowed difference in the median of the amplitudes from unity, per
      station (length = 1).
    type: string

  - id: phase_center_ra
    label: Phase center RA
    doc: |
      The RA in degrees of the phase center (length = 1).
    type: float

  - id: phase_center_dec
    label: Phase center Dec
    doc: |
      The Dec in degrees of the phase center (length = 1).
    type: float

  - id: output_slow_h5parm_joint
    label: Joint slow solve output solution table
    doc: |
      The filename of the output h5parm solution table for the first (joint) slow-gain
      solve (length = n_obs * n_freq_chunks).
    type: string[]

  - id: output_slow_h5parm_separate
    label: Separate slow solve output solution table
    doc: |
      The filename of the output h5parm solution table for the second (separate) slow-
      gain solve (length = n_obs * n_freq_chunks).
    type: string[]

  - id: combined_slow_h5parm_joint
    label: Combined joint slow output solution table
    doc: |
      The filename of the output combined h5parm solution table for the first (joint)
      slow-gain solve (length = 1).
    type: string

  - id: combined_slow_h5parm_separate
    label: Combined separate slow output solution table
    doc: |
      The filename of the output combined h5parm solution table for the second (separate)
      slow-gain solve (length = 1).
    type: string

  - id: combined_h5parms
    label: Combined output solution table
    doc: |
      The filename of the output combined h5parm solution table for the full solve
      (length = 1).
    type: string

  - id: combined_h5parms_fast_slow_joint
    label: Combined output solution table
    doc: |
      The filename of the output combined h5parm solution table for the fast-phase +
      first (joint) slow-gain solve (length = 1).
    type: string

  - id: combined_h5parms_slow_joint_separate
    label: Combined output solution table
    doc: |
      The filename of the output combined h5parm solution table for the first (joint) and
      second (separate) slow-gain solves (length = 1).
    type: string

{% if do_fulljones_solve %}
# start do_fulljones_solve
  - id: directions_fulljones
    label: Directions for full-Jones solve
    doc: |
      The calibrator patch names to use for the DDECal directions in the full-
      Jones solve. All directions are solved for together to produce direction-
      independent solutions (length = 1).
    type: string

  - id: freqchunk_filename_fulljones
    label: Filename of input MS for full-Jones solve (frequency)
    doc: |
      The filenames of input MS files for which the full-Jones gain
      calibration will be done (length = n_obs * n_freq_chunks).
    type: Directory[]

  - id: starttime_fulljones
    label: Start time of each chunk for full-Jones solve
    doc: |
      The start time (in casacore MVTime) for each time chunk used in the full-
      Jones gain calibration (length = n_obs * n_freq_chunks).
    type: string[]

  - id: ntimes_fulljones
    label: Number of times of each chunk for full-Jones solve
    doc: |
      The number of timeslots for each time chunk used in the full-Jones
      gain calibration (length = n_obs * n_freq_chunks).
    type: int[]

  - id: startchan_fulljones
    label: Start channel of each chunk for full-Jones solve
    doc: |
      The start channel for each frequency chunk used in the full-Jones gain
      calibration (length = n_obs * n_freq_chunks).
    type: int[]

  - id: nchan_fulljones
    label: Number of channels of each chunk for full-Jones solve
    doc: |
      The number of channels for each frequency chunk used in the full-Jones
      gain calibration (length = n_obs * n_freq_chunks).
    type: int[]

  - id: solint_fulljones_timestep
    label: Full-Jones solution interval in time
    doc: |
      The solution interval in number of timeslots for the full-jones gain
      solve (length = n_obs * n_freq_chunks).
    type: int[]

  - id: solint_fulljones_freqstep
    label: Full-Jones solution interval in frequency
    doc: |
      The solution interval in number of frequency channels for the full-Jones
      gain solve (length = n_obs * n_freq_chunks).
    type: int[]

  - id: smoothnessconstraint_fulljones
    label: Full-jones smoothnessconstraint
    doc: |
      The smoothnessconstraint kernel size in Hz for the full-Jones gain
      solve (length = 1).
    type: float

  - id: combined_h5parms_fast_slow_final
    label: Combined output solution table
    doc: |
      The filename of the output combined h5parm solution table for the fast solve plus
      all slow-gain solves (length = 1).
    type: string

{% endif %}
# end do_fulljones_solve

{% endif %}
# end do_slowgain_solve


outputs:
  - id: combined_solutions
    outputSource:
{% if do_fulljones_solve %}
      - combine_dd_and_fulljones_h5parms/combinedh5parm
{% else %}
      - adjust_h5parm_sources/adjustedh5parm
{% endif %}
    type: File
  - id: fast_phase_plots
    outputSource:
      - plot_fast_phase_solutions/plots
    type: File[]
{% if use_screens %}
  - id: diagonal_aterms
    outputSource:
      - merge_aterm_files/output
    type: File[]
{% endif %}
{% if do_slowgain_solve %}
  - id: slow_phase_plots
    outputSource:
      - plot_slow_phase_solutions/plots
    type: File[]
  - id: slow_amp_plots
    outputSource:
      - plot_slow_amp_solutions/plots
    type: File[]
{% endif %}


steps:
  - id: solve_fast_phases
    label: Solve for fast phases
    doc: |
      This step uses DDECal (in DP3) to solve for phase corrections on short
      timescales (< 1 minute), using the input MS files and sourcedb. These
      corrections are used to correct primarily for ionospheric effects.
{% if use_scalarphase %}
    run: {{ rapthor_pipeline_dir }}/steps/ddecal_solve_scalarphase.cwl
{% else %}
    run: {{ rapthor_pipeline_dir }}/steps/ddecal_solve_scalar.cwl
{% endif %}
{% if max_cores is not none %}
    hints:
      ResourceRequirement:
        coresMin: {{ max_cores }}
        coresMax: {{ max_cores }}
{% endif %}
    in:
      - id: msin
        source: timechunk_filename
      - id: starttime
        source: starttime
      - id: ntimes
        source: ntimes
      - id: h5parm
        source: output_fast_h5parm
      - id: solint
        source: solint_fast_timestep
      - id: nchan
        source: solint_fast_freqstep
      - id: sourcedb
        source: calibration_skymodel_file
      - id: llssolver
        source: llssolver
      - id: maxiter
        source: maxiter
      - id: propagatesolutions
        source: propagatesolutions
      - id: solveralgorithm
        source: solveralgorithm
      - id: solverlbfgs_dof
        source: solverlbfgs_dof
      - id: solverlbfgs_iter
        source: solverlbfgs_iter
      - id: solverlbfgs_minibatches
        source: solverlbfgs_minibatches
      - id: onebeamperpatch
        source: onebeamperpatch
      - id: parallelbaselines
        source: parallelbaselines
      - id: sagecalpredict
        source: sagecalpredict
      - id: stepsize
        source: stepsize
      - id: tolerance
        source: tolerance
      - id: uvlambdamin
        source: uvlambdamin
      - id: smoothnessconstraint
        source: fast_smoothnessconstraint
      - id: smoothnessreffrequency
        source: fast_smoothnessreffrequency
      - id: smoothnessrefdistance
        source: fast_smoothnessrefdistance
      - id: antennaconstraint
        source: fast_antennaconstraint
      - id: numthreads
        source: max_threads
    scatter: [msin, starttime, ntimes, h5parm, solint, nchan, smoothnessreffrequency]
    scatterMethod: dotproduct
    out:
      - id: fast_phases_h5parm

  - id: combine_fast_phases
    label: Combine fast-phase solutions
    doc: |
      This step combines all the phase solutions from the solve_fast_phases step
      into a single solution table (h5parm file). If the slow gain solves are
      not done (i.e., the calibration is phase-only), the result is the final
      solution table, used to make a-term images.
    run: {{ rapthor_pipeline_dir }}/steps/collect_h5parms.cwl
    in:
      - id: inh5parms
        source: solve_fast_phases/fast_phases_h5parm
      - id: outputh5parm
        source: combined_fast_h5parm
    out:
      - id: outh5parm

  - id: plot_fast_phase_solutions
    label: Plot fast phase solutions
    doc: |
      This step makes plots of the fast phase solutions.
    run: {{ rapthor_pipeline_dir }}/steps/plot_solutions.cwl
    in:
      - id: h5parm
        source: combine_fast_phases/outh5parm
      - id: soltype
        valueFrom: 'phase'
    out:
      - id: plots

{% if do_slowgain_solve %}
# start do_slowgain_solve

{% if do_joint_solve %}
# start do_joint_solve (solve_slow_gains_joint)

  - id: solve_slow_gains_joint
    label: Joint solve for slow gains
    doc: |
      This step uses DDECal (in DP3) to solve for diagonal gain corrections on long
      timescales (> 10 minute), using the input MS files and sourcedb. These
      corrections are used to correct primarily for beam errors. The fast-
      phase solutions are preapplied and all stations are constrained to
      have the same (joint) solutions.
    run: {{ rapthor_pipeline_dir }}/steps/ddecal_solve_diagonal_joint.cwl
{% if max_cores is not none %}
    hints:
      ResourceRequirement:
        coresMin: {{ max_cores }}
        coresMax: {{ max_cores }}
{% endif %}
    in:
      - id: msin
        source: freqchunk_filename_joint
      - id: starttime
        source: slow_starttime_joint
      - id: ntimes
        source: slow_ntimes_joint
      - id: startchan
        source: startchan_joint
      - id: nchan
        source: nchan_joint
      - id: fast_h5parm
        source: combine_fast_phases/outh5parm
      - id: h5parm
        source: output_slow_h5parm_joint
      - id: solint
        source: solint_slow_timestep_joint
      - id: solve_nchan
        source: solint_slow_freqstep_joint
      - id: sourcedb
        source: calibration_skymodel_file
      - id: llssolver
        source: llssolver
      - id: maxiter
        source: maxiter
      - id: propagatesolutions
        source: propagatesolutions
      - id: solveralgorithm
        source: solveralgorithm
      - id: solverlbfgs_dof
        source: solverlbfgs_dof
      - id: solverlbfgs_iter
        source: solverlbfgs_iter
      - id: solverlbfgs_minibatches
        source: solverlbfgs_minibatches
      - id: onebeamperpatch
        source: onebeamperpatch
      - id: parallelbaselines
        source: parallelbaselines
      - id: sagecalpredict
        source: sagecalpredict
      - id: stepsize
        source: stepsize
      - id: tolerance
        source: tolerance
      - id: uvlambdamin
        source: uvlambdamin
      - id: smoothnessconstraint
        source: slow_smoothnessconstraint_joint
      - id: antennaconstraint
        source: slow_antennaconstraint
      - id: numthreads
        source: max_threads
    scatter: [msin, starttime, ntimes, startchan, nchan, h5parm, solint, solve_nchan]
    scatterMethod: dotproduct
    out:
      - id: slow_gains_h5parm

  - id: combine_slow_gains_joint
    label: Combine joint slow-gain solutions
    doc: |
      This step combines all the gain solutions from the solve_slow_gains_joint step
      into a single solution table (h5parm file).
    run: {{ rapthor_pipeline_dir }}/steps/collect_h5parms.cwl
    in:
      - id: inh5parms
        source: solve_slow_gains_joint/slow_gains_h5parm
      - id: outputh5parm
        source: combined_slow_h5parm_joint
    out:
      - id: outh5parm

  - id: process_slow_gains_joint
    label: Process joint slow-gain solutions
    doc: |
      This step processes the joint slow-gain solutions, flagging, smoothing and
      renormalizing them.
    run: {{ rapthor_pipeline_dir }}/steps/process_slow_gains.cwl
    in:
      - id: slowh5parm
        source: combine_slow_gains_joint/outh5parm
      - id: flag
        valueFrom: 'True'
      - id: smooth
        valueFrom: 'True'
      - id: max_station_delta
        valueFrom: 0.0
      - id: phase_center_ra
        source: phase_center_ra
      - id: phase_center_dec
        source: phase_center_dec
    out:
      - id: outh5parm

  - id: combine_fast_and_joint_slow_h5parms
    label: Combine fast-phase and joint slow-gain solutions
    doc: |
      This step combines the fast-phase solutions from the solve_fast_phases step
      and the slow-gain solutions from the solve_slow_gains_joint into a single
      solution table (h5parm file).
    run: {{ rapthor_pipeline_dir }}/steps/combine_h5parms.cwl
    in:
      - id: inh5parm1
        source: combine_fast_phases/outh5parm
      - id: inh5parm2
        source: process_slow_gains_joint/outh5parm
      - id: outh5parm
        source: combined_h5parms_fast_slow_joint
      - id: mode
        valueFrom: 'p1a2'
      - id: reweight
        valueFrom: 'False'
      - id: calibrator_names
        source: calibrator_patch_names
      - id: calibrator_fluxes
        source: calibrator_fluxes
    out:
      - id: combinedh5parm

{% endif %}

  - id: solve_slow_gains_separate
    label: Separate solve for slow gains
    doc: |
      This step uses DDECal (in DP3) to solve for diagonal gain corrections on long
      timescales (> 10 minute), using the input MS files and sourcedb. These
      corrections are used to correct primarily for beam errors. The fast-
      phase solutions and first (joint) slow-gain solutions are preapplied
      and stations are solve for separately (so different stations are free
      to have different solutions).
{% if do_joint_solve %}
    run: {{ rapthor_pipeline_dir }}/steps/ddecal_solve_diagonal_separate.cwl
{% else %}
    run: {{ rapthor_pipeline_dir }}/steps/ddecal_solve_diagonal_separate_no_joint.cwl
{% endif %}
{% if max_cores is not none %}
    hints:
      ResourceRequirement:
        coresMin: {{ max_cores }}
        coresMax: {{ max_cores }}
{% endif %}
    in:
      - id: msin
        source: freqchunk_filename_separate
      - id: starttime
        source: slow_starttime_separate
      - id: ntimes
        source: slow_ntimes_separate
      - id: startchan
        source: startchan_separate
      - id: nchan
        source: nchan_separate
      - id: combined_h5parm
{% if do_joint_solve %}
        source: combine_fast_and_joint_slow_h5parms/combinedh5parm
{% else %}
        source: combine_fast_phases/outh5parm
{% endif %}
      - id: h5parm
        source: output_slow_h5parm_separate
      - id: solint
        source: solint_slow_timestep_separate
      - id: solve_nchan
        source: solint_slow_freqstep_separate
      - id: sourcedb
        source: calibration_skymodel_file
      - id: llssolver
        source: llssolver
      - id: maxiter
        source: maxiter
      - id: propagatesolutions
        source: propagatesolutions
      - id: solveralgorithm
        source: solveralgorithm
      - id: solverlbfgs_dof
        source: solverlbfgs_dof
      - id: solverlbfgs_iter
        source: solverlbfgs_iter
      - id: solverlbfgs_minibatches
        source: solverlbfgs_minibatches
      - id: onebeamperpatch
        source: onebeamperpatch
      - id: parallelbaselines
        source: parallelbaselines
      - id: sagecalpredict
        source: sagecalpredict
      - id: stepsize
        source: stepsize
      - id: tolerance
        source: tolerance
      - id: uvlambdamin
        source: uvlambdamin
      - id: smoothnessconstraint
        source: slow_smoothnessconstraint_separate
      - id: numthreads
        source: max_threads
    scatter: [msin, starttime, ntimes, startchan, nchan, h5parm, solint, solve_nchan]
    scatterMethod: dotproduct
    out:
      - id: slow_gains_h5parm

  - id: combine_slow_gains_separate
    label: Combine separate slow-gain solutions
    doc: |
      This step combines all the gain solutions from the solve_slow_gains_separate step
      into a single solution table (h5parm file).
    run: {{ rapthor_pipeline_dir }}/steps/collect_h5parms.cwl
    in:
      - id: inh5parms
        source: solve_slow_gains_separate/slow_gains_h5parm
      - id: outputh5parm
        source: combined_slow_h5parm_separate
    out:
      - id: outh5parm

  - id: process_slow_gains_separate
    label: Process separate slow-gain solutions
    doc: |
      This step processes the gain solutions from the separate solve, flagging,
      smoothing and renormalizing them.
    run: {{ rapthor_pipeline_dir }}/steps/process_slow_gains.cwl
    in:
      - id: slowh5parm
        source: combine_slow_gains_separate/outh5parm
      - id: flag
        valueFrom: 'True'
      - id: smooth
        valueFrom: 'True'
      - id: max_station_delta
        source: max_normalization_delta
      - id: scale_station_delta
        source: scale_normalization_delta
      - id: phase_center_ra
        source: phase_center_ra
      - id: phase_center_dec
        source: phase_center_dec
    out:
      - id: outh5parm

{% if do_joint_solve %}

  - id: combine_joint_and_separate_slow_h5parms
    label: Combine slow-gain solutions
    doc: |
      This step combines the gain solutions from the solve_slow_gains_joint and
      solve_slow_gains_separate steps into a single solution table (h5parm file).
      The phases and amplitudes from solve_slow_gains_separate and the amplitudes from
      solve_slow_gains_joint are used.
    run: {{ rapthor_pipeline_dir }}/steps/combine_h5parms.cwl
    in:
      - id: inh5parm1
        source: process_slow_gains_separate/outh5parm
      - id: inh5parm2
        source: combine_slow_gains_joint/outh5parm
      - id: outh5parm
        source: combined_h5parms_slow_joint_separate
      - id: mode
        valueFrom: 'p1a1a2'
      - id: reweight
        valueFrom: 'False'
      - id: calibrator_names
        source: calibrator_patch_names
      - id: calibrator_fluxes
        source: calibrator_fluxes
    out:
      - id: combinedh5parm

{% endif %}

  - id: normalize_slow_amplitudes
    label: Normalize slow-gain amplitudes
    doc: |
      This step processes the combined amplitude solutions from
      combine_joint_and_separate_slow_h5parms, flagging and renormalizing them.
    run: {{ rapthor_pipeline_dir }}/steps/process_slow_gains.cwl
    in:
      - id: slowh5parm
{% if do_joint_solve %}
        source: combine_joint_and_separate_slow_h5parms/combinedh5parm
{% else %}
        source: process_slow_gains_separate/outh5parm
{% endif %}
      - id: flag
        valueFrom: 'True'
      - id: smooth
        valueFrom: 'False'
      - id: max_station_delta
        source: max_normalization_delta
      - id: scale_station_delta
        source: scale_normalization_delta
      - id: phase_center_ra
        source: phase_center_ra
      - id: phase_center_dec
        source: phase_center_dec
    out:
      - id: outh5parm

  - id: plot_slow_phase_solutions
    label: Plot slow phase solutions
    doc: |
      This step makes plots of the slow phase solutions.
    run: {{ rapthor_pipeline_dir }}/steps/plot_solutions.cwl
    in:
      - id: h5parm
        source: normalize_slow_amplitudes/outh5parm
      - id: soltype
        valueFrom: 'phase'
    out:
      - id: plots

  - id: plot_slow_amp_solutions
    label: Plot slow amp solutions
    doc: |
      This step makes plots of the slow amplitude solutions.
    run: {{ rapthor_pipeline_dir }}/steps/plot_solutions.cwl
    in:
      - id: h5parm
        source: normalize_slow_amplitudes/outh5parm
      - id: soltype
        valueFrom: 'amplitude'
    out:
      - id: plots

  - id: combine_fast_and_full_slow_h5parms
    label: Combine fast-phase and slow-gain solutions
    doc: |
      This step combines the phase solutions from the solve_fast_phases and
      the combined (and renormalized) slow gains into a single solution table
      (h5parm file). The phases from combine_fast_phases and the phases and
      amplitudes from normalize_slow_amplitudes are used. The result is the
      final solution table, used to make a-term images.
    run: {{ rapthor_pipeline_dir }}/steps/combine_h5parms.cwl
    in:
      - id: inh5parm1
        source: combine_fast_phases/outh5parm
      - id: inh5parm2
        source: normalize_slow_amplitudes/outh5parm
      - id: outh5parm
        source: combined_h5parms_fast_slow_final
      - id: mode
{% if use_screens %}
        valueFrom: 'p1p2a2'
      - id: reweight
        valueFrom: 'True'
{% elif use_facets %}
{% if apply_diagonal_solutions %}
        valueFrom: 'p1p2a2_diagonal'
{% else %}
        valueFrom: 'p1p2a2_scalar'
{% endif %}
      - id: reweight
        valueFrom: 'False'
{% else %}
        valueFrom: 'p1p2a2_diagonal'
      - id: reweight
        valueFrom: 'False'
{% endif %}
      - id: calibrator_names
        source: calibrator_patch_names
      - id: calibrator_fluxes
        source: calibrator_fluxes
    out:
      - id: combinedh5parm

  - id: adjust_h5parm_sources
    label: Adjust h5parm sources
    doc: |
      This step adjusts the h5parm source coordinates to match those in the sky model.
    run: {{ rapthor_pipeline_dir }}/steps/adjust_h5parm_sources.cwl
    in:
      - id: skymodel
        source: calibration_skymodel_file
      - id: h5parm
        source: combine_fast_and_full_slow_h5parms/combinedh5parm
    out:
      - id: adjustedh5parm

{% if do_fulljones_solve %}

  - id: solve_fulljones_gains
    label: Solve for full-Jones gains
    doc: |
      This step uses DDECal (in DP3) to solve for full-Jones gain corrections,
      using the input MS files and sourcedb. These corrections are used to
      correct primarily for polarization errors. The direction-dependent
      solutions are preapplied. All directions are solved for jointly, resulting
      in direction-indedpendent solutions.
    run: {{ rapthor_pipeline_dir }}/steps/ddecal_solve_fulljones.cwl
{% if max_cores is not none %}
    hints:
      ResourceRequirement:
        coresMin: {{ max_cores }}
        coresMax: {{ max_cores }}
{% endif %}
    in:
      - id: msin
        source: freqchunk_filename_fulljones
      - id: starttime
        source: starttime_fulljones
      - id: ntimes
        source: ntimes_fulljones
      - id: startchan
        source: startchan_fulljones
      - id: nchan
        source: nchan_fulljones
      - id: directions
        source: directions_fulljones
      - id: combined_h5parm
        source: adjust_h5parm_sources/adjustedh5parm
      - id: h5parm
        source: output_h5parm_fulljones
      - id: solint
        source: solint_fulljones_timestep
      - id: solve_nchan
        source: solint_fulljones_freqstep
      - id: sourcedb
        source: calibration_skymodel_file
      - id: llssolver
        source: llssolver
      - id: maxiter
        source: maxiter
      - id: propagatesolutions
        source: propagatesolutions
      - id: solveralgorithm
        source: solveralgorithm
      - id: solverlbfgs_dof
        source: solverlbfgs_dof
      - id: solverlbfgs_iter
        source: solverlbfgs_iter
      - id: solverlbfgs_minibatches
        source: solverlbfgs_minibatches
      - id: onebeamperpatch
        source: onebeamperpatch
      - id: parallelbaselines
        source: parallelbaselines
      - id: sagecalpredict
        source: sagecalpredict
      - id: stepsize
        source: stepsize
      - id: tolerance
        source: tolerance
      - id: uvlambdamin
        source: uvlambdamin
      - id: smoothnessconstraint
        source: smoothnessconstraint_fulljones
      - id: numthreads
        source: max_threads
    scatter: [msin, starttime, ntimes, startchan, nchan, h5parm, solint, solve_nchan]
    scatterMethod: dotproduct
    out:
      - id: fulljonesh5parm

  - id: combine_dd_and_fulljones_h5parms
    label: Combine DD and full-Jones solutions
    doc: |
      This step combines the direction-dependent gain solutions and the direction-
      independent full-Jones solutions into a single h5parm file. The direction-dependent
      and direction-independent solutions are kept in separate solsets.
    run: {{ rapthor_pipeline_dir }}/steps/combine_h5parms.cwl
    in:
      - id: inh5parm1
        source: adjust_h5parm_sources/adjustedh5parm
      - id: inh5parm2
        source: solve_fulljones_gains/fulljonesh5parm
      - id: outh5parm
        source: combined_h5parms
      - id: mode
        valueFrom: 'separate'
      - id: reweight
        valueFrom: 'False'
      - id: calibrator_names
        source: calibrator_patch_names
      - id: calibrator_fluxes
        source: calibrator_fluxes
    out:
      - id: combinedh5parm
{% endif %}

{% if use_screens %}
# start use_screens

  - id: split_h5parms
    label: Split solution table
    doc: |
      This step splits the final solution table in time, to enable multi-node
      parallel processing of the a-term images.
    run: {{ rapthor_pipeline_dir }}/steps/split_h5parms.cwl
    in:
      - id: inh5parm
        source: adjust_h5parm_sources/adjustedh5parm
      - id: outh5parms
        source: split_outh5parm
      - id: soltabname
        valueFrom: 'gain000'
    out:
      - id: splith5parms

  - id: make_aterms
    label: Make a-term images
    doc: |
      This step makes a-term images from the split final solution tables.
    run: {{ rapthor_pipeline_dir }}/steps/make_aterm_images.cwl
    in:
      - id: h5parm
        source: split_h5parms/splith5parms
      - id: soltabname
        valueFrom: 'gain000'
      - id: screen_type
        source: screen_type
      - id: skymodel
        source: calibration_skymodel_file
      - id: outroot
        source: output_aterms_root
      - id: sector_bounds_deg
        source: sector_bounds_deg
      - id: sector_bounds_mid_deg
        source: sector_bounds_mid_deg
      - id: ncpu
        source: max_threads
    scatter: [h5parm, outroot]
    scatterMethod: dotproduct
    out:
      - id: output_images

{% endif %}
# end use_screens

{% else %}
# start not do_slowgain_solve

  - id: adjust_h5parm_sources
    label: Adjust h5parm sources
    doc: |
      This step adjusts the h5parm source coordinates to match those in the sky model.
    run: {{ rapthor_pipeline_dir }}/steps/adjust_h5parm_sources.cwl
    in:
      - id: skymodel
        source: calibration_skymodel_file
      - id: h5parm
        source: combine_fast_phases/outh5parm
    out:
      - id: adjustedh5parm

{% if use_screens %}
# start use_screens

  - id: split_h5parms
    label: Split solution table
    doc: |
      This step splits the final solution table in time, to enable multi-node
      parallel processing of the a-term images.
    run: {{ rapthor_pipeline_dir }}/steps/split_h5parms.cwl
    in:
      - id: inh5parm
        source: adjust_h5parm_sources/adjustedh5parm
      - id: outh5parms
        source: split_outh5parm
      - id: soltabname
        valueFrom: 'phase000'
    out:
      - id: splith5parms

  - id: make_aterms
    label: Make a-term images
    doc: |
      This step makes a-term images from the split final solution tables.
    run: {{ rapthor_pipeline_dir }}/steps/make_aterm_images.cwl
    in:
      - id: h5parm
        source: split_h5parms/splith5parms
      - id: soltabname
        valueFrom: 'phase000'
      - id: screen_type
        source: screen_type
      - id: skymodel
        source: calibration_skymodel_file
      - id: outroot
        source: output_aterms_root
      - id: sector_bounds_deg
        source: sector_bounds_deg
      - id: sector_bounds_mid_deg
        source: sector_bounds_mid_deg
      - id: ncpu
        source: max_threads
    scatter: [h5parm, outroot]
    scatterMethod: dotproduct
    out:
      - id: output_images

{% endif %}
# end use_screens

{% endif %}
# end do_slowgain_solve / not do_slowgain_solve

{% if use_screens %}

  - id: merge_aterm_files
    in:
      - id: input
        source:
          - make_aterms/output_images
    out:
      - id: output
    run: {{ rapthor_pipeline_dir }}/steps/merge_array_files.cwl
    label: merge_aterm_files

{% endif %}

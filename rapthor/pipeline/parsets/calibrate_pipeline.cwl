cwlVersion: v1.2
class: Workflow
label: Rapthor DD calibration workflow
doc: |
  This workflow performs direction-dependent calibration. In general,
  calibration is done in three steps: (1) a fast phase-only calibration (with
  core stations constrianed to have the same solutions) to correct for
  ionospheric effects, (2) a joint slow amplitude calibration (with all stations
  constrained to have the same solutions) to correct for beam errors, and (3) a
  further unconstrained slow gain calibration to correct for station-to-station
  differences. Steps (2) and (3) are skipped if the calibration is phase-only.
  This calibration scheme works for both HBA and LBA data. The final products of
  this workflow are solution tables (h5parm files) and plots.

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
  
  - id: data_colname
    label: Input MS data column
    doc: |
      The data column to be read from the MS files (length = 1).
    type: string

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

  - id: solutions_per_direction_fast
    label: Fast number of solutions per direction
    doc: |
      The number of solutions per direction for the fast phase solve (length =
      n_obs * n_calibrators * n_time_chunks).
    type:
      type: array
      items:
        type: array
        items: int

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

  - id: smoothness_dd_factors
    label: Smoothness factors
    doc: |
      The factor by which to multiply the smoothnesscontraint, per direction (length =
      n_obs * n_calibrators * n_time_chunks).
    type:
      type: array
      items:
        type: array
        items: float

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

  - id: dp3_solve_mode_fast
    label: Solve mode for fast solve
    doc: |
      The solve mode to use for the fast-phase calibration (length = 1).
    type: string

  - id: dp3_steps_fast
    label: Steps for fast solve
    doc: |
      The list of DP3 steps to use in the fast-phase calibration (length = 1).
    type: string

  - id: dp3_applycal_steps_fast
    label: Applycal steps for fast solve
    doc: |
      The list of DP3 applycal steps to use in the fast-phase calibration (length = 1).
    type: string?

  - id: normalize_h5parm
    label: The filename of normalization h5parm
    doc: |
      The filename of the input flux-scale normalization h5parm (length = 1).
    type: File?

  - id: bda_timebase_fast
    label: BDA timebase for fast solve
    doc: |
      The baseline length (in meters) below which BDA time averaging is done in the
      fast-phase calibration (length = 1).
    type: float

  - id: bda_maxinterval_fast
    label: BDA maxinterval for fast solve
    doc: |
      The maximum interval duration (in time slots) over which BDA time averaging is
      done in the fast-phase calibration (length = n_obs * n_time_chunks).
    type: int[]

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

  - id: fast_initialsolutions_h5parm
    label: Input solution table
    doc: |
      The filename of the input h5parm solution table to use for the fast-phase
      initial solutions (length = 1).
    type: File?

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

  - id: fast_datause
    doc: |
      DDECal datause option for the fast-phase calibration (length = 1).
    type: string

  - id: stepsize
    label: Solver step size
    doc: |
      The solver step size used between iterations (length = 1).
    type: float

  - id: stepsigma
    label: Solver step size reduction factor
    doc: |
      If the solver step size mean is lower than its standard deviation by this
      factor, stop iterations (length = 1).
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

  - id: max_threads
    label: Max number of threads
    doc: |
      The maximum number of threads to use for a job (length = 1).
    type: int

{% if do_slowgain_solve %}
# start do_slowgain_solve
  - id: dp3_steps_slow_joint
    label: Steps for joint solve
    doc: |
      The list of DP3 steps to use in the first (joint) slow-gain calibration
      (length = 1).
    type: string

  - id: slow_datause
    doc: |
      DDECal datause option for the slow-gain calibration (length = 1).
    type: string

  - id: bda_timebase_slow_joint
    label: BDA timebase for joint solve
    doc: |
      The baseline length (in meters) below which BDA time averaging is done in the
      first (joint) slow-gain calibration (length = 1).
    type: float

  - id: bda_maxinterval_slow_joint
    label: BDA maxinterval for joint solve
    doc: |
      The maximum interval duration (in time slots) over which BDA time averaging is
      done in the first (joint) slow-gain calibration (length = n_obs * n_freq_chunks).
    type: int[]

  - id: dp3_steps_slow_separate
    label: Steps for separate solve
    doc: |
      The list of DP3 steps to use in the second (separate) slow-gain calibration
      (length = 1).
    type: string

  - id: dp3_applycal_steps_slow_joint
    label: Applycal steps for slow joint solve
    doc: |
      The list of DP3 applycal steps to use in the first (joint) slow-gain calibration
      (length = 1).
    type: string

  - id: dp3_applycal_steps_slow_separate
    label: Applycal steps for slow separate solve
    doc: |
      The list of DP3 applycal steps to use in the second (separate) slow-gain calibration
      (length = 1).
    type: string

  - id: bda_timebase_slow_separate
    label: BDA timebase for separate solve
    doc: |
      The baseline length (in meters) below which BDA time averaging is done in the
      second (separate) slow-gain calibration (length = 1).
    type: float

  - id: bda_maxinterval_slow_separate
    label: BDA maxinterval for separate solve
    doc: |
      The maximum interval duration (in time slots) over which BDA time averaging is
      done in the second (separate) slow-gain calibration (length = n_obs *
      n_freq_chunks).
    type: int[]

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

  - id: solutions_per_direction_slow_joint
    label: Joint slow number of solutions per direction
    doc: |
      The number of solutions per direction for the first (joint) slow-
      gain  (length = n_obs * n_directions * n_freq_chunks).
    type:
      type: array
      items:
        type: array
        items: int

  - id: solutions_per_direction_slow_separate
    label: Separate slow number of solutions per direction
    doc: |
      The number of solutions per direction for the second (separate)
      slow-gain solve (length = n_obs * n_directions * n_freq_chunks).
    type:
      type: array
      items:
        type: array
        items: int

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

  - id: slow_initialsolutions_h5parm
    label: Input solution table
    doc: |
      The filename of the input h5parm solution table to use for the second
      (separate) slow-gain initial solutions (length = 1).
    type: File?

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

  - id: solution_combine_mode
    label: Mode for combining solutions
    doc: |
      The mode used for combining the fast-phase and joint slow-gain solutions
      (length = 1).
    type: string?

{% endif %}
# end do_slowgain_solve


outputs:
  - id: fast_phase_solutions
    outputSource:
      - combine_fast_phases/outh5parm
    type: File
  - id: combined_solutions
    outputSource:
      - adjust_h5parm_sources/adjustedh5parm
    type: File
  - id: fast_phase_plots
    outputSource:
      - plot_fast_phase_solutions/plots
    type: File[]
{% if do_slowgain_solve %}
  - id: slow_gain_solutions
    outputSource:
      - combine_slow_gains_separate/outh5parm
    type: File
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
    run: {{ rapthor_pipeline_dir }}/steps/ddecal_solve.cwl
{% if max_cores is not none %}
    hints:
      ResourceRequirement:
        coresMin: {{ max_cores }}
        coresMax: {{ max_cores }}
{% endif %}
    in:
      - id: msin
        source: timechunk_filename
      - id: data_colname
        source: data_colname
      - id: starttime
        source: starttime
      - id: ntimes
        source: ntimes
      - id: h5parm
        source: output_fast_h5parm
      - id: solint
        source: solint_fast_timestep
      - id: mode
        source: dp3_solve_mode_fast
      - id: steps
        source: dp3_steps_fast
      - id: applycal_steps
        source: dp3_applycal_steps_fast
      - id: normalize_h5parm
        source: normalize_h5parm
      - id: timebase
        source: bda_timebase_fast
      - id: maxinterval
        source: bda_maxinterval_fast
      - id: solve_nchan
        source: solint_fast_freqstep
      - id: directions
        source: calibrator_patch_names
      - id: solutions_per_direction
        source: solutions_per_direction_fast
      - id: sourcedb
        source: calibration_skymodel_file
      - id: llssolver
        source: llssolver
      - id: maxiter
        source: maxiter
      - id: propagatesolutions
        source: propagatesolutions
      - id: initialsolutions_h5parm
        source: fast_initialsolutions_h5parm
      - id: initialsolutions_soltab
        valueFrom: '[phase000]'
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
      - id: datause
        source: fast_datause
      - id: stepsize
        source: stepsize
      - id: stepsigma
        source: stepsigma
      - id: tolerance
        source: tolerance
      - id: uvlambdamin
        source: uvlambdamin
      - id: smoothness_dd_factors
        source: smoothness_dd_factors
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
    scatter: [msin, starttime, ntimes, h5parm, solint, solve_nchan, maxinterval, smoothnessreffrequency, solutions_per_direction, smoothness_dd_factors]
    scatterMethod: dotproduct
    out:
      - id: output_h5parm

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
        source: solve_fast_phases/output_h5parm
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
    run: {{ rapthor_pipeline_dir }}/steps/ddecal_solve.cwl
{% if max_cores is not none %}
    hints:
      ResourceRequirement:
        coresMin: {{ max_cores }}
        coresMax: {{ max_cores }}
{% endif %}
    in:
      - id: msin
        source: freqchunk_filename_joint
      - id: data_colname
        source: data_colname
      - id: starttime
        source: slow_starttime_joint
      - id: ntimes
        source: slow_ntimes_joint
      - id: startchan
        source: startchan_joint
      - id: nchan
        source: nchan_joint
      - id: mode
        valueFrom: 'diagonal'
      - id: steps
        source: dp3_steps_slow_joint
      - id: applycal_steps
        source: dp3_applycal_steps_slow_joint
      - id: timebase
        source: bda_timebase_slow_joint
      - id: maxinterval
        source: bda_maxinterval_slow_joint
      - id: fastphase_h5parm
        source: combine_fast_phases/outh5parm
      - id: normalize_h5parm
        source: normalize_h5parm
      - id: h5parm
        source: output_slow_h5parm_joint
      - id: solint
        source: solint_slow_timestep_joint
      - id: solve_nchan
        source: solint_slow_freqstep_joint
      - id: directions
        source: calibrator_patch_names
      - id: solutions_per_direction
        source: solutions_per_direction_slow_joint
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
      - id: datause
        source: slow_datause
      - id: stepsize
        source: stepsize
      - id: stepsigma
        source: stepsigma
      - id: tolerance
        source: tolerance
      - id: uvlambdamin
        source: uvlambdamin
      - id: smoothness_dd_factors
        source: smoothness_dd_factors
      - id: smoothnessconstraint
        source: slow_smoothnessconstraint_joint
      - id: antennaconstraint
        source: slow_antennaconstraint
      - id: numthreads
        source: max_threads
    scatter: [msin, starttime, ntimes, startchan, nchan, maxinterval, h5parm, solint, solve_nchan, solutions_per_direction, smoothness_dd_factors]
    scatterMethod: dotproduct
    out:
      - id: output_h5parm

  - id: combine_slow_gains_joint
    label: Combine joint slow-gain solutions
    doc: |
      This step combines all the gain solutions from the solve_slow_gains_joint step
      into a single solution table (h5parm file).
    run: {{ rapthor_pipeline_dir }}/steps/collect_h5parms.cwl
    in:
      - id: inh5parms
        source: solve_slow_gains_joint/output_h5parm
      - id: outputh5parm
        source: combined_slow_h5parm_joint
    out:
      - id: outh5parm

  - id: process_slow_gains_joint
    label: Process joint slow-gain solutions
    doc: |
      This step processes the joint slow-gain solutions, flagging, smoothing and
      renormalizing them.
    run: {{ rapthor_pipeline_dir }}/steps/process_gains.cwl
    in:
      - id: h5parm
        source: combine_slow_gains_joint/outh5parm
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
    run: {{ rapthor_pipeline_dir }}/steps/ddecal_solve.cwl
{% if max_cores is not none %}
    hints:
      ResourceRequirement:
        coresMin: {{ max_cores }}
        coresMax: {{ max_cores }}
{% endif %}
    in:
      - id: msin
        source: freqchunk_filename_separate
      - id: data_colname
        source: data_colname
      - id: starttime
        source: slow_starttime_separate
      - id: ntimes
        source: slow_ntimes_separate
      - id: startchan
        source: startchan_separate
      - id: mode
        valueFrom: 'diagonal'
      - id: steps
        source: dp3_steps_slow_separate
      - id: applycal_steps
        source: dp3_applycal_steps_slow_separate
      - id: timebase
        source: bda_timebase_slow_separate
      - id: maxinterval
        source: bda_maxinterval_slow_separate
      - id: nchan
        source: nchan_separate
      - id: fastphase_h5parm
        source: combine_fast_phases/outh5parm
{% if do_joint_solve %}
      - id: slowgain_h5parm
        source: process_slow_gains_joint/outh5parm
{% endif %}
      - id: normalize_h5parm
        source: normalize_h5parm
      - id: h5parm
        source: output_slow_h5parm_separate
      - id: solint
        source: solint_slow_timestep_separate
      - id: solve_nchan
        source: solint_slow_freqstep_separate
      - id: directions
        source: calibrator_patch_names
      - id: solutions_per_direction
        source: solutions_per_direction_slow_separate
      - id: sourcedb
        source: calibration_skymodel_file
      - id: llssolver
        source: llssolver
      - id: maxiter
        source: maxiter
      - id: propagatesolutions
        source: propagatesolutions
      - id: initialsolutions_h5parm
        source: slow_initialsolutions_h5parm
      - id: initialsolutions_soltab
        valueFrom: '[phase000,amplitude000]'
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
      - id: datause
        source: slow_datause
      - id: stepsize
        source: stepsize
      - id: stepsigma
        source: stepsigma
      - id: tolerance
        source: tolerance
      - id: uvlambdamin
        source: uvlambdamin
      - id: smoothness_dd_factors
        source: smoothness_dd_factors
      - id: smoothnessconstraint
        source: slow_smoothnessconstraint_separate
      - id: numthreads
        source: max_threads
    scatter: [msin, starttime, ntimes, startchan, nchan, maxinterval, h5parm, solint, solve_nchan, solutions_per_direction, smoothness_dd_factors]
    scatterMethod: dotproduct
    out:
      - id: output_h5parm

  - id: combine_slow_gains_separate
    label: Combine separate slow-gain solutions
    doc: |
      This step combines all the gain solutions from the solve_slow_gains_separate step
      into a single solution table (h5parm file).
    run: {{ rapthor_pipeline_dir }}/steps/collect_h5parms.cwl
    in:
      - id: inh5parms
        source: solve_slow_gains_separate/output_h5parm
      - id: outputh5parm
        source: combined_slow_h5parm_separate
    out:
      - id: outh5parm

  - id: process_slow_gains_separate
    label: Process separate slow-gain solutions
    doc: |
      This step processes the gain solutions from the separate solve, flagging,
      smoothing and renormalizing them.
    run: {{ rapthor_pipeline_dir }}/steps/process_gains.cwl
    in:
      - id: h5parm
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
    run: {{ rapthor_pipeline_dir }}/steps/process_gains.cwl
    in:
      - id: h5parm
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
        source: combined_h5parms
      - id: mode
        source: solution_combine_mode
      - id: reweight
        valueFrom: 'False'
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

{% endif %}
# end do_slowgain_solve / not do_slowgain_solve

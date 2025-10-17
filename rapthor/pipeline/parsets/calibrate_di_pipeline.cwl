cwlVersion: v1.2
class: Workflow
label: Rapthor DI calibration workflow
doc: |
  This workflow performs direction-independent calibration. A full-Jones solve is
  done (meaning that solutions for all four polarization are found), with the
  purpose of improving the calibration done in the LINC pipelines to allow for
  Stokes IQUV imaging. The solve is done against the model data column produced by
  the PredictDI operation. The final products of this workflow are solution tables
  (h5parm files) and plots.

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
  - id: freqchunk_filename_fulljones
    label: Filename of input MS for full-Jones solve (frequency)
    doc: |
      The filenames of input MS files for which the full-Jones gain
      calibration will be done (length = n_obs * n_freq_chunks).
    type: Directory[]
    
  - id: data_colname
    label: Input MS data column
    doc: |
      The data column to be read from the MS files (length = 1).
    type: string

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

  - id: output_h5parm_fulljones
    label: Full-Jones solve output solution table
    doc: |
      The filename of the output h5parm solution table for the full-Jones
      gain solve (length = n_obs * n_freq_chunks).
    type: string[]

  - id: combined_h5parm_fulljones
    label: Combined full-Jones output solution table
    doc: |
      The filename of the output combined h5parm solution table for the full-Jones
      gain solve (length = 1).
    type: string

  - id: max_normalization_delta
    label: Maximum normalization delta
    doc: |
      The maximum allowed difference in the median of the amplitudes from unity, per
      station (length = 1).
    type: float

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

  - id: correctfreqsmearing
    label: Correct for frequency smearing
    doc: |
      Flag that determines whether the effects of frequency smearing are
      corrected for during the prediction part of the solve (length = 1).
    type: boolean

  - id: correcttimesmearing
    label: Correct for time smearing
    doc: |
      Flag that determines whether the effects of time smearing are
      corrected for during the prediction part of the solve (length = 1).
    type: boolean

  - id: max_threads
    label: Max number of threads
    doc: |
      The maximum number of threads to use for a job (length = 1).
    type: int


outputs:
  - id: combined_solutions
    outputSource:
      - process_fulljones_gains/outh5parm
    type: File
  - id: fulljones_phase_plots
    outputSource:
      - plot_fulljones_phase_solutions/plots
    type: File[]
  - id: fulljones_amp_plots
    outputSource:
      - plot_fulljones_amp_solutions/plots
    type: File[]


steps:
  - id: solve_fulljones_gains
    label: Solve for full-Jones gains
    doc: |
      This step uses DDECal (in DP3) to solve for full-Jones, direction-independent gain
      corrections, using the input MS files and model data. These corrections are used to
      correct primarily for polarization errors.
    run: {{ rapthor_pipeline_dir }}/steps/ddecal_solve.cwl
{% if max_cores is not none %}
    hints:
      ResourceRequirement:
        coresMin: {{ max_cores }}
        coresMax: {{ max_cores }}
{% endif %}
    in:
      - id: msin
        source: freqchunk_filename_fulljones
      - id: data_colname
        source: data_colname
      - id: starttime
        source: starttime_fulljones
      - id: ntimes
        source: ntimes_fulljones
      - id: solve1_mode
        valueFrom: 'fulljones'
      - id: steps
        valueFrom: '[solve1]'
      - id: solve1_h5parm
        source: output_h5parm_fulljones
      - id: solve1_solint
        source: solint_fulljones_timestep
      - id: solve1_nchan
        source: solint_fulljones_freqstep
      - id: modeldatacolumn
        valueFrom: '[MODEL_DATA]'
      - id: solve1_llssolver
        source: llssolver
      - id: solve1_maxiter
        source: maxiter
      - id: solve1_propagatesolutions
        source: propagatesolutions
      - id: solve1_solveralgorithm
        source: solveralgorithm
      - id: solve1_solverlbfgs_dof
        source: solverlbfgs_dof
      - id: solve1_solverlbfgs_iter
        source: solverlbfgs_iter
      - id: solve1_solverlbfgs_minibatches
        source: solverlbfgs_minibatches
      - id: solve1_stepsize
        source: stepsize
      - id: solve1_stepsigma
        source: stepsigma
      - id: solve1_tolerance
        source: tolerance
      - id: solve1_uvlambdamin
        source: uvlambdamin
      - id: solve1_smoothnessconstraint
        source: smoothnessconstraint_fulljones
      - id: solve1_correctfreqsmearing
        source: correctfreqsmearing
      - id: solve1_correcttimesmearing
        source: correcttimesmearing
      - id: numthreads
        source: max_threads
    scatter: [msin, starttime, ntimes, solve1_h5parm, solve1_solint, solve1_nchan]
    scatterMethod: dotproduct
    out:
      - id: output_h5parm1

  - id: combine_fulljones_gains
    label: Combine full-Jones gain solutions
    doc: |
      This step combines all the gain solutions from the solve_fulljones_gains step
      into a single solution table (h5parm file).
    run: {{ rapthor_pipeline_dir }}/steps/collect_h5parms.cwl
    in:
      - id: inh5parms
        source: solve_fulljones_gains/output_h5parm1
      - id: outputh5parm
        source: combined_h5parm_fulljones
    out:
      - id: outh5parm

  - id: process_fulljones_gains
    label: Normalize full-Jones amplitudes
    doc: |
      This step processes the combined amplitude solutions from
      combine_fulljones_gains, flagging and renormalizing them.
    run: {{ rapthor_pipeline_dir }}/steps/process_gains.cwl
    in:
      - id: h5parm
        source: combine_fulljones_gains/outh5parm
      - id: flag
        valueFrom: 'False'
      - id: smooth
        valueFrom: 'False'
      - id: max_station_delta
        source: max_normalization_delta
      - id: scale_station_delta
        valueFrom: 'False'
      - id: phase_center_ra
        valueFrom: '0.0'
      - id: phase_center_dec
        valueFrom: '0.0'
    out:
      - id: outh5parm

  - id: plot_fulljones_amp_solutions
    label: Plot full-Jones amplitude solutions
    doc: |
      This step makes plots of the full-Jones amplitude solutions.
    run: {{ rapthor_pipeline_dir }}/steps/plot_solutions.cwl
    in:
      - id: h5parm
        source: process_fulljones_gains/outh5parm
      - id: soltype
        valueFrom: 'amplitude'
    out:
      - id: plots

  - id: plot_fulljones_phase_solutions
    label: Plot full-Jones phase solutions
    doc: |
      This step makes plots of the full-Jones phase solutions.
    run: {{ rapthor_pipeline_dir }}/steps/plot_solutions.cwl
    in:
      - id: h5parm
        source: process_fulljones_gains/outh5parm
      - id: soltype
        valueFrom: 'phase'
    out:
      - id: plots

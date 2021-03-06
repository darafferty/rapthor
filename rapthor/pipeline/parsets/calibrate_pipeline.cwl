cwlVersion: v1.0
class: Workflow
label: Rapthor calibration pipeline
doc: |
  This workflow performs direction-dependent calibration. In general,
  calibration is done in three steps: (1) a fast phase-only calibration (with
  core stations constrianed to have the same solutions) to correct for
  ionospheric effects, (2) a slow amplitude calibration (with all stations
  constrained to have the same solutions) to correct for beam errors, and (3) a
  further unconstrained slow gain calibration to correct for station-to-station
  differences. Steps (2) and (3) are skipped if the calibration is phase-only.
  This calibration scheme works for both HBA and LBA data. The final products of
  this pipeline are solution tables (h5parm files) and a-term screens (FITS
  files).

requirements:
  ScatterFeatureRequirement: {}
  StepInputExpressionRequirement: {}

{% if max_cores is not none %}
hints:
  ResourceRequirement:
    coresMin: 1
    coresMax: {{ max_cores }}
{% endif %}

inputs:
  - id: timechunk_filename
    label: Filename of input MS (time)
    doc: |
      The filenames of input MS files for which calibration will be done (length =
      n_obs * n_time_chunks).
    type: string[]

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
    type: string

  - id: calibration_sourcedb
    label: Filename of sourcedb
    doc: |
      The filename of the output sourcedb sky model file (length = 1).
    type: string

  - id: fast_smoothnessconstraint
    label: Fast smoothnessconstraint
    doc: |
      The smoothnessconstraint kernel size in Hz for the fast phase solve (length = 1).
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

  - id: propagatesolutions
    label: Propagate solutions
    doc: |
      Flag that determines whether solutions are propagated as initial start values
      for the next solution interval (length = 1).
    type: string

  - id: onebeamperpatch
    doc: |
      Flag that determines whether to apply the beam once per patch or per each
      source (length = 1).
    type: string

  - id: stepsize
    label: Solver step size
    doc: |
      The solver step size used between iterations (length = 1).
    type: float

  - id: tolerance
    label: Solver tolerance
    doc: |
      The solver tolerance used to define convergance (length = 1).
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

{% if do_slowgain_solve %}
  - id: freqchunk_filename
    label: Filename of input MS (frequency)
    doc: |
      The filenames of input MS files for which calibration will be done (length =
      n_obs * n_freq_chunks).
    type: string[]

  - id: slow_starttime
    label: Start time of each chunk
    doc: |
      The start time (in casacore MVTime) for each time chunk used in the slow-gain
      calibration (length = n_obs * n_freq_chunks).
    type: string[]

  - id: slow_ntimes
    label: Number of times of each chunk
    doc: |
      The number of timeslots for each time chunk used in the slow-gain calibration
      (length = n_obs * n_freq_chunks).
    type: int[]

  - id: startchan
    label: Start channel of each chunk
    doc: |
      The start channel for each frequency chunk used in the slow-gain
      calibration (length = n_obs * n_freq_chunks).
    type: int[]

  - id: nchan
    label: Number of channels of each chunk
    doc: |
      The number of channels for each frequency chunk used in the slow-gain calibration
      (length = n_obs * n_freq_chunks).
    type: int[]

  - id: solint_slow_timestep
    label: Slow 1 solution interval in time
    doc: |
      The solution interval in number of timeslots for the first slow-gain solve (length =
      n_obs * n_freq_chunks).
    type: int[]

  - id: solint_slow_timestep2
    label: Slow 2 solution interval in time
    doc: |
      The solution interval in number of timeslots for the second slow-gain solve (length =
      n_obs * n_freq_chunks).
    type: int[]

  - id: solint_slow_freqstep
    label: Slow 1 solution interval in frequency
    doc: |
      The solution interval in number of frequency channels for the first slow-gain solve
      (length = n_obs * n_freq_chunks).
    type: int[]

  - id: solint_slow_freqstep2
    label: Slow 2 solution interval in frequency
    doc: |
      The solution interval in number of frequency channels for the second slow-gain solve
      (length = n_obs * n_freq_chunks).
    type: int[]

  - id: slow_smoothnessconstraint
    label: Slow 1 smoothnessconstraint
    doc: |
      The smoothnessconstraint kernel size in Hz for the first slow-gain solve (length = 1).
    type: float

  - id: slow_smoothnessconstraint2
    label: Slow 2 smoothnessconstraint
    doc: |
      The smoothnessconstraint kernel size in Hz for the second slow-gain solve (length = 1).
    type: float

  - id: slow_antennaconstraint
    label: Slow antenna constraint
    doc: |
      The antenna constraint for the slow-gain solve (length = 1).
    type: string

  - id: output_slow_h5parm
    label: Slow 1 output solution table
    doc: |
      The filename of the output h5parm solution table for the first slow-gain solve (length
      = n_obs * n_freq_chunks).
    type: string[]

  - id: output_slow_h5parm2
    label: Slow 2 output solution table
    doc: |
      The filename of the output h5parm solution table for the second slow-gain solve (length
      = n_obs * n_freq_chunks).
    type: string[]

  - id: combined_slow_h5parm1
    label: Combined slow 1 output solution table
    doc: |
      The filename of the output combined h5parm solution table for the first slow-gain solve
      (length = 1).
    type: string

  - id: combined_slow_h5parm2
    label: Combined slow 2 output solution table
    doc: |
      The filename of the output combined h5parm solution table for the second slow-gain solve
      (length = 1).
    type: string

  - id: combined_h5parms
    label: Combined output solution table
    doc: |
      The filename of the output combined h5parm solution table for the full solve
      (length = 1).
    type: string

  - id: combined_h5parms1
    label: Combined output solution table
    doc: |
      The filename of the output combined h5parm solution table for the fast-phase +
      first slow-gain solve (length = 1).
    type: string

  - id: combined_h5parms2
    label: Combined output solution table
    doc: |
      The filename of the output combined h5parm solution table for the first and
      second slow-gain solves (length = 1).
    type: string

{% if debug %}
  - id: output_slow_h5parm_debug
    type: string[]

  - id: combined_slow_h5parm_debug
    type: string
{% endif %}
{% endif %}

outputs: []

steps:
  - id: make_sourcedb
    label: Make a sourcedb
    doc: |
      A sourcedb (defining the model) is required by DPPP for calibration. This
      step converts the input sky model into a sourcedb.
    run: {{ rapthor_pipeline_dir }}/steps/make_sourcedb.cwl
    in:
      - id: in
        source: calibration_skymodel_file
      - id: out
        source: calibration_sourcedb
    out:
      - id: sourcedb

  - id: solve_fast_phases
    label: Solve for fast phases
    doc: |
      This step uses DDECal (in DPPP) to solve for phase corrections on short
      timescales (< 1 minute), using the input MS files and sourcedb. These
      corrections are used to correct primarily for ionospheric effects.
{% if use_scalarphase %}
    run: {{ rapthor_pipeline_dir }}/steps/ddecal_solve_scalarphase.cwl
{% else %}
    run: {{ rapthor_pipeline_dir }}/steps/ddecal_solve_scalarcomplexgain.cwl
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
        source: make_sourcedb/sourcedb
      - id: maxiter
        source: maxiter
      - id: propagatesolutions
        source: propagatesolutions
      - id: onebeamperpatch
        source: onebeamperpatch
      - id: stepsize
        source: stepsize
      - id: tolerance
        source: tolerance
      - id: uvlambdamin
        source: uvlambdamin
      - id: smoothnessconstraint
        source: fast_smoothnessconstraint
      - id: antennaconstraint
        source: fast_antennaconstraint
      - id: numthreads
        valueFrom: '{{ max_threads }}'
    scatter: [msin, starttime, ntimes, h5parm, solint, nchan]
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

{% if do_slowgain_solve %}
# Solve for slow gains

  - id: solve_slow_gains1
    label: Solve for slow gains 1
    doc: |
      This step uses DDECal (in DPPP) to solve for gain corrections on long
      timescales (> 10 minute), using the input MS files and sourcedb. These
      corrections are used to correct primarily for beam errors. The fast-
      phase solutions are preapplied and all stations are constrained to
      have the same solutions.
    run: {{ rapthor_pipeline_dir }}/steps/ddecal_solve_complexgain1.cwl
{% if max_cores is not none %}
    hints:
      ResourceRequirement:
        coresMin: {{ max_cores }}
        coresMax: {{ max_cores }}
{% endif %}
    in:
      - id: msin
        source: freqchunk_filename
      - id: starttime
        source: slow_starttime
      - id: ntimes
        source: slow_ntimes
      - id: startchan
        source: startchan
      - id: nchan
        source: nchan
      - id: fast_h5parm
        source: combine_fast_phases/outh5parm
      - id: h5parm
        source: output_slow_h5parm
      - id: solint
        source: solint_slow_timestep
      - id: solve_nchan
        source: solint_slow_freqstep
      - id: sourcedb
        source: make_sourcedb/sourcedb
      - id: maxiter
        source: maxiter
      - id: propagatesolutions
        source: propagatesolutions
      - id: onebeamperpatch
        source: onebeamperpatch
      - id: stepsize
        source: stepsize
      - id: tolerance
        source: tolerance
      - id: uvlambdamin
        source: uvlambdamin
      - id: smoothnessconstraint
        source: slow_smoothnessconstraint
      - id: antennaconstraint
        source: slow_antennaconstraint
      - id: numthreads
        valueFrom: '{{ max_threads }}'
    scatter: [msin, starttime, ntimes, startchan, nchan, h5parm, solint, solve_nchan]
    scatterMethod: dotproduct
    out:
      - id: slow_gains_h5parm

  - id: combine_slow_gains1
    label: Combine slow-gain solutions 1
    doc: |
      This step combines all the gain solutions from the solve_slow_gains1 step
      into a single solution table (h5parm file).
    run: {{ rapthor_pipeline_dir }}/steps/collect_h5parms.cwl
    in:
      - id: inh5parms
        source: solve_slow_gains1/slow_gains_h5parm
      - id: outputh5parm
        source: combined_slow_h5parm1
    out:
      - id: outh5parm

  - id: process_slow_gains1
    label: Process slow-gain solutions 1
    doc: |
      This step processes the gain solutions, smoothing and renormalizing them.
    run: {{ rapthor_pipeline_dir }}/steps/process_slow_gains.cwl
    in:
      - id: slowh5parm
        source: combine_slow_gains1/outh5parm
      - id: smooth
        valueFrom: 'True'
    out:
      - id: outh5parm

  - id: combine_fast_and_slow_h5parms1
    label: Combine fast-phase and slow-gain solutions 1
    doc: |
      This step combines the fast-phase solutions from the solve_fast_phases step
      and the slow-gain solutions from the solve_slow_gains1 into a single solution
      table (h5parm file).
    run: {{ rapthor_pipeline_dir }}/steps/combine_h5parms.cwl
    in:
      - id: inh5parm1
        source: combine_fast_phases/outh5parm
      - id: inh5parm2
        source: process_slow_gains1/outh5parm
      - id: outh5parm
        source: combined_h5parms1
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

  - id: solve_slow_gains2
    label: Solve for slow gains 2
    doc: |
      This step uses DDECal (in DPPP) to solve for gain corrections on long
      timescales (> 10 minute), using the input MS files and sourcedb. These
      corrections are used to correct primarily for beam errors. The fast-
      phase solutions and first slow-gain solutions are preapplied and stations
      are unconstrainted (so different stations are free to have different
      solutions).
    run: {{ rapthor_pipeline_dir }}/steps/ddecal_solve_complexgain2.cwl
{% if max_cores is not none %}
    hints:
      ResourceRequirement:
        coresMin: {{ max_cores }}
        coresMax: {{ max_cores }}
{% endif %}
    in:
      - id: msin
        source: freqchunk_filename
      - id: starttime
        source: slow_starttime
      - id: ntimes
        source: slow_ntimes
      - id: startchan
        source: startchan
      - id: nchan
        source: nchan
      - id: combined_h5parm
        source: combine_fast_and_slow_h5parms1/combinedh5parm
      - id: h5parm
        source: output_slow_h5parm2
      - id: solint
        source: solint_slow_timestep2
      - id: solve_nchan
        source: solint_slow_freqstep2
      - id: sourcedb
        source: make_sourcedb/sourcedb
      - id: maxiter
        source: maxiter
      - id: propagatesolutions
        source: propagatesolutions
      - id: onebeamperpatch
        source: onebeamperpatch
      - id: stepsize
        source: stepsize
      - id: tolerance
        source: tolerance
      - id: uvlambdamin
        source: uvlambdamin
      - id: smoothnessconstraint
        source: slow_smoothnessconstraint2
      - id: numthreads
        valueFrom: '{{ max_threads }}'
    scatter: [msin, starttime, ntimes, startchan, nchan, h5parm, solint, solve_nchan]
    scatterMethod: dotproduct
    out:
      - id: slow_gains_h5parm

  - id: combine_slow_gains2
    label: Combine slow-gain solutions 2
    doc: |
      This step combines all the gain solutions from the solve_slow_gains2 step
      into a single solution table (h5parm file).
    run: {{ rapthor_pipeline_dir }}/steps/collect_h5parms.cwl
    in:
      - id: inh5parms
        source: solve_slow_gains2/slow_gains_h5parm
      - id: outputh5parm
        source: combined_slow_h5parm2
    out:
      - id: outh5parm

  - id: process_slow_gains2
    label: Process slow-gain solutions 2
    doc: |
      This step processes the gain solutions, smoothing and renormalizing them.
    run: {{ rapthor_pipeline_dir }}/steps/process_slow_gains.cwl
    in:
      - id: slowh5parm
        source: combine_slow_gains2/outh5parm
      - id: smooth
        valueFrom: 'True'
    out:
      - id: outh5parm

  - id: combine_slow1_and_slow2_h5parms
    label: Combine slow-gain solutions
    doc: |
      This step combines the gain solutions from the solve_slow_gains1 and
      solve_slow_gains2 steps into a single solution table (h5parm file).
      The phases and amplitudes from solve_slow_gains1 and the amplitudes from
      solve_slow_gains2 are used.
    run: {{ rapthor_pipeline_dir }}/steps/combine_h5parms.cwl
    in:
      - id: inh5parm1
        source: process_slow_gains2/outh5parm
      - id: inh5parm2
        source: combine_slow_gains1/outh5parm
      - id: outh5parm
        source: combined_h5parms2
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

  - id: normalize_slow_amplitudes
    label: Normalize slow-gain amplitudes
    doc: |
      This step processes the combined amplitude solutions from
      combine_slow1_and_slow2_h5parms, renormalizing them.
    run: {{ rapthor_pipeline_dir }}/steps/process_slow_gains.cwl
    in:
      - id: slowh5parm
        source: combine_slow1_and_slow2_h5parms/combinedh5parm
      - id: smooth
        valueFrom: 'False'
    out:
      - id: outh5parm

  - id: combine_fast_and_slow_h5parms2
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
        valueFrom: 'p1p2a2'
      - id: reweight
        valueFrom: 'True'
      - id: calibrator_names
        source: calibrator_patch_names
      - id: calibrator_fluxes
        source: calibrator_fluxes
    out:
      - id: combinedh5parm

  - id: split_h5parms
    label: Split solution table
    doc: |
      This step splits the final solution table in time, to enable multi-node
      parallel processing of the a-term images.
    run: {{ rapthor_pipeline_dir }}/steps/split_h5parms.cwl
    in:
      - id: inh5parm
        source: combine_fast_and_slow_h5parms2/combinedh5parm
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
        valueFrom: '{{ max_threads }}'
    scatter: [h5parm, outroot]
    scatterMethod: dotproduct
    out: []

{% if debug %}
# Solve for slow gains again, applying the first ones

  - id: solve_slow_gains_debug
    label: solve_slow_gains_debug
    run: {{ rapthor_pipeline_dir }}/steps/ddecal_solve_complexgain_debug.cwl
{% if max_cores is not none %}
    hints:
      ResourceRequirement:
        coresMin: {{ max_cores }}
        coresMax: {{ max_cores }}
{% endif %}
    in:
      - id: msin
        source: freqchunk_filename
      - id: starttime
        source: slow_starttime
      - id: ntimes
        source: slow_ntimes
      - id: startchan
        source: startchan
      - id: nchan
        source: nchan
      - id: combined_h5parm
        source: combine_fast_and_slow_h5parms2/combinedh5parm
      - id: h5parm
        source: output_slow_h5parm_debug
      - id: solint
        source: solint_slow_timestep
      - id: solve_nchan
        source: solint_slow_freqstep
      - id: sourcedb
        source: make_sourcedb/sourcedb
      - id: maxiter
        source: maxiter
      - id: propagatesolutions
        source: propagatesolutions
      - id: onebeamperpatch
        source: onebeamperpatch
      - id: stepsize
        source: stepsize
      - id: tolerance
        source: tolerance
      - id: uvlambdamin
        source: uvlambdamin
      - id: smoothnessconstraint
        source: slow_smoothnessconstraint
      - id: numthreads
        valueFrom: '{{ max_threads }}'
    scatter: [msin, starttime, ntimes, startchan, nchan, h5parm, solint, solve_nchan]
    scatterMethod: dotproduct
    out:
      - id: slow_gains_h5parm

  - id: combine_slow_gains_debug
    label: combine_slow_gains_debug
    run: {{ rapthor_pipeline_dir }}/steps/collect_h5parms.cwl
    in:
      - id: inh5parms
        source: solve_slow_gains_debug/slow_gains_h5parm
      - id: outputh5parm
        source: combined_slow_h5parm_debug
    out:
      - id: outh5parm

{% endif %}

{% else %}
# Don't solve for slow gains

  - id: split_h5parms
    label: Split solution table
    doc: |
      This step splits the final solution table in time, to enable multi-node
      parallel processing of the a-term images.
    run: {{ rapthor_pipeline_dir }}/steps/split_h5parms.cwl
    in:
      - id: inh5parm
        source: combine_fast_phases/outh5parm
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
        valueFrom: '{{ max_threads }}'
    scatter: [h5parm, outroot]
    scatterMethod: dotproduct
    out: []

{% endif %}

cwlVersion: v1.0
class: Workflow
label: Rapthor calibration pipeline
doc: |
  This workflow performs direction-dependent calibration. Calibration is done in
  three steps: (1) a fast phase-only calibration to correct for ionospheric
  effects, (2) a slow amplitude calibration with station constraints to correct
  for beam errors, and (3) a further slow gain calibration to correct for
  station-to-station differences. This calibration scheme works for both HBA and
  LBA data. The final products of this pipeline are solution tables (h5parm
  files) and a-term screens (FITS files).

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

  - id: stepsize
    label: Solver stepsize
    doc: |
      The solver stepsize used between iterations (length = 1).
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
    label: solve_fast_phases
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
    label: combine_fast_phases
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
    label: solve_slow_gains1
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
    label: combine_slow_gains1
    run: {{ rapthor_pipeline_dir }}/steps/collect_h5parms.cwl
    in:
      - id: inh5parms
        source: solve_slow_gains1/slow_gains_h5parm
      - id: outputh5parm
        source: combined_slow_h5parm1
    out:
      - id: outh5parm

  - id: combine_fast_and_slow_h5parms1
    label: combine_fast_and_slow_h5parms1
    run: {{ rapthor_pipeline_dir }}/steps/combine_h5parms.cwl
    in:
      - id: inh5parm1
        source: combine_fast_phases/outh5parm
      - id: inh5parm2
        source: combine_slow_gains1/outh5parm
      - id: outh5parm
        source: combined_h5parms1
      - id: mode
        valueFrom: 'p1a2'
      - id: reweight
        valueFrom: 'False'
    out:
      - id: combinedh5parm

  - id: solve_slow_gains2
    label: solve_slow_gains2
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
    label: combine_slow_gains2
    run: {{ rapthor_pipeline_dir }}/steps/collect_h5parms.cwl
    in:
      - id: inh5parms
        source: solve_slow_gains2/slow_gains_h5parm
      - id: outputh5parm
        source: combined_slow_h5parm2
    out:
      - id: outh5parm

  - id: process_slow_gains
    label: process_slow_gains
    run: {{ rapthor_pipeline_dir }}/steps/process_slow_gains.cwl
    in:
      - id: slowh5parm
        source: combine_slow_gains2/outh5parm
      - id: smooth
        valueFrom: 'True'
    out:
      - id: outh5parm

  - id: combine_slow1_and_slow2_h5parms
    label: combine_slow1_and_slow2_h5parms
    run: {{ rapthor_pipeline_dir }}/steps/combine_h5parms.cwl
    in:
      - id: inh5parm1
        source: process_slow_gains/outh5parm
      - id: inh5parm2
        source: combine_slow_gains1/outh5parm
      - id: outh5parm
        source: combined_h5parms2
      - id: mode
        valueFrom: 'p1a1a2'
      - id: reweight
        valueFrom: 'False'
    out:
      - id: combinedh5parm

  - id: normalize_slow_amplitudes
    label: normalize_slow_amplitudes
    run: {{ rapthor_pipeline_dir }}/steps/process_slow_gains.cwl
    in:
      - id: slowh5parm
        source: combine_slow1_and_slow2_h5parms/combinedh5parm
      - id: smooth
        valueFrom: 'False'
    out:
      - id: outh5parm

  - id: combine_fast_and_slow_h5parms2
    label: combine_fast_and_slow_h5parms2
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
    out:
      - id: combinedh5parm

  - id: split_h5parms
    label: split_h5parms
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
    label: make_aterms
    run: {{ rapthor_pipeline_dir }}/steps/make_aterm_images.cwl
    in:
      - id: h5parm
        source: split_h5parms/splith5parms
      - id: soltabname
        valueFrom: 'gain000'
      - id: screen_type
        valueFrom: 'kl'
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
    label: split_h5parms
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
    label: make_aterms
    run: {{ rapthor_pipeline_dir }}/steps/make_aterm_images.cwl
    in:
      - id: h5parm
        source: split_h5parms/splith5parms
      - id: soltabname
        valueFrom: 'phase000'
      - id: screen_type
        valueFrom: 'kl'
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

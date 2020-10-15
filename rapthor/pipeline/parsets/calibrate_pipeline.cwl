cwlVersion: v1.0
class: Workflow

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
    type: string[]
  - id: starttime
    type: string[]
  - id: ntimes
    type: int[]
  - id: solint_fast_timestep
    type: int[]
  - id: solint_fast_freqstep
    type: int[]
  - id: output_fast_h5parm
    type: string[]
  - id: combined_fast_h5parm
    type: string
  - id: calibration_skymodel_file
    type: string
  - id: calibration_sourcedb
    type: string
  - id: fast_smoothnessconstraint
    type: float
  - id: fast_antennaconstraint
    type: string
  - id: maxiter
    type: int
  - id: propagatesolutions
    type: string
  - id: stepsize
    type: float
  - id: tolerance
    type: float
  - id: uvlambdamin
    type: float
  - id: sector_bounds_deg
    type: string
  - id: sector_bounds_mid_deg
    type: string
  - id: split_outh5parm
    type: string[]
  - id: output_aterms_root
    type: string[]
{% if do_slowgain_solve %}
  - id: freqchunk_filename
    type: string[]
  - id: slow_starttime
    type: string[]
  - id: startchan
    type: int[]
  - id: nchan
    type: int[]
  - id: slow_ntimes
    type: int[]
  - id: solint_slow_timestep
    type: int[]
  - id: solint_slow_timestep2
    type: int[]
  - id: solint_slow_freqstep
    type: int[]
  - id: solint_slow_freqstep2
    type: int[]
  - id: slow_smoothnessconstraint
    type: float
  - id: slow_smoothnessconstraint2
    type: float
  - id: slow_antennaconstraint
    type: string
  - id: output_slow_h5parm
    type: string[]
  - id: output_slow_h5parm2
    type: string[]
  - id: combined_slow_h5parm1
    type: string
  - id: combined_slow_h5parm2
    type: string
  - id: combined_h5parms
    type: string
  - id: combined_h5parms1
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
    label: make_sourcedb
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
      - id: outh5parm
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
      - id: outh5parm
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
      - id: outh5parm
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
        source: combined_h5parms1
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
      - id: outh5parm
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

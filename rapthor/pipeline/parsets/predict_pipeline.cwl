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
  - id: sector_filename
    type: string[]
  - id: sector_model_filename
    type: string[]
  - id: sector_starttime
    type: string[]
  - id: sector_ntimes
    type: int[]
  - id: sector_patches
    type:
      type: array
      items:
        type: array
        items: string
  - id: h5parm
    type: string
  - id: sector_skymodel
    type: string[]
  - id: sector_sourcedb
    type: string[]
  - id: sector_obs_sourcedb
    type: string[]
  - id: obs_filename
    type: string[]
  - id: obs_starttime
    type: string[]
  - id: obs_solint_sec
    type: float[]
  - id: obs_solint_hz
    type: float[]
  - id: obs_infix
    type: string[]
  - id: min_uv_lambda
    type: float
  - id: max_uv_lambda
    type: float
  - id: nr_outliers
    type: int
  - id: peel_outliers
    type: string
  - id: nr_bright
    type: int
  - id: peel_bright
    type: string
  - id: reweight
    type: string

outputs: []

steps:
  - id: make_sourcedb
    label: make_sourcedb
    run: {{ rapthor_pipeline_dir }}/steps/make_sourcedb.cwl
    in:
      - id: in
        source: sector_skymodel
      - id: out
        source: sector_sourcedb
    scatter: [in, out]
    scatterMethod: dotproduct
    out:
      - id: sourcedb

{% if do_slowgain_solve %}

  - id: predict_model_data
    label: predict_model_data
    run: {{ rapthor_pipeline_dir }}/steps/predict_model_data.cwl
{% if max_cores is not none %}
    hints:
      ResourceRequirement:
        coresMin: {{ max_cores }}
        coresMax: {{ max_cores }}
{% endif %}
    in:
      - id: msin
        source: sector_filename
      - id: msout
        source: sector_model_filename
      - id: starttime
        source: sector_starttime
      - id: ntimes
        source: sector_ntimes
      - id: h5parm
        source: h5parm
      - id: sourcedb
        source: sector_obs_sourcedb
      - id: sourcedb2
        source: make_sourcedb/sourcedb
      - id: directions
        source: sector_patches
      - id: numthreads
        valueFrom: '{{ max_threads }}'
    scatter: [msin, msout, starttime, ntimes, sourcedb, directions]
    scatterMethod: dotproduct
    out:
      - id: msmod

{% else %}

  - id: predict_model_data
    label: predict_model_data
    run: {{ rapthor_pipeline_dir }}/steps/predict_model_data_phase_only.cwl
{% if max_cores is not none %}
    hints:
      ResourceRequirement:
        coresMin: {{ max_cores }}
        coresMax: {{ max_cores }}
{% endif %}
    in:
      - id: msin
        source: sector_filename
      - id: msout
        source: sector_model_filename
      - id: starttime
        source: sector_starttime
      - id: ntimes
        source: sector_ntimes
      - id: h5parm
        source: h5parm
      - id: sourcedb
        source: sector_obs_sourcedb
      - id: sourcedb2
        source: make_sourcedb/sourcedb
      - id: directions
        source: sector_patches
      - id: numthreads
        valueFrom: '{{ max_threads }}'
    scatter: [msin, msout, starttime, ntimes, sourcedb, directions]
    scatterMethod: dotproduct
    out:
      - id: msmod

{% endif %}

  - id: subtract_models
    label: subtract_models
    run: {{ rapthor_pipeline_dir }}/steps/subtract_sector_models.cwl
{% if max_cores is not none %}
    hints:
      ResourceRequirement:
        coresMin: {{ max_cores }}
        coresMax: {{ max_cores }}
{% endif %}
    in:
      - id: msobs
        source: obs_filename
      - id: msmod
        source: predict_model_data/msmod
      - id: obs_starttime
        source: obs_starttime
      - id: solint_sec
        source: obs_solint_sec
      - id: solint_hz
        source: obs_solint_hz
      - id: infix
        source: obs_infix
      - id: min_uv_lambda
        source: min_uv_lambda
      - id: max_uv_lambda
        source: max_uv_lambda
      - id: nr_outliers
        source: nr_outliers
      - id: peel_outliers
        source: peel_outliers
      - id: nr_bright
        source: nr_bright
      - id: peel_bright
        source: peel_bright
      - id: reweight
        source: reweight
    scatter: [msobs, obs_starttime, solint_sec, solint_hz, infix]
    scatterMethod: dotproduct
    out: []

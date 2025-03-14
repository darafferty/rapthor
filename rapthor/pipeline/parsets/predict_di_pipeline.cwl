cwlVersion: v1.2
class: Workflow
label: Rapthor prediction workflow
doc: |
  This workflow performs direction-dependent prediction of sector sky models and
  adds the resulting model data together. The output MS files are suitable for
  direction-independent calibration.

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
  - id: sector_filename
    label: Filenames of input MS
    doc: |
      The filenames of input MS files for which prediction will be done (length =
      n_obs * n_sectors).
    type: Directory[]

  - id: sector_model_filename
    label: Filenames of output MS
    doc: |
      The filenames of output MS files from prediction (length = n_obs * n_sectors).
    type: string[]

  - id: sector_starttime
    label: Start times of each chunk
    doc: |
      The start time (in casacore MVTime) for each time chunk used in prediction
      (length = n_obs * n_sectors).
    type: string[]

  - id: sector_ntimes
    label: Number of times of each chunk
    doc: |
      The number of timeslots for each time chunk used in prediction (length =
      n_obs * n_sectors).
    type: int[]

  - id: onebeamperpatch
    doc: |
      Flag that determines whether to apply the beam once per patch or per each
      source (length = 1).
    type: boolean

  - id: sagecalpredict
    doc: |
      Flag that enables prediction using SAGECAl.
    type: boolean

  - id: sector_patches
    label: Names of sector calibration patches
    doc: |
      A list of lists giving the names of the calibration patches for each sector
      (length = n_obs * n_sectors).
    type:
      type: array
      items:
        type: array
        items: string

  - id: h5parm
    label: Filename of solution table
    doc: |
      The filename of the h5parm solution table from the calibration workflow (length
      = 1).
    type: File

  - id: dp3_applycal_steps
    label: Applycal steps for fast solve
    doc: |
      The list of DP3 applycal steps to use in the prediction (length = 1).
    type: string

  - id: normalize_h5parm
    label: The filename of normalization h5parm
    doc: |
      The filename of the input flux-scale normalization h5parm (length = 1).
    type: File?

  - id: sector_skymodel
    label: Filename of sky model
    doc: |
      The filename of the input sky model text file of each sector (length = n_obs
      * n_sectors).
    type: File[]

  - id: obs_filename
    label: Filename of input MS
    doc: |
      The filenames of input MS files for which subtraction will be done (length =
      n_obs).
    type: Directory[]

  - id: obs_starttime
    label: Start time of each chunk
    doc: |
      The start time (in casacore MVTime) for each time chunk used in subtraction
      (length = n_obs).
    type: string[]

  - id: obs_infix
    label: Output infix string
    doc: |
      The infix string to use when building the output MS filenames (length = n_obs).
    type: string[]

  - id: max_threads
    label: Max number of threads
    doc: |
      The maximum number of threads to use for a job (length = 1).
    type: int


outputs:
  - id: msfiles_di_cal
    outputSource:
      - add_sector_models/output_ms
    type:
      - type: array
        items:
          - type: array
            items: Directory

steps:
  - id: predict_model_data
    label: Predict the model uv data
    doc: |
      This step uses DP3 to predict uv data (using the input sky model) from the
      input MS files. It also corrupts the model data with the calibration
      solutions. For each sector, prediction is done for all observations.
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
      - id: onebeamperpatch
        source: onebeamperpatch
      - id: sagecalpredict
        source: sagecalpredict
      - id: h5parm
        source: h5parm
      - id: applycal_steps
        source: dp3_applycal_steps
      - id: normalize_h5parm
        source: normalize_h5parm
      - id: sourcedb
        source: sector_skymodel
      - id: directions
        source: sector_patches
      - id: numthreads
        source: max_threads
    scatter: [msin, msout, starttime, ntimes, sourcedb, directions]
    scatterMethod: dotproduct
    out:
      - id: msmod

  - id: add_sector_models
    label: Add the model uv data
    doc: |
      This step adds sector model uv data to the input MS files.
    run: {{ rapthor_pipeline_dir }}/steps/add_sector_models.cwl
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
      - id: infix
        source: obs_infix
    scatter: [msobs, obs_starttime, infix]
    scatterMethod: dotproduct
    out:
      - id: output_ms

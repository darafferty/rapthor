cwlVersion: v1.2
class: Workflow
label: Rapthor prediction workflow
doc: |
  This workflow performs direction-dependent prediction of sector sky models of
  non-calibrator sources and subracts the resulting model data from the input
  data. The resulting data are suitable for calibration with the calibrator
  sources only.

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
      The filename of the h5parm solution table from the calibration workflow
      (length = 1). Note that when this file is unavailable, the filename can be
      set to a dummy string, in which case it is then ignored by the script
    type:
      - string?
      - File?

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

  - id: nr_sectors
    label: Number of sectors
    doc: |
      The number of sectors to process (length = 1).
    type: int

  - id: max_threads
    label: Max number of threads
    doc: |
      The maximum number of threads to use for a job (length = 1).
    type: int


outputs:
  - id: subtract_models
    outputSource:
      - subtract_sector_models/output_models
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
      input MS files. It also optionaly corrupts the model data with the calibration
      solutions. For each sector, prediction is done for all observations.
{% if apply_solutions %}
{% if apply_amplitudes %}
    # Corrupt with both fast phases and slow gains
    run: {{ rapthor_pipeline_dir }}/steps/predict_model_data.cwl
{% else %}
    # Corrupt with fast phases only
    run: {{ rapthor_pipeline_dir }}/steps/predict_model_data_phase_only.cwl
{% endif %}
{% else %}
    # Don't corrupt
    run: {{ rapthor_pipeline_dir }}/steps/predict_model_data_no_corruptions.cwl
{% endif %}
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
{% if apply_solutions %}
      - id: h5parm
        source: h5parm
{% endif %}
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

  - id: subtract_sector_models
    label: Subtract the model uv data
    doc: |
      This step subtracts the model uv data generated in the previous step from the
      input MS files, generating data suitable for use as input to the calibration
      workflow.
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
        default: 0
      - id: solint_hz
        default: 0
      - id: infix
        source: obs_infix
      - id: min_uv_lambda
        default: 0
      - id: max_uv_lambda
        default: 0
      - id: nr_outliers
        source: nr_sectors
      - id: peel_outliers
        valueFrom: 'True'
      - id: nr_bright
        default: 0
      - id: peel_bright
        valueFrom: 'False'
      - id: reweight
        valueFrom: 'False'
    scatter: [msobs, obs_starttime, infix]
    scatterMethod: dotproduct
    out:
      - id: output_models

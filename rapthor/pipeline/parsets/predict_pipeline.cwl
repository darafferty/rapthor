cwlVersion: v1.2
class: Workflow
label: Rapthor prediction pipeline
doc: |
  This workflow performs direction-dependent prediction of sector sky models and
  subracts the resulting model data from the input data, reweighting if desired.
  The resulting data are suitable for imaging.

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
      The filename of the h5parm solution table from the calibration pipeline (length
      = 1).
    type: File

  - id: sector_skymodel
    label: Filename of sky model
    doc: |
      The filename of the input sky model text file of each sector (length = n_sectors).
    type: File[]

  - id: sector_sourcedb
    label: Filename of sourcedb
    doc: |
      The filename of the output sourcedb sky model file of each sector (length =
      n_sectors).
    type: string[]

  - id: sector_obs_sourcedb
    label: Filename of sourcedb
    doc: |
      The filename of the output sourcedb sky model file of each sector, repeated for
      each observation  (length = n_obs * n_sectors).
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

  - id: obs_solint_sec
    label: Solution interval in sec
    doc: |
      The solution interval in sec used during the fast-phase calibration (length =
      n_obs).
    type: float[]

  - id: obs_solint_hz
    label: Solution interval in Hz
    doc: |
      The solution interval in Hz used during the slow-gain calibration (length =
      n_obs).
    type: float[]

  - id: obs_infix
    label: Output infix string
    doc: |
      The infix string to use when building the output MS filenames (length = n_obs).
    type: string[]

  - id: min_uv_lambda
    label: Minimum uv distance in lambda
    doc: |
      The minimum uv distance used during the calibration (length = 1).
    type: float

  - id: max_uv_lambda
    label: Maximum uv distance in lambda
    doc: |
      The maximum uv distance used during the calibration (length = 1).
    type: float

  - id: nr_outliers
    label: Number outlier sectors
    doc: |
      The number of outlier sectors to process (length = 1).
    type: int

  - id: peel_outliers
    label: Outlier flag
    doc: |
      The flag that sets peeling of outlier sources (length = 1).
    type: boolean

  - id: nr_bright
    label: Number bright-source sectors
    doc: |
      The number of bright-source sectors to process (length = 1).
    type: int

  - id: peel_bright
    label: Bright-source flag
    doc: |
      The flag that sets peeling of bright sources (length = 1).
    type: boolean

  - id: reweight
    label: Reweight flag
    doc: |
      The flag that sets reweighting of uv data (length = 1).
    type: boolean

outputs:
  - id: make_sourcedb
    outputSource:
      - make_sourcedb/sourcedb
    type: File[]
  - id: subtract_models
    outputSource:
      - merge_subtract_models/output
    type: Directory[]

steps:
  - id: make_sourcedb
    label: Make a sourcedb
    doc: |
      A sourcedb (defining the model) is required by DPPP for prediction. This
      step converts the input sky model into a sourcedb (one per sector).
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

  - id: predict_model_data
    label: Predict the model uv data
    doc: |
      This step uses DPPP to predict uv data (using the input sourcedb) from the
      input MS files. It also corrupts the model data with the calibration
      solutions. For each sector, prediction is done for all observations.
{% if do_slowgain_solve %}
    # Corrupt with both fast phases and slow gains
    run: {{ rapthor_pipeline_dir }}/steps/predict_model_data.cwl
{% else %}
    # Corrupt with fast phases only
    run: {{ rapthor_pipeline_dir }}/steps/predict_model_data_phase_only.cwl
{% endif %}
{% if max_cores is not none %}
    hints:
      ResourceRequirement:
        coresMin: 1
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
      - id: h5parm
        source: h5parm
      - id: sourcedb
        source: make_sourcedb/sourcedb
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

  - id: subtract_models
    label: Subtract the model uv data
    doc: |
      This step subtracts the model uv data generated in the previous step from the
      input MS files. For each sector, all sources that lie outside of the sector are
      subtracted (or peeled), generating data suitable for use as input to the imaging
      pipeline. Reweighting by the residuals can also be done, by generating data in
      which all sources have been subtracted.
    run: {{ rapthor_pipeline_dir }}/steps/subtract_sector_models.cwl
{% if max_cores is not none %}
    hints:
      ResourceRequirement:
        coresMin: 1
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
    out:
      - id: output_models

  - id: merge_subtract_models
    in:
      - id: input
        source:
          - subtract_models/output_models
    out:
      - id: output
    run: /project/rapthor/Software/rapthor.rap-423/lib/python3.6/site-packages/rapthor/pipeline/steps/merge_array_directories.cwl
    label: merge_subtract_models

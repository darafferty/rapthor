cwlVersion: v1.0
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
    type: string[]
    label: Filename of input MS
    doc: |
       The filenames of input MS files for which predition will be done. There should
       be n_obs * n_sector filenames.
  - id: sector_model_filename
    type: string[]
    label: Filename of output MS
    doc: |
       The filenames of output MS files from prediction. There should be n_obs *
       n_sector filenames.
  - id: sector_starttime
    type: string[]
    label: Start time of each chunk
    doc: |
       The start time (in casacore MVTime) for each time chunk used in processing. There should be n_obs *
       n_sector times.
  - id: sector_ntimes
    type: int[]
    label: Number of times of each chunk
    doc: |
       The number of timeslots for each time chunk used in processing. There should be n_obs *
       n_sector entries.
  - id: sector_patches
    type:
      type: array
      items:
        type: array
        items: string
    label: Names of sector calibration patches
    doc: |
       A list of lists giving the names of the calibration patches for each sector.
  - id: h5parm
    type: string
    label: Filename of solution table
    doc: |
       The filename of the h5parm solution table from the calibration pipeline.
  - id: sector_skymodel
    label: Filename of sky model
    doc: |
       The filename of the input sky model text file of each sector.
    type: string[]
  - id: sector_sourcedb
    type: string[]
    label: Filename of sourcedb
    doc: |
       The filename of the output sourcedb sky model file of each sector.
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

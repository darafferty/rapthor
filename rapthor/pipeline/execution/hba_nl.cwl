cwlVersion: v1.2
class: Workflow
label: Rapthor HBA NL pipeline

doc: |
  This is the top-level workflow for the Rapthor HBA NL pipeline.

requirements:
- class: MultipleInputFeatureRequirement
- class: ScatterFeatureRequirement
- class: SubworkflowFeatureRequirement

inputs:
- id: surls
  doc: List of SURLs to the input data
  type: string[]
- id: settings
  doc: Pipeline settings, used to generate a parset file
  type:
    type: record
    fields:
      - name: global
        type: Any
      - name: calibration
        type: Any
      - name: imaging
        type: Any
      - name: cluster
        type: Any
- id: virtualenv
  type: Any

outputs:
- id: parset
  type: File
  outputSource:
    run_rapthor/parset
- id: tarballs
  type: File[]
  outputSource:
    tar_directory/tarball

steps:
# fetch MS files
- id: fetch_data
  label: Fetch MS files
  doc: |
    Fetch all the data from a single target observation produced by LINC
    (as specified in the SURLs) and uncompress them
  in:
    - id: surl_link
      source: surls
  out:
    - id: uncompressed
  run: fetch_data.cwl
  scatter: surl_link

# concat MS files into one
- id: concat_ms
  label: Concatenate MS files
  doc: |
    Concatenate input MS files into a single MS that Rapthor can process
  in:
    - id: msin
      source: fetch_data/uncompressed
  out:
    - id: msout
  run: concat_ms.cwl

# fetch sky model files
- id: fetch_tgss_model
  label: Fetch TGSS sky model
  doc: |
    Fetch TGSS sky model (for photometry checks)
  in:
    - id: msin
      source: concat_ms/msout
    - id: survey
      valueFrom: "TGSS"
    - id: radius
      valueFrom: "5.0"
    - id: skymodel
      valueFrom: "TGSS_model.txt"
  out:
    - id: model
  run: fetch_model.cwl

- id: fetch_panstarrs_model
  label: Fetch Pan-STARRS sky model
  doc: |
    Fetch Pan-STARRS sky model (for astrometry checks)
  in:
    - id: msin
      source: concat_ms/msout
    - id: survey
      valueFrom: "PANSTARRS"
    - id: radius
      valueFrom: "5.0"
    - id: skymodel
      valueFrom: "PANSTARRS_model.txt"
  out:
    - id: model
  run: fetch_model.cwl

- id: fetch_vlssr_model
  label: Fetch VLSSr sky model
  doc: |
    Fetch VLSSr sky model (for photometry normalization)
  in:
    - id: msin
      source: concat_ms/msout
    - id: survey
      valueFrom: "VLSSR"
    - id: radius
      valueFrom: "5.0"
    - id: skymodel
      valueFrom: "VLSSR_model.txt"
  out:
    - id: model
  run: fetch_model.cwl

- id: fetch_wenss_model
  label: Fetch WENSS sky model
  doc: |
    Fetch WENSS sky model (for photometry normalization)
  in:
    - id: msin
      source: concat_ms/msout
    - id: survey
      valueFrom: "WENSS"
    - id: radius
      valueFrom: "5.0"
    - id: skymodel
      valueFrom: "WENSS_model.txt"
  out:
    - id: model
  run: fetch_model.cwl

# run Rapthor pipeline
- id: run_rapthor
  label: Run Rapthor pipeline
  doc: |
    Run the Rapthor pipeline in a virtual environment that is created on
    the fly
  in:
    - id: msin
      source: concat_ms/msout
    - id: photometry_check_model
      source: fetch_tgss_model/model
    - id: astrometry_check_model
      source: fetch_panstarrs_model/model
    - id: photometry_normalization_model_1
      source: fetch_vlssr_model/model
    - id: photometry_normalization_frequecy_1
      valueFrom: "74e6"
    - id: photometry_normalization_model_2
      source: fetch_wenss_model/model
    - id: photometry_normalization_frequecy_2
      valueFrom: "327e6"
    - id: settings
      source: settings
    - id: virtualenv
      source: virtualenv
  out:
    - id: images
    - id: logs
    - id: parset
    - id: plots
    - id: regions
    - id: skymodels
    - id: solutions
  run: run_rapthor.cwl

- id: tar_directory
  label: Tar the pipeline results
  doc: |
    Create tar-balls for each of the output directories produced by Rapthor.
  in:
   - id: directory
     linkMerge: merge_flattened
     source:
      - run_rapthor/images
      - run_rapthor/logs
      - run_rapthor/plots
      - run_rapthor/regions
      - run_rapthor/skymodels
      - run_rapthor/solutions
  out:
  - id: tarball
  run: tar_directory.cwl
  scatter: directory

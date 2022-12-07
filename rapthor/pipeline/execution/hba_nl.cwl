cwlVersion: v1.2
class: Workflow
label: Rapthor HBA NL pipeline

doc: |
  This is the top-level workflow for the Rapthor HBA NL pipeline.

requirements:
- class: ScatterFeatureRequirement
- class: SubworkflowFeatureRequirement

inputs:
- id: surls
  doc: List of SURLs to the input data
  type: string[]
- id: config
  type: File
- id: skymodel
  type: File

outputs:
- id: images
  type: Directory[]
  outputSource:
    - run_rapthor/images
- id: logs
  type: Directory[]
  outputSource:
    - run_rapthor/logs
- id: parset
  type: File
  outputSource:
    - run_rapthor/parset
- id: plots
  type: Directory[]
  outputSource:
    - run_rapthor/plots
- id: regions
  type: Directory[]
  outputSource:
    - run_rapthor/regions
- id: skymodels
  type: Directory[]
  outputSource:
    - run_rapthor/skymodels
- id: solutions
  type: Directory[]
  outputSource:
    - run_rapthor/solutions

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
  run: ../steps/fetch_data.cwl
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
  run: ../steps/concat_ms.cwl

# run Rapthor pipeline
- id: run_rapthor
  label: Run Rapthor pipeline
  doc: |
    Run the Rapthor pipeline in a virtual environment that is created on
    the fly
  in:
    - id: json
      source: config
    - id: ms
      source: concat_ms/msout
    - id: skymodel
      source: skymodel
  out:
    - id: images
    - id: logs
    - id: parset
    - id: plots
    - id: regions
    - id: skymodels
    - id: solutions
  run: ../steps/run_rapthor.cwl

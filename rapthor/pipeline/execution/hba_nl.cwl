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
- id: ms
  doc: |
    Name of the MS containing the concatenated data. This file will be use as input
    to the actual Rapthor pipeline. Its name must match with the name specified in
    the parset file (which will also be generated).
  type: string

outputs: []
# Add all outputs generated by Rapthor pipeline here

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
    - id: msout_name
      source: ms
  out:
    - id: msout
  run: ../steps/concat_ms.cwl

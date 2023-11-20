cwlVersion: v1.2
class: Workflow
label: Rapthor concatenation workflow
doc: |
  This workflow performs the frequency concatenation of MS files. The concatenation
  is done per epoch, with all files from a given epoch concatenated into one output
  file.

requirements:
  MultipleInputFeatureRequirement: {}
  ScatterFeatureRequirement: {}
  StepInputExpressionRequirement: {}

{% if max_cores is not none %}
hints:
  ResourceRequirement:
    coresMin: {{ max_cores }}
    coresMax: {{ max_cores }}
{% endif %}

inputs:
  - id: input_filenames
    label: Filenames of input files
    doc: |
      The filenames of the input MS files  (length = n_epochs, each of which has
      n_files_per_epoch elements).
    type:
      type: array
      items:
        type: array
        items: Directory

  - id: output_filenames
    label: Filenames of output files
    doc: |
      The filenames of the output concatenated MS files (length = n_epochs).
    type: string[]

outputs:
  - id: concatenated_filenames
    outputSource:
      - concatenate_per_epoch/msconcat
    type: Directory[]

steps:
  - id: concatenate_per_epoch
    label: Concatenate MS files per epoch
    doc: |
      This step concatenates the input MS files over frequency for a
      single epoch.
    run: {{ rapthor_pipeline_dir }}/steps/concat_ms_files.cwl
    in:
    - id: mslist
      source: input_filenames
    - id: msout
      source: output_filenames
    - id: concat_property
      valueFrom: 'frequency'
    scatter: [mslist, msout]
    scatterMethod: dotproduct
    out:
      - id: msconcat

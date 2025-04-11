cwlVersion: v1.2
class: Workflow
label: Rapthor mosaicking workflow
doc: |
  This workflow performs the mosaicking of images made with the imaging workflow.
  If only a single image was made, processing is (mostly) skipped.

{% if not skip_processing %}
# start not skip_processing

requirements:
  MultipleInputFeatureRequirement: {}
  ScatterFeatureRequirement: {}
  StepInputExpressionRequirement: {}
  SubworkflowFeatureRequirement: {}

{% if max_cores is not none %}
hints:
  ResourceRequirement:
    coresMin: {{ max_cores }}
    coresMax: {{ max_cores }}
{% endif %}

inputs:
  - id: sector_image_filename
    label: Filenames of images
    doc: |
      The filenames of the sector FITS images (length = n_sectors * n_img_types).
    type:
      type: array
      items:
        type: array
        items: File

  - id: sector_vertices_filename
    label: Filenames of vertices files
    doc: |
      The filenames of the sector vertices files (length = n_sectors * n_img_types).
    type:
      type: array
      items:
        type: array
        items: File

  - id: template_image_filename
    label: Filename of template image
    doc: |
      The filename of the temporary mosaic template image (length = n_img_types).
    type: string[]

  - id: regridded_image_filename
    label: Filenames of images
    doc: |
      The filenames of the regridded sector images (length = n_sectors * n_img_types).
    type:
      type: array
      items:
        type: array
        items: string

  - id: mosaic_filename
    label: Filename of mosiac image
    doc: |
      The filename of the final mosaic image (length = n_img_types).
    type: string[]

  - id: skip_processing
    label: Flag to skip processing
    doc: |
      The flag that sets whether processing is skipped or not (length = 1).
    type: boolean

outputs:
  - id: mosaic_image
    outputSource:
      - mosaic_by_type/mosaic_image
    type: File[]

steps:
  - id: mosaic_by_type
    label: Mosiac an image type
    doc: |
      This step is a subworkflow that performs the processing for each image
      type.
    run: {{ pipeline_working_dir }}/subpipeline_parset.cwl
    in:
    - id: sector_image_filename
      source: sector_image_filename
    - id: sector_vertices_filename
      source: sector_vertices_filename
    - id: template_image_filename
      source: template_image_filename
    - id: regridded_image_filename
      source: regridded_image_filename
    - id: mosaic_filename
      source: mosaic_filename
    - id: skip_processing
      source: skip_processing
    scatter: [sector_image_filename, sector_vertices_filename, template_image_filename,
              regridded_image_filename, mosaic_filename]
    scatterMethod: dotproduct
    out:
      - id: mosaic_image

{% else %}
# start skip_processing

inputs: []
outputs: []
steps: []

{% endif %}
# end skip_processing / not skip_processing

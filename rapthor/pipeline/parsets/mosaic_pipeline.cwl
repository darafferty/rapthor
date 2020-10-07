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
  - id: sector_image_filename
    type: string[]
  - id: sector_vertices_filename
    type: string[]
  - id: template_image_filename
    type: string
  - id: regridded_image_filename
    type: string[]
  - id: mosaic_filename
    type: string
  - id: skip_processing
    type: string

outputs: []

steps:
  - id: make_mosaic_template
    label: make_mosaic_template
    run: {{ rapthor_pipeline_dir }}/steps/make_mosaic_template.cwl
    in:
      - id: input_image_list
        source: sector_image_filename
      - id: vertices_file_list
        source: sector_vertices_filename
      - id: output_image
        source: template_image_filename
      - id: skip
        source: skip_processing
    out:
      - id: template_image

  - id: regrid_image
    label: regrid_image
    run: {{ rapthor_pipeline_dir }}/steps/regrid_image.cwl
    in:
      - id: input_image
        source: sector_image_filename
      - id: template_image
        source: make_mosaic_template/template_image
      - id: vertices_file
        source: sector_vertices_filename
      - id: output_image
        source: regridded_image_filename
      - id: skip
        source: skip_processing
    scatter: [input_image, vertices_file, output_image]
    scatterMethod: dotproduct
    out:
      - id: regridded_image

  - id: make_mosaic
    label: make_mosaic
    run: {{ rapthor_pipeline_dir }}/steps/make_mosaic.cwl
    in:
      - id: input_image_list
        source: regrid_image/regridded_image
      - id: template_image
        source: make_mosaic_template/template_image
      - id: output_image
        source: mosaic_filename
      - id: skip
        source: skip_processing
    out: []

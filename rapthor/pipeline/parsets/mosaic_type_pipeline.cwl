cwlVersion: v1.2
class: Workflow
label: Rapthor mosaicking subworkflow
doc: |
  This subworkflow performs the mosaicking of a single type of image made with the
  imaging workflow. If only a single image was made, processing is (mostly) skipped.

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
  - id: sector_image_filename
    label: Filenames of images
    doc: |
      The filenames of the sector FITS images (length = n_sectors).
    type: File[]

  - id: sector_vertices_filename
    label: Filenames of vertices files
    doc: |
      The filenames of the sector vertices files (length = n_sectors).
    type: File[]

  - id: template_image_filename
    label: Filename of template image
    doc: |
      The filename of the temporary mosaic template image (length = 1).
    type: string

  - id: regridded_image_filename
    label: Filenames of images
    doc: |
      The filenames of the regridded sector images (length = n_sectors).
    type: string[]

  - id: mosaic_filename
    label: Filename of mosiac image
    doc: |
      The filename of the final mosaic image (length = 1).
    type: string

  - id: skip_processing
    label: Flag to skip processing
    doc: |
      The flag that sets whether processing is skipped or not (length = 1).
    type: boolean

outputs:
{% if compress_images %}
  - id: mosaic_image
    outputSource:
      - compress/compressed_mosaic_image
    type: File
{% else %}
  - id: mosaic_image
    outputSource:
      - make_mosaic/mosaic_image
    type: File
{% endif %}

steps:
  - id: make_mosaic_template
    label: Make mosaic template
    doc: |
      This step makes a temporary template FITS image that is used
      in the regrid_image and make_mosaic steps.
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
    label: Regrid image
    doc: |
      This step regrids FITS images to the grid of the template FITS
      image made in the make_mosaic_template step.
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
    label: Make mosaic
    doc: |
      This step makes the final mosaic FITS image from the regridded
      sector images.
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
    out:
      - id: mosaic_image

{% if compress_images %}
# start compress_images
  - id: compress
    label: Compress mosaic image
    doc: |
      This step uses cfitsio fpack to compress the mosaic FITS format image file
    run: {{ rapthor_pipeline_dir }}/steps/compress_mosaic_image.cwl
    in:
      - id: mosaic_image
        source: make_mosaic/mosaic_image
    out:
      - id: compressed_mosaic_image

{% endif %}
# end compress_images

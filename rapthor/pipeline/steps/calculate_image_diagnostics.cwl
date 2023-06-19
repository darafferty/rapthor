cwlVersion: v1.2
class: CommandLineTool
baseCommand: [calculate_image_diagnostics.py]
label: Calculate image diagnostics
doc: |
  This tool calculates various diagnostics from the input images

requirements:
  - class: InlineJavascriptRequirement

inputs:
  - id: flat_noise_image
    label: Flat-noise image
    doc: |
      The filename of the input flat-noise FITS image.
    type: File
    inputBinding:
      position: 1
  - id: flat_noise_rms_image
    label: Flat-noise RMS image
    doc: |
      The filename of the input flat-noise background RMS FITS image.
    type: File
    inputBinding:
      position: 2
  - id: true_sky_image
    label: True-sky image
    doc: |
      The filename of the input true-sky (primary-beam-corrected) FITS image.
    type: File
    inputBinding:
      position: 3
  - id: true_sky_rms_image
    label: True-sky RMS image
    doc: |
      The filename of the input true-sky (primary-beam-corrected) RMS FITS image.
    type: File
    inputBinding:
      position: 4
  - id: input_catalog
    label: Input source FITS catalog
    doc: |
      The filename of the input FITS catalog.
    type: File
    inputBinding:
      position: 5
  - id: input_skymodel
    label: Input sky model
    doc: |
      The filename of the input sky model.
    type: File
    inputBinding:
      position: 6
  - id: beamMS
    label: Filename of MS file for beam
    doc: |
      The filenames of the MS files to use for beam calculations.
    type: Directory[]
    inputBinding:
      position: 7
      itemSeparator: ","
  - id: diagnostics_file
    label: Input diagnostics file
    doc: |
      The filename of the input diagnostics JSON file.
    type: File
    inputBinding:
      position: 8
  - id: output_root
    label: Output root name
    doc: |
      The root of the filenames of the output filtered sky models.
    type: string
    inputBinding:
      position: 9

outputs:
  - id: diagnostics
    label: Image diagnostics
    doc: |
      The image diagnostics, including RMS noise, dynamic range, frequency, and beam.
    type: File
    outputBinding:
      glob: '$(inputs.output_root).image_diagnostics.json'


hints:
  - class: DockerRequirement
    dockerPull: 'astronrd/rapthor'

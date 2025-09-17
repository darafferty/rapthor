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
      The filename of the input flat-noise background RMS FITS image
      (made by PyBDSF).
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
      The filename of the input true-sky (primary-beam-corrected) RMS FITS
      image (made by PyBDSF).
    type: File
    inputBinding:
      position: 4
  - id: input_catalog
    label: Input source FITS catalog
    doc: |
      The filename of the input source FITS catalog (made by PyBDSF).
    type: File
    inputBinding:
      position: 5
  - id: input_skymodel
    label: Input sky model
    doc: |
      The filename of the input sky model (in makesourcedb format). This model
      should be grouped into source patches.
    type: File
    inputBinding:
      position: 6
  - id: obs_ms
    label: Filenames of MS files
    doc: |
      The filenames of the MS files to use for observation properties, such
      as the theoretical noise. This MS should have the original phase center of
      the observation.
    type: Directory[]
    inputBinding:
      position: 7
      itemSeparator: ","
  - id: obs_starttime
    label: Start time of obs
    doc: |
      The start times in casacore MVTime of the observations corresponding to
      those in obs_ms.
    type: string[]
    inputBinding:
      position: 8
      itemSeparator: ","
  - id: obs_ntimes
    label: Number of time slots of obs
    doc: |
      The number of time slots of the observations corresponding to those in
      obs_ms.
    type: int[]
    inputBinding:
      position: 9
      itemSeparator: ","
  - id: diagnostics_file
    label: Input diagnostics file
    doc: |
      The filename of the input diagnostics JSON file.
    type: File
    inputBinding:
      position: 10
  - id: output_root
    label: Output root name
    doc: |
      The root of the filenames of the output filtered sky models.
    type: string
    inputBinding:
      position: 11
  - id: facet_region_file
    label: Input ds9 region file
    doc: |
      The filename of the input ds9 region file that defines the facets. Note
      that when this file is unavailable, the filename can be set to a dummy
      string, in which case it is then ignored by the script
    type:
      - string?
      - File?
    inputBinding:
      prefix: --facet_region_file=
      separate: false

outputs:
  - id: diagnostics
    label: Image diagnostics
    doc: |
      The image diagnostics, including RMS noise, dynamic range, frequency, and beam.
    type: File
    outputBinding:
      glob: '$(inputs.output_root).image_diagnostics.json'
  - id: offsets
    label: Astrometry offsets
    doc: |
      The astrometry offsets in RA and Dec, per facet.
    type: File?
    outputBinding:
      glob: '$(inputs.output_root).astrometry_offsets.json'
  - id: plots
    label: Diagnostic plots
    doc: |
      Various diagnostic plots of the photometry and astrometry.
    type: File[]?
    outputBinding:
      glob: '*.pdf'


hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor

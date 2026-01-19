cwlVersion: v1.2
class: CommandLineTool
baseCommand: [check_image_beam.py]
label: Check image beam information
doc: |
  This tool ensures that valid beam information is present in the image header.

requirements:
  - class: InlineJavascriptRequirement
  - class: InitialWorkDirRequirement
    listing:
      - entry: $(inputs.input_image)
        writable: true

inputs:
  - id: input_image
    label: Filename of input image
    doc: |
      The filename of the input FITS image.
    type: File
    inputBinding:
      position: 1
  - id: beam_size_arcsec
    label: Beam size in arcsec
    doc: |
      The beam size in arcsec to use when no beam information is found in the input
      image. A circular beam is assumed.
    type: float
    inputBinding:
      position: 2

outputs:
  - id: validated_image
    doc: |
      The validated image, with beam information added if needed. The value is taken
      from the input parameter "input_image".
    type: File
    outputBinding:
      glob: $(inputs.input_image.basename)

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor

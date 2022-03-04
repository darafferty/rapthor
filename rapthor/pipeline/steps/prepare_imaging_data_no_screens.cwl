cwlVersion: v1.0
class: CommandLineTool
baseCommand: [DP3]
label: Prepares a dataset for imaging
doc: |
  This tool prepares the input data for imaging without screens, including
  applying the beam model, phase shifting, averaging, and applying all
  solutions. See prepare_imaging_data.cwl for a detailed description of the
  inputs and outputs

requirements:
  InlineJavascriptRequirement: {}
  ShellCommandRequirement: {}

arguments:
  - msin.datacolumn=DATA
  - msout.overwrite=True
  - msout.writefullresflag=False
  - steps=[applybeam,shift,applycal,avg]
  - shift.type=phaseshifter
  - avg.type=squash
  - applycal.type=applycal
  - applycal.correction=phase000
  - applycal.steps=[slowamp,fastphase]
  - applycal.slowamp.correction=amplitude000
  - applycal.fastphase.correction=phase000
  - msout.storagemanager=Dysco

inputs:
  - id: msin
    type: string
    inputBinding:
      prefix: msin=
      separate: False
  - id: msout
    type: string
    inputBinding:
      prefix: msout=
      separate: False
  - id: starttime
    type: string
    inputBinding:
      prefix: msin.starttime=
      separate: False
  - id: ntimes
    type: int
    inputBinding:
      prefix: msin.ntimes=
      separate: False
  - id: phasecenter
    type: string
    inputBinding:
      prefix: shift.phasecenter=
      separate: False
      shellQuote: False
  - id: freqstep
    type: int
    inputBinding:
      prefix: avg.freqstep=
      separate: False
  - id: timestep
    type: int
    inputBinding:
      prefix: avg.timestep=
      separate: False
  - id: beamdir
    type: string
    inputBinding:
      prefix: applybeam.direction=
      separate: False
      shellQuote: False
  - id: h5parm
    label: Filename of h5parm
    doc: |
      The filename of the h5parm file with the calibration solutions.
    type: string
    inputBinding:
      prefix: applycal.parmdb=
      separate: False
  - id: central_patch_name
    label: Name of central patch
    doc: |
      The name of the central-most patch of the sector.
    type: string
    inputBinding:
      valueFrom: $('['+self+']')
      prefix: applycal.direction=
      separate: False
  - id: numthreads
    type: string
    inputBinding:
      prefix: numthreads=
      separate: False

outputs:
  - id: msimg
    type: string
    outputBinding:
      outputEval: $(inputs.msout)
hints:
  - class: DockerRequirement
    dockerPull: 'loose/rapthor'

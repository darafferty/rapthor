cwlVersion: v1.2
class: CommandLineTool
baseCommand: [DP3]
label: Prepares a dataset for imaging
doc: |
  This tool prepares the input data for imaging without screens or applying
  calibration solutions, including applying the beam model, phase shifting,
  averaging. See prepare_imaging_data.cwl for a detailed description of the
  inputs and outputs

requirements:
  InlineJavascriptRequirement: {}
  ShellCommandRequirement: {}

arguments:
  - msin.datacolumn=DATA
  - msout.overwrite=True
  - msout.writefullresflag=False
  - steps=[applybeam,shift,avg]
  - shift.type=phaseshifter
  - avg.type=squash
  - msout.storagemanager=Dysco

inputs:
  - id: msin
    type: Directory
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
  - id: numthreads
    type: int
    inputBinding:
      prefix: numthreads=
      separate: False

outputs:
  - id: msimg
    type: Directory
    outputBinding:
      glob: $(inputs.msout)

hints:
  - class: DockerRequirement
    dockerPull: 'astronrd/rapthor'

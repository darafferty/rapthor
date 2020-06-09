cwlVersion: v1.0
class: CommandLineTool
baseCommand: [DPPP]
label: "Prepares a dataset for imaging"

requirements:
  InlineJavascriptRequirement: {}
  ShellCommandRequirement: {}

arguments:
  - numthreads=0
  - msin.datacolumn=DATA
  - msout.overwrite=True
  - msout.writefullresflag=False
  - steps=[applybeam,shift,avg]
  - shift.type=phaseshifter
  - avg.type=squash
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

outputs:
  - id: msimg
    type: string
    outputBinding:
      outputEval: $(inputs.msout)

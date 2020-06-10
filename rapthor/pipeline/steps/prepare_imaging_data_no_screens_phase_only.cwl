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
  - steps=[applybeam,shift,applycal,avg]
  - shift.type=phaseshifter
  - avg.type=squash
  - applycal.type=applycal
  - applycal.correction=phase000
  - applycal.steps=[fastphase]
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
    type: string
    inputBinding:
      prefix: applycal.parmdb=
      separate: False
  - id: central_patch_name
    type: string
    inputBinding:
      valueFrom: $('['+self+']')
      prefix: applycal.direction=
      separate: False

outputs:
  - id: msimg
    type: string
    outputBinding:
      outputEval: $(inputs.msout)

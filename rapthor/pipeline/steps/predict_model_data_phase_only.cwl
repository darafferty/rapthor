cwlVersion: v1.2
class: CommandLineTool
baseCommand: [DP3]
label: Predicts model visibilities (phase only)
doc: |
  This tool predicts and corrupts model visibilities for the given MS file,
  using the input sourcedb and h5parm. Phase solutions must be present in the
  input h5parm. See predict_model_data.cwl for a detailed description of the
  inputs and outputs.

requirements:
  - class: InlineJavascriptRequirement
  - class: InitialWorkDirRequirement
    listing:
      - entry: $(inputs.msin)
        writable: false
  - class: InplaceUpdateRequirement
    inplaceUpdate: true

arguments:
  - msin.datacolumn=DATA
  - msout.overwrite=True
  - msout.writefullresflag=False
  - steps=[predict]
  - predict.type=h5parmpredict
  - predict.operation=replace
  - predict.applycal.correction=phase000
  - predict.applycal.steps=[fastphase]
  - predict.applycal.fastphase.correction=phase000
  - predict.usebeammodel=True
  - predict.beammode=array_factor
  - msout.storagemanager=Dysco
  - msout.storagemanager.databitrate=0

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
  - id: onebeamperpatch
    type: boolean
    inputBinding:
      prefix: predict.onebeamperpatch=
      valueFrom: "$(self ? 'True': 'False')"
      separate: False
  - id: h5parm
    type: File
    inputBinding:
      prefix: predict.applycal.parmdb=
      separate: False
  - id: sourcedb
    type: File
    inputBinding:
      prefix: predict.sourcedb=
      separate: False
  - id: directions
    type: string[]
    inputBinding:
      valueFrom: $('['+self+']')
      prefix: predict.directions=
      itemSeparator: ','
      separate: False
  - id: numthreads
    type: string
    inputBinding:
      prefix: numthreads=
      separate: False

outputs:
  - id: msmod
    type: Directory
    outputBinding:
      glob: $(inputs.msout)

hints:
  - class: DockerRequirement
    dockerPull: 'astronrd/rapthor'

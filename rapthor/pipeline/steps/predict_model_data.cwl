cwlVersion: v1.2
class: CommandLineTool
baseCommand: [DP3]
label: Predicts model visibilities
doc: |
  This tool predicts and corrupts model visibilities for the given MS file,
  using the input sourcedb and h5parm. Both phase and amplitude solutions must
  be present in the input h5parm, with the phases being the sum of the fast
  and slow phase solutions. Output is an MS file containing the predicted
  visibilities in the DATA column.

requirements:
  InlineJavascriptRequirement: {}

arguments:
  - msout.overwrite=True
  - steps=[predict]
  - predict.operation=replace
  - predict.applycal.correction=phase000
  - predict.applycal.fastphase.correction=phase000
  - predict.applycal.fastphase.solset=sol000
  - predict.applycal.slowgain.correction=amplitude000
  - predict.applycal.slowgain.solset=sol000
  - predict.applycal.normalization.correction=amplitude000
  - predict.applycal.normalization.solset=sol000
  - predict.usebeammodel=True
  - predict.beam_interval=120
  - predict.beammode=array_factor
  - predict.correctfreqsmearing=True
  - predict.correcttimesmearing=True
  - msout.storagemanager=Dysco
  - msout.storagemanager.databitrate=0  # don't compress data, as they are noiseless

inputs:
  - id: msin
    label: Input MS filename
    doc: |
      The filename of the input MS file.
    type: Directory
    inputBinding:
      prefix: msin=
      separate: False

  - id: data_colname
    label: Input MS data column
    doc: |
      The MS data column to be read.
    type: string
    inputBinding:
      prefix: msin.datacolumn=
      separate: false

  - id: msout
    label: Output MS filename
    doc: |
      The filename of the output MS file.
    type: string
    inputBinding:
      prefix: msout=
      separate: False

  - id: starttime
    label: Start time
    doc: |
      The start time (in casacore MVTime) for the time chunk to be predicted.
    type: string
    inputBinding:
      prefix: msin.starttime=
      separate: False

  - id: ntimes
    label: Number of times
    doc: |
      The number of time slots for the time chunk to be predicted.
    type: int
    inputBinding:
      prefix: msin.ntimes=
      separate: False

  - id: onebeamperpatch
    label: One beam per patch
    doc: |
      Flag that sets beam correction per patch or per source.
    type: boolean
    inputBinding:
      prefix: predict.onebeamperpatch=
      valueFrom: "$(self ? 'True': 'False')"
      separate: False

  - id: sagecalpredict
    label: Use SAGECal predict
    doc: |
      Flag that enables prediction using SAGECal.
    type: boolean
    inputBinding:
      prefix: predict.type=
      valueFrom: "$(self ? 'sagecalpredict': 'h5parmpredict')"
      separate: False

  - id: applycal_steps
    label: List of applycal steps
    doc: |
      The list of applycal steps to perform. Allowed steps are "fastphase",
      "slowgain", and "normalization".
    type: string?
    inputBinding:
      prefix: predict.applycal.steps=
      separate: False

  - id: h5parm
    label: Solution table
    doc: |
      The filename of the h5parm file with the fast-phase and slow-gain corrections.
    type: File?
    inputBinding:
      prefix: predict.applycal.parmdb=
      separate: False

  - id: normalize_h5parm
    label: Filename of h5parm
    doc: |
      The filename of the h5parm file with the flux-scale normalization corrections.
    type: File?
    inputBinding:
      prefix: predict.applycal.normalization.parmdb=
      separate: False

  - id: sourcedb
    label: Sky model
    doc: |
      The sky model to use to predict the model visibilities.
    type: File
    inputBinding:
      prefix: predict.sourcedb=
      separate: False

  - id: directions
    label: Direction names
    doc: |
      The list of direction names (matching those in the h5parm and sky model)
      used in the calibration.
    type: string[]?
    inputBinding:
      valueFrom: $('['+self+']')
      prefix: predict.directions=
      itemSeparator: ','
      separate: False

  - id: numthreads
    label: Number of threads
    doc: |
      The number of threads for DPPP.
    type: int
    inputBinding:
      prefix: numthreads=
      separate: False

outputs:
  - id: msmod
    label: Output MS filename
    doc: |
      The filename of the output MS file.
    type: Directory
    outputBinding:
      glob: $(inputs.msout)

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor

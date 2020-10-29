cwlVersion: v1.0
class: CommandLineTool
baseCommand: [DPPP]
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
  - msin.datacolumn=DATA
  - msout.overwrite=True
  - msout.writefullresflag=False
  - steps=[predict]
  - predict.type=h5parmpredict  # needed to do multi-direction predict
  - predict.operation=replace
  - predict.applycal.correction=phase000
  - predict.applycal.steps=[slowamp,totalphase]
  - predict.applycal.slowamp.correction=amplitude000
  - predict.applycal.totalphase.correction=phase000
  - predict.usebeammodel=True
  - predict.beammode=array_factor  # element beam was removed at phase center
  - predict.onebeamperpatch=True  # better set to False, but slow
  - msout.storagemanager=Dysco
  - msout.storagemanager.databitrate=0  # don't compress data, as they are noiseless

inputs:
  - id: msin
    label: Input MS filename
    doc: |
      The filename of the input MS file.
    type: string
    inputBinding:
      prefix: msin=
      separate: False

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

  - id: h5parm
    label: Solution table
    doc: |
      The solution table to use to corrupt the model visibilities.
    type: string
    inputBinding:
      prefix: predict.applycal.parmdb=
      separate: False

  - id: sourcedb
    label: Sky model
    doc: |
      The sourcedb sky model to use to predict the model visibilities.
    type: string
    inputBinding:
      prefix: predict.sourcedb=
      separate: False

  - id: sourcedb2
    label: Dummy parameter
    doc: |
      A dummy parameter used to enforce step order.
    type: string[]
    inputBinding:
      valueFrom: ''

  - id: directions
    label: Direction names
    doc: |
      The list of direction names (matching those in the h5parm and sourcedb)
      used in the calibration.
    type: string[]
    inputBinding:
      valueFrom: $('['+self+']')
      prefix: predict.directions=
      itemSeparator: ','
      separate: False

  - id: numthreads
    label: Number of threads
    doc: |
      The number of threads for DPPP.
    type: string
    inputBinding:
      prefix: numthreads=
      separate: False

outputs:
  - id: msmod
    label: Output MS filename
    doc: |
      The filename of the output MS file.
    type: string
    outputBinding:
      outputEval: $(inputs.msout)

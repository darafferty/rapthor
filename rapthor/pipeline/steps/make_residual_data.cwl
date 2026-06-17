cwlVersion: v1.2
class: CommandLineTool
baseCommand: [DP3]
label: Makes residual visibility data
doc: |
  This tool makes residual visibility data by subtracting the MODEL_DATA
  column from the DATA column of the input MS file. Output is an MS file
  containing the residual visibilities in the DATA column.

requirements:
  - class: InlineJavascriptRequirement
  - class: ShellCommandRequirement
  - class: InitialWorkDirRequirement
    listing:
      - $(inputs.msin)

arguments:
  - msin.extradatacolumns=[MODEL_DATA]
  - msout.overwrite=True
  - msout.storagemanager=Dysco
  - steps=[combine]
  - combine.type=combine
  - combine.buffername=MODEL_DATA
  - combine.operation=subtract

inputs:
  - id: msin
    label: Filename of input MS
    doc: |
      The filename of input MS file with DATA and MODEL_DATA columns.
    type: Directory
    inputBinding:
      prefix: msin=
      separate: False

  - id: msout
    label: Filename of output MS
    doc: |
      The filename of output MS file.
    type: string
    inputBinding:
      prefix: msout=
      separate: False

  - id: numthreads
    label: Number of threads
    doc: |
      The maximum number of threads to use.
    type: int
    inputBinding:
      prefix: numthreads=
      valueFrom: $(runtime.cores)
      separate: False

outputs:
  - id: msresid
    label: Filename of output MS
    doc: |
      The filename of the output MS with the residual visibilities in the DATA
      column.
    type: Directory
    outputBinding:
      glob: $(inputs.msout)

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor

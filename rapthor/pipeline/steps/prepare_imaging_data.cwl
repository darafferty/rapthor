cwlVersion: v1.2
class: CommandLineTool
baseCommand: [DP3]
label: Prepares a dataset for imaging
doc: |
  This tool prepares the input data for imaging, including applying solutions,
  applying the beam model, phase shifting, and averaging. See the relevant
  parameters below for the allowed processing step names and solution types.
  Output is an MS file containing the imaging visibilities in the DATA column.

requirements:
  InlineJavascriptRequirement: {}
  ShellCommandRequirement: {}

arguments:
  - msin.datacolumn=DATA
  - msout.overwrite=True
  - shift.type=phaseshifter
  - avg.type=squash
  - applycal.type=applycal
  - applycal.correction=phase000
  - applycal.slowamp.correction=amplitude000
  - applycal.slowamp.solset=sol000
  - applycal.fastphase.correction=phase000
  - applycal.fastphase.solset=sol000
  - applycal.fulljones.correction=fulljones
  - applycal.fulljones.solset=sol000
  - applycal.fulljones.soltab=[amplitude000,phase000]
  - applycal.normalization.correction=amplitude000
  - applycal.normalization.solset=sol000
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
  - id: h5parm
    label: Filename of h5parm
    doc: |
      The filename of the h5parm file with the direction-dependent calibration solutions.
    type: File?
    inputBinding:
      prefix: applycal.parmdb=
      separate: False
  - id: fulljones_h5parm
    label: Filename of h5parm
    doc: |
      The filename of the h5parm file with the full-Jones calibration solutions.
    type: File?
    inputBinding:
      prefix: applycal.fulljones.parmdb=
      separate: False
  - id: normalize_h5parm
    label: Filename of h5parm
    doc: |
      The filename of the h5parm file with the flux-scale normalization corrections.
    type: File?
    inputBinding:
      prefix: applycal.normalization.parmdb=
      separate: False
  - id: central_patch_name
    label: Name of central patch
    doc: |
      The name of the central-most patch of the sector.
    type: string?
    inputBinding:
      valueFrom: $('['+self+']')
      prefix: applycal.direction=
      separate: False
  - id: numthreads
    label: Number of threads
    doc: |
      The maximum number of threads to use.
    type: int
    inputBinding:
      prefix: numthreads=
      separate: False
  - id: steps
    label: List of steps
    doc: |
      The list of steps to perform. Allowed steps are "applybeam", "shift", "avg", and
      "applycal".
    type: string
    inputBinding:
      prefix: steps=
      separate: False
  - id: applycal_steps
    label: List of applycal steps
    doc: |
      The list of applycal steps to perform. Allowed steps are "fastphase", "slowamp",
      "fulljones", and "normalization"
    type: string
    inputBinding:
      prefix: applycal.steps=
      separate: False

outputs:
  - id: msimg
    type: Directory
    outputBinding:
      glob: $(inputs.msout)

hints:
  - class: DockerRequirement
    dockerPull: 'astronrd/rapthor'

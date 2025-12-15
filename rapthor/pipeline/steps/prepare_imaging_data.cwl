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
  - msout.overwrite=True
  - shift.type=phaseshifter
  - avg.type=squash
  - bdaavg.type=bdaaverager
  - bdaavg.minchannels=1
  - bdaavg.frequencybase=0.0
  - applycal.type=applycal
  - applycal.correction=phase000
  - applycal.slowgain.correction=amplitude000
  - applycal.slowgain.solset=sol000
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
    label: Filename of input MS
    doc: |
      The filename of input MS file.
    type: Directory
    inputBinding:
      prefix: msin=
      separate: False

  - id: data_colname
    type: string
    inputBinding:
      prefix: msin.datacolumn=
      separate: false

  - id: msout
    label: Filename of output MS
    doc: |
      The filename of output MS file.
    type: string
    inputBinding:
      prefix: msout=
      separate: False

  - id: starttime
    label: Start time
    doc: |
      The start time (in casacore MVTime) of the chunk of the MS to be processed.
    type: string
    inputBinding:
      prefix: msin.starttime=
      separate: False

  - id: ntimes
    label: Number of times
    doc: |
      The number of timeslots of the chunk of the MS to be processed.
    type: int
    inputBinding:
      prefix: msin.ntimes=
      separate: False

  - id: phasecenter
    label: Phase center
    doc: |
      The phase center in deg to phase shift to.
    type: string
    inputBinding:
      prefix: shift.phasecenter=
      separate: False
      shellQuote: False

  - id: freqstep
    label: Averaging interval in frequency
    doc: |
      The averaging interval in number of frequency channels.
    type: int
    inputBinding:
      prefix: avg.freqstep=
      separate: False

  - id: timestep
    label: Averaging interval in time
    doc: |
      The averaging interval in number of timeslots.
    type: int
    inputBinding:
      prefix: avg.timestep=
      separate: False

  - id: timebase
    label: BDA timebase
    doc: |
      The baseline length (in meters) below which BDA time averaging is done.
    type: float?
    inputBinding:
      prefix: bdaavg.timebase=
      separate: False

  - id: maxinterval
    label: BDA maxinterval
    doc: |
      The maximum interval duration (in time slots) over which BDA time averaging is done.
    type: int?
    inputBinding:
      prefix: bdaavg.maxinterval=
      separate: False

  - id: beamdir
    label: Direction in which to apply beam
    doc: |
      The direction in deg in which to apply the beam.
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
      The list of applycal steps to perform. Allowed steps are "fastphase", "slowgain",
      "fulljones", and "normalization".
    type: string?
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
    dockerPull: astronrd/rapthor:2.1.post1

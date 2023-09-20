cwlVersion: v1.2
class: CommandLineTool
baseCommand: [DP3]
label: Prepares a dataset for imaging
doc: |
  This tool prepares the input data for imaging with direction-dependent corrections,
  including applying the direction-independent full-Jones solutions and the beam model,
  phase shifting, and averaging.

requirements:
  InlineJavascriptRequirement: {}
  ShellCommandRequirement: {}

arguments:
  - msin.datacolumn=DATA
  - msout.overwrite=True
  - msout.writefullresflag=False
  - steps=[applybeam,applycal,shift,avg]
  - applycal.type=applycal
  - applycal.correction=fulljones
  - applycal.solset=sol000
  - applycal.soltab=[amplitude000,phase000]
  - shift.type=phaseshifter
  - avg.type=squash
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
  - id: beamdir
    label: Direction in which to apply beam
    doc: |
      The direction in deg in which to apply the beam.
    type: string
    inputBinding:
      prefix: applybeam.direction=
      separate: False
      shellQuote: False
  - id: numthreads
    label: Number of threads
    doc: |
      The number of threads to use during solve (0 = all).
    type: int
    inputBinding:
      prefix: numthreads=
      separate: False
  - id: h5parm
    label: Filename of h5parm
    doc: |
      The filename of the h5parm file with the calibration solutions.
    type: File
    inputBinding:
      prefix: applycal.parmdb=
      separate: False

outputs:
  - id: msimg
    label: Output MS
    doc: |
      The filename of the output MS file The value is taken from the input
      parameter "msout"
    type: Directory
    outputBinding:
      glob: $(inputs.msout)

hints:
  - class: DockerRequirement
    dockerPull: 'astronrd/rapthor'

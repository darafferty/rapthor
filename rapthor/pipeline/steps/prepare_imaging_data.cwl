cwlVersion: v1.0
class: CommandLineTool
baseCommand: [DP3]
label: Prepares a dataset for imaging
doc: |
  This tool prepares the input data for imaging with screens, including applying
  the beam model, phase shifting, and averaging.

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
    label: Filename of input MS
    doc: |
      The filename of input MS file.
    type: string
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
    type: string
    inputBinding:
      prefix: numthreads=
      separate: False

outputs:
  - id: msimg
    label: Output MS
    doc: |
      The filename of the output MS file The value is taken from the input
      parameter "msout"
    type: string
    outputBinding:
      outputEval: $(inputs.msout)
hints:
  - class: DockerRequirement
    dockerPull: 'loose/rapthor'

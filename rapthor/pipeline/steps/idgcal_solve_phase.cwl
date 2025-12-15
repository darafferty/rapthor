cwlVersion: v1.2
class: CommandLineTool
baseCommand: [DP3]
id: idgcal_solve_phase
label: Calibrates a dataset using IDGCal
doc: |
  This tool solves for phase screens using IDGCal for the given MS file,
  using the input model image. Output is the solution table in h5parm
  format.

requirements:
  - class: InlineJavascriptRequirement

arguments:
  - msin.datacolumn=DATA
  - msout=
  - steps=[solve]
  - solve.type=python
  - solve.python.module=idg.idgcaldpstep_phase_only_dirac
  - solve.python.class=IDGCalDPStepPhaseOnlyDirac
  - solve.nrcorrelations=4
  - solve.subgridsize=32
  - solve.tapersupport=7
  - solve.wtermsupport=5
  - solve.atermsupport=5
  - solve.solverupdategain=0.5
  - solve.tolerancepinv=1e-9
  - solve.polynomialdegphase=2
  - solve.nr_channels_per_block=30
  - solve.lbfgshistory=10
  - solve.lbfgsminibatches=3
  - solve.lbfgsepochs=3

inputs:
  - id: msin
    label: Input MS directory name
    doc: |
      The name of the input MS directory.
    type: Directory
    inputBinding:
      prefix: msin=
      separate: False

  - id: starttime
    label: Start time
    doc: |
      The start time (in casacore MVTime) for the time chunk to be processed.
    type: string
    inputBinding:
      prefix: msin.starttime=
      separate: False

  - id: ntimes
    label: Number of times
    doc: |
      The number of time slots for the time chunk to be processed.
    type: int
    inputBinding:
      prefix: msin.ntimes=
      separate: False

  - id: h5parm
    label: Solution table
    doc: |
      The filename of the output solution table.
    type: string
    inputBinding:
      prefix: solve.h5parm=
      separate: False

  - id: solint
    label: Solution interval
    doc: |
      The solution interval in timeslots for the solve.
    type: int
    inputBinding:
      prefix: solve.solintphase=
      separate: False

  - id: model_image
    label: Sky model image
    doc: |
      The sky model FITS image(s) (one per spectral term) to use for the solve.
      Note: currently, IDGCal does not support multiple spectra terms, so we just
      use the flux image (the first one).
    type: File[]
    inputBinding:
      valueFrom: "$(self[0].path)"
      prefix: solve.modelimage=
      separate: False

  - id: maxiter
    label: Maximum iterations
    doc: |
      The maximum number of iterations in the solve.
    type: int
    inputBinding:
      prefix: solve.maxiter=
      separate: False

  - id: antennaconstraint
    label: Antenna constraint
    doc: |
      A list of antennas that will be constrained to have the same solutions.
    type: string
    inputBinding:
      prefix: solve.antennaconstraint=
      separate: False

  - id: numthreads
    label: Number of threads
    doc: |
      The number of threads to use during solve (0 = all).
    type: int
    inputBinding:
      prefix: numthreads=
      separate: False

outputs:
  - id: output_h5parm
    label: Solution table
    doc: |
      The filename of the output solution table. The value is taken from the input
      parameter "h5parm"
    type: File
    outputBinding:
      glob: $(inputs.h5parm)
hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor

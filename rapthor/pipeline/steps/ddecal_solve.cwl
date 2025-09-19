cwlVersion: v1.2
class: CommandLineTool
baseCommand: [DP3]
label: Calibrates a dataset using DDECal
doc: |
  This tool solves for corrections in multiple directions simultaneously for
  the given MS file, using the input sourcedb and (optionally) baseline-
  dependent averaging. See the relevant parameters below for the allowed
  processing step names and solve types. Output is the solution table in
  h5parm format.

requirements:
  - class: InlineJavascriptRequirement

arguments:
  - msout=
  - applybeam.type=applybeam
  - applybeam.beammode=array_factor
  - applybeam.usemodeldata=True
  - applybeam.invert=False
  - applycal.type=applycal
  - applycal.fastphase.correction=phase000
  - applycal.fastphase.solset=sol000
  - applycal.fastphase.usemodeldata=True
  - applycal.fastphase.invert=False
  - applycal.normalization.correction=amplitude000
  - applycal.normalization.solset=sol000
  - applycal.normalization.usemodeldata=True
  - applycal.normalization.invert=False
  - avg.type=bdaaverager
  - avg.minchannels=1
  - avg.frequencybase=0.0
  - predict.type=wgridderpredict
  - solve.type=ddecal
  - solve.usebeammodel=True
  - solve.beam_interval=120
  - solve.beammode=array_factor
  - solve.correctfreqsmearing=True
  - solve.correcttimesmearing=True
  - solve.applycal.fastphase.correction=phase000
  - solve.applycal.fastphase.solset=sol000
  - solve.applycal.normalization.correction=amplitude000
  - solve.applycal.normalization.solset=sol000
  - solve.initialsolutions.missingantennabehavior=unit

inputs:
  - id: msin
    label: Input MS directory name
    doc: |
      The name of the input MS directory.
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

  - id: startchan
    label: Start channel
    doc: |
      The start index for the frequency chunk to be processed.
    type: int?
    inputBinding:
      prefix: msin.startchan=
      separate: False

  - id: nchan
    label: Number of channels
    doc: |
      The number of channels for the frequency chunk to be processed.
    type: int?
    inputBinding:
      prefix: msin.nchan=
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
      The solution interval in number of time slots for the solve.
    type: int
    inputBinding:
      prefix: solve.solint=
      separate: False

  - id: solve_nchan
    label: Solution interval
    doc: |
      The solution interval in number of channels for the solve.
    type: int
    inputBinding:
      prefix: solve.nchan=
      separate: False

  - id: mode
    label: Solver mode
    doc: |
      The solver mode to use for the solve.
    type: string
    inputBinding:
      prefix: solve.mode=
      separate: False

  - id: steps
    label: Processing steps
    doc: |
      The list of processing steps to perform.
    type: string
    inputBinding:
      prefix: steps=
      separate: False

  - id: applycal_steps
    label: List of applycal steps
    doc: |
      The list of applycal steps to perform. Allowed steps are "fastphase",
      "slowgain", and "normalization".
    type: string?
    inputBinding:
      prefix: applycal.steps=
      separate: False

  - id: ddecal_applycal_steps
    label: List of DDECal applycal steps
    doc: |
      The list of DDECal applycal steps to perform. Allowed steps are "fastphase",
      "slowgain", and "normalization".
    type: string?
    inputBinding:
      prefix: solve.applycal.steps=
      separate: False

  - id: fastphase_h5parm
    label: Solution table
    doc: |
      The filename of the input solution table containing the fast-phase
      solutions. These solutions are preapplied before the solve is done.
    type: File?
    inputBinding:
      prefix: applycal.fastphase.parmdb=
      separate: False

  - id: ddecal_fastphase_h5parm
    label: Solution table
    doc: |
      The filename of the input solution table containing the fast-phase
      solutions. These solutions are applied during the solve.
    type: File?
    inputBinding:
      prefix: solve.applycal.fastphase.parmdb=
      separate: False

  - id: normalize_h5parm
    label: Filename of h5parm
    doc: |
      The filename of the h5parm file with the flux-scale normalization corrections.
      These solutions are preapplied before the solve is done.
    type: File?
    inputBinding:
      prefix: applycal.normalization.parmdb=
      separate: False

  - id: ddecal_normalize_h5parm
    label: Filename of h5parm
    doc: |
      The filename of the h5parm file with the flux-scale normalization corrections.
      These solutions are applied during the solve.
    type: File?
    inputBinding:
      prefix: solve.applycal.normalization.parmdb=
      separate: False

  - id: timebase
    label: BDA timebase
    doc: |
      The baseline length (in meters) below which BDA time averaging is done.
    type: float?
    inputBinding:
      prefix: avg.timebase=
      separate: False

  - id: maxinterval
    label: BDA maxinterval
    doc: |
      The maximum interval duration (in time slots) over which BDA time averaging is done.
    type: int?
    inputBinding:
      prefix: avg.maxinterval=
      separate: False

  - id: directions
    label: Direction names
    doc: |
      The names of the directions for the solve.
    type: string[]?
    inputBinding:
      valueFrom: $('['+self+']')
      prefix: solve.directions=
      itemSeparator: ','
      separate: False

  - id: solutions_per_direction
    label: Solutions per direction
    doc: |
      The number of solution intervals (in time) per direction for the solve.
    type: int[]?
    inputBinding:
      valueFrom: $('['+self+']')
      prefix: solve.solutions_per_direction=
      itemSeparator: ','
      separate: False

  - id: sourcedb
    label: Sky model
    doc: |
      The sourcedb sky model to use for the solve.
    type: File?
    inputBinding:
      prefix: solve.sourcedb=
      separate: False

  - id: modeldatacolumn
    label: Model data column
    doc: |
      The name of the model data to use for the solve (used if no sourcedb is given).
    type: string?
    inputBinding:
      prefix: solve.modeldatacolumns=
      separate: False

  - id: llssolver
    label: Linear least-squares solver
    doc: |
      The linear least-squares solver to use (one of 'qr', 'svd', or 'lsmr').
    type: string
    inputBinding:
      prefix: solve.llssolver=
      separate: False

  - id: maxiter
    label: Maximum iterations
    doc: |
      The maximum number of iterations in the solve.
    type: int
    inputBinding:
      prefix: solve.maxiter=
      separate: False

  - id: propagatesolutions
    label: Propagate solutions
    doc: |
      Flag that determines whether solutions are propagated as initial start values
      for the next solution interval.
    type: boolean
    inputBinding:
      prefix: solve.propagatesolutions=
      valueFrom: "$(self ? 'True': 'False')"
      separate: False

  - id: initialsolutions_h5parm
    label: Initial solutions H5parm
    doc: |
      The input H5parm file containing initial solutions.
    type: File?
    inputBinding:
      prefix: solve.initialsolutions.h5parm=
      separate: False

  - id: initialsolutions_soltab
    label: Initial solutions soltab
    doc: |
      The solution table containing initial solutions.
    type: string?
    inputBinding:
      prefix: solve.initialsolutions.soltab=
      separate: False

  - id: solveralgorithm
    label: Solver algorithm
    doc: |
      The algorithm used for solving.
    type: string
    inputBinding:
      prefix: solve.solveralgorithm=
      separate: False

  - id: solverlbfgs_dof
    label: LBFGS solver DOF
    doc: |
      The degrees of freedom for the LBFGS solve algorithm.
    type: float
    inputBinding:
      prefix: solve.solverlbfgs.dof=
      separate: False

  - id: solverlbfgs_iter
    label: LBFGS solver iterations
    doc: |
      The number of iterations for the LBFGS solve algorithm.
    type: int
    inputBinding:
      prefix: solve.solverlbfgs.iter=
      separate: False

  - id: solverlbfgs_minibatches
    label: LBFGS solver minibatches
    doc: |
      The number of minibatches for the LBFGS solve algorithm.
    type: int
    inputBinding:
      prefix: solve.solverlbfgs.minibatches=
      separate: False

  - id: onebeamperpatch
    label: One beam per patch
    doc: |
      Flag that sets beam correction per patch or per source.
    type: boolean?
    inputBinding:
      prefix: solve.onebeamperpatch=
      valueFrom: "$(self ? 'True': 'False')"
      separate: False

  - id: parallelbaselines
    label: Parallelize over baselines
    doc: |
      Flag that enables parallel prediction over baselines.
    type: boolean?
    inputBinding:
      prefix: solve.parallelbaselines=
      valueFrom: "$(self ? 'True': 'False')"
      separate: False

  - id: sagecalpredict
    label: predict using SAGECal
    doc: |
      Flag that enables prediction using SAGECal.
    type: boolean?
    inputBinding:
      prefix: solve.sagecalpredict=
      valueFrom: "$(self ? 'True': 'False')"
      separate: False

  - id: datause
    label: Datause parameter
    doc: |
      The datause parameter that determines how the visibilies are used in
      the solves.
    type: string?
    inputBinding:
      prefix: solve.datause=
      separate: False

  - id: stepsize
    label: Solver step size
    doc: |
      The solver step size used between iterations.
    type: float
    inputBinding:
      prefix: solve.stepsize=
      separate: False

  - id: stepsigma
    label: Solver step size standard deviation factor.
    doc: |
      If the solver step size mean is lower than its standard deviation by this
      factor, stop iterations.
    type: float
    inputBinding:
      prefix: solve.stepsigma=
      separate: False

  - id: tolerance
    label: Solver tolerance
    doc: |
      The solver tolerance used to define convergence.
    type: float
    inputBinding:
      prefix: solve.tolerance=
      separate: False

  - id: uvlambdamin
    label: Minimum uv distance in lambda
    doc: |
      The minimum uv distance to use in the calibration.
    type: float
    inputBinding:
      prefix: solve.uvlambdamin=
      separate: False

  - id: smoothness_dd_factors
    label: Smoothness factors
    doc: |
      The factor by which to multiply the smoothnesscontraint, per direction, for the solve.
    type: float[]?
    inputBinding:
      valueFrom: $('['+self+']')
      prefix: solve.smoothness_dd_factors=
      itemSeparator: ','
      separate: False

  - id: smoothnessconstraint
    label: Smoothness constraint kernel size
    doc: |
      The smoothness constraint kernel size in Hz, used to enforce a smooth frequency
      dependence of the phase solutions.
    type: float?
    inputBinding:
      prefix: solve.smoothnessconstraint=
      separate: False

  - id: smoothnessreffrequency
    label: Smoothness constraint reference frequency
    doc: |
      The smoothness constraint reference frequency in Hz.
    type: float?
    inputBinding:
      prefix: solve.smoothnessreffrequency=
      separate: False

  - id: smoothnessrefdistance
    label: Smoothness constraint reference distance
    doc: |
      The smoothness constraint reference distance in m.
    type: float?
    inputBinding:
      prefix: solve.smoothnessrefdistance=
      separate: False

  - id: antennaconstraint
    label: Antenna constraint
    doc: |
      A list of antennas that will be constrained to have the same solutions.
    type: string?
    inputBinding:
      prefix: solve.antennaconstraint=
      separate: False

  - id: predict_regions
    label: Regions for image-based predict
    doc: |
      A ds9 file that defines the regions to use with image-based predict.
    type: File?
    inputBinding:
      prefix: predict.regions=
      separate: False

  - id: predict_images
    label: Images for image-based predict
    doc: |
      The model images, one per spectral term, to use with image-based predict.
    type: File[]?
    inputBinding:
      valueFrom: "[$(self.map(function(file){ return file.path; }).join(','))]"
      prefix: predict.images=
      separate: False

  - id: reuse_model
    label: Reuse model list
    doc: |
      A list of model data columns that will be reused from the predict step.
    type: string?
    inputBinding:
      prefix: solve.reusemodel=
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
      parameter "h5parm".
    type: File
    outputBinding:
      glob: $(inputs.h5parm)

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor

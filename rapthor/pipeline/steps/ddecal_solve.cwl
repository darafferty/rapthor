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
  - predict.type=wgridderpredict
  - solve1.type=ddecal
  - solve1.usebeammodel=True
  - solve1.beam_interval=120
  - solve1.beammode=array_factor
  - solve1.applycal.normalization.correction=amplitude000
  - solve1.applycal.normalization.solset=sol000
  - solve1.initialsolutions.missingantennabehavior=unit
  - solve1.correctfreqsmearing=False
  - solve1.correcttimesmearing=False
  - solve2.type=ddecal
  - solve2.initialsolutions.missingantennabehavior=unit

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
      The list of DDECal applycal steps to perform in solve1. Currently, only "normalization"
      is allowed.
    type: string?
    inputBinding:
      prefix: solve1.applycal.steps=
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

  - id: timebase
    label: BDA timebase
    doc: |
      The baseline length (in meters) below which BDA time averaging is done.
    type: float?
    inputBinding:
      prefix: avg.timebase=
      separate: False

  - id: frequencybase
    label: BDA frequencybase
    doc: |
      The baseline length (in meters) below which BDA frequency averaging is done.
    type: float?
    inputBinding:
      prefix: avg.frequencybase=
      separate: False

  - id: maxinterval
    label: BDA maxinterval
    doc: |
      The maximum interval duration (in sec) over which BDA time averaging is done.
    type: float?
    inputBinding:
      prefix: avg.maxinterval=
      separate: False

  - id: minchannels
    label: BDA minchannels
    doc: |
      The minimum number of channels remaining after BDA frequency averaging is done.
    type: int?
    inputBinding:
      prefix: avg.minchannels=
      separate: False

  - id: directions
    label: Direction names
    doc: |
      The names of the directions for solve1.
    type: string[]?
    inputBinding:
      valueFrom: $('['+self+']')
      prefix: solve1.directions=
      itemSeparator: ','
      separate: False

  - id: sourcedb
    label: Sky model
    doc: |
      The sourcedb sky model to use for solve1.
    type: File?
    inputBinding:
      prefix: solve1.sourcedb=
      separate: False

  - id: modeldatacolumn
    label: Model data column
    doc: |
      The name of the model data to use for solve1 (used if no sourcedb is given).
    type: string?
    inputBinding:
      prefix: solve1.modeldatacolumns=
      separate: False

  - id: onebeamperpatch
    label: One beam per patch
    doc: |
      Flag that sets beam correction per patch or per source.
    type: boolean?
    inputBinding:
      prefix: solve1.onebeamperpatch=
      valueFrom: "$(self ? 'True': 'False')"
      separate: False

  - id: parallelbaselines
    label: Parallelize over baselines
    doc: |
      Flag that enables parallel prediction over baselines in solve1.
    type: boolean?
    inputBinding:
      prefix: solve1.parallelbaselines=
      valueFrom: "$(self ? 'True': 'False')"
      separate: False

  - id: sagecalpredict
    label: predict using SAGECal
    doc: |
      Flag that enables prediction using SAGECal in solve1.
    type: boolean?
    inputBinding:
      prefix: solve1.sagecalpredict=
      valueFrom: "$(self ? 'True': 'False')"
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

  - id: solve1_normalize_h5parm
    label: Filename of h5parm
    doc: |
      The filename of the h5parm file with the flux-scale normalization corrections.
      These solutions are applied during solve1.
    type: File?
    inputBinding:
      prefix: solve1.applycal.normalization.parmdb=
      separate: False

  - id: solve1_h5parm
    label: Solution table
    doc: |
      The filename of the output solution table for solve1.
    type: string
    inputBinding:
      prefix: solve1.h5parm=
      separate: False

  - id: solve1_solint
    label: Solution interval
    doc: |
      The solution interval in number of time slots for solve1.
    type: int
    inputBinding:
      prefix: solve1.solint=
      separate: False

  - id: solve1_nchan
    label: Solution interval
    doc: |
      The solution interval in number of channels for solve1.
    type: int
    inputBinding:
      prefix: solve1.nchan=
      separate: False

  - id: solve1_mode
    label: Solver mode
    doc: |
      The solver mode to use for solve1.
    type: string
    inputBinding:
      prefix: solve1.mode=
      separate: False

  - id: solve1_solutions_per_direction
    label: Solutions per direction
    doc: |
      The number of solution intervals (in time) per direction for solve1.
      Note: this parameter is not yet supported in multi-calibration and so
      should either not be set or be set to a list of ones.
    type: int[]?
    inputBinding:
      valueFrom: $('['+self+']')
      prefix: solve1.solutions_per_direction=
      itemSeparator: ','
      separate: False

  - id: solve1_llssolver
    label: Linear least-squares solver
    doc: |
      The linear least-squares solver to use for solve1 (one of 'qr', 'svd', or 'lsmr').
    type: string
    inputBinding:
      prefix: solve1.llssolver=
      separate: False

  - id: solve1_maxiter
    label: Maximum iterations
    doc: |
      The maximum number of iterations in solve1.
    type: int
    inputBinding:
      prefix: solve1.maxiter=
      separate: False

  - id: solve1_propagatesolutions
    label: Propagate solutions
    doc: |
      Flag that determines whether solutions are propagated as initial start values
      for the next solution interval in solve1.
    type: boolean
    inputBinding:
      prefix: solve1.propagatesolutions=
      valueFrom: "$(self ? 'True': 'False')"
      separate: False

  - id: solve1_initialsolutions_h5parm
    label: Initial solutions H5parm
    doc: |
      The input H5parm file containing initial solutions for solve1.
    type: File?
    inputBinding:
      prefix: solve1.initialsolutions.h5parm=
      separate: False

  - id: solve1_initialsolutions_soltab
    label: Initial solutions soltab
    doc: |
      The solution table containing initial solutions for solve1.
    type: string?
    inputBinding:
      prefix: solve1.initialsolutions.soltab=
      separate: False

  - id: solve1_solveralgorithm
    label: Solver algorithm
    doc: |
      The algorithm used for solving in solve1.
    type: string
    inputBinding:
      prefix: solve1.solveralgorithm=
      separate: False

  - id: solve1_solverlbfgs_dof
    label: LBFGS solver DOF
    doc: |
      The degrees of freedom for the LBFGS solve algorithm in solve1.
    type: float
    inputBinding:
      prefix: solve1.solverlbfgs.dof=
      separate: False

  - id: solve1_solverlbfgs_iter
    label: LBFGS solver iterations
    doc: |
      The number of iterations for the LBFGS solve algorithm in solve1.
    type: int
    inputBinding:
      prefix: solve1.solverlbfgs.iter=
      separate: False

  - id: solve1_solverlbfgs_minibatches
    label: LBFGS solver minibatches
    doc: |
      The number of minibatches for the LBFGS solve algorithm in solve1.
    type: int
    inputBinding:
      prefix: solve1.solverlbfgs.minibatches=
      separate: False

  - id: solve1_datause
    label: Datause parameter
    doc: |
      The datause parameter that determines how the visibilies are used in
      the solve1.
    type: string?
    inputBinding:
      prefix: solve1.datause=
      separate: False

  - id: solve1_stepsize
    label: Solver step size
    doc: |
      The solver step size used between iterations in solve1.
    type: float
    inputBinding:
      prefix: solve1.stepsize=
      separate: False

  - id: solve1_stepsigma
    label: Solver step size standard deviation factor.
    doc: |
      If the solver step size mean is lower than its standard deviation by this
      factor, stop iterations.
    type: float
    inputBinding:
      prefix: solve1.stepsigma=
      separate: False

  - id: solve1_tolerance
    label: Solver tolerance
    doc: |
      The solver tolerance used to define convergence in solve1.
    type: float
    inputBinding:
      prefix: solve1.tolerance=
      separate: False

  - id: solve1_uvlambdamin
    label: Minimum uv distance in lambda
    doc: |
      The minimum uv distance to use in the calibration for solve1.
    type: float
    inputBinding:
      prefix: solve1.uvlambdamin=
      separate: False

  - id: solve1_smoothness_dd_factors
    label: Smoothness factors
    doc: |
      The factor by which to multiply the smoothnesscontraint, per direction, for solve1.
    type: float[]?
    inputBinding:
      valueFrom: $('['+self+']')
      prefix: solve1.smoothness_dd_factors=
      itemSeparator: ','
      separate: False

  - id: solve1_smoothnessconstraint
    label: Smoothness constraint kernel size
    doc: |
      The smoothness constraint kernel size in Hz, used to enforce a smooth frequency
      dependence, for solve1.
    type: float?
    inputBinding:
      prefix: solve1.smoothnessconstraint=
      separate: False

  - id: solve1_smoothnessreffrequency
    label: Smoothness constraint reference frequency
    doc: |
      The smoothness constraint reference frequency in Hz for solve1.
    type: float?
    inputBinding:
      prefix: solve1.smoothnessreffrequency=
      separate: False

  - id: solve1_smoothnessrefdistance
    label: Smoothness constraint reference distance
    doc: |
      The smoothness constraint reference distance in m for solve1.
    type: float?
    inputBinding:
      prefix: solve1.smoothnessrefdistance=
      separate: False

  - id: solve1_antennaconstraint
    label: Antenna constraint
    doc: |
      A list of antennas that will be constrained to have the same solutions in solve1.
    type: string?
    inputBinding:
      prefix: solve1.antennaconstraint=
      separate: False

  - id: solve1_keepmodel
    label: Keep model data
    doc: |
      Flag that determines whether model data used in solve1 is kept for
      subsequent steps.
    type: string?
    inputBinding:
      prefix: solve1.keepmodel=
      separate: False

  - id: solve1_reusemodel
    label: Reuse model list
    doc: |
      A list of model data columns that will be reused from an earlier
      step for solve1.
    type: string?
    inputBinding:
      prefix: solve1.reusemodel=
      separate: False

  - id: solve2_normalize_h5parm
    label: Filename of h5parm
    doc: |
      The filename of the h5parm file with the flux-scale normalization corrections.
      These solutions are applied during solve1.
    type: File?
    inputBinding:
      prefix: solve2.applycal.normalization.parmdb=
      separate: False

  - id: solve2_h5parm
    label: Solution table
    doc: |
      The filename of the output solution table for solve2.
    type: string?
    inputBinding:
      prefix: solve2.h5parm=
      separate: False

  - id: solve2_solint
    label: Solution interval
    doc: |
      The solution interval in number of time slots for solve2.
    type: int?
    inputBinding:
      prefix: solve2.solint=
      separate: False

  - id: solve2_nchan
    label: Solution interval
    doc: |
      The solution interval in number of channels for solve2.
    type: int?
    inputBinding:
      prefix: solve2.nchan=
      separate: False

  - id: solve2_llssolver
    label: Linear least-squares solver
    doc: |
      The linear least-squares solver to use for solve2 (one of 'qr', 'svd', or 'lsmr').
    type: string?
    inputBinding:
      prefix: solve2.llssolver=
      separate: False

  - id: solve2_mode
    label: Solver mode
    doc: |
      The solver mode to use for solve2.
    type: string?
    inputBinding:
      prefix: solve2.mode=
      separate: False

  - id: solve2_solutions_per_direction
    label: Solutions per direction
    doc: |
      The number of solution intervals (in time) per direction for solve2.
      Note: this parameter is not yet supported in multi-calibration and so
      should either not be set or be set to a list of ones.
    type: int[]?
    inputBinding:
      valueFrom: $('['+self+']')
      prefix: solve2.solutions_per_direction=
      itemSeparator: ','
      separate: False

  - id: solve2_maxiter
    label: Maximum iterations
    doc: |
      The maximum number of iterations in solve2.
    type: int?
    inputBinding:
      prefix: solve2.maxiter=
      separate: False

  - id: solve2_propagatesolutions
    label: Propagate solutions
    doc: |
      Flag that determines whether solutions are propagated as initial start values
      for the next solution interval in solve2.
    type: boolean?
    inputBinding:
      prefix: solve2.propagatesolutions=
      valueFrom: "$(self ? 'True': 'False')"
      separate: False

  - id: solve2_initialsolutions_h5parm
    label: Initial solutions H5parm
    doc: |
      The input H5parm file containing initial solutions for solve2.
    type: File?
    inputBinding:
      prefix: solve2.initialsolutions.h5parm=
      separate: False

  - id: solve2_initialsolutions_soltab
    label: Initial solutions soltab
    doc: |
      The solution table containing initial solutions for solve2.
    type: string?
    inputBinding:
      prefix: solve2.initialsolutions.soltab=
      separate: False

  - id: solve2_solveralgorithm
    label: Solver algorithm
    doc: |
      The algorithm used for solving in solve2.
    type: string?
    inputBinding:
      prefix: solve2.solveralgorithm=
      separate: False

  - id: solve2_solverlbfgs_dof
    label: LBFGS solver DOF
    doc: |
      The degrees of freedom for the LBFGS solve algorithm in solve2.
    type: float?
    inputBinding:
      prefix: solve2.solverlbfgs.dof=
      separate: False

  - id: solve2_solverlbfgs_iter
    label: LBFGS solver iterations
    doc: |
      The number of iterations for the LBFGS solve algorithm in solve2.
    type: int?
    inputBinding:
      prefix: solve2.solverlbfgs.iter=
      separate: False

  - id: solve2_solverlbfgs_minibatches
    label: LBFGS solver minibatches
    doc: |
      The number of minibatches for the LBFGS solve algorithm in solve2.
    type: int?
    inputBinding:
      prefix: solve2.solverlbfgs.minibatches=
      separate: False

  - id: solve2_datause
    label: Datause parameter
    doc: |
      The datause parameter that determines how the visibilies are used in
      the solve2.
    type: string?
    inputBinding:
      prefix: solve2.datause=
      separate: False

  - id: solve2_stepsize
    label: Solver step size
    doc: |
      The solver step size used between iterations in solve2.
    type: float?
    inputBinding:
      prefix: solve2.stepsize=
      separate: False

  - id: solve2_stepsigma
    label: Solver step size standard deviation factor.
    doc: |
      If the solver step size mean is lower than its standard deviation by this
      factor, stop iterations.
    type: float?
    inputBinding:
      prefix: solve2.stepsigma=
      separate: False

  - id: solve2_tolerance
    label: Solver tolerance
    doc: |
      The solver tolerance used to define convergence in solve2.
    type: float?
    inputBinding:
      prefix: solve2.tolerance=
      separate: False

  - id: solve2_uvlambdamin
    label: Minimum uv distance in lambda
    doc: |
      The minimum uv distance to use in the calibration for solve2.
    type: float?
    inputBinding:
      prefix: solve2.uvlambdamin=
      separate: False

  - id: solve2_smoothness_dd_factors
    label: Smoothness factors
    doc: |
      The factor by which to multiply the smoothnesscontraint, per direction, for solve2.
    type: float[]?
    inputBinding:
      valueFrom: $('['+self+']')
      prefix: solve2.smoothness_dd_factors=
      itemSeparator: ','
      separate: False

  - id: solve2_smoothnessconstraint
    label: Smoothness constraint kernel size
    doc: |
      The smoothness constraint kernel size in Hz, used to enforce a smooth frequency
      dependence, for solve2.
    type: float?
    inputBinding:
      prefix: solve2.smoothnessconstraint=
      separate: False

  - id: solve2_smoothnessreffrequency
    label: Smoothness constraint reference frequency
    doc: |
      The smoothness constraint reference frequency in Hz for solve2.
    type: float?
    inputBinding:
      prefix: solve2.smoothnessreffrequency=
      separate: False

  - id: solve2_smoothnessrefdistance
    label: Smoothness constraint reference distance
    doc: |
      The smoothness constraint reference distance in m for solve2.
    type: float?
    inputBinding:
      prefix: solve2.smoothnessrefdistance=
      separate: False

  - id: solve2_antennaconstraint
    label: Antenna constraint
    doc: |
      A list of antennas that will be constrained to have the same solutions in solve2.
    type: string?
    inputBinding:
      prefix: solve2.antennaconstraint=
      separate: False

  - id: solve2_keepmodel
    label: Keep model data
    doc: |
      Flag that determines whether model data used in solve2 is kept for
      subsequent steps.
    type: string?
    inputBinding:
      prefix: solve2.keepmodel=
      separate: False

  - id: solve2_reusemodel
    label: Reuse model list
    doc: |
      A list of model data columns that will be reused from an earlier
      step for solve2.
    type: string?
    inputBinding:
      prefix: solve2.reusemodel=
      separate: False

  - id: solve2_reusemodel
    label: Reuse model list
    doc: |
      A list of model data columns that will be reused from an earlier
      step for solve2.
    type: string?
    inputBinding:
      prefix: solve2.reusemodel=
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
  - id: output_h5parm1
    label: Solution table
    doc: |
      The filename of the output solution table of solve1. The value is taken from the input
      parameter "solve1_h5parm".
    type: File
    outputBinding:
      glob: $(inputs.solve1_h5parm)

  - id: output_h5parm2
    label: Solution table
    doc: |
      The filename of the output solution table of solve1. The value is taken from the input
      parameter "solve2_h5parm".
    type: File?
    outputBinding:
      glob: $(inputs.solve2_h5parm)

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor

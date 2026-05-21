{% set active_di_solves = di_solves | default(["full_jones"]) %}
{% set active_nr_di_solves = nr_di_solves | default(active_di_solves|length) %}
{% set active_has_slow_gains = has_slow_gains | default("slow_gains" in active_di_solves) %}
{% set active_is_full_jones = is_full_jones | default(active_di_solves == ["full_jones"]) %}
{% set active_needs_combine_fast_medium = needs_combine_fast_medium | default(active_nr_di_solves >= 2 and active_di_solves[0] == "fast_phase" and active_di_solves[1] == "medium_phase") %}
{% set active_needs_combine_slow = needs_combine_slow | default(active_nr_di_solves == 3 and active_di_solves[2] == "slow_gains") %}
cwlVersion: v1.2
class: Workflow
label: Rapthor DI calibration workflow
requirements:
  ScatterFeatureRequirement: {}
  StepInputExpressionRequirement: {}
  InlineJavascriptRequirement: {}

{% if max_cores is not none %}
hints:
  ResourceRequirement:
    coresMin: {{ max_cores }}
    coresMax: {{ max_cores }}
{% endif %}

inputs:
  - id: timechunk_filename
    type: Directory[]
  - id: data_colname
    type: string
  - id: starttime
    type: string[]
  - id: ntimes
    type: int[]
  - id: steps
    type: string
  - id: maxiter
    type: int
  - id: llssolver
    type: string
  - id: propagatesolutions
    type: boolean
  - id: solveralgorithm
    type: string
  - id: solverlbfgs_dof
    type: float
  - id: solverlbfgs_iter
    type: int
  - id: solverlbfgs_minibatches
    type: int
  - id: stepsize
    type: float
  - id: stepsigma
    type: float
  - id: tolerance
    type: float
  - id: uvlambdamin
    type: float
  - id: correctfreqsmearing
    type: boolean
  - id: correcttimesmearing
    type: boolean
  - id: max_threads
    type: int
  - id: max_normalization_delta
    type: float
  - id: scale_normalization_delta
    type: string
  - id: phase_center_ra
    type:
      - float
      - string
  - id: phase_center_dec
    type:
      - float
      - string
  - id: calibrator_names
    type: string[]
  - id: calibrator_fluxes
    type: float[]
  - id: final_h5parm
    type: string?
{% for solve in active_di_solves %}
{% set solve_index = loop.index %}
  - id: solve{{ solve_index }}_mode
    type: string
  - id: solve{{ solve_index }}_h5parm
    type: string[]
  - id: solve{{ solve_index }}_solint
    type: int[]
  - id: solve{{ solve_index }}_nchan
    type: int[]
  - id: solve{{ solve_index }}_collected_h5parm
    type: string
{% if solve in ["slow_gains", "full_jones"] %}
  - id: solve{{ solve_index }}_processed_h5parm
    type: string
{% endif %}
  - id: solve{{ solve_index }}_initialsolutions_h5parm
    type: File?
  - id: solve{{ solve_index }}_initialsolutions_soltab
    type: string?
  - id: solve{{ solve_index }}_llssolver
    type: string
  - id: solve{{ solve_index }}_maxiter
    type: int
  - id: solve{{ solve_index }}_propagatesolutions
    type: boolean
  - id: solve{{ solve_index }}_solveralgorithm
    type: string
  - id: solve{{ solve_index }}_solverlbfgs_dof
    type: float
  - id: solve{{ solve_index }}_solverlbfgs_iter
    type: int
  - id: solve{{ solve_index }}_solverlbfgs_minibatches
    type: int
  - id: solve{{ solve_index }}_stepsize
    type: float
  - id: solve{{ solve_index }}_stepsigma
    type: float
  - id: solve{{ solve_index }}_tolerance
    type: float
  - id: solve{{ solve_index }}_uvlambdamin
    type: float
{% if active_nr_di_solves > 1 and solve_index == 1 %}
  - id: solve1_keepmodel
    type: string
{% endif %}
{% if solve_index > 1 %}
  - id: solve{{ solve_index }}_reusemodel
    type: string
{% endif %}
{% endfor %}
{% if active_needs_combine_fast_medium %}
  - id: combined_fast_medium_h5parm
    type: string
  - id: solution_combine_mode
    type: string
{% endif %}
{% if active_needs_combine_slow %}
  - id: combined_slow_h5parm
    type: string
{% endif %}

outputs:
  - id: combined_solutions
    outputSource:
{% if active_is_full_jones %}
      - process_solve1_gains/outh5parm
{% elif active_needs_combine_slow %}
      - combine_slow/combinedh5parm
{% elif active_needs_combine_fast_medium %}
      - combine_fast_medium/combinedh5parm
{% elif active_has_slow_gains %}
      - process_solve1_gains/outh5parm
{% else %}
      - collect_solve1/outh5parm
{% endif %}
    type: File
{% for solve in active_di_solves %}
{% set solve_index = loop.index %}
  - id: solve{{ solve_index }}_solutions
    outputSource:
{% if solve in ["slow_gains", "full_jones"] %}
      - process_solve{{ solve_index }}_gains/outh5parm
{% else %}
      - collect_solve{{ solve_index }}/outh5parm
{% endif %}
    type: File
  - id: solve{{ solve_index }}_phase_plots
    outputSource:
      - plot_solve{{ solve_index }}_phase_solutions/plots
    type: File[]
{% if solve in ["slow_gains", "full_jones"] %}
  - id: solve{{ solve_index }}_amp_plots
    outputSource:
      - plot_solve{{ solve_index }}_amp_solutions/plots
    type: File[]
{% endif %}
{% endfor %}

steps:
  - id: solve_di
    run: {{ rapthor_pipeline_dir }}/steps/ddecal_solve.cwl
    in:
      - id: msin
        source: timechunk_filename
      - id: data_colname
        source: data_colname
      - id: starttime
        source: starttime
      - id: ntimes
        source: ntimes
      - id: steps
        source: steps
      - id: modeldatacolumn
        valueFrom: "[MODEL_DATA]"
      - id: numthreads
        source: max_threads
      - id: solve1_correctfreqsmearing
        source: correctfreqsmearing
      - id: solve1_correcttimesmearing
        source: correcttimesmearing
{% for solve in active_di_solves %}
{% set solve_index = loop.index %}
      - id: solve{{ solve_index }}_mode
        source: solve{{ solve_index }}_mode
      - id: solve{{ solve_index }}_h5parm
        source: solve{{ solve_index }}_h5parm
      - id: solve{{ solve_index }}_solint
        source: solve{{ solve_index }}_solint
      - id: solve{{ solve_index }}_nchan
        source: solve{{ solve_index }}_nchan
      - id: solve{{ solve_index }}_initialsolutions_h5parm
        source: solve{{ solve_index }}_initialsolutions_h5parm
      - id: solve{{ solve_index }}_initialsolutions_soltab
        source: solve{{ solve_index }}_initialsolutions_soltab
      - id: solve{{ solve_index }}_llssolver
        source: solve{{ solve_index }}_llssolver
      - id: solve{{ solve_index }}_maxiter
        source: solve{{ solve_index }}_maxiter
      - id: solve{{ solve_index }}_propagatesolutions
        source: solve{{ solve_index }}_propagatesolutions
      - id: solve{{ solve_index }}_solveralgorithm
        source: solve{{ solve_index }}_solveralgorithm
      - id: solve{{ solve_index }}_solverlbfgs_dof
        source: solve{{ solve_index }}_solverlbfgs_dof
      - id: solve{{ solve_index }}_solverlbfgs_iter
        source: solve{{ solve_index }}_solverlbfgs_iter
      - id: solve{{ solve_index }}_solverlbfgs_minibatches
        source: solve{{ solve_index }}_solverlbfgs_minibatches
      - id: solve{{ solve_index }}_stepsize
        source: solve{{ solve_index }}_stepsize
      - id: solve{{ solve_index }}_stepsigma
        source: solve{{ solve_index }}_stepsigma
      - id: solve{{ solve_index }}_tolerance
        source: solve{{ solve_index }}_tolerance
      - id: solve{{ solve_index }}_uvlambdamin
        source: solve{{ solve_index }}_uvlambdamin
{% if active_nr_di_solves > 1 and solve_index == 1 %}
      - id: solve1_keepmodel
        source: solve1_keepmodel
{% endif %}
{% if solve_index > 1 %}
      - id: solve{{ solve_index }}_reusemodel
        source: solve{{ solve_index }}_reusemodel
{% endif %}
{% endfor %}
    scatter: [msin, starttime, ntimes, solve1_h5parm, solve1_solint, solve1_nchan{% if active_nr_di_solves >= 2 %}, solve2_h5parm, solve2_solint, solve2_nchan{% endif %}{% if active_nr_di_solves >= 3 %}, solve3_h5parm, solve3_solint, solve3_nchan{% endif %}]
    scatterMethod: dotproduct
    out:
{% for solve in active_di_solves %}
      - id: output_h5parm{{ loop.index }}
{% endfor %}

{% for solve in active_di_solves %}
{% set solve_index = loop.index %}
  - id: collect_solve{{ solve_index }}
    run: {{ rapthor_pipeline_dir }}/steps/collect_h5parms.cwl
    in:
      - id: inh5parms
        source: solve_di/output_h5parm{{ solve_index }}
      - id: outputh5parm
        source: solve{{ solve_index }}_collected_h5parm
    out:
      - id: outh5parm

{% if solve in ["slow_gains", "full_jones"] %}
  - id: process_solve{{ solve_index }}_gains
    run: {{ rapthor_pipeline_dir }}/steps/process_gains.cwl
    in:
      - id: h5parm
        source: collect_solve{{ solve_index }}/outh5parm
      - id: flag
{% if solve == "slow_gains" %}
        valueFrom: "True"
      - id: smooth
        valueFrom: "True"
      - id: scale_station_delta
        source: scale_normalization_delta
      - id: phase_center_ra
        source: phase_center_ra
      - id: phase_center_dec
        source: phase_center_dec
{% else %}
        valueFrom: "False"
      - id: smooth
        valueFrom: "False"
      - id: scale_station_delta
        valueFrom: "False"
      - id: phase_center_ra
        valueFrom: "0.0"
      - id: phase_center_dec
        valueFrom: "0.0"
{% endif %}
      - id: max_station_delta
        source: max_normalization_delta
    out:
      - id: outh5parm
{% endif %}

  - id: plot_solve{{ solve_index }}_phase_solutions
    run: {{ rapthor_pipeline_dir }}/steps/plot_solutions.cwl
    in:
      - id: h5parm
{% if solve in ["slow_gains", "full_jones"] %}
        source: process_solve{{ solve_index }}_gains/outh5parm
{% else %}
        source: collect_solve{{ solve_index }}/outh5parm
{% endif %}
      - id: soltype
        valueFrom: "phase"
      - id: root
        valueFrom: "solve{{ solve_index }}_phase_"
    out:
      - id: plots
{% if solve in ["slow_gains", "full_jones"] %}

  - id: plot_solve{{ solve_index }}_amp_solutions
    run: {{ rapthor_pipeline_dir }}/steps/plot_solutions.cwl
    in:
      - id: h5parm
        source: process_solve{{ solve_index }}_gains/outh5parm
      - id: soltype
        valueFrom: "amplitude"
      - id: root
        valueFrom: "solve{{ solve_index }}_amplitude_"
    out:
      - id: plots
{% endif %}

{% endfor %}
{% if active_needs_combine_fast_medium %}
  - id: combine_fast_medium
    run: {{ rapthor_pipeline_dir }}/steps/combine_h5parms.cwl
    in:
      - id: inh5parm1
        source: collect_solve1/outh5parm
      - id: inh5parm2
        source: collect_solve2/outh5parm
      - id: outh5parm
        source: combined_fast_medium_h5parm
      - id: mode
{% if active_needs_combine_slow %}
        valueFrom: "p1p2_scalar"
{% else %}
        source: solution_combine_mode
{% endif %}
      - id: reweight
        valueFrom: "False"
      - id: calibrator_names
        source: calibrator_names
      - id: calibrator_fluxes
        source: calibrator_fluxes
    out:
      - id: combinedh5parm
{% endif %}
{% if active_needs_combine_slow %}

  - id: combine_slow
    run: {{ rapthor_pipeline_dir }}/steps/combine_h5parms.cwl
    in:
      - id: inh5parm1
        source: combine_fast_medium/combinedh5parm
      - id: inh5parm2
        source: process_solve3_gains/outh5parm
      - id: outh5parm
        source: combined_slow_h5parm
      - id: mode
        source: solution_combine_mode
      - id: reweight
        valueFrom: "False"
      - id: calibrator_names
        source: calibrator_names
      - id: calibrator_fluxes
        source: calibrator_fluxes
    out:
      - id: combinedh5parm
{% endif %}


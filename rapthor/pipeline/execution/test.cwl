cwlVersion: v1.2
class: CommandLineTool
baseCommand:
  - bash
  - script.sh

inputs:
- id: msin
  type: Directory
  doc: |
    Input Measurement Set
    In principle, Rapthor can handle multiple input Measurement Sets with data
    from different epochs containing the same frequency bands (e.g., from
    multiple nights of observations). This CWL step does _not_ support this.
# - id: global
#   type:
#     type: record
#     fields:
#       - name: input_ms
#         type: Directory
  # default: |
  #   $(
  #     {
  #       'unsupportedarray' : ['a', 'b'],
  #       'unsupportedobject': {'a': 1, 'b': 2}
  #     }
  #   )
#   type:
#     type: record
#     fields:
#       - name: input_skymodel
#         type: File?
#         doc: |
#           Full path to the input sky model file, with true-sky fluxes (required when not automatically downloading). If you also have a sky model with apparent flux densities, specify it with the apparent_skymodel option (note that the source names must be identical in both sky models)
#       - name: apparent_skymodel
#         type: File?
#       - name: download_initial_skymodel
#         type: boolean?
# #        default: true
#         doc: Automatically download a target skymodel. This will ignore input_skymodel when set.
#       - name: download_initial_skymodel_radius
#         type: float?
# #        default: 5.0
#         doc: Radius out to which a skymodel should be downloaded (default is 5 degrees).
#       - name: download_initial_skymodel_server
#         type: string?
# #        default: "TGSS"
#         doc: Service from which a skymodel should be downloaded (default is TGSS).
#       - name: unsupportedarray
#         type: string[]?
#       - name: unsupportedobject
#         type: Any?


# Regroup input skymodel as needed to meet target flux (default = True). If False, the existing
# patches are used for the calibration
# regroup_input_skymodel = True

# Processing strategy to use (default = selfcal):
# - selfcal: standard self calibration
# - image: (not yet supported) image using the input solutions (no calibration is done)
# - user-supplied file: full path to Python file defining custom strategy
# strategy = selfcal

# Fraction of data to process (default = 0.2 for self calibration and 1.0 for the final pass).
# If less than one, the input data are divided by time into chunks (of no less
# than slow_timestep_sec below) that sum to the requested fraction, spaced out
# evenly over the full time range. A final fraction can also be specified
# (default = selfcal_data_fraction) such that a final processing pass (i.e.,
# after selfcal finishes) is done with a different fraction
# selfcal_data_fraction = 0.2
# final_data_fraction = 1.0
- id: settings
  type:
    type: record
    fields:
      - name: global
        type: Any
      - name: calibration
        type: Any
      - name: imaging
        type: Any
      - name: cluster
        type: Any

# - id: global
#   type: Any
# - id: calibration
#   type: Any
# - id: imaging
#   type: Any
# - id: cluster
#   type: Any

outputs:
# - id: images
#   type: Directory[]
#   outputBinding:
#     glob: images
- id: logs
  type: Directory
  outputBinding:
    glob:
    - logs
    # - logs/*.log
    # - logs/*/*.log
- id: parset
  type: File
  outputBinding:
    loadContents: true
    glob: rapthor.parset
# - id: plots
#   type: Directory[]
#   outputBinding:
#     glob: plots
# - id: regions
#   type: Directory[]
#   outputBinding:
#     glob: regions
# - id: skymodels
#   type: Directory[]
#   outputBinding:
#     glob: skymodels
# - id: solutions
#   type: Directory[]
#   outputBinding:
#     glob: solutions

requirements:
- class: InlineJavascriptRequirement
  expressionLib:
    - { $include: utils.js}
- class: InitialWorkDirRequirement
  listing:
  - entryname: script.sh
    entry: |
        #!/bin/bash
        mkdir -p images/img_{1,2}
        touch images/img_{1,2}/img_{1,2}.fits
        mkdir -p logs/log_{1,2}
        touch logs/log_{1,2}/log_{1,2}.log
        touch logs/rapthor.log
  - entryname: rapthor.parset
    writable: True
    entry: |
        ${
            var settings = inputs.settings;
            settings.global.dir_working = runtime.outdir;
            settings.global.input_ms = inputs.msin.path;
            var result = "";
            ["global", "calibration", "imaging", "cluster"].forEach(element => {
                result += objectToParsetString(settings[element], element) + "\n\n"
            });
            return result;
        }

id: run_rapthor
label: Run Rapthor pipeline
class: CommandLineTool
cwlVersion: v1.2

doc: |
    This step does the following:
        * generates a parset file for Rapthor
        * generates a virtual environment where Rapthor will run;
        * runs the Rapthor pipeline.
    These actions must all be done in a single step, because the paths that
    are put in the parset file need to be the same as the paths used when
    Rapthor is actually being run.

# Input should be a JSON file with the correct pipeline settings. The contents of
# this file will be converted to a valid Rapthor parset file.
inputs:
- id: settings
  type: File
  doc: File containing the settings for the Rapthor pipeline in JSON format
- id: ms
  type: Directory
  doc: |
    Input Measurement Set
    In principle, Rapthor can handle multiple input Measurement Sets with data
    from different epochs containing the same frequency bands (e.g., from
    multiple nights of observations). This CWL step does _not_ support this.
- id: skymodel
  type: File?
  doc: Optional input sky model
- id: apparent_sky
  type: File?
  doc: Optional apparent sky model
- id: strategy
  type:
  - File?
  - string?
  doc: |
    Optional strategy; either a name (e.g., "selfcal"), or a path to a python
    strategy file (e.g., "/path/to/my_fancy_strategy.py")
- id: global
  type: Any
# Anything that the Rapthor pipeline will produce as outputs.
outputs:
- id: images
  type: Directory[]
  outputBinding:
    glob: images
- id: logs
  type: Directory[]
  outputBinding:
    glob: logs
- id: parset
  type: File
  outputBinding:
    glob: rapthor.parset
- id: plots
  type: Directory[]
  outputBinding:
    glob: plots
- id: regions
  type: Directory[]
  outputBinding:
    glob: regions
- id: skymodels
  type: Directory[]
  outputBinding:
    glob: skymodels
- id: solutions
  type: Directory[]
  outputBinding:
    glob: solutions

baseCommand:
  - bash
  - runner.sh

requirements:
- class: InlineJavascriptRequirement
  expressionLib:
    - { $include: utils.js}
- class: InitialWorkDirRequirement
  listing:
  - entryname: rapthor.parset
    entry: |
      #
      [global]
      dir_working = $(runtime.outdir)
      input_ms = $(inputs.ms.path)
      $(inputs.skymodel? "input_skymodel=": "") $(inputs.skymodel.path)
      #apparent_skymodel = /project/rapthor/Data/rapthor/HBA_short/apparent_sky.txt
      download_initial_skymodel = False
      regroup_input_skymodel = $(inputs.settings.config.global.regroup_input_skymodel)
      selfcal_data_fraction = 0.01
      strategy = $(inputs.strategy.path)
      $(inputs.strategy? "input_strategy=": "") $(
        typeof inputs.strategy === "string" ? inputs.strategy : inputs.strategy.path
        )

      ${
        function objectToParsetString(obj) {
          result = ""
          for(var item 0in obj) {
            var value = obj[item]
            value = typeof value === 'string' ? value : typeof value === 'number' ? String(value) : value.path 
            result += "\n" + item + "=" + value
          }
          return result
        }
        var somt = inputs.settings.global
        var parset = ""
        parset += "[global]\n" + objectToParsetString(inputs.global) 
        parset += "\ndir_working = " + runtime.outdir
        parset += "\ninput_ms =" + inputs.ms.path

        return parset
      }

  - entryname: json2ini.jq
    entry: |
      # Convert JSON to INI using `jq`
      # See: https://stackoverflow.com/a/50902853
      def kv: to_entries[] | "\(.key)=\(.value)";
      if type == "array" then .[] else . end
      | to_entries[]
      | "[\(.key)]", (.value|kv)

  - entryname: runner.sh
    entry: |
      #!/bin/bash
      set -ex

      # Compose a filter to update the relevant items in the input JSON file.
      # NOTE: the JavaScript expressions are *always* evaluated. Hence, the
      # extra check in the `then`-clause. Also note that `.global.strategy`
      # can either be a string or a File, hence the ternary expression.
      filter='.config
        | .global.dir_working |= "$(runtime.outdir)"
        | .global.input_ms |= "$(inputs.ms.path)"
        | .global.input_skymodel |=
            if $(inputs.skymodel != null)
            then "$(inputs.skymodel && inputs.skymodel.path)"
            else empty
            end
        | .global.apparent_skymodel |=
            if $(inputs.apparent_sky != null)
            then "$(inputs.apparent_sky && inputs.apparent_sky.path)"
            else empty
            end
        | .global.strategy |=
            if $(inputs.strategy != null)
            then "$(inputs.strategy && inputs.strategy.path
                    ? inputs.strategy.path
                    : inputs.strategy)"
            else empty
            end
      '

      # Read input JSON file, update paths, and write to INI (parset) file
      jq -c "\${filter}" "$(inputs.settings.path)" | \
        jq -rf json2ini.jq > rapthor.parset

      # Create virtual environment and activate it.
      python_version=`jq -r .virtualenv.python.version "$(inputs.settings.path)"`
      virtualenv --python="python\${python_version}" venv
      . venv/bin/activate

      # Install rapthor
      rapthor_version=`jq -r .virtualenv.rapthor.version "$(inputs.settings.path)"`
      pip install --no-cache-dir \
        git+https://git.astron.nl/RD/rapthor.git@\${rapthor_version}

      # Download and install casacore data files, and make them findable
      data_dir="\${VIRTUAL_ENV}/share/casacore/data"
      mkdir -p "\${data_dir}"
      wget -qO - ftp://ftp.astron.nl/outgoing/Measures/WSRT_Measures.ztar | \
        tar -C "\${data_dir}" -xzf -
      echo "measures.directory: \${data_dir}" > "\${HOME}/.casarc"

      # Run rapthor
      rapthor -v rapthor.parset

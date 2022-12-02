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

# An easy way to convert JSON to INI:
#     with open("rapthor_full.json", "r") as f:
#         settings = json.load(f)
#     for sect in settings.keys():
#         print(f"[{sect}]")
#         for key, val in settings[sect].items():
#             print(f"{key} = {val}")
#         print()

# Input should be a JSON file with the correct pipeline settings. The contents of
# this file will be converted to a valid Rapthor parset file.
inputs:
- id: json
  type: File
  doc: File containing the configuration for the Rapthor pipeline in JSON format
- id: ms
  type: Directory
  doc: Input Measurement Set
- id: skymodel
  type: File
  doc: Input Sky model

# Anything that the Rapthor pipeline will produce as outputs.
outputs:
- id: log
  type: File
  outputBinding:
    glob: log.txt
- id: parset
  type: File
  outputBinding:
    glob: rapthor.parset
# - id: strace
#   type: File
#   outputBinding:
#     glob: strace.log

baseCommand:
  - bash
  - runner.sh

requirements:
- class: InlineJavascriptRequirement
- class: InitialWorkDirRequirement
  listing:
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

      # Read input JSON file, update paths, and write to INI (parset) file
      filter='
        .config |
        .global.dir_working |= "$(runtime.outdir)" |
        .global.input_ms |= "$(inputs.ms.path)" |
        .global.input_skymodel |= "$(inputs.skymodel.path)"'
      jq -c "\${filter}" "$(inputs.json.path)" | \
        jq -rf json2ini.jq > rapthor.parset

      # Create virtual environment and activate it.
      python_version=`jq -r .virtualenv.python.version "$(inputs.json.path)"`
      virtualenv --python="python\${python_version}" venv
      . venv/bin/activate

      echo "PWD: \${PWD}" >> log.txt
      echo "python: `which pyhton`" >> log.txt
      echo "  version: `python --version`" >> log.txt

      # Install rapthor
      rapthor_version=`jq -r .virtualenv.rapthor.version "$(inputs.json.path)"`
      echo "pip: `which pip`" >> log.txt
      pip install git+https://git.astron.nl/RD/rapthor.git@\${rapthor_version}
      echo "pip list:" >> log.txt
      pip list >> log.txt

      # Download and install casacore data files, and make them findable
      data_dir="\${VIRTUAL_ENV}/share/casacore/data"
      mkdir -p "\${data_dir}"
      wget -qO - ftp://ftp.astron.nl/outgoing/Measures/WSRT_Measures.ztar | \
        tar -C "\${data_dir}" -xzf -
      echo "measures.directory: \${data_dir}" > "\${HOME}/.casarc"

      # Run rapthor
      rapthor -v rapthor.parset

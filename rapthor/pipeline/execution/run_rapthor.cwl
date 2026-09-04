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

inputs:
- id: msin
  type: Directory
  doc: |
    Input Measurement Set
    In principle, Rapthor can handle multiple input Measurement Sets with data
    from different epochs containing the same frequency bands (e.g., from
    multiple nights of observations). This CWL step does _not_ support this.
- id: astrometry_check_model
  type: File
  doc: |
    Sky model used for astrometry checks.
- id: photometry_check_model
  type: File
  doc: |
    Sky model used for photometry checks.
- id: photometry_normalization_model_1
  type: File
  doc: |
    Sky model 1 used for photometry normalization/correction.
- id: photometry_normalization_frequency_1
  type: string
  doc: |
    Reference frequency in Hz for photometry normalization sky model 1.
- id: photometry_normalization_model_2
  type: File
  doc: |
    Sky model 2 used for photometry normalization/correction.
- id: photometry_normalization_frequency_2
  type: string
  doc: |
    Reference frequency in Hz for photometry normalization sky model 2.
- id: settings
  doc: Pipeline settings, used to generate a parset file
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
- id: virtualenv
  doc: Description of the virtual environment used to run Rapthor.
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
    writable: True
    entry: |
        ${
            var settings = inputs.settings;
            settings.global.dir_working = runtime.outdir;
            settings.global.input_ms = inputs.msin.path;
            settings.imaging.photometry_skymodel = inputs.photometry_check_model.path;
            settings.imaging.astrometry_skymodel = inputs.astrometry_check_model.path;
            settings.imaging.normalization_skymodels = [inputs.photometry_normalization_model_1.path, inputs.photometry_normalization_model_2.path];
            settings.imaging.normalization_reference_frequencies = [inputs.photometry_normalization_frequency_1, inputs.photometry_normalization_frequency_2];
            var result = "";
            ["global", "calibration", "imaging", "cluster"].forEach(element => {
                result += objectToParsetString(settings[element], element) + "\n\n"
            });
            return result;
        }

  - entryname: runner.sh
    entry: |
      #!/bin/bash
      set -ex

      # Create virtual environment and activate it.
      python_version="$(inputs.virtualenv.python.version)"
      virtualenv --python="python\${python_version}" venv
      . venv/bin/activate

      # Install rapthor
      rapthor_version="$(inputs.virtualenv.rapthor.version)"
      pip install --no-cache-dir \
        git+https://git.astron.nl/RD/rapthor.git@\${rapthor_version}

      # Download and install casacore data files, and make them findable
      data_dir="\${VIRTUAL_ENV}/share/casacore/data"
      mkdir -p "\${data_dir}"
      wget -qO - https://iers.astron.nl/WSRT_Measures.ztar | \
        tar -C "\${data_dir}" -xzf -
      echo "measures.directory: \${data_dir}" > "\${HOME}/.casarc"

      # Run rapthor
      rapthor -v rapthor.parset

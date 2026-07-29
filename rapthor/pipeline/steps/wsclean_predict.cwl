cwlVersion: v1.2
class: CommandLineTool
baseCommand: [wsclean_predict.py]
label: Predict using WSClean
doc: |
  This tool uses WSClean to predict model data into separate columns.

requirements:
  InlineJavascriptRequirement: {}

  ResourceRequirement:
    coresMin: $(inputs.numthreads)
    coresMax: $(inputs.numthreads)

inputs:
  - id: region_file
    label: DS9 region file
    doc: |
      The filename of the region file. 
    type:
      - File
    inputBinding:
      prefix: --region

  - id: msin
    label: Input MS directory name
    doc: |
      The name of the input MS directory.
    type: Directory
    inputBinding:
      prefix: --msin

  - id: skymodel
    label: Filename of input sky model
    doc: |
      The filename of the input sky model in makesourcedb format used to
      generate the output model images.
    type: File
    inputBinding:
      prefix: --skymodel

  - id: ra_dec
    label: RA and Dec of center
    doc: |
      The RA and Dec in hmsdms of the center of the output images.
    type: string[]
    inputBinding:
      prefix: --ra_dec

  - id: frequency_bandwidth
    label: Frequency and full bandwidth of data
    doc: |
      The central frequency and bandwidth in Hz of the data.
    type: float[]
    inputBinding:
      prefix: --frequency_bandwidth

  - id: predict_bandwidth
    label: Bandwidth of model images
    doc: |
      The bandwidth in Hz to create separate model images.
    type: float
    inputBinding:
      prefix: --predict_bandwidth

  - id: cellsize_deg
    label: Pixel size
    doc: |
      The size of one pixel of the image in deg.
    type: float
    inputBinding:
      prefix: --cellsize

  - id: imsize
    label: Image size
    doc: |
      The size of the image in pixels.
    type: int[]
    inputBinding:
      prefix: --imsize

  - id: numthreads
    label: Number of threads
    doc: |
      The number of threads to use (will ask this value, but use what is allocated).
    type: int
    inputBinding:
      prefix: --threads
      valueFrom: $(runtime.cores)

  - id: time_freq_smearing
    label: Enable time frequency smearing
    doc: |
      If true, enable time frequency smearing in prediction.
    type: boolean
    inputBinding:
      prefix: --time_freq_smearing

outputs:
  - id: msout
    label: Output MS
    doc: |
      The directory list of the output MS. The input msin list is returned if it
      is writable, otherwise a copy with temp name is made.
    type: Directory
    outputBinding:
      loadContents: true
      glob: "msout_names.json"
      outputEval: |
        ${ 
           var msname=JSON.parse(self[0].contents).msout;

           return {
               "class": "Directory",
               "location" : msname,
           };
        }

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor

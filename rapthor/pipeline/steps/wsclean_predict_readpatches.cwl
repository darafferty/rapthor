cwlVersion: v1.2
class: CommandLineTool
baseCommand: [wsclean_predict.py]
label: Provide patch names when predicting using WSClean
doc: |
  This tool reads a region file and provides a string with patch names matching the column names being created while predicting.

requirements:
  InlineJavascriptRequirement: {}

inputs:
  - id: region_file
    label: DS9 region file
    doc: |
      The filename of the region file. 
    type:
      - File
    inputBinding:
      prefix: --region

outputs:
  - id: patches
    label: Model data patch names
    doc: |
      The list of patch names for a model data column is created.
    type: string
    outputBinding:
      loadContents: true
      glob: "msout_names.json"
      outputEval: |
        ${ 
           var patch_list=JSON.parse(self[0].contents).patches;

           return patch_list;
        }

hints:
  - class: DockerRequirement
    dockerPull: astronrd/rapthor

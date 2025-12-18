cwlVersion: v1.2
class: ExpressionTool
label: Pick a file from a list
doc: |
  This tool returns the file from the input list that matches the input filename.

requirements:
    - class: InlineJavascriptRequirement
    - class: MultipleInputFeatureRequirement
    - class: InitialWorkDirRequirement
      listing:
        - $(inputs.input_file_list)

inputs:
  - id: input_file_list
    label: Filenames of input files
    doc: |
      The filenames of the input files.
    type: File[]
  - id: filename_to_match
    label: Filename of picked file
    doc: |
      The filename of the file to match.
    type: string
  - id: suffix
    label: Suffix to add to filename of picked file
    doc: |
      The suffix to add to the filename of the file to match.
    type: string

outputs:
  - id: picked_file
    label: Output image
    doc: |
      The file that matches the input filename. If no file was matched, a null value is
      output.
    type: File?

expression: |
  ${
    var picked_file
    var filename_to_match = inputs.filename_to_match + inputs.suffix
    for(var i=0; i<inputs.input_file_list.length; i++){
        var item = inputs.input_file_list[i]
        if(item.basename == filename_to_match){
            picked_file = item
        }
    }
    return {'picked_file': picked_file}
  }

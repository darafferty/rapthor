id: merge_array_directories
label: merge_array_directories
class: ExpressionTool

cwlVersion: v1.2
inputs:
    - id: input
      type:
        - type: array
          items:
            - type: array
              items: Directory
outputs:
    - id: output
      type: Directory[]

expression: |
  ${
    var out_directory = []
    for(var i=0; i<inputs.input.length; i++){
        var item = inputs.input[i]
        if(item != null){
            out_directory = out_directory.concat(item)
        }
    }
    return {'output': out_directory}
  }


requirements:
  - class: InlineJavascriptRequirement

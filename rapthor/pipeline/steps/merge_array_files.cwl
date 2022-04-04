id: merge_array_files
label: merge_array_files
class: ExpressionTool

cwlVersion: v1.2
inputs: 
    - id: input
      type:
        - type: array
          items:
            - type: array
              items: File
outputs: 
    - id: output
      type: File[]

expression: |
  ${
    var out_file = []
    for(var i=0; i<inputs.input.length; i++){
        var item = inputs.input[i]
        if(item != null){
            out_file = out_file.concat(item)
        }
    }
    return {'output': out_file}
  }


requirements:
  - class: InlineJavascriptRequirement

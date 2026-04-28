class: ExpressionTool
cwlVersion: v1.2
requirements:
  InlineJavascriptRequirement: {}

inputs:
  use_option_a: boolean
  option_a:
    type: File?
  option_b:
    type: File?

outputs:
  selected:
    type: File

expression: |
  ${
    return {
      selected: inputs.use_option_a ? inputs.option_a : inputs.option_b
    };
  }
id: concat_ms
label: Concatenate MS files
class: CommandLineTool
cwlVersion: v1.2
inputs:
- id: msin_dirs
  type: Directory[]
# Directories containing input MS files
outputs:
- id: msout
  type: Directory
# Name of output MS
doc: Concatenate all MS files in given directory, and save as one output MS
baseCommand:
  - concat_linc_files

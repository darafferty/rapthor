#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: CommandLineTool
baseCommand: [bash, -c]

inputs:
  message:
    type: string
    default: "Hello CWL"

arguments:
  - |
    # Create a single file
    echo "$(inputs.message)" > output.txt
    
    # Create a directory
    mkdir -p output_dir
    echo "Directory content" > output_dir/content.txt
    
    # Create multiple files (fixed number for simplicity)
    echo "File 0" > output_0.txt
    echo "File 1" > output_1.txt
    echo "File 2" > output_2.txt
    
    # Create multiple directories
    mkdir -p dir_0
    echo "Dir 0 content" > dir_0/content.txt
    mkdir -p dir_1
    echo "Dir 1 content" > dir_1/content.txt
    mkdir -p dir_2
    echo "Dir 2 content" > dir_2/content.txt

outputs:
  single_file:
    type: File
    outputBinding:
      glob: output.txt
  
  single_dir:
    type: Directory
    outputBinding:
      glob: output_dir
  
  file_array:
    type: File[]
    outputBinding:
      glob: "output_*.txt"
  
  dir_array:
    type: Directory[]
    outputBinding:
      glob: "dir_*"

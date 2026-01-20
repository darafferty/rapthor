#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: Workflow

requirements:
  ScatterFeatureRequirement: {}

inputs:
  messages:
    type: string[]
    default: ["First", "Second", "Third"]

steps:
  process_messages:
    run: simple_tool.cwl
    scatter: [message]
    scatterMethod: dotproduct
    in:
      message: messages
    out: [single_file, single_dir, file_array, dir_array]

outputs:
  all_files:
    type: File[]
    outputSource: process_messages/single_file
  
  all_dirs:
    type: Directory[]
    outputSource: process_messages/single_dir
  
  nested_files:
    type:
      type: array
      items:
        type: array
        items: File
    outputSource: process_messages/file_array
  
  nested_dirs:
    type:
      type: array
      items:
        type: array
        items: Directory
    outputSource: process_messages/dir_array

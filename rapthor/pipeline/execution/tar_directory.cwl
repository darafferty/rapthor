id: tar_directory
label: Tar directory contents
class: CommandLineTool
cwlVersion: v1.2

doc: |
    Create a tar-ball of the contents of the input directory.

inputs:
- id: directory
  type: Directory

outputs:
- id: tarball
  type: File
  outputBinding:
    glob: $(inputs.directory.basename).tar.gz

baseCommand:
  - tar

arguments:
  - --create
  - --dereference
  - --gzip
  - prefix: --directory
    valueFrom: $(inputs.directory.dirname)
  - prefix: --file
    valueFrom: $(inputs.directory.basename).tar.gz
  - valueFrom: $(inputs.directory.basename)

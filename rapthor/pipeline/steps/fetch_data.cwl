id: fetchdata
label: fetch_data
class: CommandLineTool
cwlVersion: v1.2
inputs: 
  - id: surl_link
    type: string
    inputBinding:
      position: 0
    
outputs: 
  - id: uncompressed
    type: Directory
    outputBinding:
      glob: 'out/*'
baseCommand: 
  - 'bash'
  - 'fetch.sh'
doc: 'Fetch a file from surl and uncompresses it'
requirements:
  InitialWorkDirRequirement:
    listing:
      - entryname: 'fetch.sh' 
        entry: |
          #!/bin/bash
          mkdir out
          cd out
          turl=`echo $1 | awk '{gsub("srm://srm.grid.sara.nl[:0-9]*","gsiftp://gridftp.grid.sara.nl"); print}'`
          echo "Downloading $turl"
          globus-url-copy $turl - | tar -xvf -

    

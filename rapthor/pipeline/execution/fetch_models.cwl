id: fetchmodel
label: fetch_model
class: CommandLineTool
cwlVersion: v1.2
inputs:
  - id: msin
    type: Directory
  - id: survey
    doc: Survey to query
    type: string
  - id: radius
    doc: Query radius in deg
    type: float
  - id: skymodel
    doc: Output sky model filename
    type: string
outputs:
  - id: skymodel
    type: File
    outputBinding:
      glob: $(inputs.skymodel)
baseCommand:
  - python3
  - fetch_model.py
doc: Download sky model needed for Rapthor.
requirements:
  - class: InlineJavascriptRequirement
  - class: InitialWorkDirRequirement
    listing:
     - entry: $(inputs.msin)
       writable: false
     - entryname: fetch_model.py
       entry: |
         import json
         import sys

         import casacore.tables as ct
         from lsmtool.download_skymodel import download_skymodel
         from lsmtool.operations_lib import normalize_ra_dec
         import numpy as np

         msin = sys.argv[1]
         inputs = json.loads(r'''$(inputs)''')
         survy = inputs['survey']
         radius = float(inputs['radius'])
         skymodel = inputs['skymodel']

         with ct.table(msin + "::FIELD", ack=False) as obs:
             ra, dec = normalize_ra_dec(
                 np.degrees(float(obs.col("REFERENCE_DIR")[0][0][0])),
                 np.degrees(float(obs.col("REFERENCE_DIR")[0][0][1])),
             )

         download_skymodel({"ra": ra, "dec": dec, "radius": radius},
                           survey=survey, skymodel_path=skymodel, overwrite=True)

id: run_rapthor
label: Run Rapthor pipeline
class: CommandLineTool
cwlVersion: v1.2

doc: |
    This step does the following:
        * generates a parset file for Rapthor
        * generates a virtual environment where Rapthor will run;
        * runs the Rapthor pipeline.
    These actions must all be done in a single step, because the paths that
    are put in the parset file need to be the same as the paths used when
    Rapthor is actually being run.

# An easy way to convert JSON to INI:
#     with open("rapthor_full.json", "r") as f: 
#         settings = json.load(f)
#     for sect in settings.keys(): 
#         print(f"[{sect}]") 
#         for key, val in settings[sect].items(): 
#             print(f"{key} = {val}") 
#         print() 

# Input should be a JSON file with the correct pipeline settings. The contents of
# this file will be converted to a valid Rapthor parset file.
inputs: []

# Anything that the Rapthor pipeline will produce as outputs. These artefacts will
# all 
outputs: []


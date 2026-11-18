"""Command-line entry point for the LDV executer service."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

from rapthor.cli import _run_pipeline

ALLOWED_RAPTHOR_SECTIONS = ["global", "calibration", "imaging", "cluster"]


def parse_inputs(input_file: str, error_file: str):
    """Parse input JSON file and return working directory and inputs as a dict"""
    shared_dir = Path(input_file).parent
    try:
        # Read input file as json
        with open(input_file, "r") as file:
            inputs = json.load(file)

        return shared_dir, inputs

    except Exception as e:
        # Log error to the error file
        with open(error_file, "w") as f:
            f.write(f"Error: {str(e)}")
        print(f"Error occurred: {str(e)}", file=sys.stderr)


def untar_input_dataproducts(input_data: list, tmp_dir: Path) -> None:
    """Untar input data products"""
    os.makedirs(tmp_dir, exist_ok=True)

    # Iterate over each item in input_data
    for item in input_data:
        # Check if the item is a tar file
        if item.endswith(".tar"):
            # Open the tar file and extract its contents to 'out'
            with tarfile.open(item, "r") as tar:
                tar.extractall(path=tmp_dir)
        else:
            # Copy the file to the 'out' directory
            shutil.copy2(item, tmp_dir)


def make_rapthor_parset(inputs: dict, tmp_dir: Path, error_file: Path) -> Path:
    """Make a Rapthor-compatible JSON parset file from the inputs"""
    rapthor_inputs = inputs.copy()

    for k, v in inputs.items():
        if k not in ALLOWED_RAPTHOR_SECTIONS:
            rapthor_inputs.pop(k)
        elif not isinstance(v, dict):
            # Log error to the error file
            error_msg = "Rapthor entries must be dictionaries, one per parset section."
            with open(error_file, "w") as f:
                f.write(f"Error: {error_msg}")
            print(f"Error occurred: {error_msg}", file=sys.stderr)

    # Add any missing sections
    for section in ALLOWED_RAPTHOR_SECTIONS:
        if section not in rapthor_inputs:
            rapthor_inputs[section] = {}

    # Add path to untarred input data
    rapthor_inputs["global"]["input_ms"] = str(tmp_dir / "*.ms")

    # Add path to working directory
    rapthor_inputs["global"]["dir_working"] = str(tmp_dir)

    # allow_internet_access should always be false
    allow_internet_access = rapthor_inputs["cluster"].get("allow_internet_access", False)
    if not allow_internet_access:
        # Set the various parameters for input sky models needed when run without internet
        # access (see rapthor/execution/image/catalog_preparation.py)
        rapthor_inputs["imaging"]["photometry_skymodel"] = inputs.get("photometry_skymodel", None)
        rapthor_inputs["imaging"]["astrometry_skymodel"] = inputs.get("astrometry_skymodel", None)
        rapthor_inputs["imaging"]["normalization_skymodels"] = [
            ", ".join(inputs.get("normalization_skymodels", []))
        ]
        rapthor_inputs["imaging"]["normalization_reference_frequencies"] = [
            ", ".join(inputs.get("normalization_reference_frequencies", []))
        ]

    rapthor_parset = tmp_dir / "inputs.json"
    with open(rapthor_parset, "w") as handle:
        json.dump(rapthor_inputs, handle)

    return rapthor_parset


def get_final_cycle_number(working_dir: Path):
    """Determines the final cycle number from the pipeline outputs"""
    image_dir = working_dir / "images"
    images = image_dir.glob("image_*")
    if images:
        final_cycle_number = max([int(im.name.split("_")[1]) for im in images])
        return final_cycle_number
    else:
        return None


def run_pipeline(
    parset_file: Path, shared_dir: Path, tmp_dir: Path, output_sas_id: str
) -> (list, list):
    """Run the Rapthor pipeline, tar the outputs, and return output filenames"""
    _run_pipeline(parset_file, logging_level="debug")

    # Collect outputs
    final_cycle = get_final_cycle_number(tmp_dir)
    if final_cycle is None:
        return [], []

    primary_output_data = []
    primary_outputs = {
        "images": tmp_dir / "images" / f"image_{final_cycle}",
        "solutions": tmp_dir / "solutions" / f"calibrate_{final_cycle}",
        "skymodels": tmp_dir / "skymodels" / f"calibrate_{final_cycle}",
    }
    ancillary_output_data = []
    ancillary_outputs = {
        "calibration_plots": tmp_dir / "plots" / f"calibrate_{final_cycle}",
        "imaging_plots": tmp_dir / "plots" / f"image_{final_cycle}",
    }

    for output_type, output_path in primary_outputs.items():
        # Tar the directory
        tar_path = output_path.parent.with_suffix(".tar")
        with tarfile.open(tar_path, "w") as tar:
            tar.add(output_path, arcname=output_type)

        # Copy the new tar file to shared_dir
        shutil.copy2(tar_path, shared_dir)
        output_tar_path = shared_dir / tar_path.name
        primary_output_data.append(str(output_tar_path))

    for output_type, output_path in ancillary_outputs.items():
        # Tar the directory
        tar_path = output_path.with_suffix(".tar")
        with tarfile.open(tar_path, "w") as tar:
            tar.add(output_path, arcname=output_type)

        # Copy the new tar file to shared_dir
        shutil.copy2(tar_path, shared_dir)
        output_tar_path = shared_dir / tar_path.name
        ancillary_output_data.append(output_tar_path)

    return primary_output_data, ancillary_output_data


def create_dataproduct_structure(key, location):
    """Create the required json output structure for a single dataproduct (filename)"""

    filename = os.path.basename(location)
    structure = {
        "path": key,
        "metadata": {
            "project": {"projectCode": "DDT2_001"},
            "dataProduct": {
                "size": os.path.getsize(location),
                "_type": "CorrelatedDataProduct",
                "subband": "SB-115",
                "duration": "PT9M59.832658S",
                "fileName": filename,
                "startTime": "2018-05-24T18:43:28.000",
                "fileFormat": "AIPS++/CASA",
                "md5checksum": "0",
                "channelWidth": {"_attrs": {"units": "kHz"}, "_value": 97.65625},
                "storageWriter": "DyscoStorageManager",
                "stationSubband": "SB-115",
                "dataProductType": "Correlator data",
                "centralFrequency": {"_attrs": {"units": "MHz"}, "_value": 146.18988037109375},
                "channelsPerSubband": 40,
                "integrationInterval": {"_attrs": {"units": "s"}, "_value": 8.01112064},
                "storageWriterVersion": "None",
            },
            "pipelineRun": {
                "_type": "CalibrationPipeline",
                "demixing": "false",
                "pipelineName": "calibration_pipeline",
                "strategyName": "POC-5 python pipeline",
                "pipelineVersion": "v1.0",
                "skyModelDatabase": "",
                "strategyDescription": "POC-5 python",
                "timeIntegrationStep": "8",
                "flagAutoCorrelations": "true",
                "frequencyIntegrationStep": "8",
                "numberOfInstrumentModels": "1",
                "numberOfCorrelatedDataProducts": "1",
            },
        },
        "file_name": filename,
    }
    return structure


def create_location_structure(location, i):
    """Create the required json output structure for a single dataproduct (location)"""
    filename = os.path.basename(location)

    structure = {
        "size": os.path.getsize(location),
        "class": "File",
        "nameext": ".tar",
        "basename": filename,
        "checksum": str(i),
        "location": location,
        "nameroot": filename,
    }
    return structure


def create_ancillary_structure(location, i):
    """Encapsulate the ancillary file in its required json structure"""
    filename = os.path.basename(location)

    structure = {
        "location": location,
        "path": location,
        "size": os.path.getsize(location),
        "class": "File",
        "basename": filename,
        "checksum": str(i),
    }
    return structure


def create_outputs_metadata(primary_output_data: list, ancillary_output_data: list) -> dict:
    """Creates the required outputs json for Poppy, based on the tarred files."""
    outputs = {}

    key = "compressed"

    # Construct the 'ingest' section (primary dataproducts to be ingested)
    ingest = []
    for location in primary_output_data:
        primary_dataproduct = create_dataproduct_structure(key, location)
        ingest.append(primary_dataproduct)
    outputs["ingest"] = ingest

    # Construct the quality section
    # Construct the quality.plots section (ancillary dataproducts to be ingested)
    plots = []
    i = 0
    for ancillary in ancillary_output_data:
        if ancillary:
            ancillary_dataproduct = create_ancillary_structure(ancillary, i)
            plots.append(ancillary_dataproduct)
            i += 1

    quality = {}
    quality["plots"] = plots
    outputs["quality"] = quality

    # construct the files section (identified by the key)
    output_files = []
    i = 0
    for location in primary_output_data:
        output_file = create_location_structure(location, i)
        output_files.append(output_file)
        i += 1
    outputs[key] = output_files

    return outputs


def write_outputs(output_file: str, outputs: str) -> None:
    """Write the outputs to the output JSON file"""
    with open(output_file, "w") as file:
        file.write(json.dumps(outputs, indent=4))


def main(input_file: str, output_file: str, error_file: str) -> None:
    """Set up the pipeline inputs, run the pipeline, and handle the outputs"""
    # Read the input parameters
    shared_dir, inputs = parse_inputs(input_file, error_file)

    # The tmp_dir simulates a location on the processing node. So make it the current local directory + 'out'
    tmp_dir = Path.cwd() / "out"

    # Copy the input dataproducts from a shared location and update inputs
    # so that Rapthor can find the untarred files
    untar_input_dataproducts(inputs["data"], tmp_dir)

    # Make the Rapthor parset
    rapthor_parset = make_rapthor_parset(inputs, tmp_dir, error_file)

    # Run the pipeline
    output_sas_id = inputs.get("output_sas_id")
    primary_output_data, ancillary_output_data = run_pipeline(
        rapthor_parset, shared_dir, tmp_dir, output_sas_id
    )

    # Cleanup - remove the tmp_dir
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    # Create outputs metadata
    outputs = create_outputs_metadata(primary_output_data, ancillary_output_data)

    # Write the output json to the output_file
    write_outputs(output_file, outputs)


if __name__ == "__main__":
    """
    This (python) pipeline can be run with bash on a headnode, or as slurm job on processing nodes.
    """
    # Parse the input arguments
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    error_file = sys.argv[3]

    main(input_file, output_file, error_file)

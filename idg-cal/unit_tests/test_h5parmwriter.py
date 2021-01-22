import sys

sys.path.append("..")
from h5parmwriter import H5ParmWriter
import pytest
import os
import h5py as h5
from datetime import datetime
import numpy as np


def test_h5parmwriter():
    """
    Test H5ParmWriter, simultaneously serving demonstrating its usage
    """
    h5file_name = "test.h5"

    solution_set_name = "test000"

    # Antenna/station names
    antenna_names = ["ant1", "ant2"]
    antenna_positions = np.array([[0, 1, 2], [4, 5, 6]])

    # Source names/directions
    source_names = ["POINTING"]
    source_dirs = np.array([[2, 3]])

    coeffs = np.ones((2, 3, 1, 2, 4))

    # Axis info
    axes_labels = ["ant", "dir", "freq", "pol", "time"]
    axes_data = dict(zip(axes_labels, coeffs.shape))
    h5writer = H5ParmWriter(h5file_name, solution_set_name=solution_set_name)

    # Add antenna info
    h5writer.add_antennas(antenna_names, antenna_positions)

    # Add sources info
    h5writer.add_sources(source_names, source_dirs)

    # Add phase solution table (withou data)
    h5writer.create_solution_table(
        "phase000",
        "phase",
        axes_data,
        values=coeffs,
        history=f'CREATED at {datetime.today().strftime("%Y/%m/%d")}',
    )

    # Add amplitude solution table (without data)
    h5writer.create_solution_table(
        "amplitude000",
        "amplitude",
        axes_data,
        dtype=coeffs.dtype,
        history=f'CREATED at {datetime.today().strftime("%Y/%m/%d")}',
    )

    # Fill solution table with a slice of data
    amplitude_offset = (1, 1, 0, 1, 1)
    amplitude_coeffs = np.ones((1, 2, 1, 1, 3)) * 500
    h5writer.fill_solution_table("amplitude000", amplitude_coeffs, amplitude_offset)

    # Add meta data to ant axis in amplitude soltab
    h5writer.create_axis_meta_data(
        "amplitude000", "ant", meta_data=np.array(antenna_names)
    )

    # Add meta data to dir axis in phase soltab
    h5writer.create_axis_meta_data(
        "phase000",
        "dir",
        attributes={
            "basisfunction_type": "lagrange",
            "image_size": 1024,
            "subgrid_size": 32,
        },
    )
    h5writer.close_file()

    h5reader = h5.File(h5file_name, "r+")

    # Check top level group name
    assert list(h5reader["/"].keys())[0] == solution_set_name

    # Check (compound) antenna info
    assert "antenna" in h5reader[solution_set_name]
    antenna = h5reader[solution_set_name + "/antenna"]
    np.testing.assert_equal(
        antenna[:]["name"], np.array(antenna_names).astype(antenna[:]["name"].dtype)
    )
    np.testing.assert_equal(antenna[:]["position"], np.array(antenna_positions))

    # Check (compound) source info
    assert "source" in h5reader[solution_set_name]
    sources = h5reader[solution_set_name + "/source"]
    np.testing.assert_equal(
        sources[:]["name"], np.array(source_names).astype(sources[:]["name"].dtype)
    )
    np.testing.assert_equal(sources[:]["dir"], np.array(source_dirs))

    # Check (phase) solution table
    assert "phase000" in h5reader[solution_set_name]
    assert (
        h5reader[solution_set_name + "/phase000"].attrs["TITLE"] == "phase"
        or h5reader[solution_set_name + "/phase000"].attrs["TITLE"] == "phase".encode()
    )
    values = h5reader[solution_set_name + "/phase000/val"]
    np.testing.assert_equal(coeffs, values)

    # Check (amplitude) solution table
    assert "amplitude000" in h5reader[solution_set_name]
    assert (
        h5reader[solution_set_name + "/amplitude000"].attrs["TITLE"] == "amplitude"
        or h5reader[solution_set_name + "/amplitude000"].attrs["TITLE"]
        == "amplitude".encode()
    )
    slicer = tuple(
        slice(off, off + shape)
        for (off, shape) in zip(amplitude_offset, amplitude_coeffs.shape)
    )
    np.testing.assert_equal(
        h5reader[solution_set_name + "/amplitude000/val"][slicer], amplitude_coeffs
    )

    np.testing.assert_equal(
        np.char.decode(h5reader[solution_set_name + "/amplitude000/ant"]),
        np.array(antenna_names),
    )

    assert (
        values.attrs["AXES"] == ",".join(axes_labels)
        or values.attrs["AXES"] == ",".join(axes_labels).encode()
    )
    assert (
        values.attrs["HISTORY000"]
        == f'CREATED at {datetime.today().strftime("%Y/%m/%d")}'
        or values.attrs["HISTORY000"]
        == f'CREATED at {datetime.today().strftime("%Y/%m/%d")}'.encode()
    )

    # Check meta data of dir axis
    assert "dir" in h5reader[solution_set_name + "/phase000"]
    assert (
        h5reader[solution_set_name + "/phase000/dir"].attrs["basisfunction_type"]
        == "lagrange"
        or h5reader[solution_set_name + "/phase000/dir"].attrs["basisfunction_type"]
        == "lagrange".encode()
    )
    assert h5reader[solution_set_name + "/phase000/dir"].attrs["image_size"] == 1024
    assert h5reader[solution_set_name + "/phase000/dir"].attrs["subgrid_size"] == 32

    # Close and remove
    h5reader.close()
    os.remove(h5file_name)

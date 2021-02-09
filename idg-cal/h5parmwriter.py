# Copyright (C) 2020 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later

import h5py
import numpy as np
import re
import os, logging, sys
from packaging import version

H5PARM_VERSION = "1.0"

if version.parse(
    str(sys.version_info.major) + "." + str(sys.version_info.minor)
) < version.parse("3.6"):
    raise Exception("Must be using Python >= 3.6")


class H5ParmWriter:
    """
    A thin writer wrapper around H5Parm, which itself is a tailored HDF5 for storing
    radio astronomical data.
    """

    def __init__(
        self, file_name, solution_set_name="sol000", attributes=None, overwrite=True
    ):
        """
        Initialize H5Parmwriter. The file, specified by file_name, should not yet
        exist. If it does, an error is thrown.

        Parameters
        ----------
        file_name : str
            Full path to the h5parm file.
        solution_set_name : str, optional
            Name of the top level solution set, by default "sol000"
        attributes: dict
            Dictionary containing attributes for the sol set
        """

        if os.path.isfile(file_name) and not overwrite:
            raise ValueError(
                f"File {file_name} already exists, choose a different name!"
            )

        # Create file for writing
        self.h5file = h5py.File(file_name, "w")
        self.solution_set = solution_set_name
        self.h5file.create_group(self.solution_set)

        # Write a version attribute
        self.__write_version_stamp(self.solution_set)

        # Add other attributes when specified
        if attributes is not None:
            self.__write_attributes(self.solution_set, attributes)

    def add_antennas(self, names, positions):
        """
        Add antennas dataset to h5 file

        Parameters
        ----------
        names : list, np.ndarray
            list or 1-d numpy array of strings containing the antenna names
        positions : np.array
            numpy array of shape nantenna x 3 (x, y, z), containing antenna positions in ITRF ECEF
            coordinates
        """

        assert len(names) == positions.shape[0]
        assert positions.ndim == 2 and positions.shape[1] == 3

        if isinstance(names, list):
            names = np.array(names)

        assert names.ndim == 1
        assert names.dtype.type == np.str_

        # Set proper dtype for names string (replace U by S):
        data_type_h5 = re.sub(r"[a-zA-Z]", "S", names.dtype.str)

        antenna_type = np.dtype(
            {
                "names": ["name", "position"],
                "formats": [data_type_h5, (positions.dtype, (3,))],
            }
        )

        arr = np.ndarray(len(names), antenna_type)
        arr["name"] = names.astype(data_type_h5)
        arr["position"] = positions

        self.h5file[self.solution_set].create_dataset(
            "antenna", (names.size,), dtype=antenna_type
        )
        self.h5file[self.solution_set + "/antenna"][:] = arr

    def add_sources(self, names, directions):
        """
        Add metadata (directions on the sky in J2000) about named sources

        Parameters
        ----------
        names : list, np.1darray
            Source names
        directions : np.2darray
            Directions (in J2000) as ra, dec (both radians)
        """

        # TODO: could optionally be merged with add_antennas()
        assert len(names) == directions.shape[0]
        assert directions.ndim == 2 and directions.shape[1] == 2

        if isinstance(names, list):
            names = np.array(names)

        assert names.ndim == 1
        assert names.dtype.type == np.str_

        # Convert names to bytes
        names = self.__convert_np_str_to_bytes(names)
        # Set proper dtype for names string (replace U by S):
        antenna_type = np.dtype(
            {
                "names": ["name", "dir"],
                "formats": [names.dtype, (directions.dtype, (2,))],
            }
        )
        arr = np.ndarray(len(names), antenna_type)
        arr["name"] = names
        arr["dir"] = directions

        self.h5file[self.solution_set].create_dataset(
            "source", (names.size,), dtype=antenna_type
        )
        self.h5file[self.solution_set + "/source"][:] = arr

    def create_solution_table(
        self, name, type_name, axes, values=None, dtype=None, weights=None, history=None
    ):
        """
        Create a solution table

        Parameters
        ----------
        name : str
            Name of the solution table (an alias for a H5 group) to create within the solution set
        type_name : str
            Type name of the field
        axes : dict
            Dictionary containing the axes labels (string) and axis length (int)
            as keys and values, respectively
        values : np.ndarray, optional
            Values to store in "val". Shape should match the specified dimensions in "axes".
        dtype = str, optional
            String specifying the dtype. Only needed (required!) when values=None
        weight : np.ndarray, optional
            Weights to store in "weight". Shape should match the specified dimensions in "axes".
        history: str
            Add history info as an attribute to the var dataset containing the actual data
        """

        hdf_location = self.solution_set + "/" + name
        self.h5file[self.solution_set].create_group(name)
        self.h5file[hdf_location].attrs["TITLE"] = np.asarray(
            type_name, dtype=f"<S{len(type_name)}"
        )
        self.__write_version_stamp(self.solution_set + "/" + name)

        # Convert axes labels to S-type
        axes_labels = list(axes.keys())
        axes_labels = ",".join(axes_labels).encode()
        axes_labels = np.asarray(axes_labels, dtype=f"<S{len(axes_labels)}")
        axes_shape = tuple(axes.values())

        dtype = values.dtype if values is not None else dtype
        if dtype is None:
            raise ValueError(
                "values and dtype are both None. Please specify either one of these, since I need to know the dtype of the new solution table"
            )

        # Solution table always has a val dataset, containing the actual values
        self.h5file[hdf_location].create_dataset("val", axes_shape, dtype=dtype)
        # Write axes names to attribute
        self.h5file[hdf_location + "/val"].attrs["AXES"] = axes_labels

        # Write values, if given
        if values is not None:
            if values.shape != axes_shape:
                raise ValueError(
                    f"values shape {values.shape} and axes shape {axes_shape} mismatch!"
                )
            self.h5file[hdf_location + "/val"][:] = values

        # Write weights, if given
        self.h5file[hdf_location].create_dataset("weight", axes_shape)
        if not weights:
            # write all ones
            weights = np.ones(axes_shape)

        if weights.shape != axes_shape:
            raise ValueError(
                f"weights shape {weights.shape} and axes shape {axes_shape} mismatch!"
            )

        # Fill nan values with 0.
        weights = np.nan_to_num(weights, copy=True, nan=0.0)
        self.h5file[hdf_location + "/weight"][:] = weights

        if history is not None:
            self.h5file[hdf_location + "/val"].attrs["HISTORY000"] = np.asarray(
                history, dtype=f"<S{len(history)}"
            )

    def fill_solution_table(self, name, values, offset=None):
        """
        Fill solution table from given offset

        Parameters
        ----------
        name : str
            Solution table name
        values : np.ndarray
            Numpy ndarray containing the values to store in the solution table.
            Number of dimensions should match the dimension of the solution table.
        offset : tuple, optional
            Integer tuple containing the axis offsets at which to insert the values
            into the solution table.
        """

        hdf_location = self.solution_set + "/" + name
        sol_table = self.h5file[hdf_location + "/val"]

        if offset is None:
            if values.shape != sol_table.shape:
                raise ValueError(
                    f"Shape of specified values array {values.shape} should match the solution table shape {sol_table.shape}"
                )
            sol_table[:] = values
        else:
            # Check whether offsets have correct length
            if len(offset) != values.ndim:
                raise ValueError(
                    f"Offset tuple of size {len(offset)} does not match dimension of input values {values.ndim})"
                )
            if len(offset) != sol_table.ndim:
                raise ValueError(
                    f"Offset tuple of size {len(offset)} does not match dimension of target solution table {sol_table.ndim}"
                )

            # Check whether offset are within range of axes
            if (
                len(tuple(filter(lambda x: x[0] >= x[1], zip(offset, sol_table.shape))))
                != 0
            ):
                raise ValueError(
                    f"Specified offset is not within range of solution table. Check your axis dimensions and offsets"
                )

            # Convert offset and values shape into slice objects
            slicer = tuple(
                slice(off, off + shape) for (off, shape) in zip(offset, values.shape)
            )
            sol_table[slicer] = values

    def create_axis_meta_data(
        self, solution_table, axis, meta_data=None, attributes=None, overwrite=True
    ):
        """
        Add meta data for an axis in the solution table

        Parameters
        ----------
        solution_table : str
            Name of the solution table
        axis : str
            Axis name
        meta_data : np.ndarray, optional
            Numpy array containing meta data values
        attributes : dict, optional
            Dictionary containing key-value pairs of axis attributes.
            If meta_data is None, an empty data set containing only the
            attributes is generated.
        overwrite : bool, optional
            Allow overwriting existing meta data or attributes? Defaults to True.
        """

        if meta_data is None and attributes is None:
            return

        hdf_location = self.solution_set + "/" + solution_table

        # TODO: optionally, we could check whether the axis name is permitted at all

        if hdf_location not in self.h5file:
            raise RuntimeError(
                f"Solution table name {solution_table} not present in solution set!"
            )

        try:
            axes_info = self.h5file[hdf_location + "/val"].attrs["AXES"]
        except:
            raise RuntimeError(
                f"AXES attribute not found in solution table {solution_table}!"
            )

        # Only decode if axes_info a bytes-object
        axes_info = axes_info.decode() if isinstance(axes_info, bytes) else axes_info
        if axis not in axes_info:
            raise ValueError(
                f"axis {axis} not found in solution table {solution_table}"
            )

        if meta_data is not None:
            if meta_data.dtype.type == np.str_:
                meta_data = self.__convert_np_str_to_bytes(meta_data)

            if hdf_location + "/" + axis not in self.h5file:
                # Create axis meta data set
                self.h5file[hdf_location].create_dataset(
                    axis, meta_data.shape, meta_data.dtype
                )
                self.h5file[hdf_location + "/" + axis][:] = meta_data
            elif overwrite is True:
                axis_dataset = self.h5file[hdf_location + "/" + axis]
                if axis_dataset.shape == meta_data.shape:
                    # Write immediately to dataset
                    axis_dataset[...] = meta_data
                else:
                    raise ValueError(
                        "Shape mismatch of data to be added, I can't handle this!"
                    )
            else:
                pass

        # Add attributes to axis data set
        if attributes is not None:
            # Create (empty) dataset if not yet existing
            if hdf_location + "/" + axis not in self.h5file:
                self.h5file[hdf_location].create_dataset(axis, (0,))

            self.__write_attributes(
                hdf_location + "/" + axis, attributes, overwrite=overwrite
            )

    def close_file(self):
        self.h5file.close()

    def __write_version_stamp(self, hdf_location):
        """
        Add version stamp to specified location

        Parameters
        ----------
        hdf_location : str
            Location where the version stamp will be added.
        """
        self.h5file[hdf_location].attrs["h5parm_version"] = np.asarray(
            H5PARM_VERSION, dtype=f"<S{len(H5PARM_VERSION)}"
        )

    def __write_attributes(self, hdf_location, attributes, overwrite):
        """
        Write attributes to given location in hdf5 file

        Parameters
        ----------
        hdf_location : str
            hdf5 location
        attributes : dict
            key-value pair of attributes to write
        overwrite : bool
            Overwrite attribute if it already exists?
        """

        for key, value in attributes.items():
            if (
                key in set(self.h5file[hdf_location].attrs.keys())
                and overwrite is False
            ):
                pass
            else:
                if isinstance(value, str):
                    # Byte encoded string
                    value = np.asarray(value, dtype=f"<S{len(value) + 1}")
                elif isinstance(value, np.ndarray) and value.dtype.type == np.str_:
                    # Convert to bytes string
                    value = self.__convert_np_str_to_bytes(value)
                self.h5file[hdf_location].attrs[key] = value

    @staticmethod
    def __convert_np_str_to_bytes(array):
        """
        Convert a numpy array of strings to a (H5) writable/readable
        bytes array. In order to do so, the length is extended with one additional
        position (to get the correct result in the CPP H5 reader)

        Parameters
        ----------
        array : np.ndarray
            Input array

        Returns
        -------
        np.ndarray
            Numpy array with converted dtype
        """

        assert array.dtype.type == np.str_

        # Capture the length of the string, and extend with 1 position
        str_length = int(re.match(r"<\w(\d*)", array.dtype.str).group(1)) + 1
        # dtype is S string, rather than U (unicode) string
        dtype_h5 = f"<S{str_length}"
        return array.astype(dtype=dtype_h5)

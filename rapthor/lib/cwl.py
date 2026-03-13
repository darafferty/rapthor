"""
Definition of CWL-related classes
"""

import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger("rapthor:cwl")


class CWLPath(object):
    """
    CWL path class

    Parameters
    ----------
    path : str or list of str
        Path or list of paths
    path_type : str
        Type of path: 'file' or 'directory'
    """

    def __init__(self, path, path_type):
        if type(path) not in [str, list]:
            raise ValueError("path must be a string or a list of strings")
        self.path = path
        if path_type.lower() == "file":
            self.path_type = "File"
        elif path_type.lower() == "directory":
            self.path_type = "Directory"
        else:
            raise ValueError('path type must be one of "file" or "directory"')

    def to_json(self):
        """
        Returns a dict suitable for use with json.dumps()
        """
        if type(self.path) is str:
            # File type
            cwl_value = {"class": self.path_type, "path": self.path}
        else:
            # File[] type
            cwl_value = []
            for p in self.path:
                cwl_value.append({"class": self.path_type, "path": p})

        return cwl_value
    
            
class CWLFile(CWLPath):
    """
    CWL File class

    Parameters
    ----------
    filename : str or list of str
        Filename or list of filenames
    """

    def __init__(self, filename):
        super(CWLFile, self).__init__(filename, "file")


class CWLDir(CWLPath):
    """
    CWL Directory class

    Parameters
    ----------
    dirname : str or list of str
        Directory name or list of directory names
    """

    def __init__(self, dirname):
        super(CWLDir, self).__init__(dirname, "directory")


class NpEncoder(json.JSONEncoder):
    """
    Numpy to JSON encoder class

    Numpy types cannot be serialized to JSON by default, so this
    class is used in json.dumps() calls when numpy types are
    present
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    

class PosixPathEncoder(json.JSONEncoder):
    """
    JSON encoder that converts Path objects to their string representation.
    """

    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj.absolute())
        return super(PosixPathEncoder, self).default(obj)


class MultiEncoder(PosixPathEncoder, NpEncoder):
    """
    JSON encoder that combines the functionality of PosixPathEncoder and NpEncoder.
    """
    def default(self, obj):
        return super().default(obj)
    

def is_cwl_file(cwl_obj):
    """
    Check if the given object is a CWL file representation.

    A CWL file representation is a dictionary with 'class' key set to 'File'.
    """
    return isinstance(cwl_obj, dict) and cwl_obj.get("class") == "File"


def is_cwl_directory(cwl_obj):
    """
    Check if the given object is a CWL directory representation.

    A CWL directory representation is a dictionary with 'class' key set to 'Directory'.
    """
    return isinstance(cwl_obj, dict) and cwl_obj.get("class") == "Directory"


def is_cwl_file_or_directory(cwl_obj):
    """
    Check if the given object is a CWL file or directory representation.
    """
    return is_cwl_file(cwl_obj) or is_cwl_directory(cwl_obj)


def naturalize_cwl_output(cwl_output):
    """
    Convert a CWL output object with more elements per field
    into a list of CWL output objects with one element per field.
    The output with just one element per field will be repeated.

    For example, if the input is:
    {
        "output1": [{"class": "File", "path": "file1.txt"}, {"class": "File", "path": "file2.txt"}],
        "output2": {"class": "File", "path": "file3.txt"}
    }
    the output will be:
    [
        {
            "output1": {"class": "File", "path": "file1.txt"},
            "output2": {"class": "File", "path": "file3.txt"}
        },
        {
            "output1": {"class": "File", "path": "file2.txt"},
            "output2": {"class": "File", "path": "file3.txt"}
        }
    ]
    """
    # Check if output is already naturalized (i.e., all fields have one element)
    if isinstance(cwl_output, list):
        return cwl_output
    if not isinstance(cwl_output, dict):
        raise ValueError("CWL output must be a dictionary or a list of dictionaries")

    # First, determine the number of items in the output
    num_items = 1
    single_valued_keys = set()
    for key, value in cwl_output.items():
        if isinstance(value, list):
            num_items = max(num_items, len(value))
        else:
            single_valued_keys.add(key)

    # Then, create a list of output objects with one element per field
    naturalized_output = []
    for i in range(num_items):
        item = {key: cwl_output[key] for key in single_valued_keys}
        for key, value in cwl_output.items():
            if key not in single_valued_keys:
                item[key] = value[i] if i < len(value) else value[-1]
        naturalized_output.append(item)

    return naturalized_output


def copy_cwl_object(src_obj, dest_dir, move=False):
    """
    Copy a CWL file or directory object to the specified destination directory.

    Parameters
    ----------
    src_obj : object
        Source object of the copy
    dest_dir: str
        Path of destination directory to which src_obj will be copied
    move : bool, optional
        If True, move files instead of copying them
    """
    if is_cwl_file_or_directory(src_obj) and os.path.exists(src_obj["path"]):
        os.makedirs(dest_dir, exist_ok=True)
        src = Path(src_obj["path"])
        dest = Path(dest_dir) / src.name
        if move:
            shutil.move(src, dest)
        else:
            if is_cwl_file(src_obj):
                shutil.copy(src, dest)
            elif is_cwl_directory(src_obj):
                shutil.copytree(src, dest, dirs_exist_ok=True)
    # Otherwise, do nothing


def copy_cwl_recursive(src_obj, dest_dir, index=None, move=False):
    """
    Recursively copy CWL file or directory objects to the specified destination
    directory.

    Parameters
    ----------
    src_obj : object or list of objects
        Source object(s) of the copy
    dest_dir: str
        Path of destination directory to which src_obj will be copied
    index : int
        If src_obj is a list and index is specified, only the item with the specified index is
        copied (other items in the list are ignored)
    move : bool, optional
        If True, move files instead of copying them
    """
    if isinstance(src_obj, list):
        for i, item in enumerate(src_obj):
            if index is None or i == index:
                copy_cwl_recursive(item, dest_dir, None, move)
    elif is_cwl_file_or_directory(src_obj):
        copy_cwl_object(src_obj, dest_dir, move)
    # Otherwise, do nothing


def remove_or_log_error(path: Path):
    """
    Remove a file or directory at the specified path.
    Log a warning if the file or directory does not exist.

    Parameters
    ----------
    path: Path object
        Path of file or directory to remove
    """
    try:
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except FileNotFoundError:
        logger.warning("Cannot remove non-existing path: %s", path)


def clean_if_cwl_file_or_directory(src_obj):
    """
    Remove CWL file or directory objects from the filesystem.

    Parameters
    ----------
    src_obj : object or list of objects
        Source object(s) to be removed
    """
    if isinstance(src_obj, list):
        for item in src_obj:
            clean_if_cwl_file_or_directory(item)
    elif is_cwl_file_or_directory(src_obj):
        remove_or_log_error(Path(src_obj["path"]))


def parse_cwl_output_recursive(cwl_object):
    """
    Recursively parse a CWL output object, converting any CWL file or directory
    representations into CWLFile or CWLDir objects.

    Parameters
    ----------
    object : object
        Object to be parsed

    Returns
    -------
    object
        Parsed object with CWL file and directory representations converted to
        CWLFile and CWLDir objects
    """
    if isinstance(cwl_object, list):
        return [parse_cwl_output_recursive(item) for item in cwl_object]
    elif is_cwl_file(cwl_object) or is_cwl_directory(cwl_object):
        return {**cwl_object, "path": Path(cwl_object["path"])}
    elif isinstance(cwl_object, dict):
        return {key: parse_cwl_output_recursive(value) for key, value in cwl_object.items()}
    else:
        return cwl_object
    

def store_cwl_output(output_obj, output_file):
    """
    Store a CWL output object to a JSON file.

    Parameters
    ----------
    output_obj : object
        CWL output object to be stored
    output_file : str
        Path of JSON file to which the output object will be stored
    """
    with open(output_file, "w") as f:
        json.dump(output_obj, f, cls=MultiEncoder)
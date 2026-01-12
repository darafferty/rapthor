"""
Definition of CWL-related classes
"""
import json
import numpy as np
import os
import shutil

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
            raise ValueError('path must be a string or a list of strings')
        self.path = path
        if path_type.lower() == 'file':
            self.path_type = 'File'
        elif path_type.lower() == 'directory':
            self.path_type = 'Directory'
        else:
            raise ValueError('path type must be one of "file" or "directory"')

    def to_json(self):
        """
        Returns a dict suitable for use with json.dumps()
        """
        if type(self.path) is str:
            # File type
            cwl_value = {'class': self.path_type, 'path': self.path}
        else:
            # File[] type
            cwl_value = []
            for p in self.path:
                cwl_value.append({'class': self.path_type, 'path': p})

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
        super(CWLFile, self).__init__(filename, 'file')


class CWLDir(CWLPath):
    """
    CWL Directory class

    Parameters
    ----------
    dirname : str or list of str
        Directory name or list of directory names
    """
    def __init__(self, dirname):
        super(CWLDir, self).__init__(dirname, 'directory')


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

def is_cwl_file(cwl_obj):
    """
    Check if the given object is a CWL file representation.

    A CWL file representation is a dictionary with 'class' key set to 'File'.
    """
    return isinstance(cwl_obj, dict) and cwl_obj.get('class') == 'File'

def is_cwl_directory(cwl_obj):
    """
    Check if the given object is a CWL directory representation.

    A CWL directory representation is a dictionary with 'class' key set to 'Directory'.
    """
    return isinstance(cwl_obj, dict) and cwl_obj.get('class') == 'Directory'

def is_cwl_file_or_directory(cwl_obj):
    """
    Check if the given object is a CWL file or directory representation.
    """
    return is_cwl_file(cwl_obj) or is_cwl_directory(cwl_obj)

def copy_cwl_object(src_obj, dest_dir):
    """
    Copy a CWL file or directory object to the specified destination directory.
    """
    if is_cwl_file_or_directory(src_obj):
        os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
        src = src_obj['path']
        dest = os.path.join(dest_dir, os.path.basename(src_obj['path']))
        if is_cwl_file(src_obj):
            shutil.copy(src, dest)
        elif is_cwl_directory(src_obj):
            shutil.copytree(src, dest)
    # Otherwise, do nothing

def copy_cwl_recursive(src_obj, dest_dir):
    """
    Recursively copy CWL file or directory objects to the specified destination directory.
    """
    if isinstance(src_obj, list):
        for item in src_obj:
            copy_cwl_recursive(item, dest_dir)
    elif is_cwl_file_or_directory(src_obj):
        copy_cwl_object(src_obj, dest_dir)
    # Otherwise, do nothing

def clean_if_cwl_file_or_directory(src_obj):
    """
    Remove CWL file or directory objects from the filesystem.
    """
    if isinstance(src_obj, list):
        for item in src_obj:
            clean_if_cwl_file_or_directory(item)
    elif is_cwl_file(src_obj):
        try:
            os.remove(src_obj['path'])
        except FileNotFoundError:
            pass
    elif is_cwl_directory(src_obj):
        try:
            shutil.rmtree(src_obj['path'])
        except FileNotFoundError:
            pass
    # Otherwise, do nothing
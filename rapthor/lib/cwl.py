"""
Definition of CWL-related classes
"""
import json
import numpy as np


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

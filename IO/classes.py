import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, patches
import hystorian as hy
import h5py

import os
from glob import glob
import sys


class CypherBatch:
    def __init__(self, path, filename, **kwargs):
        """
        A class for batch processing of Cypher scans in a folder.

        Attributes:
        ----------
        path: str
            path to folder containing scans TODO: Choose between parent path and filename structure or the entire path.
        **kwargs: keyword arguments for all classes. Universal.
        Methods
        -------
        __call__:

        """
        self.path = path
        self.filename = filename
        self.scans = glob(
            os.path.join(path, "*.ibw")
        )  # TODO: Implement an option to change the number of files.
        self.kwargs = (
            kwargs  # TODO: The kwargs dictionary is to be universal for all classes.
        )

        self.container = [CypherFile(scan, **kwargs) for scan in self.scans]

    def __call__(self):
        """
        Creates one hdf5 file for data analysis, containing all scans in the folder.
        """
        from IO import ibw_io

        ibw_io.convert_batch_ibw(self)
        return


class CypherFile:
    def __init__(self, path, filename, **kwargs):
        """
        path: path to folder
        filename: filename of scan
        """
        self.fullpath = os.path.join(path, filename+".ibw")
        self.kwargs = kwargs

    def __call__(self):
        """
        Creates the hdf5 file for data analysis.
        """
        from IO import ibw_io

        ibw_io.convert_ibw(self)
        return

    def hy_apply(self, function, *args, **kwargs):
        """
        Applies a function to the data in the hdf5 file.
        """
        hy.m_apply(self.fullpath, function, *args, **kwargs)
        return

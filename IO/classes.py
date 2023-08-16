import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, patches
import hystorian as hy
import h5py
import pandas as pd
import os
from glob import glob
import sys


class CypherBatch:
    def __init__(self, path:str, filename:str, **kwargs):
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
    def __init__(self, path:str, filename:str, **kwargs):
        """
        path: path to folder
        filename: filename of scan
        """
        self.path = path
        self.filename = filename
        self.fullpath = os.path.join(path, filename+".ibw")
        self.opath = os.path.join(self.path, self.filename + ".hdf5")
        self.kwargs = kwargs

    def __call__(self):
        """
        Creates the hdf5 file for data analysis.
        """
        from IO import ibw_io

        ibw_io.convert_ibw(self)
        return
    
    def __getitem__(self, key: str) -> np.ndarray:

        def create_path(string: str) -> str:
            """
            Creates a hdf5 path from a string.
            """
            return r"datasets/"+rf"{os.path.join(self.path, self.filename)}" + r"/" + rf"{string}"

        keys2paths = {
            "CR2": create_path("Current2Retrace"),
            "CR": create_path("CurrentRetrace"),
            "CT": create_path("CurrentTrace"),
            "DR": create_path("DeflectionRetrace"),
            "HR": create_path("HeightRetrace"),
            "ZR": create_path("ZSensorRetrace"),
        }

        print(keys2paths[key])
         
        try:
            with h5py.File(self.opath, "r") as f:

                return np.array(f[keys2paths[key]])
        except KeyError:
            print("Key not found in hdf5 file.")
            print(keys2paths[key])
            return
    
    def get_metadata(self):
        """
        Encodes and returns the metadata from the hdf5 file as a pd.DataFrame.
        """
        with h5py.File(self.opath, "r") as f:
            metadata = f[r"metadata/"+rf"{os.path.join(self.path, self.filename)}"]
            strng= metadata.tolist().decode('ascii', errors='replace')
            df = pd.DataFrame([sub.split(":") for sub in strng.split("\r")])
        df.columns = ["key", "value"]
        df = df.set_index("key")
        return df


    def hy_apply(self, function: callable, *args, **kwargs):
        """
        Applies a function to the data in the hdf5 file.
        """
        hy.m_apply(self.fullpath, function, *args, **kwargs)
        return
    
    def get_dataset_keys(self,f):
        """
        Returns all dataset keys in a hdf5 file.
        """
        keys = []
        f.visit(lambda key: keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
        return keys
    






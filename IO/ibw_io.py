

import numpy as np
import pandas as pd
import hystorian as hy
import h5py

import os
# Use glob or os.listdir to get a list of files in a directory. (glob is better, but os.listdir was used in the master's thesis)
from glob import glob
from universal_reader import find_key


"""
Would like a code structure that utilises Hystorian to retrieve SPM data, export to HDF5, be able to do batch processing, 
log the processing, and have much functionality. The functionality should include leveling, cropping, range-limiting, plotting etc. 

Initially, reading and writing of data should be implemented.
Then, plotting should be implemented.

Idea is that data analysis and preparation of results should be done in a juptyer notebook.
This fact imposes the issue that the convertion of data only needs to be done once. 
Convertion of data could be the __call__ method of a class, which is called by the notebook manually if the hdf5 file is not present.

Code structure is the first important decision,
and will be as follows:

1. A Batch wrapper that can convert all scans in a folder of data. Argparse-file perhaps? Or Wrapper class. 
2. A File wrapper that can convert a single file.



"""





class CypherFile:


    keywords = {
            "H": ["Height", r"^H."],
            "C": ["Current", r"^C."],
            "D": ["DFL", "Deflection", r"^D."],
            "Z": ["ZSensor", r"^Z."],
            # "V": ["Voltage"],
            # "Amplitude": ["Amp", "Amplitude", "Mag"], TODO: Add these channels. Bit jalla, but okay.
            # "Phase": ["Phase"],
        }
    modes = {
            "T": ["F:", "Trace", "Forward", ".T$"],
            "R": ["B:", "Retrace", "Backward", ".R$" ],
            "R2": ["B2:", "Retrace2", "Backward2", ".R2$" ],
        }
    

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
    
        self.convert_ibw()

        im = self["CR"]
        self.x_dim = im.shape[1]
        self.y_dim = im.shape[0]

        self.x_res = float(self.get_metadata_key("ScanSize")) / float(self.get_metadata_key("PointsLines"))
        self.y_res = self.x_res

        return
    
    def __getitem__(self, key_input: str) -> np.ndarray:

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
         
        try:
            with h5py.File(self.opath, "r") as f:
                key_input = f"{key_input}" #TODO: Add space or not?
                key = find_key(CypherFile.keywords, key_input) + find_key(CypherFile.modes, key_input)

                return np.array(f[keys2paths[key]])
        except KeyError:
            print("Key not found in hdf5 file.")
            print(keys2paths[key])
            return
    
    def get_metadata(self)->pd.DataFrame:
        """
        Encodes and returns the metadata from the hdf5 file as a pd.DataFrame.
        """
        with h5py.File(self.opath, "r") as f:
            metadata = f[r"metadata/"+rf"{os.path.join(self.path, self.filename)}"]
            strng= np.array(metadata).tolist().decode('ascii', errors='replace')
            df = pd.DataFrame([sub.split(":") for sub in strng.split("\r")]).loc[:, :1]
            df.set_index(0, inplace=True)
        
        self.meta = df
        return df
    
    def get_metadata_key(self, key:str) ->pd.DataFrame.items:
        """
        Returns the value of a metadata key.
        """
        try:
            return self.meta.loc[key].item()
        except AttributeError:
            self.get_metadata()
            return self.meta.loc[key].item()


    def hy_apply(self, function: callable, *args, **kwargs):
        """
        Applies a function to the data in the hdf5 file.
        """
        hy.m_apply(self.fullpath, function, *args, **kwargs)
        return
    
    def get_dataset_keys(self):
        """
        Returns all dataset keys in a hdf5 file.
        """
        keys = []
        with h5py.File(self.opath, "r") as f:
            f.visit(lambda key: keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
        return keys

    
    def convert_ibw(self):
        """
        Part of the CypherFile class.
        Converts the ibw file to hdf5.
        """

        hy.io.read_file.tohdf5(self.fullpath)

        return


    def convert_batch_ibw(self):
        """
        Part of the CypherBatch class.
        Converts all ibw files in the folder to hdf5.
        TODO: Figure out the naming when several files are merged. 
        """

        hy.io.read_file.merge_hdf5(self.scans, self.filename)
        self.opath = os.path.join(self.path, self.filename + ".hdf5")

        return
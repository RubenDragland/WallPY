from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, patches
import h5py
import pandas as pd
import os
from glob import glob
import sys
import gwyfile as gwy
import re



"""
GwyFile from classes.py was copied over so that the class could be imported in environments without all the different dependencies.
TODO: Do the same for CypherFile, and use classes to do relative imports of the different classes in case one wants to import all of them. 
"""

def find_key(keywords: dict, title: str) -> str:
        """
        Finds the key in the keywords dictionary based on channel title.
        Universal function for all classes.
        """

        for key, elems in keywords.items():
            for elem in elems:
                match = re.search(elem, title)
                if match is not None:
                    return key
                
        print(title, "not found in keywords dictionary.")
        return None



class GwyFile:

    """
        Believed to be an universal class from reading AFM scans in python.
        The key is to use Gwyddion to do an initial filtering of saved scans from experiments. 
        Next, use Gwyddion to export all the different data formats to .gwy-file. 
        The .gwy-file can then be read by this class using the gwyfile package.
        CypherFile is suspected to perform better than this class, but will nevertheless be compatible. However, less information.
        Note limited amount of metadata extractable. CypherFile is better for this. Tip type needs additional logging regardless. 

        Container for python-processing of data from a gwy-file.

        Attributes
        ----------

        path: str
            path to folder containing scans
        filename: str
            filename of scan
        **kwargs: dict, optional
            keyword arguments for all classes. Universal.
        
        Methods
        -------

        __call__:
            Idea: Redefine keys based on metadata, and store the data in a hdf5 file.
        __getitem__:
            Returns the data of a channel based on category and mode. Returns a list if multiple channels are found.
        get_dataset_keys:
            Returns all dataset keys in a hdf5 file.
        
    """


    keywords = {
            "Height": ["Height"],
            "Current": ["Current", "Iprobe"],
            "Deflection": ["DFL", "Deflection"],
            "ZSensor": ["ZSensor", "Z-Axis"],
            "Voltage": ["Voltage", "Ext1"], #TODO: Know how to sort these. Include Channels etc. Remember to update both or remove gwyfile from classes. This one is the updated one. 
            "VPFM": ["VPFM", "Ext2"],
            "LPFM": ["LPFM", "Ext3"],
            "Amplitude": ["Amp", "Amplitude", "Mag"],
            "Phase": ["Phase"],
        }
    modes = {
            "Forward": ["F:", "Trace", "Forward", "forward"],
            "Backward": ["B:", "Retrace", "Backward", "backward" ]
        }

    settings = {
            #TODO: Get more meta data from scan. Current/voltage/frerquency etc. Possibly need for manual logging system.

        }


    def __init__(self, path:str, filename:str, **kwargs):
        """
        path: path to folder
        filename: filename of scan
        """
        self.path = path
        self.filename = filename[:-4] if filename.endswith(".gwy") else filename
        self.fullpath = os.path.join(self.path, self.filename +".gwy") 
        self.opath = os.path.join(self.path, self.filename  + ".hdf5")
        self.kwargs = kwargs

        try:

            with h5py.File(self.opath, "r") as f:
                self.channel_names = list(f["channel_names"]) #TODO: Stored as bytes. Issue?
            print("H5 Structure exists.")
        except:

            self.channel_names = []
            print("Convert to hdf5.")
        finally:
            pass
    
    def __call__(self):
        """
        Currently no functionality. Keep it as gwy-dict. Perhaps store transformed data in hdf5 file or something?
        Idea: Redefine keys based on metadata, and store the data in a hdf5 file.
        """

        obj = gwy.load(self.fullpath)
        channels = gwy.util.get_datafields(obj)

        self.channel_names = []
        print("Channels found: ")

        with h5py.File(self.opath, "w") as f:
            
            for id, key in enumerate(channels.keys()):

                category = str(find_key(GwyFile.keywords, key))
                mode = str(find_key(GwyFile.modes, key))
                unique = str(id) #str(np.round(np.max(1e6*channels[key].data)-np.min(1e6*channels[key].data),3)) 

                name = f"{category}_{mode}_{unique}"
                self.channel_names.append(name)
                print(name)
                                
                f.create_group(name)
                f[name].create_dataset("data", data=channels[key].data)
                f[name].attrs["title"] = name
                f[name].attrs["category"] = category
                f[name].attrs["mode"] = mode
                f[name].attrs["unit"] = channels[key].si_unit_z.unitstr
                f[name].attrs["xsize"] = channels[key].xreal
                f[name].attrs["ysize"] = channels[key].yreal
                f[name].attrs["xres"] = channels[key].xreal / channels[key].data.shape[1]

            f.create_dataset("channel_names", data=np.array(self.channel_names, dtype="S") )
        return
    
    def get_by_key(self, key: str) -> list:

        """
        Returns the data of a channel based on category and mode. Returns a list if multiple channels are found.
        TODO: Adjust so that B and F are valid modes.
        TODO: Can both modes work?
        """
        datas = []

        with h5py.File(self.opath, "r") as f:

            # self.channel_names = list(f["channel_names"])
            for channel in self.channel_names:

                keyword = find_key(GwyFile.keywords, key)
                mode = find_key(GwyFile.modes, key)
                if ( re.search( f[channel].attrs["category"], keyword ) is not None) and (re.search(f[channel].attrs["mode"], mode) is not None):
                    datas.append(np.array(f[channel]["data"]))

        if datas == []:
            print("No data found.")
            return None
        else:
            return datas


    
    def __getitem__(self, index: str) -> np.ndarray:
        """
        Returns the data of a channel based on index.
        """
        name = self.channel_names[index]

        with h5py.File(self.opath, "r") as f:
            return np.array(f[name]["data"]) 

    
    def get_dataset_keys(self):
        """
        Returns all dataset keys in a hdf5 file.
        """
        keys = []
        with h5py.File(self.opath, "r") as f:
            f.visit(lambda key: keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
        return keys
    
    def index_metadata(self, index: int, feature=None) -> Any:
        """
        Returns the metadata of a channel based on index.
        """
        name = self.channel_names[index]
        if feature is None:
            with h5py.File(self.opath, "r") as f:
                return f[name].attrs.items()
        else:
            with h5py.File(self.opath, "r") as f:
                return f[name].attrs[feature]
            


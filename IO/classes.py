from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, patches
import h5py
import pandas as pd
import os
from glob import glob
import sys
import tifffile
import gwyfile as gwy
import re

from universal_reader import find_key, GwyFile

import hyperspy.api as hs
import hystorian as hy

    
# def find_key(keywords: dict, title: str) -> str:
#         """
#         Finds the key in the keywords dictionary based on channel title.
#         Universal function for all classes.
#         """

#         for key, elems in keywords.items():
#             for elem in elems:
#                 match = re.search(elem, title)
#                 if match is not None:
#                     return key
                
#         print(title, "not found in keywords dictionary.")
#         return None

# class GwyFile:

#     """
#         Believed to be an universal class from reading AFM scans in python.
#         The key is to use Gwyddion to do an initial filtering of saved scans from experiments. 
#         Next, use Gwyddion to export all the different data formats to .gwy-file. 
#         The .gwy-file can then be read by this class using the gwyfile package.
#         CypherFile is suspected to perform better than this class, but will nevertheless be compatible. However, less information.
#         Note limited amount of metadata extractable. CypherFile is better for this. Tip type needs additional logging regardless. 

#         Container for python-processing of data from a gwy-file.

#         Attributes
#         ----------

#         path: str
#             path to folder containing scans
#         filename: str
#             filename of scan
#         **kwargs: dict, optional
#             keyword arguments for all classes. Universal.
        
#         Methods
#         -------

#         __call__:
#             Idea: Redefine keys based on metadata, and store the data in a hdf5 file.
#         __getitem__:
#             Returns the data of a channel based on category and mode. Returns a list if multiple channels are found.
#         get_dataset_keys:
#             Returns all dataset keys in a hdf5 file.
        
#     """


#     keywords = {
#             "Height": ["Height"],
#             "Current": ["Current"],
#             "Deflection": ["DFL", "Deflection"],
#             "ZSensor": ["ZSensor"],
#             "Voltage": ["Voltage", "Ext", "Iprobe"], #TODO: Know how to sort these.
#             "Amplitude": ["Amp", "Amplitude", "Mag"],
#             "Phase": ["Phase"],
#         }
#     modes = {
#             "Forward": ["F:", "Trace", "Forward"],
#             "Backward": ["B:", "Retrace", "Backward" ]
#         }

#     settings = {
#             #TODO: Get more meta data from scan. Current/voltage/frerquency etc. Possibly need for manual logging system.

#         }


#     def __init__(self, path:str, filename:str, **kwargs):
#         """
#         path: path to folder
#         filename: filename of scan
#         """
#         self.path = path
#         self.filename = filename
#         self.fullpath = os.path.join(path, filename+".gwy")
#         self.opath = os.path.join(self.path, self.filename + ".hdf5")
#         self.kwargs = kwargs
    
#     def __call__(self):
#         """
#         Currently no functionality. Keep it as gwy-dict. Perhaps store transformed data in hdf5 file or something?
#         Idea: Redefine keys based on metadata, and store the data in a hdf5 file.
#         """

#         obj = gwy.load(self.fullpath)
#         channels = gwy.util.get_datafields(obj)

#         self.channel_names = []
#         print("Channels found: ")

#         with h5py.File(self.opath, "w") as f:
#             for id, key in enumerate(channels.keys()):

#                 #TODO: Figure out name of channels. F/B are for forward/backward, respectively. 

#                 category = str(find_key(GwyFile.keywords, key))
#                 mode = str(find_key(GwyFile.modes, key))
#                 unique = id #str(np.round(np.max(1e6*channels[key].data)-np.min(1e6*channels[key].data),3)) 

#                 name = f"{category}_{mode}_{unique}"
#                 self.channel_names.append(name)
#                 print(name)
                                
#                 f.create_group(name)
#                 f[name].create_dataset("data", data=channels[key].data)
#                 f[name].attrs["title"] = name
#                 f[name].attrs["category"] = category
#                 f[name].attrs["mode"] = mode
#                 f[name].attrs["unit"] = channels[key].si_unit_z.unitstr
#                 f[name].attrs["xsize"] = channels[key].xreal
#                 f[name].attrs["ysize"] = channels[key].yreal
#                 f[name].attrs["xres"] = channels[key].xreal / channels[key].data.shape[1]

#         return
    
#     def get_by_key(self, key: str) -> list:

#         """
#         Returns the data of a channel based on category and mode. Returns a list if multiple channels are found.
#         TODO: Adjust so that B and F are valid modes.
#         TODO: Can both modes work?
#         """
#         datas = []

#         with h5py.File(self.opath, "r") as f:
#             for channel in f.keys():

#                 keyword = find_key(GwyFile.keywords, key)
#                 mode = find_key(GwyFile.modes, key)
#                 if ( re.search( f[channel].attrs["category"], keyword ) is not None) and (re.search(f[channel].attrs["mode"], mode) is not None):
#                     datas.append(np.array(f[channel]["data"]))

#         if datas == []:
#             print("No data found.")
#             return None
#         else:
#             return datas


    
#     def __getitem__(self, index: str) -> np.ndarray:
#         """
#         Returns the data of a channel based on index.
#         """
#         name = self.channel_names[index]

#         with h5py.File(self.opath, "r") as f:
#             return np.array(f[name]["data"]) 

    
#     def get_dataset_keys(self):
#         """
#         Returns all dataset keys in a hdf5 file.
#         """
#         keys = []
#         with h5py.File(self.opath, "r") as f:
#             f.visit(lambda key: keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
#         return keys
    
#     def index_metadata(self, index: int, feature=None) -> Any:
#         """
#         Returns the metadata of a channel based on index.
#         """
#         name = self.channel_names[index]
#         if feature is None:
#             with h5py.File(self.opath, "r") as f:
#                 return f[name].attrs.items()
#         else:
#             with h5py.File(self.opath, "r") as f:
#                 return f[name].attrs[feature]



# class CypherBatch:

#     def __init__(self, path:str, filename:str, **kwargs):
#         """
#         A class for batch processing of Cypher scans in a folder.

#         Attributes:
#         ----------
#         path: str
#             path to folder containing scans TODO: Choose between parent path and filename structure or the entire path.
#         **kwargs: keyword arguments for all classes. Universal.
#         Methods
#         -------
#         __call__:

#         """
#         self.path = path
#         self.filename = filename
#         self.scans = glob(
#             os.path.join(path, "*.ibw")
#         )  # TODO: Implement an option to change the number of files.
#         self.kwargs = (
#             kwargs  # TODO: The kwargs dictionary is to be universal for all classes.
#         )

#         self.container = [CypherFile(scan, **kwargs) for scan in self.scans]

#     def __call__(self):
#         """
#         Creates one hdf5 file for data analysis, containing all scans in the folder.
#         """
#         from IO import ibw_io

#         ibw_io.convert_batch_ibw(self)
#         return


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
        from IO import ibw_io

        ibw_io.convert_ibw(self)

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
    


class TifFile(CypherFile):
    """
    Class for handling SEM images
    TODO: Why inheritance to CypherFile?
    """


    def __init__(self,path:str, filename:str, **kwargs):

        self.path = path
        self.filename = filename
        self.fullpath = os.path.join(path, filename+".tif") #TODO: Or tiff? What to do?
        self.opath = os.path.join(self.path, self.filename + ".hspy")
        self.kwargs = kwargs

        return
    
    def __call__(self, hyperspy:bool = True):
        """
        Creates the hdf5 file for data analysis.
        """
        try: 
            self.hyperspy = self.kwargs["hyperspy"] and hyperspy
        except KeyError:
            self.hyperspy = hyperspy
        finally:

            if self.hyperspy:
                img = hs.load(self.fullpath)
                img.save(self.opath) #TODO: Test and find res

            else:
                img = tifffile.imread(self.fullpath)
                meta = tifffile.tiffFile(self.fullpath)

                self.x_dim = img.shape[1]
                self.y_dim = img.shape[0]

                self.x_res = meta.fei_metadata['EScan']['PixelWidth']
                self.y_res = meta.fei_metadata['EScan']['PixelHeight']

                #TODO: Save to hdf5 file but necessary?
                self.opath = os.path.join(self.path, self.filename + ".hdf5")


        return img
    
    def __getitem__(self, key: str) -> np.ndarray:

        if self.hyperspy:
            h = hs.load(self.opath)
            if key == "all":
                return (h.data, h.metadata)
            else:
                return h[key].data #TODO: Fix and test.
        else:
            with h5py.File(self.opath, "r") as f:
                if key == "all":
                    return (np.array(f["data"]), np.array(f["meta"]) )
                else:
                    return np.array(f[key])






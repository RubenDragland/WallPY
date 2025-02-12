
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import igor2 as ig
import os
# Use glob or os.listdir to get a list of files in a directory. (glob is better, but os.listdir was used in the master's thesis)
from glob import glob
from universal_reader import find_key

from igor2 import binarywave
import h5py
from numpy import flipud








class IgorFile:


    keywords = {
            "H": ["Height", r"^H."],
            "C": ["Current", r"^C."],
            "D": ["DFL", "Deflection", r"^D."],
            "Z": ["ZSensor", r"^Z."],
            # "V": ["Voltage"],
            "A": ["A","Amp", "Amplitude", "Mag"], #TODO: Add these channels. Bit jalla, but okay.
            "P": ["P","Phase"],
        }
    modes = {
            "T": ["F:", "Trace", "Forward", ".T$"],
            "R": ["B:", "Retrace", "Backward", ".R$" ],
            "R2": ["B2:", "Retrace2", "Backward2", ".R2$" ],
        }
    

    def load_meta_to_memory(self):

        try:
            with h5py.File(self.opath, "r") as f:
                self.exists = True
        except FileNotFoundError:
            self.exists = False
        finally:
            if self.exists:
                print("File already exists.")
                with h5py.File(self.opath, "r") as f:

                    if self.id is not None:
                        f = f[self.id]

                    self.metadata = f["meta"]["metadata"]
                    self.label_list = [s.decode('utf8') for s in f["meta"]["label_list"]]
                    self.type = f["meta"].attrs["type"]

                    
                    self.shape = f["data"]["0000"].attrs["name"]
                    self.size = f["data"]["0000"].attrs["size"]
                    self.offset = f["data"]["0000"].attrs["offset"]

                    self.shape = f["data"]["0000"].attrs["shape"]
                    self.x_res = f["data"]["0000"].attrs["x_res"]
                    self.y_res = f["data"]["0000"].attrs["y_res"]

            else:
                print("Conversion required.")
            return

    

    def __init__(self, path:str, filename:str, id=None, **kwargs):

        """
        path: path to folder
        filename: filename of scan

        kwargs = {
            "opath": None
            "oname": None,
        }
        """
        self.kwargs = {
            "opath": None,
            "oname": None,
            "fullpath": None,

        }

        for key, value in kwargs.items():
            if key in self.kwargs:
                self.kwargs[key] = value


        if self.kwargs["fullpath"] is not None:
            self.fullpath = kwargs["fullpath"]
            self.path = os.path.dirname(self.fullpath)
            self.filename = os.path.basename(self.fullpath)
            self.filename = self.filename[:-4] if filename.endswith(".ibw") else self.filename
        else:
            self.path = path
            self.filename = filename[:-4] if filename.endswith(".ibw") else filename
        
        self.fullpath = os.path.join(self.path, self.filename +".ibw") 
        opath = self.path if self.kwargs["opath"] is None else self.kwargs["opath"]
        oname = self.filename if self.kwargs["oname"] is None else self.kwargs["oname"]

        self.opath = os.path.join(opath, oname + ".hdf5")
        self.oname = oname

        self.id = str(id).zfill(4) if id is not None else None

        self.exists= False

        self.load_meta_to_memory()

        pass


    def __getitem__(self, ind: int) -> np.ndarray:

        ind = len(self.label_list) + ind if ind < 0 else ind

        if self.id is not None:
            with h5py.File(self.opath, "r") as f:
                return np.array(f[self.id]["data"][str(ind).zfill(4)])
        else:
            with h5py.File(self.opath, "r") as f:
                return np.array(f["data"][str(ind).zfill(4)])
        
    def get(self, name=None, ind=None, meta=False ) -> np.ndarray:

        with h5py.File(self.opath, "r") as f:

            if ind is not None:
                data = np.array(f["data"][str(ind).zfill(4)])
                meta = f["data"][str(ind).zfill(4)].attrs
                if not meta:
                    return data
                else:
                    return data, meta
            elif name is not None:
                data = self.get_by_name(name)
                meta = None
                if not meta:
                    return data
                #TODO: Complete
            else:
                print("No index or name provided.")
    
    # def __setitem__(self, ind: int, data: np.ndarray) -> None:

    #     with h5py.File(self.opath, "a") as f:
    #         f["data"][str(ind).zfill(4)] = data
    #     return

    def get_by_name(self, name: str) -> np.ndarray:
        #TODO: Make abbreviation for this function.

        with h5py.File(self.opath, "r") as f:
            for key in f["data"].keys():
                if f["data"][key].attrs["name"] == name:
                    return np.array(f["data"][key])
        return


    def __call__(self):
        """
        Creates the hdf5 file for data analysis.
        """
    
        self.convert_ibw()

        self.load_meta_to_memory()

        return
    
    def correct_label(self, label):
        label = [x for x in label if x]  # Remove the empty lists
        label = label[0]  # Remove the unnecessary inception

        corrected_label = []

        for i in label:
            i = i.decode('UTF-8')
            if len(i) == 0:  # Skip empty channel names
                pass
            else:  # Correct the duplicate letters
                if 'Trace' in i:
                    i = i.split('Trace')[0]
                    corrected_label.append(i + 'Trace')
                elif 'Retrace' in i:
                    i = i.split('Retrace')[0]
                    corrected_label.append(i + 'Retrace')
                else:
                    corrected_label.append(i)
        corrected_label = [x.encode() for x in corrected_label]
        return corrected_label
        

        # def ibw2hdf5(self,filename, filepath=None):
    def ibw2hdf5(self, f):
        #TODO: filename and then oname possibly. Do not need parameters, can access them directly.

        tmpdata = binarywave.load(self.fullpath)['wave']
        note = tmpdata['note']
        label_list = self.correct_label(tmpdata['labels'])

        fastsize = float(str(note).split('FastScanSize:')[-1].split('\\r')[0])
        slowsize = float(str(note).split('SlowScanSize:')[-1].split('\\r')[0])
        xoffset = float(str(note).split('XOffset:')[1].split('\\r')[0])
        yoffset = float(str(note).split('YOffset:')[1].split('\\r')[0])

        # with h5py.File(self.opath, "w") as f:
            # f.require_group("process") # Open a group; creating if not existing. # Do not know if necessary with this group.
        data_group = f.require_group("data")
        meta_group = f.require_group("meta")

        metadata_dataset = meta_group.create_dataset("metadata", data=tmpdata['note'])
        meta_group.attrs['type'] = self.fullpath.split('.')[-1]
        # metadata_dataset.attrs["label_list"] = label_list
        meta_group.create_dataset("label_list", data=label_list)

        for i, k in enumerate(label_list):

            dataset_name = str(i).zfill(4)

            if len(np.shape(tmpdata['wData'])) == 2:
                data_group.create_dataset(dataset_name, data=np.flipud(tmpdata['wData'][:, i].T))
                data_group[dataset_name].attrs['shape'] = tmpdata['wData'][:, i].T.shape # 
                data_group[dataset_name].attrs['x_res'] = fastsize/tmpdata['wData'][:, i].T.shape[0]
                data_group[dataset_name].attrs['y_res'] = slowsize/tmpdata['wData'][:, i].T.shape[1]
            else:
                data_group.create_dataset(dataset_name, data=np.flipud(tmpdata['wData'][:, :, i].T))
                data_group[dataset_name].attrs['shape'] = tmpdata['wData'][:, :, i].T.shape
                data_group[dataset_name].attrs['x_res'] = fastsize/tmpdata['wData'][:, :, i].T.shape[0]
                data_group[dataset_name].attrs['y_res'] = slowsize/tmpdata['wData'][:, :, i].T.shape[1]

            data_group[dataset_name].attrs['name'] = k.decode('utf8')
            data_group[dataset_name].attrs['size'] = (fastsize, slowsize)
            data_group[dataset_name].attrs['offset'] = (xoffset, yoffset)
            # data_group[dataset_name].attrs['path'] = ("data/"+str(k).split('\'')[1])

            if "Phase" in str(k):
                data_group[dataset_name].attrs['unit'] = ('m', 'm', 'deg')
            elif "Amplitude" in str(k):
                data_group[dataset_name].attrs['unit'] = ('m', 'm', 'V')
            elif "Height" in str(k):
                data_group[dataset_name].attrs['unit'] = ('m', 'm', 'm')
            else:
                data_group[dataset_name].attrs['unit'] = ('m', 'm', 'unknown')

            print(f"Dataset {k.decode('utf8')} created.")
        return
    
    
    def convert_ibw(self):

        #TODO: Check if other things to do here. 
        with h5py.File(self.opath, "w") as f:
            self.ibw2hdf5(f)

        return
    

class IgorBatch(IgorFile):

    def __init__(self, path, filenames, oname, opath=None, **kwargs):

        # self.batch = True

        self.IgorFiles = [IgorFile(path, filename, oname=oname, opath=opath, id=i,  **kwargs) for i, filename in enumerate(filenames)]

        opath = path if opath is None else opath

        self.opath = os.path.join(opath, oname + ".hdf5")
        self.oname = oname

        self.exists= False

        # self.load_meta_to_memory() #TODO: Make this for the batch as well.


        pass

    def __call__(self):
        """
        Creates the hdf5 file for data analysis.
        """
        #TODO: Complete
    
        self.convert_ibw()

        # self.load_meta_to_memory()

        return
    
    def __getitem__(self, ind:int) -> IgorFile:
        
        return self.IgorFiles[ind]
    
    def __len__(self) -> int:

        return len(self.IgorFiles)
    
    def convert_ibw(self):

        with h5py.File(self.opath, "w") as parent_f:
            for i, file in enumerate(self.IgorFiles):
                fi = str(i).zfill(4)
                f = parent_f.create_group(fi)
                file.ibw2hdf5(f)

            
            return
        
    




    # def __getitem__(self, key_input: str) -> np.ndarray:

    #     def create_path(string: str) -> str:
    #         """
    #         Creates a hdf5 path from a string.
    #         """
    #         return r"datasets/"+rf"{os.path.join(self.path, self.filename)}" + r"/" + rf"{string}"

    #     keys2paths = {
    #         "CR2": create_path("Current2Retrace"),
    #         "CR": create_path("CurrentRetrace"),
    #         "CT": create_path("CurrentTrace"),
    #         "DR": create_path("DeflectionRetrace"),
    #         "HR": create_path("HeightRetrace"),
    #         "HT": create_path("HeightTrace"),
    #         "ZR": create_path("ZSensorRetrace"),
    #         "AT": create_path("AmplitudeTrace"),
    #         "AR": create_path("AmplitudeRetrace"),
    #         "PR": create_path("PhaseRetrace"),
    #         "PT": create_path("PhaseTrace"),
    #     }
        
    #     try:
    #         with h5py.File(self.opath, "r") as f:
    #             key_input = f"{key_input}" #TODO: Add space or not?
    #             key = find_key(CypherFile.keywords, key_input) + find_key(CypherFile.modes, key_input)

    #             print(key)

    #             print(keys2paths[key])

    #             return np.array(f[keys2paths[key]])
    #     except KeyError:
    #         print("Key not found in hdf5 file.")
    #         print(keys2paths[key])
    #         return





    # def ibw2hdf5(self,filename, filepath=None):
    #     #TODO: Look into all the metadata that can be extracted from the igor file.
    #     #Rewrite so that we have a universal reader for all file types directly...
    #     tmpdata = binarywave.load(filename)['wave']
    #     note = tmpdata['note']
    #     label_list = self.correct_label(tmpdata['labels'])

    #     fastsize = float(str(note).split('FastScanSize:')[-1].split('\\r')[0])
    #     slowsize = float(str(note).split('SlowScanSize:')[-1].split('\\r')[0])
    #     xoffset = float(str(note).split('XOffset:')[1].split('\\r')[0])
    #     yoffset = float(str(note).split('YOffset:')[1].split('\\r')[0])

    #     with h5py.File(filename.split('.')[0] + ".hdf5", "w") as f:
    #         f.require_group("process")
    #         metadatagrp = f.require_group("metadata")
    #         if filepath is not None:
    #             metadatagrp.create_dataset(filepath.split('.')[0], data=tmpdata['note'])
    #             datagrp = f.require_group("datasets/" + filepath.split('.')[0])
    #             datagrp.attrs.__setattr__('type', filepath.split('.')[-1])

    #         else:
    #             metadatagrp.create_dataset(filename.split('.')[0], data=tmpdata['note'])
    #             datagrp = f.require_group("datasets/" + filename.split('.')[0])
    #             datagrp.attrs.__setattr__('type', filename.split('.')[-1])

    #         for i, k in enumerate(label_list):
    #             if len(np.shape(tmpdata['wData'])) == 2:
    #                 datagrp.create_dataset(k, data=flipud(tmpdata['wData'][:, i].T))
    #                 datagrp[label_list[i]].attrs['shape'] = tmpdata['wData'][:, i].T.shape
    #                 datagrp[label_list[i]].attrs['scale_m_per_px'] = fastsize/tmpdata['wData'][:, i].T.shape[0]
    #             else:
    #                 datagrp.create_dataset(k, data=flipud(tmpdata['wData'][:, :, i].T))
    #                 datagrp[label_list[i]].attrs['shape'] = tmpdata['wData'][:, :, i].T.shape
    #                 datagrp[label_list[i]].attrs['scale_m_per_px'] = fastsize/tmpdata['wData'][:, :, i].T.shape[0]
    #             datagrp[label_list[i]].attrs['name'] = k.decode('utf8')
    #             datagrp[label_list[i]].attrs['size'] = (fastsize, slowsize)
    #             datagrp[label_list[i]].attrs['offset'] = (xoffset, yoffset)
    #             datagrp[label_list[i]].attrs['path'] = ("datasets/" + filename.split('.')[0]+"/"+str(k).split('\'')[1])


    #             if "Phase" in str(k):
    #                 datagrp[label_list[i]].attrs['unit'] = ('m', 'm', 'deg')
    #             elif "Amplitude" in str(k):
    #                 datagrp[label_list[i]].attrs['unit'] = ('m', 'm', 'V')
    #             elif "Height" in str(k):
    #                 datagrp[label_list[i]].attrs['unit'] = ('m', 'm', 'm')
    #             else:
    #                 datagrp[label_list[i]].attrs['unit'] = ('m', 'm', 'unknown')
    #         # f.create_dataset("channelsdata/pxs", data=sizes)

    #     print('file successfully converted')
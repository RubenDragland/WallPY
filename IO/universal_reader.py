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
        Believed to be a universal class for reading AFM scans in python.
        The key is to use Gwyddion to do an initial filtering of saved scans from experiments. 
        Next, use Gwyddion to export all files containing interesting data to the .gwy-file format.
        Unfortunatly, not all metadata is stored in the .gwy-file, and only reading of the essential metadata is currently implemented. 
        The .gwy-file is initially read by this class using the gwyfile package.
        CypherFile is a specialized class for .ibw files from the Cypher AFMs. Generally, use this class for Cypher files.
        Note that other essential metadata not recordable or extractable should be manually logged.

        Container for python-processing of data from a gwy-file.

        Attributes
        ----------

        path: str
            path to folder containing scans
        filename: str
            filename of scan
        **kwargs: dict, optional
            keyword arguments for all classes. The default is:
                
                {
                "opath": None,
                    output path for hdf5 file. If None, the path of the raw data is used.
                "oname": None,
                    output name for hdf5 file. If None, the name of the raw data is used.
                }
        
        Methods
        -------

        __call__:
            Utilises gwy-package to load datafields, 
            and transfers the data to a hdf5 file with the AFM scan and essential metadata easily accessible.
        get_by_key:
            Returns the data of a channel based on category and mode. 
            Returns a list if multiple channels are found.
        __getitem__:
            Returns the data of a channel based on its original index, which is the last number in the channel name, 
            and is considered the identification of an individual scan.  
        get_nth:
            Returns the data of the nth channel, meaning the data with index n in self.channel_names.
        get_dataset_keys:
            Returns all dataset keys in a hdf5 file.
        copy_2_other:
            Copies all channels to keep to a new hdf5 file.
        name_index_2_list_index:
            Converts the original index to the index of the channel list.
        index_metadata:
            Returns the metadata of a channel based on index; original index if name is True (default), else based on nth element in channel_names.
        


        
    """


    keywords = {
            "ampMFM": ["AmplitudeActual"],
            "Height": ["Height"],
            "Current": ["Current", "Iprobe"],
            "Deflection": ["DFL", "Deflection",],
            "ZSensor": ["ZSensor", "Z-Axis"],
            "Voltage": ["Voltage", "Ext1", "Peak Force Error"], #TODO: Know how to sort these. Include Channels etc. Remember to update both or remove gwyfile from classes. This one is the updated one. Ext1 and Iprobe have weird current amplifiers, which make phase not retrievable.
            "VPFM": ["VPFM", "Ext2"],
            "LPFM": ["LPFM", "Ext3"],
            "Amplitude": ["Amp", "Amplitude", "Mag", "AmplitudeError"],
            "Phase": ["Phase"], #TODO: Ambigious; many different phases going into the same. 
            "Potential": ["Surf", "Pot", "Potential"],
             #TODO: Check if these work. And check cypher. 
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

        kwargs = {
            "opath": None
            "oname": None,
        }
        """
        self.kwargs = {
            "opath": None,
            "oname": None,

        }

        for key, value in kwargs.items():
            if key in self.kwargs:
                self.kwargs[key] = value


        self.path = path
        self.filename = filename[:-4] if filename.endswith(".gwy") else filename
        self.fullpath = os.path.join(self.path, self.filename +".gwy") 
        opath = self.path if self.kwargs["opath"] is None else self.kwargs["opath"]
        oname = self.filename if self.kwargs["oname"] is None else self.kwargs["oname"]

        self.opath = os.path.join(opath, oname + ".hdf5")
        self.oname = oname

        try:

            with h5py.File(self.opath, "r") as f:
                self.channel_names = np.array(f["channel_names"], dtype=str) #list(np.array(f["channel_names"], dtype="S")) #TODO: Stored as bytes. Issue?
                # print(self.channel_names)
            print("H5 Structure exists.")
        except:

            self.channel_names = []
            print("Convert to hdf5.")
        finally:
            print("Channels found: ")
            for channel in self.channel_names:
                print(channel)
    
    def __call__(self):
        """
        Transforms the gwy-file to a hdf5-file that can be processed and more easily read.
        """

        obj = gwy.load(self.fullpath)
        channels = gwy.util.get_datafields(obj)


        self.channel_names = []
        print("Channels found: ")

        with h5py.File(self.opath, "w") as f:
            
            for id, key in enumerate(channels.keys()):
                
                category = str(find_key(GwyFile.keywords, key))
                mode = str(find_key(GwyFile.modes, key))
                unique = str(id) 

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
        Returns the data of a channel based on category and mode. 
        Returns a list if multiple channels are found.
        Must currently specify both category/keyword and mode.

        Parameters
        ----------
        key: str
            The keyword to search for.
        
        Returns
        -------
        datas: list
            A list of the found data.
        """
        datas = []

        assert len(self.channel_names)>0, "No channels found. Run __call__ first."

        with h5py.File(self.opath, "r") as f:

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
        

    def name_index_2_list_index(self, index: int) -> int:
        """
        Converts the original index to the index of the channel list.

        index: int
            The original index of the channel.

        Returns
        -------
        int
            The index of the channel in the list of channel names.
        """
        indices = [int(str(channel).split("_")[-1]) for channel in self.channel_names]
        return indices.index(index)
    
    def if_name(self, index:int, name: bool) -> int:
        """
        Returns the index of the channel in the list of channel names by translating the original index to the list index if name is True.

        index: int
            The original index of the channel.
        name: bool
            Whether provided index is based on the original names or the list of channel names.

        Returns
        -------
        int
            The index of the channel in the list of channel names.
        """
        assert len(self.channel_names) > 0, "No channels found. Run __call__ first."
        if name:
            index = int(str(index).split("_")[-1])
            index = self.name_index_2_list_index(index)
        
        return index

    
    def __getitem__(self, index: str) -> np.ndarray:
        """
        Returns the data of a channel based on original index.

        index: str/int
            The identification number of the desired channel.
        """

        index = self.if_name(index, name=True)
        name = self.channel_names[index]

        with h5py.File(self.opath, "r") as f:
            return np.array(f[name]["data"]) 
    
    def get_nth(self, n: int) -> np.ndarray:
        """
        Returns the data of the nth channel.

        n: int
            The index of the desired channel in the list of channels.
        """
        assert len(self.channel_names) > 0, "No channels found. Run __call__ first."
        with h5py.File(self.opath, "r") as f:
            return np.array(f[self.channel_names[n]]["data"])

    
    def get_dataset_keys(self):
        """
        Returns all dataset keys in a hdf5 file.
        """
        keys = []
        with h5py.File(self.opath, "r") as f:
            f.visit(lambda key: keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
        return keys
    

    def copy_2_other(self, new_name: str, keep: list, name=True, **kwargs):
        """

        Copies all channels to keep to a new hdf5 file.


        Parameters
        ----------
        new_name: str
            The name of the new hdf5 file.
        keep: str
            The list containing names/indices of the channel to keep.
        name: bool
            If True, the indices are based on the original names. If False, the indices are based on the list of channel
        **kwargs: dict
            opath: str
                The path to save the new file. If None, the path of the original file is used.
        
        Returns
        -------
        other: GwyFile
            A new GwyFile instance with the copied channels.
        """

        def_kwargs = {
            "opath": None,
        }

        for key, value in def_kwargs.items(): #TODO: Possibly the better way of doing this. 
            if key not in kwargs.keys():
                kwargs[key] = value
        
        # Creates new path and name for the new file.
        new_path = self.path if kwargs['opath'] is None else kwargs["opath"]
        new_path = new_path[:-5] if new_path.endswith(".hdf5") else new_path 
        new_opath = os.path.join(new_path, new_name + ".hdf5")

        # Translates the indices to the list of channel names.
        if name:
            # keep = [int(str(index).split("_")[-1]) for index in keep]
            # keep = [self.name_index_2_list_index(index) for index in keep]
            keep = [self.if_name(index, name=name) for index in keep]

        keep = [self.channel_names[index] for index in keep]

        # Copies the channels to the new file, and initializes a new instance. 
        with h5py.File(new_opath, "w") as f:

            for channel in keep:
                f.create_group(channel)
                f[channel].create_dataset("data", data=self[channel])
                f[channel].attrs["title"] = str(channel)
                f[channel].attrs["category"] = channel.split("_")[0]
                f[channel].attrs["mode"] = channel.split("_")[1]
                f[channel].attrs["unit"] =  self.index_metadata(index= channel, feature= "unit", name=True )
                f[channel].attrs["xsize"] = self.index_metadata(index=channel, feature="xsize", name=True) 
                f[channel].attrs["ysize"] = self.index_metadata(index=channel, feature="ysize", name=True) 
                f[channel].attrs["xres"]  = self.index_metadata(index=channel, feature="xres", name=True)  
            
            f.create_dataset("channel_names", data=np.array(keep, dtype="S") )

        other = GwyFile(path=self.path, filename=self.filename, opath=new_path, oname=new_name)
        other.channel_names = keep
        return other
    
    def index_metadata(self, index: str, feature=None, name=True):
        """
        Returns the metadata of a channel based on index.

        index: str
            The index of the desired channel.
        feature: str, optional
            The specific metadata to return. The default is None, which returns all metadata.
        name: bool, optional
            If True, the index is based on the original names. If False, the index is based on the list of channel names.
        """
        assert len(self.channel_names) > 0, "No channels found. Run __call__ first."

        # Translates the index to the list of channel names.
        if name:
            # index = int(str(index).split("_")[-1])
            # index = self.name_index_2_list_index(index)  #TODO: Make common function for this.     
            index = self.if_name(index, name=name)

        ch_name = self.channel_names[index]

        #Returns either all items, or a single feature. 
        if feature is None:
            with h5py.File(self.opath, "r") as f:
                return f[ch_name].attrs.items()
        else:
            with h5py.File(self.opath, "r") as f:
                return f[ch_name].attrs[feature]
    
    def remove_redundant(self, keep: list, name=True):
        """
        Removes all channels except those in the keep list.
        Keep list with name=True means that the number included refers to the originally assigned index, and/or that a whole name is written. 
        Note that the original file size is not reduced, but the removed channels are deleted, and the channel_names list is updated.

        keep: list
            The list of channels to keep. Either indices (original or current), or entire channel names.
        name: bool, optional
            If True, the indices are based on the original names. If False, the indices are based on the list of channel names.

        Returns
        -------
        None
        """

        assert len(self.channel_names) > 0, "No channels found. Run __call__ first."

        if name:
            # keep = [int(str(index).split("_")[-1]) for index in keep]
            # keep = [self.name_index_2_list_index(index) for index in keep]
            keep = [self.if_name(index, name=name) for index in keep]
        
        keep = [self.channel_names[index] for index in keep] 
        
        with h5py.File(self.opath, "r+") as f:
            for channel in self.channel_names:
                if channel not in keep:
                    del f[channel]
            del f["channel_names"]
            f.create_dataset("channel_names", data=np.array(keep, dtype="S") )
            self.channel_names = keep

        return
    
    def copy_n_remove(self, new_name: str, keep: list, name=True, **kwargs):
        """
        Copies the channels in keep to a new file and removes the original file. 

        Parameters
        ----------
        new_name: str
            The name of the new file.
        keep: list
            The list of channels to keep.
        name: bool, optional
            If True, the indices are based on the original names. If False, the indices are based on the list of channel names.
        **kwargs: dict
            opath: str
                The path to save the new file. If None, the path of the original file is used.
            oname: str
                The name of the new file. If None, the name of the original file is used.
        
        Returns
        -------
        new: GwyFile
            A new GwyFile instance with the copied channels
        """

        assert os.path.exists(self.opath), "Original file not found."

        new = self.copy_2_other(new_name, keep, name, **kwargs)
        with h5py.File(self.opath, "r+") as f:
            del f
        try:
            assert os.path.exists(self.opath) == False, "Original file not removed."
        except:
            os.remove(self.opath)
        finally:
            return new
    

    def create_overview_file(self, quantile: float = 0.69)->None:
        """
        Creates an overview file of the channels in the hdf5 file.
        The overview file is saved in the same folder as the hdf5 file.
        The overview file is named after the original file with "_overview" appended, as png.

        Parameters
        ----------
        quantile: float, optional
            The quantile of the histogram to use for the colorbar. The default is 0.69.
            The higher the quantile, the more saturated the colors will be.
        
        Returns
        -------
        None
        """

        fig, axes = plt.subplots(1, len(self.channel_names), figsize=(len(self.channel_names)*5, 5))

        for ax, channel in zip(axes.reshape(-1), self.channel_names):
            ind = int(channel.split("_")[-1])
            data = self[ind]

            #TODO: Consider using kde instead to automatically account for data size via Scott's rule. 
            hist, edges = np.histogram(data, bins=100)

            # Find peak value
            pvalue = np.max(hist)
            peak = edges[np.argmax(hist)]

            # Find lower and upper bounds
            lower = edges[np.where(hist > pvalue*quantile)[0][0]]
            upper = edges[np.where(hist > pvalue*quantile)[0][-1]]

            vmin = lower
            vmax = upper


            ax.imshow(data, cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_title(channel)
        
        savepath = self.path if self.kwargs["opath"] is None else self.kwargs["opath"]
        savename = self.filename
        plt.savefig(os.path.join(savepath, savename + "_overview.png"))
        plt.show()
        return
    
    def save_processed(self, index:str, data:np.ndarray, name=True)->None:
        """
        Saves processed data to the hdf5 file.

        Parameters
        ----------
        index: str
            The index of the channel to save the data to.
        data: np.ndarray
            The processed data to save.
        name: bool, optional
            If True, the index is based on the original names. If False, the index is based on the list of channel names.

        Returns
        -------
        None
        """
        if name:
            # index = int(str(index).split("_")[-1])
            # index = self.name_index_2_list_index(index)
            index = self.if_name(index, name=name)
        channel = self.channel_names[index]

        with h5py.File(self.opath, "r+") as f:
            f[channel].create_dataset("processed", data=data)
            #TODO: Add more metadata?
        return
            


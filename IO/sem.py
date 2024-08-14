
from typing import Any
import numpy as np
import h5py
import os
from glob import glob
import tifffile
import hyperspy.api as hs

def tif_init_setup(obj, kwargs: dict, extension: str):
    """
    Shared function for tif files. 
    Finds full path and output path for the object in a folder.

    Parameters
    ----------

    obj: object
    kwargs: dictionary
    extension: str

    Returns
    -------
    None
    """

    for key, value in kwargs.items():
        if key in obj.kwargs:
            obj.kwargs[key] = value


    possible_files = glob(os.path.join(obj.path, obj.filename+"*"))
    possible_files = [file for file in possible_files if file.endswith(".tif") or file.endswith(".tiff") ]

    assert len(possible_files) == 1, "Multiple files found. Please specify."
    obj.fullpath = possible_files[0]

    obj.opath = os.path.join(obj.path, obj.filename  + extension) if obj.kwargs["opath"] is None else os.path.join(obj.kwargs["opath"], obj.filename  + extension)

    return



class Tif2Hspy:
    """
    Class for handling SEM images using hyperspy

    Attributes
    ----------
    path : str
        path to folder
    filename : str
        filename of scan
    kwargs : dict
        keyword arguments for the class
            opath : str
                output path for the hyperspy file
    
    Methods
    -------
    __call__()
        Loads the tif file and saves it as a hyperspy file.
    __getitem__(key:str)
        Returns the hyperspy object

    """

    def __init__(self, path:str, filename:str, **kwargs):
        """
        Parameters
        ----------

        path: str
            path to folder
        filename: str
            filename of scan

        kwargs: dict
            keyword arguments for the class
                opath : str, optional
                    output path for the hyperspy file. Default is None.
        """
        self.kwargs = {
            "opath": None,

        }

        self.path = path
        self.filename = filename

        tif_init_setup(self, kwargs, ".hspy")

        return
    
    def __call__(self):
        """
        Loads the tif file and saves it as a hyperspy file.
        """

        img = hs.load(self.fullpath)
        img.save(self.opath)

        meta = img.metadata
        self.x_dim = img.axes_manager["width"].size #meta["Signal"].axes_manager[0].size
        self.x_res = img.axes_manager["width"].scale #meta["Signal"].axes_manager[0].scale #TODO: Check if correct.

        return
    
    def __getitem__(self, key: str = '') -> np.ndarray:
        """
        Returns the data from hyperspy, possibly with metadata.

        Parameters
        ----------
        key : str
            The key to return. Default is "".
        
        Returns
        -------
        np.ndarray
            The hyperspy data possibly with metadata.
        """


        h = hs.load(self.opath)
        if key == "all":
            return (h.data, h.metadata) #TODO: Keep this?
        else:
            return h.data




class Tif2H5:
    """
    Class for handling SEM images using tiffile and hdf5

    Attributes
    ----------
    path : str
        path to folder
    filename : str
        filename of scan
    kwargs : dict
        keyword arguments for the class
            opath : str
                output path for the hdf5 file
    
    Methods
    -------
    __call__(index:int)
        Creates the hdf5 file for data analysis.
    __getitem__(index:int)
        Returns the data from the hdf5 file.

    """


    def __init__(self,path:str, filename:str, **kwargs):
        """
        Parameters
        ----------
        path : str
            path to folder
        filename : str
            filename of scan
        
        kwargs : dict
            keyword arguments for the class
                opath : str, optional
                    output path for the hdf5 file. Default is None.
        """

        self.kwargs = {
            "opath": None,
        }

        self.path = path
        self.filename = filename

        tif_init_setup(self, kwargs, ".hdf5")

        return
    
    def __call__(self, index = 0):
        """
        Creates the hdf5 file for data analysis.

        Parameters
        ----------
        index : int, optional
            The index of the file. Default is 0.
        """

        img = tifffile.imread(self.fullpath)
        meta = tifffile.TiffFile(self.fullpath) #TODO: Currently not working. 

        print(meta)

        self.x_dim = img.shape[1]
        self.y_dim = img.shape[0]

        self.x_res = meta.fei_metadata['EScan']['PixelWidth']
        self.y_res = meta.fei_metadata['EScan']['PixelHeight']

        self.name = str(index).zfill(2)

        #TODO: Save to hdf5 file but necessary?
        with h5py.File(self.opath, "w") as f:
            
            f.create_group(self.name)
            f[self.name].create_dataset("data", data=img)
            # f.attrs["title"] = name
            # f.attrs["category"] = category
            # f.attrs["mode"] = mode
            # f.attrs["unit"] = channels[key].si_unit_z.unitstr
            f[self.name].attrs["name"] = self.name
            f[self.name].attrs["xsize"] = self.x_dim
            f[self.name].attrs["ysize"] = self.y_dim
            f[self.name].attrs["xres"] = self.x_res

            # f.create_dataset("channel_names", data=np.array(self.channel_names, dtype="S") ) TODO: In batch


        return 
    
    def __getitem__(self, index: int = 0 ) -> np.ndarray:

        with h5py.File(self.opath, "r") as f:
            return np.array(f[str(index).zfill(2)]["data"])
            

class BatchTif2H5(Tif2H5):
    """
    Gathers all tif files in a folder and saves them to a single hdf5 file.

    Attributes
    ----------

    path : str
        path to folder
    filename : str
        filename of scan
    kwargs : dict
        keyword arguments for the class
            opath : str
                output path for the hdf5 file
            oname : str
                name of the hdf5 file
            existing : bool
                whether the file already exists
    
    Methods
    -------
    __call__()
        Creates the hdf5 file for data analysis.
    get_meta(index:int)
        Returns the metadata of the hdf5 file.
    
    """

    def __init__(self, path:str, filename:str, **kwargs):
        """
        Parameters
        ----------
        path : str
            path to folder
        filename : str
            filename of scan
        
        kwargs : dict
            keyword arguments for the class
                opath : str, optional
                    output path for the hdf5 file. Default is None.
                oname : str, optional
                    name of the hdf5 file. Default is "SEM_batch".
                existing : bool, optional
                    whether the file already exists. Default is False.
        """

        self.kwargs = {
            "opath": None,
            "oname": "SEM_batch",
            "existing": False,

        }

        self.path = path
        self.filename = filename

        for key, value in kwargs.items():
            if key in self.kwargs:
                self.kwargs[key] = value

        if self.kwargs["existing"]:
            self.opath = os.path.join(self.path, self.kwargs["oname"]  + ".hdf5") if self.kwargs["opath"] is None else os.path.join(self.kwargs["opath"], self.kwargs["oname"]  + ".hdf5")
        else:
            possible_files = glob(os.path.join(self.path, self.filename+"*"))

            for file in possible_files:
                if not(file.endswith(".tif")) or not(file.endswith(".tiff")):
                    possible_files.remove(file)

            assert len(possible_files) > 0, "No tif/tiff files found."

            self.fullpaths = possible_files
            self.opath = os.path.join(self.path, self.kwargs["oname"]  + ".hdf5") if self.kwargs["opath"] is None else os.path.join(self.kwargs["opath"], self.kwargs["oname"]  + ".hdf5")

        return


    def __call__(self):
        """
        Creates the hdf5 file for data analysis.
        """

        with h5py.File(self.opath, "w") as f: #TODO: Option for adding and editing files?
            self.channel_names = []
            for index, file in enumerate(self.fullpaths):
                img = tifffile.imread(file)
                meta = tifffile.tiffFile(file) #TODO: Currently not working. 

                name = str(index).zfill(2)
                self.channel_names.append(name)

                f.create_group(name)
                f[name].create_dataset("data", data=img)
                f[name].attrs["name"] = name
                f[name].attrs["xsize"] = img.shape[1]
                f[name].attrs["ysize"] = img.shape[0]
                f[name].attrs["xres"] = meta.fei_metadata['EScan']['PixelWidth']

            f.create_dataset("channel_names", data=np.array(self.channel_names, dtype="S") )
        return

    def get_meta(self, index: int = 0) -> dict:
        """
        Returns the metadata of the hdf5 file.
        TODO: Something like this. 
        """
        name = str(index).zfill(2)
        with h5py.File(self.opath, "r") as f:

            meta = {
                "xsize": f[name].attrs["xsize"],
                "ysize": f[name].attrs["ysize"],
                "xres": f[name].attrs["xres"],
            }
        return meta


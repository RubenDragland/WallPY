

import numpy as np
import hystorian as hy
import h5py

import os
# Use glob or os.listdir to get a list of files in a directory. (glob is better, but os.listdir was used in the master's thesis)
from glob import glob


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
    """

    hy.io.read_file.merge_hdf5(self.scans, self.filename)

    return


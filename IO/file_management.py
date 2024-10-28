import os
import sys
import numpy as np
import glob

def find_files_by_extension(path:str, extension:str) -> list:
    """
    Find all files with a specific extension in a folder.
    
    Parameters
    ----------
    path : str
        The path to the folder.
    extension : str
        The extension of the files to find.
    
    Returns
    -------
    list
        The list of files.
    """
    full_name = glob.glob(os.path.join(path, f"*{extension}"))
    filtered_name = [os.path.basename(name) for name in full_name]
    filtered_name = [name[:-len(extension)] for name in filtered_name]
    return filtered_name

def create_oname(filename:str, add_on_name:str)->str:
    """
    Create a new filename by adding an add-on name to the original filename.
    
    Parameters
    ----------
    filename : str
        The original filename.
    add_on_name : str
        The add-on name.
    
    Returns
    -------
    str
        The new filename.
    """
    return filename + "_" + add_on_name

def create_onames(filenames:list, add_on_names:list)->list:
    """
    Create a list of new filenames by adding an add-on name to the original filenames.
    
    Parameters
    ----------
    filenames : list
        The list of original filenames.
    add_on_name : list
        The add-on names.
    
    Returns
    -------
    list
        The list of new filenames.
    """
    return list(map(create_oname, filenames, add_on_names))
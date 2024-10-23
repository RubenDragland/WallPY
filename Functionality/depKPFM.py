import numpy as np
import os
import sys

import scipy.ndimage as ndi
import scipy.stats as stats
import scipy.optimize as opt
import scipy as sp

import skimage.filters

#TODO: Consider line streak artefact correction.

wallpy_cmd = r"C:\Users\rubensd\OneDrive - NTNU\PhD\Analysis\WallPY"
sys.path.append(os.path.join(wallpy_cmd, "Display"))
from figures import FigureSinglePlot, FigureSubplots
import rcParams
import matplotlib.pyplot as plt


def align_profiles(data_raw: np.ndarray, search_value=None, x_plus_minus=50)->np.ndarray:
    """
    Aligns the line profiles of a KPFM image around a given search value.

    Parameters
    ----------
    data_raw : np.array
        The raw data.
    search_value : float, optional
        The value to search for in the data. The default is None, which means that the mean of the data is used.
    x_plus_minus : int, optional
        The number of pixels to include before and after the search value. The default is 50.

    Returns
    -------
    data_aligned : np.array
        The aligned data.
    """

    search_value = np.mean(data_raw) if search_value is None else search_value

    aligned_data = np.zeros((data_raw.shape[0], 2*x_plus_minus))

    for i in range(data_raw.shape[0]):
        ind = np.argmin(np.abs(data_raw[i] - search_value))
        aligned_data[i] = data_raw[i, ind - x_plus_minus:ind + x_plus_minus]

    return aligned_data


def group_profiles(data: np.ndarray, width=10)->np.ndarray:
    """
    Merge line profiles of a KPFM image into groups of a given width.

    Parameters
    ----------
    data : np.array
        The data to group.
    width : int, optional
        The width of the floating average. The default is 10.

    Returns
    -------
    data_grouped : np.array
        The grouped data.    
    centers : np.array
        The centers of the groups.
    """

    data_grouped = np.zeros((data.shape[0]//width, data.shape[1]))
    centers = np.arange(data.shape[0]//width)*width + width//2

    # NOTE: Out of index error? 
    for i in range(data_grouped.shape[0]):
        data_grouped[i] = np.mean(data[i*width:(i+1)*width], axis=0)


    return data_grouped, centers

def filter_profile(y: np.array)->np.array:
    """
    Filters the line profiles of a KPFM image.

    Parameters
    ----------
    y : np.array
        The y-axis of the line profiles.

    Returns
    -------
    y_filtered : np.array
        The filtered line profiles.
    """

    y_filtered = np.zeros_like(y)

    y_filtered = sp.signal.medfilt(y, kernel_size=9)
    y_filtered = sp.signal.savgol_filter(y_filtered, window_length=11, polyorder=2)

    return y_filtered


def find_depletion_region(x:np.ndarray, y:np.ndarray, high_low = False )->tuple:
    """
    Automatically finds the depletion region in a KPFM line profile.

    Parameters
    ----------
    x : np.array
        The x-axis of the line profile.
    y : np.array
        The y-axis of the line profile.
    high_low : bool, optional
        Whether to invert the jerk profile. The default is False, and is used when the potential goes from low to high.

    Returns
    -------
    W : float
        The width of the depletion region.
    dep_edge1 : float
        The position of the first edge of the depletion region.
    dep_edge2 : float
        The position of the second edge of the depletion region.
    orig_ind1 : int
        The index of the first edge in the original data.
    orig_ind2 : int
        The index of the second edge in the original data.
    """

    jerk = np.gradient(np.gradient(np.gradient(y, x, edge_order=2), x, edge_order=2), x, edge_order=2)

    jerk = -jerk if high_low else jerk

    jerk_peak_indices = sp.signal.find_peaks(jerk)[0]
    jerk_peak_values = jerk[jerk_peak_indices]
    x_peaks = x[jerk_peak_indices]

    sorted_indices = np.argsort(jerk_peak_values)
    jerk_peak_values = jerk_peak_values[sorted_indices]
    x_peaks = x_peaks[sorted_indices]

    dep_edge1 = x_peaks[-1]
    dep_edge2 = x_peaks[-2]

    W = np.abs(dep_edge1 - dep_edge2)

    orig_ind1 = np.argmin(np.abs(x - dep_edge1))
    orig_ind2 = np.argmin(np.abs(x - dep_edge2))

    return W, dep_edge1, dep_edge2, orig_ind1, orig_ind2, jerk


class KPFM_interface:

    def __init__(self, raw_data: np.ndarray, xres: float, search_value: float, x_plus_minus=25, group_width=15, high_low=False):
        self.raw_data = raw_data
        self.xres = xres
        self.search_value = search_value
        self.x_plus_minus = x_plus_minus
        self.group_width = group_width
        self.high_low = high_low

        self.aligned_data = np.zeros((self.raw_data.shape[0], 2*self.x_plus_minus))
        self.x = np.arange(self.aligned_data.shape[1])*self.xres # SI units
        self.grouped_data = np.zeros((self.raw_data.shape[0]//self.group_width, 2*self.x_plus_minus))
        self.filtered_data = np.zeros_like(self.grouped_data)

        self.W = np.zeros(self.grouped_data.shape[0])
        self.dep_edge1 = np.zeros_like(self.W)
        self.dep_edge2 = np.zeros_like(self.W)
        self.orig_ind1 = np.zeros_like(self.W)
        self.orig_ind2 = np.zeros_like(self.W)
        self.jerk = np.zeros_like(self.filtered_data)
    
    def append(self, find_depletion_output: tuple, i: int):
        self.W[i], self.dep_edge1[i], self.dep_edge2[i], self.orig_ind1[i], self.orig_ind2[i], self.jerk[i] = find_depletion_output

    def get_width(self):
        return np.mean(self.W), np.std(self.W)


def process_kpfm_roi(data: np.ndarray, xres, search_value, x_plus_minus=25, group_width=15, high_low=False)->KPFM_interface:
    """
    Aligns, groups, filters, and finds the depletion region in a KPFM region of interest.

    Parameters
    ----------
    data : np.array
        The raw data.
    xres : float
        The resolution of the x-axis.
    search_value : float
        The value to search for in the data.
    x_plus_minus : int, optional
        The number of pixels to include before and after the search value. The default is 25.
    group_width : int, optional
        The width of the floating average. The default is 15.
    high_low : bool, optional
        Whether to invert the jerk profile. The default is False, and is used when the potential goes from low to high.

    Returns
    -------
    data_aligned : np.array
    """

    kpfm_results = KPFM_interface(data, xres, search_value, x_plus_minus, group_width, high_low)

    kpfm_results.aligned_data = align_profiles(data, search_value=search_value, x_plus_minus=x_plus_minus)
    kpfm_results.grouped_data, kpfm_results.centers = group_profiles(kpfm_results.aligned_data, group_width)
    kpfm_results.centers = kpfm_results.centers*xres

    print(kpfm_results.grouped_data.shape)
    print(kpfm_results.filtered_data.shape)

    for i in range(kpfm_results.grouped_data.shape[0]):

        
        kpfm_results.filtered_data[i] = filter_profile(kpfm_results.grouped_data[i])

        out= find_depletion_region(kpfm_results.x, kpfm_results.filtered_data[i], high_low)

        kpfm_results.append(out, i)


    return kpfm_results



def find_sep_distance(data, line, x, ):
    """
    Automatically finds the separation distance between two domains in a KPFM line profile.

    Parameters
    ----------
    data : np.array
        The image data.
    line : np.array
        The line profile.
    x : np.array
        The x-axis of the line profile.

    Returns
    -------
    W : float
        The separation distance between the two domains.
    sep_edge1 : float
        The position of the first edge of the separation region.
    sep_edge2 : float
        The position of the second edge of the separation region.
    peaks : tuple
        The indices of the peaks in the gradient of the line profile.
    """

    thresh = skimage.filters.threshold_otsu(data)
    binary = (line > thresh).astype(int)

    edges = np.abs(np.gradient(binary, x, edge_order=2))
    edges = edges / np.max(edges) * np.isclose(edges, 1, atol=0.1)

    peaks = sp.signal.find_peaks(edges)[0]
    sep_edge1 = x[peaks[0]]
    sep_edge2 = x[peaks[1]]

    W = np.abs(sep_edge1 - sep_edge2)

    return W, sep_edge1, sep_edge2, peaks[0], peaks[1]
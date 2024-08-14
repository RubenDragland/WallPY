import numpy as np
import os
import sys

import scipy.ndimage as ndi
import scipy.stats as stats
import scipy.optimize as opt
import scipy as sp

import skimage.filters


wallpy_cmd = r"C:\Users\rubensd\OneDrive - NTNU\PhD\Analysis\WallPY"
sys.path.append(os.path.join(wallpy_cmd, "Display"))
from figures import FigureSinglePlot, FigureSubplots
import rcParams
import matplotlib.pyplot as plt


def find_depletion_region(x:np.ndarray, y:np.ndarray )->tuple:
    """
    Automatically finds the depletion region in a KPFM line profile.

    Parameters
    ----------
    x : np.array
        The x-axis of the line profile.
    y : np.array
        The y-axis of the line profile.

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

    return W, dep_edge1, dep_edge2, orig_ind1, orig_ind2

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
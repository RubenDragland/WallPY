import numpy as np
import json
import matplotlib.pyplot as plt
import skimage.measure as skm
import scipy as sp
import os
from tqdm import tqdm, trange



def filter_savoy_crossections(Y, window_length=15, polyorder=3, mode="interp", **kwargs ):

    for i, y in enumerate(Y):
        Y[i] = sp.signal.savgol_filter(y, window_length=window_length, polyorder=polyorder, mode=mode)
    
    return Y

def bisection_prominence_peak_detection(Y, savoy_wl=10, min_p=0.1, max_p=1, revs=10, atol=10, rtol=0.01, savoy_wl_max=25, **kwargs):
    #TODO: Find peaks in all crossections, and assure that the distance between peaks is not too great
    #TODO: Maybe do grid search if not all peaks are found.
    decrease_prominence = lambda p, lp: p - (p - lp) / 2
    increase_prominence = lambda p, hp: p + (hp - p) / 2

    prominence = increase_prominence(min_p, max_p)

    Y_filtered = filter_savoy_crossections(Y, window_length=savoy_wl, **kwargs)

    peaks = [sp.signal.find_peaks(y, prominence=prominence)[0] for y in Y_filtered]

    # for rev in trange(revs): 
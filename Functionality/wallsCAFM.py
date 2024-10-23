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

    for rev in trange(revs):








































# def filter_savoy_peak_prominence(y,**kwargs ):
#     #TODO: Add docstring
#     #TODO: Change from kwargs to args?

#     window_length = 5 if "window_length" not in kwargs.keys() else kwargs["window_length"]
#     polyorder = 2 if "polyorder" not in kwargs.keys() else kwargs["polyorder"]
#     mode = "interp" if "mode" not in kwargs.keys() else kwargs["mode"]
#     prominence_factor = 0.3 if "prominence_factor" not in kwargs.keys() else kwargs["prominence_factor"]
#     prominence = np.ptp(y)*prominence_factor

#     # print(window_length, polyorder, mode, prominence_factor)


#     y_filtered = sp.signal.savgol_filter(y, window_length=window_length, polyorder=polyorder, mode=mode)
#     peaks, peak_dict = sp.signal.find_peaks(y_filtered, prominence=prominence, )

#     return y_filtered, peaks, peak_dict




# def iteratively_retrieve_peaks(y, max_revs=10, min_window_length=5, max_window_length=25, min_prominence=0.01, max_prominence=1, w_tol=4, p_tol =0.05, **kwargs):
#     #TODO: Add docstring

#     calc_rel_prominence = lambda p: 1 - p - min_prominence / (max_prominence - min_prominence)
#     calc_rel_window_length = lambda w: (w - min_window_length) / (max_window_length - min_window_length)

#     decrease_prominence = lambda p, lp: p - (p - lp) / 2
#     increase_prominence = lambda p, hp: p + (hp - p) / 2

#     increase_window_length = lambda w, wh: int(w + (wh - w) / 2)
#     decrease_window_length = lambda w, wl: int(w - (w - wl) / 2)

#     #NOTE: Want min filtering, max prominence.

#     kwargs["window_length"] = increase_window_length(min_window_length, max_window_length)
#     kwargs["prominence_factor"] = decrease_prominence(max_prominence, min_prominence)

#     high_window = max_window_length
#     low_window = min_window_length

#     high_prominence = max_prominence
#     low_prominence = min_prominence

#     y_filtered, peaks, peak_dict = filter_savoy_peak_prominence(y, **kwargs)

#     print(len(peaks), kwargs["window_length"], kwargs["prominence_factor"], high_window, low_window, high_prominence, low_prominence)
#     w_last = True

#     for rev in trange(max_revs):

#         # if rev == 0 and len(peaks) == 1:
#             # return y_filtered, peaks, peak_dict
#         if len(peaks) == 1:

#             low_prominence = kwargs["prominence_factor"]
#             high_window = kwargs["window_length"]

#             kwargs["prominence_factor"] = increase_prominence(kwargs["prominence_factor"], high_prominence)
#             kwargs["window_length"] = decrease_window_length(kwargs["window_length"], low_window)
#             w_last = not w_last
            
#         elif len(peaks) > 1:
#             # rel_prominence = calc_rel_prominence(kwargs["prominence_factor"])
#             # rel_window_length = calc_rel_window_length(kwargs["window_length"])

#             # if rel_prominence < rel_window_length or w_last:
#             if w_last:
#                 low_prominence = kwargs["prominence_factor"]
#                 w_last = not w_last
#                 # low_window = min_window_length
#                 # high_window = max_window_length
#                 kwargs["prominence_factor"] = increase_prominence(kwargs["prominence_factor"], high_prominence)
                
#             else:
#                 low_window = kwargs["window_length"]
#                 w_last = not w_last
#                 # high_prominence = max_prominence
#                 # low_prominence = min_prominence
#                 kwargs["window_length"] = increase_window_length(kwargs["window_length"], high_window)
                
#         else:
#             # rel_prominence = calc_rel_prominence(kwargs["prominence_factor"])
#             # rel_window_length = calc_rel_window_length(kwargs["window_length"])
            
#             # if rel_prominence < rel_window_length:
#             if w_last:
#                 high_prominence = kwargs["prominence_factor"]
#                 w_last = not w_last
#                 # high_window = max_window_length
#                 # low_window = min_window_length
#                 kwargs["prominence_factor"] = decrease_prominence(kwargs["prominence_factor"], low_prominence)
#                 # print(kwargs["prominence_factor"])
#             else:
#                 high_window = kwargs["window_length"]
#                 w_last = not w_last
#                 # low_prominence = min_prominence
#                 # high_prominence = max_prominence
#                 kwargs["window_length"] = decrease_window_length(kwargs["window_length"], low_window)
#                 # print(kwargs["window_length"])
        
#         y_filtered, peaks, peak_dict = filter_savoy_peak_prominence(y, **kwargs)

#         print(len(peaks), kwargs["window_length"], kwargs["prominence_factor"], high_window, low_window, high_prominence, low_prominence)

#         # if len(peaks) == 1:
#         #     if np.abs(kwargs["window_length"] - low_window) <  w_tol:
#         #         if np.abs(kwargs["prominence_factor"] - low_prominence) < p_tol:
#         #             return y_filtered, peaks, peak_dict, kwargs["window_length"], kwargs["prominence_factor"]
                
#     return y_filtered, peaks, peak_dict, kwargs["window_length"], kwargs["prominence_factor"]#
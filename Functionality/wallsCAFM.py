import numpy as np
import json
import matplotlib.pyplot as plt
import skimage.measure as skm
import skimage.filters as skf
import scipy as sp
import os
from tqdm import tqdm, trange
import copy



def filter_savoy_crossections(Y, window_length=15, polyorder=3, mode="interp", **kwargs ):

    for i, y in enumerate(Y):
        Y[i] = sp.signal.savgol_filter(y, window_length=window_length, polyorder=polyorder, mode=mode)
    
    return Y

def bisection_prominence_peak_detection(Y, savoy_wl=10, min_p=0.1, max_p=1, revs=10, atol=10, rtol=0.01, savoy_wl_max=25, **kwargs):
    #TODO: Find peaks in all crossections, and assure that the distance between peaks is not too great
    #TODO: Maybe do grid search if not all peaks are found.
    #TODO: Perhaps update min_p and max_p?
    #NOTE: Vulnerable to noise...
    #NOTE: Could work with clear contrast
    decrease_prominence = lambda p, lp: p - (p - lp) / 2
    increase_prominence = lambda p, hp: p + (hp - p) / 2

    bool_found_peaks = lambda p: not np.any([len(peak)==0 for peak in p])
    count_ones = lambda p: np.sum([len(peak)==1 for peak in p])
    count_peaks = lambda p: np.sum([len(peak) for peak in p])

    def assess_displacement(peaks, atol):
        for i, p in enumerate(peaks):
            if np.abs( p- peaks[i-1]) > atol:
                return False
    
    def manual_completion(peaks, atol):
        single_peak_i = np.array([ i for (i, peak) in enumerate(peaks) if len(peak)==1])
        single_peak_p = np.array([ peak[0] for peak in peaks if len(peak)==1])
        for i, p in enumerate(peaks):
            if len(p) > 1:
                closest_single = np.argmin(np.abs(i - single_peak_i))
                peak_diff = np.abs(p - single_peak_p[closest_single])
                tol = np.min(peak_diff )
                if tol < atol:
                    chosen_peak_i = np.argmin(peak_diff)
                    peaks[i] = [p[chosen_peak_i]]
                else:
                    peaks[i] = [] #TODO: Come up with something
        return peaks
    
    Y_filtered = filter_savoy_crossections(Y, window_length=savoy_wl, **kwargs)
    min_p *= np.ptp(Y_filtered)
    max_p *= np.ptp(Y_filtered)
    prominence = increase_prominence(min_p, max_p)


    plt.show()
    plt.imshow(Y_filtered)
    plt.show()

    [plt.plot(y) for y in Y_filtered]
    plt.show()

    peaks = [sp.signal.find_peaks(y, prominence=prominence)[0] for y in Y_filtered]

    filled_bool = bool_found_peaks(peaks)
    peaks_ones = count_ones(peaks)
    peaks_count = count_peaks(peaks)
    length = len(peaks)

    best_result = copy.deepcopy(peaks)
    best_count = peaks_count
    best_ones = peaks_ones
    print(prominence)
    print(filled_bool)
    print(peaks_ones)



    for rev in trange(revs): 

        if peaks_ones == length:

            if assess_displacement(peaks, atol):
                return best_result

        elif not filled_bool:

            prominence = decrease_prominence(prominence, min_p)
            peaks = [sp.signal.find_peaks(y, prominence=prominence)[0] for y in Y_filtered]

        # elif peaks_count > length:
        else:
            prominence = increase_prominence(prominence, max_p)
            peaks = [sp.signal.find_peaks(y, prominence=prominence)[0] for y in Y_filtered]

        filled_bool = bool_found_peaks(peaks)
        peaks_ones = count_ones(peaks)
        peaks_count = count_peaks(peaks)

        print(prominence)
        print(filled_bool)
        print(peaks_ones)
        print(peaks)

        if (peaks_count < best_count and peaks_ones > best_ones):
            best_result = copy.deepcopy(peaks)
            best_count = peaks_count
            best_ones = peaks_ones
    
    if assess_displacement(peaks, atol):
        return best_result
    elif filled_bool:
        peaks = manual_completion(peaks, atol)
        return peaks


        

#TODO: Function that binerize into domains (Becayse slight domain contrast)

def retrieve_walls_from_domains(Y, ):
    #TODO: Filter or not to filter?
    #TODO: Either give image, or give crossections. Also, either make KDE or just let the function do its thing

    Y = filter_savoy_crossections(Y, window_length=15, polyorder=3, mode="interp")
    binerized = np.zeros_like(Y)
    for i, y in enumerate(Y):
        thresh = skf.threshold_otsu(y)
        binerized[i] = (y > thresh).astype(float)
    
    binerized = sp.ndimage.binary_erosion(binerized, iterations=1)
    binerized = sp.ndimage.binary_dilation(binerized, iterations=1)
    binerized = sp.ndimage.binary_fill_holes(binerized)

    binerized = binerized.astype(float)

    # thresh = skf.threshold_otsu(Y)
    # binerized = (Y > thresh).astype(float)
    gradient_x = np.gradient(binerized, axis=0)**2 + np.gradient(binerized, axis=1)**2

    TT_walls_x = np.zeros(len(Y))
    TT_walls_y = np.zeros(len(Y))
    HH_walls_x = np.zeros(len(Y))
    HH_walls_y = np.zeros(len(Y))

    # plt.show()
    # plt.imshow(gradient_x)
    # plt.show()

    # plt.plot(gradient_x[0])
    # plt.show()


    walls = []

    for i, y in enumerate(gradient_x):
        walls, dic = sp.signal.find_peaks(y)

        values = y[walls]
        TT_wall_ind = walls[np.argmax(values)]
        TT_walls_x[i] = TT_wall_ind
        TT_walls_y[i] = Y[i, TT_wall_ind]

        HH_wall_ind = walls[np.argmin(values)]
        HH_walls_x[i] = HH_wall_ind
        HH_walls_y[i] = Y[i, HH_wall_ind]


    return TT_walls_x, TT_walls_y, HH_walls_x, HH_walls_y #TODO: Does not work that well...









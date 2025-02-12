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

#NOTE: Filtering? Affects the width quite a bit. Should not be used if not necessary.
#TODO: Find out the effect of filtering. Kernel size 9 for medfilt is probably too large.
def filter_profile(y: np.array, median_kernel= 21, savgol_kernel=20, savgol_order=5)->np.array:
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

    y_filtered = sp.signal.medfilt(y, kernel_size=median_kernel) #TODO: Find optimal 9, 
    y_filtered = sp.signal.savgol_filter(y_filtered, window_length=savgol_kernel, polyorder=savgol_order, mode="interp") #TODO: Find optimal 11, 2

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

    jerk_peak_indices = sp.signal.find_peaks(jerk)[0] #prominence=np.ptp(jerk)*0.2)[0] #TODO: Hard coded 0.33
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

        self.median_kernel = x_plus_minus-1 if (x_plus_minus-1) % 2 == 1 else x_plus_minus-2
        self.savgol_kernel = x_plus_minus-1 if (x_plus_minus-1) % 2 == 1 else x_plus_minus-2
        self.savgol_order = np.max([3, x_plus_minus//4])
        
        self.raw_data = raw_data

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

    def get_mean_width(self):
        return np.mean(self.W), np.std(self.W)
    
    def get_median_width(self):
        return np.median(self.W), stats.median_abs_deviation(self.W)
    
    #NOTE: Included now as a class function. The other functions are however external helper functions, as these can have a general purpose.
    def process_kpfm_roi(self, filter=False, median_kernel=None, savgol_kernel=None, savgol_order= None,  ): #TODO: A bit messy. 
        #TODO: Fix docstring
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
        kpfm_results : KPFM_interface
        """

        # kpfm_results = KPFM_interface(data, xres, search_value, x_plus_minus, group_width, high_low)
        kpfm_results = self
        kpfm_results.median_kernel = median_kernel if median_kernel is not None else kpfm_results.median_kernel
        kpfm_results.savgol_kernel = savgol_kernel if savgol_kernel is not None else kpfm_results.savgol_kernel
        kpfm_results.savgol_order = savgol_order if savgol_order is not None else kpfm_results.savgol_order



        kpfm_results.aligned_data = align_profiles(self.raw_data, search_value=self.search_value, x_plus_minus=self.x_plus_minus)
        kpfm_results.grouped_data, kpfm_results.centers = group_profiles(kpfm_results.aligned_data, self.group_width)
        kpfm_results.centers = kpfm_results.centers*self.xres

        for i in range(kpfm_results.grouped_data.shape[0]):

            
            kpfm_results.filtered_data[i] = filter_profile(kpfm_results.grouped_data[i], 
                                                           median_kernel=kpfm_results.median_kernel, 
                                                           savgol_kernel=kpfm_results.savgol_kernel, 
                                                           savgol_order=kpfm_results.savgol_order) if filter else kpfm_results.grouped_data[i]

            out= find_depletion_region(kpfm_results.x, kpfm_results.filtered_data[i], self.high_low)

            kpfm_results.append(out, i)


        return kpfm_results


class Metal:

    def __init__(self, name, wf, Si_n=None , Si_p= None):

        self.name = name
        self.wf = wf
        self.Si_n = Si_n
        self.Si_p = Si_p

class Semiconductor:
    def __init__(self, name, type, Eg, chi_s, e_r, N_dos):
        self.name = name
        self.type = type
        self.Eg = Eg
        self.chi_s = chi_s
        self.e_r = e_r
        self.N_dos = N_dos


class MS_model_interface:
    
    material = { "p-Si" : Semiconductor(name="Si", type="p", Eg=1.12, chi_s=4.05, e_r=11.7, N_dos=2.78e19*1e6),
                "n-Si" : Semiconductor(name="Si", type="n", Eg=1.12, chi_s=4.05, e_r=11.7, N_dos=2.78e19*1e6),
                "EMO_IP": Semiconductor(name="EMO_IP", type="p", Eg=1.6, chi_s=3.83, e_r=11, N_dos=2.78e19*1e6), # T. S. Holstad, D. M. Evans, A. Ruff, D. R. Småbråten, J. Schaab, Ch. Tzschaschel, Z. Yan, E. Bourret, S. M. Selbach, S. Krohns, and D. Meier Phys. Rev. B 97, 085143 – Published 22 February 2018 But this is actually Ti-doped... Then 22.
                "EMO_OOP": Semiconductor(name="EMO_OOP", type="p", Eg=1.6, chi_s=3.83, e_r=15, N_dos=2.78e19*1e6), # Frequency dependent polarisation switching in h-ErMnO3 (high frequency)
                }
    
    metal = {
        "Al":Metal('Al', 4.28),
        "Ag": Metal('Ag', 4.26),
        "Au": Metal('Au', 5.1, 0.8, 0.3),
        "Pt": Metal('Pt', 5.65, 0.9),
        "Pd": Metal('Pd', 5.12, 0.7), 
        "Ni": Metal('Ni', 5.01), 
        "Ti": Metal('Ti', 4.33, 0.5, 0.61), 
        "Cr": Metal('Cr', 4.5, 0.61, 0.5), 
        "W": Metal('W', 4.54, 0.67), 
        "Mg": Metal("Mg", 3.7, 0.4), 
        "Mo": Metal("Mo", 4.6, 0.68, 0.42) 
    }

    def __init__(self, material_name=None, material_dict=None, metal_name=None, metal_dict=None, kpfm_depletion_width=None, kpfm_contact_potential=None):
        try:
            self.material = material_dict if material_dict is not None else self.material[material_name]
        except:
            "Material not loaded properly"
        
        try:
            self.metal = metal_dict if metal_dict is not None else self.metal[metal_name]
        except:
            "Metal not loaded properly"

        self.W = kpfm_depletion_width
        self.dCPD = kpfm_contact_potential

        self.barrier_height = None
        
        pass

    def set_material_kpfm_potential(self, V):
        self.material.kpfm_pot = V

    def set_metal_kpfm_potential(self, V):
        self.metal.kpfm_pot = V
    
    def set_material_kpfm_workfunction(self, V, wf_tip):
        self.material.kpfm_wf = wf_tip - V
    
    def set_metal_kpfm_workfunction(self, V, wf_tip):
        self.metal.kpfm_wf = wf_tip - V

    def set_depletion_width(self, W):
        self.W = W

    
    def ideal_barrier_height(self,wf_m, ea_s, Eg):
        """
        Returns barrier height in Joules by multiplying by q
        """
        q = 1.6e-19
        return (wf_m - ea_s - Eg)*q
    
    def n_barrier(self,phi_m, chi_s):
        """
        Return in eV?
        """
        return (phi_m - chi_s)


    def p_barrier(self,phi_m, chi_s, Eg):
        """
        Returns in eV?
        """
        return (Eg+chi_s) - phi_m # Rectifying if positive #(Eg -(phi_m - chi_s))#*q
    
    def contact_potential(self,phi_Bn, fermi_potential):

        return phi_Bn - fermi_potential
    
    def fermi_level(self,Nc, Nd, T, k=1.380649e-23,  e=1.6e-19):
        return (k*T/e) * np.log(Nc/Nd) # Volt?

    def depletion_width(self,contact_potential, bias, Nd, e_r):

        q = 1.6e-19
        e_0 = 8.854e-12

        w = np.sqrt(2*e_0*e_r*(contact_potential-bias)/(q*Nd) )
        return w
    
    def calculate_depletion_width(self, bias=0, Nd=1e14*1e6, T=300, use_kpfm_metal_wf=False, use_kpfm_semi_wf=False, ):

        barrier_height = None

        if use_kpfm_metal_wf:
            try:
                phi_m = self.metal.kpfm_wf
            except:
                "Metal workfunction not set"
                phi_m = self.metal.wf
        else:
            phi_m = self.metal.wf

        if use_kpfm_semi_wf:
            try:
                phi_s = self.material.kpfm_wf
                barrier_height = self.n_barrier(phi_m, phi_s) if self.material.type == "n" else self.p_barrier(phi_m, phi_s, 0)
            except:
                "Material potential not set"
                phi_s = self.material.chi_s
        else:
            phi_s = self.material.chi_s 
        
        if barrier_height is None:
            barrier_height = self.n_barrier(phi_m, phi_s) if self.material.type == "n" else self.p_barrier(phi_m, phi_s, self.material.Eg)
        else:
            barrier_height = barrier_height

        fermi_level_adjustment = self.fermi_level(Nc=self.material.N_dos, Nd=Nd, T=T) 

        adjusted_potential_difference = self.contact_potential(barrier_height, fermi_level_adjustment)

        return self.depletion_width(adjusted_potential_difference, bias, Nd, self.material.e_r)
    
    def estimate_charge_carrier_density(self, measured_width=None, initial_guess = 1e14*1e6, use_kpfm_metal_wf = False, use_kpfm_semi_wf = True): #TODO

        measured_width = self.W if measured_width is None else measured_width
        if measured_width is None:
            raise ValueError("Measured width not set")

        barrier_height = None

        if use_kpfm_metal_wf:
            try:
                phi_m = self.metal.kpfm_wf
            except:
                "Metal workfunction not set"
                phi_m = self.metal.wf
        else:
            phi_m = self.metal.wf
        
        if use_kpfm_semi_wf:
            try:
                phi_s = self.material.kpfm_wf
                barrier_height = self.n_barrier(phi_m, phi_s) if self.material.type == "n" else self.p_barrier(phi_m, phi_s, 0)
            except:
                "Material potential not set"
                phi_s = self.material.chi_s
        else:
            phi_s = self.material.chi_s

        if barrier_height is None:
            barrier_height = self.n_barrier(phi_m, phi_s) if self.material.type == "n" else self.p_barrier(phi_m, phi_s, self.material.Eg)
        else:
            barrier_height = barrier_height

        print(barrier_height)
        assert barrier_height > 0, "Barrier height must be positive"

        self.barrier_height = barrier_height

        equation = lambda Nd: self.depletion_width(self.contact_potential(barrier_height, self.fermi_level(self.material.N_dos, Nd, T=300)), 0, Nd, self.material.e_r) - measured_width

        # solution = opt.fsolve(equation, initial_guess, xtol=1e-12, maxfev=1000) # Consider to do root_scalar of newton or brentq or something. 
        solution = opt.root_scalar(equation, bracket=[1e10*1e6, 1e20*1e6], method='brentq')

        self.estimated_Nd = solution.root
        if solution.converged:
            print(f"Converged with Nd = {solution.root}")

        return self.estimated_Nd

    def get_spatial_charge_carrier_density(self, x,  bias, Nd, T=300, ):

        def n_x(x, Vd, Nd, e_r, T=300,):
            #TODO: A bit unsure about the signs for p-type semiconductor. Boundaries: Must be equal to barrier
            #TODO: First is positive because positive charge in metal. Second comes from max electric field by metal, Also positive field, so negative potential. Last is integration of field dependence, should be negative field, so positive potential.
            e_0 = 8.854e-12
            e = 1.6e-19
            k = 1.380649e-23
            psi_x = Vd - np.sqrt(2*e*Nd*Vd/(e_0*e_r))*x + (e/(2*e_r*e_0))*Nd*x**2
            return Nd*np.exp(-e*(psi_x)/(k*T))

        n = n_x(x, self.barrier_height, Nd, self.material.e_r, T)

        return n
    
    def get_spatial_potential(self, x, bias, Nd, T=300):

        def phi_x(x, Vd, Nd, e_r, T=300):
            e_0 = 8.854e-12
            e = 1.6e-19
            k = 1.380649e-23
            return Vd - np.sqrt(2*e*Nd*Vd/(e_0*e_r))*x + (e/(2*e_r*e_0))*Nd*x**2
        
        return phi_x(x, self.barrier_height, Nd, self.material.e_r, T)

    def calculate_built_in_voltage_PN(Na, Nd, e_r, e_0= 8.854e-12, e = 1.6e-19):
        return e/(2*e_r*e_0) * (Na-Nd) * 1e-6 #SI
    
    def calculate_depletion_width_PN(Na, Nd, e_r, V_bi, V_b, e_0= 8.854e-12, e = 1.6e-19):
        return np.sqrt( 2*e_0*e_r/(e)  * (Na+Nd)/(Nd*Na) * (  V_bi - V_b ) * 1e6) #SI

    def calculate_fermi_potential_PN(Na, Nd, ni, T, k=1.380649e23,  e=1.6e-19):
        return (k*T/e) * np.log(Na*Nd/ni**2)
    

    # def calculate_PN_width(self, )




    

    





#NOTE: Filtering? Affects the width quite a bit. Should not be used. 
def process_kpfm_roi(data: np.ndarray, xres, search_value, x_plus_minus=25, group_width=15, high_low=False, filter=False)->KPFM_interface: #TODO: A bit messy. 
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
    kpfm_results : KPFM_interface
    """

    kpfm_results = KPFM_interface(data, xres, search_value, x_plus_minus, group_width, high_low)

    kpfm_results.aligned_data = align_profiles(data, search_value=search_value, x_plus_minus=x_plus_minus)
    kpfm_results.grouped_data, kpfm_results.centers = group_profiles(kpfm_results.aligned_data, group_width)
    kpfm_results.centers = kpfm_results.centers*xres

    print(kpfm_results.grouped_data.shape)
    print(kpfm_results.filtered_data.shape)

    for i in range(kpfm_results.grouped_data.shape[0]):

        
        kpfm_results.filtered_data[i] = filter_profile(kpfm_results.grouped_data[i]) if filter else kpfm_results.grouped_data[i]

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
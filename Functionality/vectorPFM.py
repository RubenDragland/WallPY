
import numpy as np
import os
import sys

import scipy.ndimage as ndi
import scipy.stats as stats
import scipy.optimize as opt


wallpy_cmd = r"C:\Users\rubensd\OneDrive - NTNU\PhD\Analysis\WallPY"
sys.path.append(os.path.join(wallpy_cmd, "Display"))
from figures import FigureSinglePlot, FigureSubplots
import rcParams
import matplotlib.pyplot as plt
import skimage.restoration as rest
import skimage.filters as filters


'''
Code structure will be similar to the poly crystal class. 
#TODO: vectorPFM class holding instances of the orientationPFM class.
#TODO: each orientationPFM class holds hyperparameters for cropping and at which orientation it was obtained
#TODO: Processing in orientationPFM includes to sort pixel intensities by kernel density estimation, 
# curve-fitting gaussians in the mixture model, calculate the d from the condition of unimodality, perform a Shapiro test
#TODO: Possibly find other robust ways to evaluate the signal... and evaluate the fitting is also important. 
#TODO: The vectorPFM class will hold the further processing of the orientationPFM instances.

'''



class vectorPFM:
    """
    Class for processing cropped PFM images at same locations 
    """
    
    def __init__(self, orientationPFMs:list, ofile:str, opath = '' , **kwargs):
        """
        orientationPFMs: list of orientationPFM instances
        ofile: filename of the output file
        opath: path to save the output file


        """

        self.instances = orientationPFMs
        self.ofile = ofile
        self.opath = opath
        self.kwargs = kwargs

        return
    
    def __call__(self):
        """
        Process the orientationPFM instances
        """

        self.process_instances()
        self.save()

        return
    
    def roi_shapiro_processing(self, filtering=True):
        """
        Process the orientationPFM instances
        """

        for instance in self.instances:
            instance.preprocess_LPFM(filtering=filtering)
            instance.roi_shapiro_test()

        return
    
    def roi_otsu_processing(self, filtering=False):

        for instance in self.instances:
            instance.preprocess_LPFM(filtering=filtering)
            instance.otsu_thresholding()
        
        return
    
    def roi_processing(self, sanity_check=False):
        """
        Process the orientationPFM instances
        """

        for instance in self.instances:
            instance.preprocess_LPFM()
            instance.roi_analysis_LPFM()

        return
    

    def plot_shapiro(self, instances):

        fig = plt.figure(figsize=(rcParams.DEFAULT_FIGSIZE[0]*2*len(instances), 3*rcParams.DEFAULT_FIGSIZE[1]*2) )
        gs = fig.add_gridspec(4, 2*len(instances))

        angles = [instance.orientation_deg for instance in instances]

        vmin = np.min([np.min(instance.data) for instance in instances])
        vmax = np.max([np.max(instance.data) for instance in instances])

        for i, instance in enumerate(instances):
                
                ax1 = fig.add_subplot(gs[0, 2*i:2*i+2])
                # ax11 = fig.add_subplot(gs[0, 2*i+1])
                ax2 = fig.add_subplot(gs[1, 2*i:2*i+2])
    
                ax1.imshow(instance.grain_array, vmin=None, vmax=None)

                ax1.plot([instance.roi_crop_x[0], instance.roi_crop_x[1], instance.roi_crop_x[1], instance.roi_crop_x[0], instance.roi_crop_x[0]], 
                        [instance.roi_crop_y[0], instance.roi_crop_y[0], instance.roi_crop_y[1], instance.roi_crop_y[1], instance.roi_crop_y[0]], "r-")

                # ax_inlet = ax1.inset_axes([0.65, 0.1, 0.35, 0.35])

                # ax_inlet.imshow(instance.data,  cmap="gray")
    
                x = np.linspace(np.min(instance.data), np.max(instance.data), 2000)

                ax2.hist(instance.data.flatten(), bins=69, density=True, label = "Data" )
                ax2.plot(x, instance.LPFM_params["kde"](x), label="KDE")
                ax2.legend()
        
        ax3 = fig.add_subplot(gs[2, :])
        ax3_twin = ax3.twinx()

        shapiro_stat = [instance.LPFM_params["S_stat"] for instance in instances]
        shapiro_p = [instance.LPFM_params["S_p"] for instance in instances]

        ax3.plot(angles, shapiro_stat, "D:", label="Shapiro_stat")
        ax3_twin.plot(angles, shapiro_p, "ro--", label="Shapiro_p")
        ax3_twin.set_yscale("log")
        ax3.legend(loc="upper left")
        ax3_twin.legend(loc="upper right")

        angles = [instance.orientation_deg%180 for instance in instances]

        ax3.scatter(angles, shapiro_stat, s=100, c="blue",)
        ax3_twin.scatter(angles, shapiro_p, s=100, c="red",)

        ax3.set_xlabel("Orientation [deg]")
        ax3.set_ylabel("Shapiro_stat")
        ax3_twin.set_ylabel("Shapiro_p")

        return
    
    def sanity_check(self, instances):
        """
        Check if the data is processed correctly
        """

        def gauss(x,mu,sigma,A):
            return A**2*np.exp(-(x-mu)**2/2/sigma**2)

        def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
            return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

        fig = plt.figure(figsize=(rcParams.DEFAULT_FIGSIZE[0]*2*len(instances), 3*rcParams.DEFAULT_FIGSIZE[1]*2) )
        gs = fig.add_gridspec(4, 2*len(instances))

        angles = [instance.orientation_deg for instance in instances]

        vmin = np.min([np.min(instance.data) for instance in instances])
        vmax = np.max([np.max(instance.data) for instance in instances])

        for i, instance in enumerate(instances):

            ax1 = fig.add_subplot(gs[0, 2*i])
            ax11 = fig.add_subplot(gs[0, 2*i+1])
            ax2 = fig.add_subplot(gs[1, 2*i:2*i+2])

            # vmin= np.min(instance.data)
            # vmax = np.max(instance.data)
            ax1.imshow(instance.grain_array, vmin=vmin, vmax=vmax)
            ax11.imshow(instance.data, vmin=vmin, vmax=vmax)

            x = np.linspace(np.min(instance.data), np.max(instance.data), 2000) #len(instance.roi_arrays[instance.LPFM_indices[0]].flatten()) )
            ax2.hist(instance.data.flatten(), bins=69, density=True, label = "Data" ) 

            random_ind = np.random.randint(0, len(instance.auto_stats["kde"]), 1)[0]
            x = np.linspace(np.min(instance.data), np.max(instance.data), 1000) #len(instance.roi_arrays[instance.LPFM_indices[0]].flatten()) )
            ax2.plot(x, instance.auto_stats["kde"][random_ind](x), label = "random_KDE" )
            # ax2.plot(x, instance.LPFM_params[instance.LPFM_indices[0]]["kde"](x), label="KDE")
            ax2.plot(x, bimodal(x, *instance.auto_stats["params"][random_ind]), label=f"Fit d: {instance.auto_stats['d'][random_ind]:.2f}")

            ax2.legend()
        
        ax3 = fig.add_subplot(gs[2, :])

        ds = [instance.auto_stats["d"] for instance in instances]
        ps = [instance.auto_stats["e_d"] for instance in instances]

        # ax3.plot(angles, ds, "D:", label="d")
        # ax3.plot(angles,ps, "ro--", label="p-safe")
        d = np.mean(ds, axis=1)
        p = np.mean(ps, axis=1)
        d_std = np.std(ds, axis=1)
        p_std = np.std(ps, axis=1)
        ax3.errorbar(angles, d, yerr=d_std, fmt="o--", label="d")
        ax3.plot(angles, p, "o--", label="p-safe")

        ax3.scatter(np.array([angles]*len(ds[0])).T, ds, s=100, c="blue", label="d_scatter")
        ax3.legend()
        # ax3_twin.legend(loc="upper right")

        ax3.set_xlabel("Orientation [deg]")
        ax3.set_ylabel("evaluator")
        # ax3.set_ylim(0, np.max([d, p, d_std, p_std])*1.1)

        ax4 = fig.add_subplot(gs[3, :])

        # mean_diffs = []
        # for instance in instances:
        #     mean_diffs.append(np.abs(instance.auto_stats["params"][:][3] - instance.auto_stats["params"][:][0]))
        mean_diffs = np.array(np.abs([instance.auto_stats["params"][:][3] - instance.auto_stats["params"][:][0] for instance in instances])).mean(axis=1)
        std_diffs = np.array(np.abs([instance.auto_stats["params"][:][3] - instance.auto_stats["params"][:][0] for instance in instances])).std(axis=1)

        ax4.errorbar(angles, mean_diffs, fmt="o--", yerr=std_diffs,  label="mean_diff")

        
        print(d)
        print(p)
        shap_stat = np.array([instance.auto_stats["S_stat"] for instance in instances]).mean(axis=1)
        shap_p = np.array([instance.auto_stats["S_p"] for instance in instances]).mean(axis=1)
        print(shap_stat, shap_p)

        for instance in instances:
            for j in range(len(instance.auto_stats["params"])):
                print(instance.auto_stats["params"])

        plt.show()     
        return
    

    def big_data_processing(self, auto_x=50, auto_y=50,  margin_x=10, margin_y=10, counts=10):
        """
        #TODO: see if an entire grain can be processed automatically, and more data can be collected. 
        """
        for instance in self.instances:
            instance.preprocess_LPFM()
            instance.automatic_roi(instance.kde_biomodality, auto_x, auto_y, margin_x, margin_y, counts)
        return
    
    def save(self):
        """
        Save the processed data to a new hfile holding all necessary data for plotting etc. 
        """

        return
    

class orientationPFM:
    """
    Class for processing a single PFM image used for vectorPFM. 
    """


    
    def __init__(self, gwyfile:object, LPFM_indices:list=None, VPFM_indices:list=None, orientation_deg:float=None, grain_crop_x:list=None, grain_crop_y:list=None, roi_crop_x=None, roi_crop_y=None, processed_array:np.array = None, **kwargs):
        """
        #TODO: Additional cropping parameters for the supercrop. List of tuples with the cropping parameters? Can be same region or not same region. 
        Assume entire grain in grain_crop, and roi_crop is the region of interest if difficult to perform analysis automatically
        #TODO: Check if necessary with the keyword arguments when processed is an option. Make optional... 
        """

        def read_LPFM_indices_from_channels(channel_names):
            """
            Read the LPFM indices from the gwyfile
            #TODO: Should this be a part of universal reader instead?
            """
            LPFM_indices = [int(str(ch).split("_")[-1]) for ch in channel_names if "LPFM" in ch]

            return LPFM_indices


        self.data = None
        self.LPFM_params = {}

        if processed_array is not None:
            self.grain_array = processed_array
        else:
            self.grain_array = None

        if isinstance(gwyfile, dict):
            self.gwyfile = gwyfile["gwyfile"]
            self.LPFM_indices = gwyfile["LPFM_indices"]
            self.VPFM_indices = gwyfile["VPFM_indices"]
            self.kwargs = gwyfile["kwargs"]
            self.orientation_deg = gwyfile["orientation_deg"]
            self.orientation_rad = np.deg2rad(self.orientation_deg)
            self.grain_crop_x = gwyfile["grain_crop_x"]
            self.grain_crop_y = gwyfile["grain_crop_y"]
            try:
                self.roi_crop_x = gwyfile["roi_crop_x"] 
                self.roi_crop_y = gwyfile["roi_crop_y"] 
            except:
                self.roi_crop_x = None
                self.roi_crop_y = None
        else:
            try:
                self.gwyfile = gwyfile
                self.LPFM_indices = read_LPFM_indices_from_channels(self.gwyfile.channel_names) if LPFM_indices is None else LPFM_indices
                self.VPFM_indices = VPFM_indices
                self.kwargs = kwargs
                self.orientation_deg = orientation_deg
                self.orientation_rad = np.deg2rad(self.orientation_deg)
                self.grain_crop_x = grain_crop_x
                self.grain_crop_y = grain_crop_y
                self.roi_crop_x = roi_crop_x
                self.roi_crop_y = roi_crop_y    
            except:
                raise ValueError("Not all necessary parameters provided")   

        if (self.roi_crop_x and self.roi_crop_y):
            self.automatic_whole_grain = False
            # self.roi_arrays = {} #TODO: What with this?
        else:
            self.automatic_whole_grain = True
            # self.roi_arrays = {}
        return
    

    def inspect_rotated_cropped(self, crop_x = None, crop_y = None):
        fig, axes = plt.subplots(1, len(self.gwyfile.channel_names), figsize=(len(self.gwyfile.channel_names)*5, 5))

        fig.suptitle(f"Rotated {self.orientation_deg} degrees")
        

        for ax, channel in zip(axes.reshape(-1), self.gwyfile.channel_names):
            ind = int(channel.split("_")[-1])

            data = self.gwyfile[ind]

            hist, edges = np.histogram(data, bins=100)

            pvalue = np.max(hist)
            peak = edges[np.argmax(hist)]

            lower = edges[np.where(hist > pvalue*0.5)[0][0]]
            upper = edges[np.where(hist > pvalue*0.5)[0][-1]]

            vmin = lower
            vmax = upper



            data = ndi.rotate(data, angle=self.orientation_deg, reshape=False)

            if crop_x is not None and crop_y is not None:
                data = data[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]


            ax.imshow(data, cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_title(channel)
        plt.show()
        return
    

    def preprocess_LPFM(self, filtering=True):
        """
        #TODO: Decide what to do with forward and backward...
        Process the LPFM indices:
        - align rotation
        - crop the image
        - sort the pixel intensities
        - fit the pixel intensities to a mixture model
        - calculate the d from the condition of unimodality
        - perform a Shapiro test
        - evaluate the fitting
        """

        def merge_forward_backward():
            """
            Merge the forward and backward scans
            """
            data = np.mean([self.gwyfile[self.LPFM_indices[i]] for i in range(len(self.LPFM_indices))], axis=0)
            return data

        def align_orientation(data):
            """
            Align the PFM image to the defined 0 orientation
            """
            # for index in self.LPFM_indices:
            #     self.grain_arrays[index] = ndi.rotate(self.gwyfile[index], self.orientation_deg, reshape=False)
            data = ndi.rotate(data, self.orientation_deg, reshape=False)
            return data

        def crop_grain(data):
            """
            Crop the grain
            """
            # for index in self.LPFM_indices:
            #     #TODO: Prob x and y are switched.
            #     self.grain_arrays[index] = self.grain_arrays[index][self.grain_crop_y[0]:self.grain_crop_y[1], self.grain_crop_x[0]:self.grain_crop_x[1]]

            data = data[self.grain_crop_y[0]:self.grain_crop_y[1], self.grain_crop_x[0]:self.grain_crop_x[1]]
            self.grain_array = data
            if not self.automatic_whole_grain:
                    data = data[self.roi_crop_y[0]:self.roi_crop_y[1], self.roi_crop_x[0]:self.roi_crop_x[1]]
                  
            return data
        
        
        def filtering(data):

            # for index in self.LPFM_indices:
            #     # self.grain_arrays[index] = rest.denoise_bilateral(self.grain_arrays[index], win_size=5, mode='wrap')
            #     if not self.automatic_whole_grain:
            #         self.roi_arrays[index] = rest.denoise_bilateral(self.roi_arrays[index], win_size=5, sigma_color=np.std(self.roi_arrays[index]), sigma_spatial=1, mode='wrap')
            data = rest.denoise_bilateral(data, win_size=5, mode='symmetric')
            return data

        if self.grain_array is None: 
            data = merge_forward_backward()
            data = align_orientation(data)
            data = crop_grain(data)
            if filtering:
                data= filtering(data) #TODO: Move forwards and or make optional possibly. 
        else:
            data = merge_forward_backward()
            data = filtering(data)

        self.data = data
        return 
    

    def otsu_thresholding(self):

        kde = stats.gaussian_kde(self.data.flatten())
        # x = np.linspace(np.min(self.data), np.max(self.data), 5000)  #TODO: Evaluate if useful with this number of points, or just a random number.

        # kernel_data = kde(x) #TODO: Threshold does not work on processed data.
        thresh_value = filters.threshold_otsu(self.data)

        print(thresh_value)

        d1_bool = self.data < thresh_value
        d2_bool = self.data >= thresh_value
        domain1 = self.data[d1_bool]
        domain2 = self.data[d2_bool]

        print(domain1.std(), domain2.std(), self.data.std())
        print(d1_bool.sum(), d2_bool.sum(), self.data.size)

        otsu_param = (d1_bool.sum()*domain1.std()**2 + d2_bool.sum()*domain2.std()**2) / (self.data.std()**2 * self.data.size)

        self.LPFM_params["otsu_thresh"] = thresh_value
        self.LPFM_params["otsu"] = otsu_param
        return


    

    def roi_shapiro_test(self):

        assert self.data is not None, "No data provided"

        kde = stats.gaussian_kde(self.data.flatten())
        S_stat, S_p = stats.shapiro(self.data.flatten())

        self.LPFM_params["kde"] = kde
        self.LPFM_params["S_stat"] = S_stat
        self.LPFM_params["S_p"] = S_p
        return S_stat, S_p
    
    
    def roi_analysis_LPFM(self):
        """
        Performs analysis in case roi is provided, and entire grain is not used. 
        """
        
        # assert self.roi_arrays != {}, "No roi arrays provided"
        assert self.data is not None, "No data provided"
        

        def gauss(x,mu,sigma,A):
            return A**2*np.exp(-(x-mu)**2/2/sigma**2)

        def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
            return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)
        
        def initial_guess(data):
            sigma1 = np.std(data)
            mu1 = np.mean(data) - sigma1*1.0
            A1 = 0.3 #1/(sigma1*np.sqrt(2*np.pi))
            mu2 = np.mean(data) + sigma1*1.0
            sigma2 = sigma1
            A2 = A1
            return [mu1,sigma1,A1,mu2,sigma2,A2]
        
        def unimodal_condition(mu1, sigma1, mu2, sigma2):
            """
            Condition for unimodality
            Behboodian, J (1970). "On the modes of a mixture of two normal distributions". Technometrics. 12 (1): 131–139. doi:10.2307/1267357. JSTOR 1267357.
            """
            return np.abs(mu1 - mu2) / (2*np.min(np.abs([sigma1 , sigma2])))
        #TODO: Not taking p into account. Iterative approach from same source...

        def eisenberger_unimodal_condition(mu1, sigma1, mu2, sigma2):

            return (mu1-mu2)**2 * 4*(sigma1**2 + sigma2**2) / (27*sigma1**2*sigma2**2)
        

        # for index in self.LPFM_indices:

        kde = stats.gaussian_kde(self.data.flatten()) 
        x = np.linspace(np.min(self.data), np.max(self.data), 5000)  #TODO: Evaluate if useful with this number of points, or just a random number. 
        
        params, cov = opt.curve_fit(bimodal, x, kde(x), p0=initial_guess(self.data.flatten()))

        d = unimodal_condition(params[0], params[1], params[3], params[4])

        e_d = eisenberger_unimodal_condition(params[0], params[1], params[3], params[4])

        S_stat, S_p = stats.shapiro(self.data.flatten())

        self.LPFM_params = {
            "kde": kde,
            "params": params,
            "cov": cov,
            "d": d,
            "e_d": e_d,
            "S_stat": S_stat,
            "S_p": S_p}
        return
    
    def kde_biomodality(self, roi):
    

        def gauss(x,mu,sigma,A):
            return A**2*np.exp(-(x-mu)**2/2/sigma**2)

        def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
            return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)
        
        def initial_guess(data):
            sigma1 = np.std(data)
            mu1 = np.mean(data) - sigma1*1.0
            A1 = 0.3 #1/(sigma1*np.sqrt(2*np.pi))
            mu2 = np.mean(data) + sigma1*1.0
            sigma2 = sigma1
            A2 = A1
            return [mu1,sigma1,A1,mu2,sigma2,A2]
        
        def unimodal_condition(mu1, sigma1, mu2, sigma2):
            """
            Condition for unimodality
            Behboodian, J (1970). "On the modes of a mixture of two normal distributions". Technometrics. 12 (1): 131–139. doi:10.2307/1267357. JSTOR 1267357.
            """
            return np.abs(mu1 - mu2) / (2*np.min(np.abs([sigma1 , sigma2])))
        #TODO: Not taking p into account. Iterative approach from same source...

        def eisenberger_unimodal_condition(mu1, sigma1, mu2, sigma2):

            return (mu1-mu2)**2 * 4*(sigma1**2 + sigma2**2) / (27*sigma1**2*sigma2**2)
        

        flat_roi = roi.flatten()

        kde = stats.gaussian_kde(flat_roi) 
        x = np.linspace(np.min(roi), np.max(roi), 5000)  #TODO: Evaluate if useful with this number of points, or just a random number. 
        
        params, cov = opt.curve_fit(bimodal, x, kde(x), p0=initial_guess(flat_roi))

        d = unimodal_condition(params[0], params[1], params[3], params[4])

        e_d = eisenberger_unimodal_condition(params[0], params[1], params[3], params[4])

        S_stat, S_p = stats.shapiro(flat_roi)

        anal_stats = {
            "kde": kde,
            "params": params,
            "cov": cov,
            "d": d,
            "e_d": e_d,
            "S_stat": S_stat,
            "S_p": S_p}
        return anal_stats
    

    def automatic_roi(self, analysis, auto_x=50, auto_y=50,  margin_x=10, margin_y=10, counts=10,):
        """
        #TODO: Need settings: auto_x, auto_y, counts, margin_x, margin_y
        # Find rois randomly
        #Fits the data to a bimodal distribution ==> analysis
        """ 
        self.auto_x = auto_x
        self.auto_y = auto_y
        self.margin_x = margin_x
        self.margin_y = margin_y
        self.counts = counts

        def find_rois():
            """
            Find the regions of interest
            """
            x = np.random.randint(self.margin_x, self.data.shape[1]-self.margin_x-self.auto_x, self.counts)
            y = np.random.randint(self.margin_y, self.data.shape[0]-self.margin_y-self.auto_y, self.counts)
            return x, y
        
        def crop_roi(x, y):
            """
            Crop the regions of interest
            """
            roi = self.data[y:y+self.auto_y, x:x+self.auto_x]
            return roi
        
        self.auto_stats = {
            "kde": [],
            "params": [],
            "cov": [],
            "d": [],
            "e_d": [],
            "S_stat": [],
            "S_p": []}
        #...etc...

        x, y = find_rois()
        
        for cc in range(self.counts):
            roi = crop_roi(x[cc], y[cc])
            anal_stats = analysis(roi)
            for key, item in anal_stats.items():
                self.auto_stats[key].append(item)
            
        return      

        

    
    def quick_view(self):
        """
        Quick view of the processed data
        #TODO: Outdated, switch to data...
        """
        fig = plt.figure(figsize=(3*rcParams.DEFAULT_FIGSIZE[0], len(self.LPFM_indices)*rcParams.DEFAULT_FIGSIZE[1]) )
        gs = fig.add_gridspec(3, len(self.LPFM_indices))

        for i, index in enumerate(self.LPFM_indices):
            ax1 = fig.add_subplot(gs[0, i])
            ax2 = fig.add_subplot(gs[1, i])
            ax3 = fig.add_subplot(gs[2, i])

            ax1.imshow(self.grain_arrays[index])
            ax2.imshow(self.roi_arrays[index])
            ax3.hist(self.grain_arrays[index].flatten(), bins=69)
            ax3.hist(self.roi_arrays[index].flatten(), bins=69)

            # x = np.linspace(np.min(self.roi_arrays[index]), np.max(self.roi_arrays[index]), len(self.roi_arrays[index].flatten()), label = "Data" ) 
            # ax2.plot(x, self.LPFM_params[index]["kde"](x), label="KDE")
            # ax2.plot(x, bimodal(x, *self.LPFM_params[index]["params"]), label="Fit")

            # ax2.legend()

        plt.show()
        return




    



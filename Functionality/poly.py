

from typing import Any
import numpy as np
import pyclesperanto_prototype as cle
import scipy as sp
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import skimage.measure as skm
import scipy.ndimage as ndi
import os



class PolyCrystal:
    '''
    Class for storing grains and their properties, specifically their grain/domain size relation for the polycrystal investigated.

    Attributes
    ----------

    grains : list
        List of grains in the polycrystal.
    ofile : str
        Path to the output file.
    kwargs : dict
        Dictionary of keyword arguments.

    Methods
    -------
    __call__
        Stores the calculated grains properties in a dataframe and saves it to file.
    '''

    def __init__(self, ofile:str, **kwargs):
        """
        ofile: path to hdf5 file
        """
        self.ofile = ofile
        self.kwargs = kwargs

        self.grains = []

        return
    
    def __call__(self):

        """
        Stores the calculated grains properties in a dataframe and saves it to file.
        """

        df = pd.DataFrame(columns=["id", "filename", "grain_size", "mean_domain_size", "std_domain_size", "final_domain_size", "final_domain_size_std", "grain_area", "tot_avg", "tot_std", "tot_median", "tot_mad"]) 

        for grain in self.grains:
            df.loc[len(df.index)] = [grain.id, grain.filename, grain.grain_size, grain.mean_domain_size, grain.std_domain_size, grain.final_domain_size, grain.final_domain_size_std, grain.grain_area, grain.tot_avg, grain.tot_std, grain.tot_median, grain.tot_mad]
        
        print(df)
        df.to_csv(self.ofile, index=True)


        return df
    


class Grain:
    '''
    Class for storing grain properties and methods for calculating domain sizes.

    Attributes
    ----------
    filename : str
        The name for saving the grain properties
    channel : int
        The channel information for saving
    id : int
        The id of the grain
    pfm : np.array
        The piezo force microscopy image
    borders : list
        The corners in polygon defining the grain
    res : float
        The resolution of pfm image
    lines : dict
        The lines to draw cressections from. Should be leftmost line. (Not needed for orientational sampling)
    kwargs : dict
        Dictionary of keyword arguments.
    
    Methods
    -------
    __call__
        Old method for processing the grain. Deprecated.
    save_grain
        Saves the grain to file using pickle.
    calc_weighted_avg
        Calculates the weighted average of a list of dictionaries.
    call_projection_counting
        Uses the entire grain to calculate the domain size. Draws projections, binarizes, finds gradients, and samples peak-2-peak distances. Orientational sampling.
    ...

    '''



    default_kwargs = {
        "repeats": 10,
        "lw": 4,
        "order": 5,
        "reduce_func": np.median,
        "filter_width": 2,
        "plot" : False,
    }


    def __init__(self, filename: str, channel: int,  pfm : np.array, borders: list, res: float, lines: dict, **kwargs):
        """
        filename: str
            The name for saving the grain properties
        channel: int
            The channel information for saving
        pfm: np.array
            The piezo force microscopy image
        borders: list
            The corners in polygon defining the grain
        res: float
            The resolution of pfm image
        lines: dict
            The lines to draw cressections from. Should be leftmost line. (Not needed for orientational sampling)
        kwargs: dict
            Dictionary of keyword arguments.
        """

        self.filename = filename
        self.channel = channel
        self.id = None

        self.pfm = pfm
        self.borders = borders
        self.mask = None
        self.res = res
        self.lines = lines
        self.kwargs = kwargs

        for key, value in self.default_kwargs.items():
            if key not in self.kwargs:
                self.kwargs[key] = value

        self.grain_area = None
        self.grain_size = None

        self.domain_binaries = {"S": [], "M": [], "L": []}
        self.domain_peakcounts = {"S": [], "M": [], "L": []}

        self.domain_binaries_avg = {"S": None , "M": None , "L": None }
        self.domain_peakcounts_avg = {"S": None , "M": None , "L": None }

        self.domain_binaries_std = {"S": None , "M": None , "L": None }
        self.domain_peakcounts_std = {"S": None , "M": None , "L": None }

        self.ns = {"S": None , "M": None , "L": None }


        self.domain_grad_binaries_mean = {"S": [], "M": [], "L": []}
        self.domain_grad_binaries_std = {"S": [], "M": [], "L": []}
        self.domain_grad_binaries_ws = {"S": [], "M": [], "L": []}

        self.mean_domain_sizes = {"S": None , "M": None , "L": None }
        self.std_domain_sizes = {"S": None , "M": None , "L": None }

        self.final_domain_size = None
        self.final_domain_size_std = None

        self.sizes = []
        self.tot_avg = None
        self.tot_std = None

        self.tot_median = None
        self.tot_mad = None

        self.sizes_old = []

        return
    
    def save_grain(self, path:str = ""):
        """
        Saves the grain to file using pickle.
        """

        self.savename = os.path.join(path, f"{self.filename}_{self.channel}_{self.id}.pkl") 
        with open(self.savename, "wb") as f:
            pd.to_pickle(self, self.savename)

        return

    

    def __call__(self, poly: PolyCrystal):
        '''
        NOT THE ORIENTATIONAL SAMPLING METHOD. Deprecated.
        '''

        self.id = len(poly.grains)

        self.grain_size, self.grain_area, self.mask = calc_grain_size(self.pfm, self.borders, self.res)

        # fig, ax = plt.subplots(1,1)
        

        #TODO: Ensure that lines comes from dictionary with keys S, M, L. 
        for i, (key, xys) in enumerate(self.lines.items()):

            for j, xy in enumerate(xys):

                x = xy[0][1] - xy[0][0]
                y = xy[1][1] - xy[1][0]

                if np.abs(x) > np.abs(y):
                    for k in range(self.kwargs["repeats"]):
                        xy[1][0] += self.kwargs["lw"]
                        xy[1][1] += self.kwargs["lw"]

                        # try:#TODO: Update.
                            # self.binarize_domain_size(self.pfm, key, xy, plot=True if k==0 else False, id=self.id, lw=self.kwargs["lw"], order=self.kwargs["order"], reduce_func=self.kwargs["reduce_func"], filter_width= self.kwargs["filter_width"]  )
                        self.binarize_grad_domain_size(self.pfm, key, xy, plot=True if k==0 else False, id=self.id, lw=self.kwargs["lw"], order=self.kwargs["order"], reduce_func=self.kwargs["reduce_func"], filter_width= self.kwargs["filter_width"]  )
                        # except:
                            # print("Failed")
                            # continue # TODO: Bit lazy, improve. 
                else:
                    for k in range(self.kwargs["repeats"]):

                        xy[0][0] += self.kwargs["lw"]
                        xy[0][1] += self.kwargs["lw"]

                        # try:
                        self.binarize_grad_domain_size(self.pfm, key, xy, plot=True if k==0 else False, id=self.id, lw=self.kwargs["lw"], order=self.kwargs["order"], reduce_func=self.kwargs["reduce_func"], filter_width= self.kwargs["filter_width"]  )
                            # self.binarize_domain_size(self.pfm, key, xy, plot=True if k==0 else False, id=self.id, lw=self.kwargs["lw"], order=self.kwargs["order"], reduce_func=self.kwargs["reduce_func"], filter_width= self.kwargs["filter_width"]  )
                        # except:
                            # print("Failed")
                            # continue # TODO: Bit lazy, improve. 
        for k, v in self.domain_binaries.items(): #Note: Averages over possibly more areas of ish similar size. 

            self.ns[k] = len(v)
            
            self.domain_binaries_avg[k] = np.mean( self.domain_binaries[k] )
            self.domain_binaries_std[k] = np.std( self.domain_binaries[k] )

            # self.domain_peakcounts_avg[k] = np.mean( self.domain_peakcounts[k] )
            # self.domain_peakcounts_std[k] = np.std( self.domain_peakcounts[k] )

            self.mean_domain_sizes[k], self.std_domain_sizes[k] = calc_combined_mean_std(self.domain_grad_binaries_mean[k], self.domain_grad_binaries_std[k], self.domain_grad_binaries_ws[k])
        

        # self.mean_domain_size = self.calc_weighted_avg(self.domain_binaries_avg, self.ns)
        # self.std_domain_size = self.calc_weighted_avg(self.domain_binaries_std, self.ns, var = True)**0.5
        self.mean_domain_size, self.std_domain_size = self.calc_combined_domain_size(self.domain_binaries_avg, self.domain_binaries_std, self.ns)

        final_weights = [np.sum(list(v)) for k, v in self.domain_grad_binaries_ws.items() ] #TODO: Uncertain about this. Sum or three? Or Do all as one. Also, median could be used to remove outliers. 
        self.final_domain_size, self.final_domain_size_std = calc_combined_mean_std(list(self.mean_domain_sizes.values()), list(self.std_domain_sizes.values() ), final_weights)

        self.sizes = np.array(self.sizes).flatten()*self.res
        try:
            self.sizes_old = np.array(self.sizes_old).flatten()*self.res
        except:
            pass #TODO: HATLØSNING

        self.tot_avg = np.mean(self.sizes)
        self.tot_std = np.std(self.sizes)

        self.tot_median = np.median(self.sizes)
        self.tot_mad = np.median(np.abs(self.sizes - self.tot_median))


        poly.grains.append(self)
        return
    
    #THIS IS THE ORIENTATIONAL SAMPLING METHOD.
    def call_projection_counting(self, poly: PolyCrystal, projections=100):
        '''
        Uses the entire grain to calculate the domain size.
        Draws projections, binarizes, finds gradients, and samples peak-2-peak distances. 
        Orientational sampling.

        Parameters
        ----------
        poly : PolyCrystal
            The polycrystal object to store the grain in.
        projections : int
            The number of projections to sample. The default is 100.
        
        Returns
        -------
        None
        '''

        self.sizes=[]

        self.id = len(poly.grains)

        self.grain_size, self.grain_area, self.mask = calc_grain_size(self.pfm, self.borders, self.res)

        grain = np.abs(self.pfm*self.mask) 

        # Batch everything in 2D.
        # Do [:,] and [,:] to minimise computational time. Needs to rotate 90 degrees.
        # Note rotation angle in degrees. https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html
        self.angles = np.linspace(0, 90, projections, endpoint=False)
        self.sizes0 = []
        self.sizes90 = []

        self.angles_points0 = []
        self.angles_points90 = []

        self.sizes_old = []

        for i, angle in enumerate(self.angles):

            print(f"ID: {self.id} Angle: {angle}")

            rotated = ndi.rotate(grain, angle, reshape=False)

            reset0 = np.isclose(rotated, 0, atol=0.1) # Minimises rotation artifacts, noise, and influcences from outside the mask. 
            rotated[reset0] = np.nan

            proj0 = ndi.maximum_filter1d(rotated, size=self.kwargs["filter_width"]//2, axis=1)
            proj90 = ndi.maximum_filter1d(rotated, size=self.kwargs["filter_width"]//2, axis=0)

            proj0_binary = proj0 > np.nanmean(proj0, axis=1)[:, np.newaxis] 
            proj90_binary = proj90 > np.nanmean(proj90, axis=0)[np.newaxis, :]

            proj0_binary = ndi.maximum_filter1d(proj0_binary, size=self.kwargs["filter_width"], axis=1)
            proj90_binary = ndi.maximum_filter1d(proj90_binary, size=self.kwargs["filter_width"], axis=0)

            proj0_binary = np.gradient(proj0_binary.astype(float), axis=1)
            proj90_binary = np.gradient(proj90_binary.astype(float), axis=0) 

            proj0_ppeaks = [sp.signal.find_peaks(proj0_binary[x,:])[0] for x in range(proj0_binary.shape[0])] #List because variable length.
            proj90_ppeaks = [sp.signal.find_peaks(proj90_binary[:, x])[0] for x in range(proj90_binary.shape[1])] 

            proj0_npeaks = [sp.signal.find_peaks(-proj0_binary[x,:])[0] for x in range(proj0_binary.shape[0])]
            proj90_npeaks = [sp.signal.find_peaks(-proj90_binary[:, x])[0] for x in range(proj90_binary.shape[1])] 

            proj0_len_p = np.array([len(x) for x in proj0_ppeaks])
            proj90_len_p = np.array([len(x) for x in proj90_ppeaks])

            proj0_len_n = np.array([len(x) for x in proj0_npeaks])
            proj90_len_n = np.array([len(x) for x in proj90_npeaks])

            proj0_len_min = np.min([proj0_len_p, proj0_len_n], axis=0) #TODO: Check. axis=1
            proj90_len_min = np.min([proj90_len_p, proj90_len_n], axis=0)

            proj0_grad1 = []
            proj90_grad1 = []

            proj0_grad2 = []
            proj90_grad2 = []

            # Now calculates peak-2-peak distances.
            for x in range(len(proj0_ppeaks)):

                proj0_grad1.extend(np.abs(np.array(proj0_ppeaks[x])[:proj0_len_min[x]] - np.array(proj0_npeaks[x])[:proj0_len_min[x]]) )
                proj0_grad2.extend(np.abs( np.array(proj0_ppeaks[x])[ int(proj0_len_p[x] > proj0_len_n[x]) : ] - np.array(proj0_npeaks[x])[ int(proj0_len_p[x] < proj0_len_n[x]) : ] ) )
                

            for x in range(len(proj90_ppeaks)):

                proj90_grad1.extend(np.abs(np.array(proj90_ppeaks[x])[:proj90_len_min[x]] - np.array(proj90_npeaks[x])[:proj90_len_min[x]]) )
                proj90_grad2.extend(np.abs( np.array(proj90_ppeaks[x])[ int(proj90_len_p[x] > proj90_len_n[x]) : ] - np.array(proj90_npeaks[x])[ int(proj90_len_p[x] < proj90_len_n[x]) : ] ) )



            combined = np.concatenate((proj0_grad1, proj90_grad1, proj0_grad2, proj90_grad2))
            self.sizes.extend(combined)

            self.sizes0.extend(proj0_grad1)
            self.sizes90.extend(proj90_grad1)
            self.sizes0.extend(proj0_grad2)
            self.sizes90.extend(proj90_grad2)

            self.angles_points0.extend([angle]*(len(proj0_grad1) + len(proj0_grad2)))
            self.angles_points90.extend([angle]*(len(proj90_grad1) + len(proj90_grad2)))

        self.sizes = np.array(self.sizes)*self.res
        self.sizes0 = np.array(self.sizes0)*self.res
        self.sizes90 = np.array(self.sizes90)*self.res

        self.angles_points0 = np.array(self.angles_points0)
        self.angles_points90 = np.array(self.angles_points90) + 90

        # Add grain to polycrystal object.
        poly.grains.append(self)
        return


    
    def calc_weighted_avg(self, list_of_dicts: list, weights: dict, var=False):
        '''
        Calculates weighted average. Not used for orientational sampling.
        '''

        values = list(list_of_dicts[0].values())
        ws = list(weights.values())
        for x in range(1, len(list_of_dicts)):
            values.extend(list(list_of_dicts[x].values()))
            ws.extend(list(weights.values()))
        values = np.array(values)**2 if var else np.array(values)
        ws = np.array(ws)

        avg = np.sum(values*ws)/np.sum(ws)
        
        return avg


    
    def calc_combined_domain_size(self, means: dict, stds:dict,  weights: dict):
        '''
        Calculates the combined mean and standard deviation of a list of dictionaries, with weights corresponding to sample size.
        Not used for orientational sampling.

        Parameters
        ----------
        means : list(dict)
            The list of dictionaries of means to combine. Several entries of small, medium, and large domains.
        stds : list(dict)
            The list of dictionaries of standard deviations to combine. Several entries of small, medium, and large domains.
        weights : list(dict)
            The weights corresponding to the sample size of each list of dictionaries. Several entries of small, medium, and large domains.
        '''
        def retrieve_values(dictionary):
            """
            Rearranges the dictionaries into a single array.
            """
            values = list(dictionary.values())
            # for x in range(1, len(list_of_dicts)):
            #     values.extend(list(means[x].values()))
            values = np.array(values)
            return values
        
        mean_values = retrieve_values(means)
        std_values = retrieve_values(stds)
        ws_values = retrieve_values(weights)

        combined_mean = np.sum(mean_values*ws_values)/np.sum(ws_values)

        nom = np.sum(ws_values*(std_values**2)) + np.sum(ws_values*(mean_values - combined_mean)**2)
        den = np.sum(ws_values)

        combined_std = np.sqrt(nom/den)
        
        return combined_mean, combined_std


    def calc_domain_size(self,domain, key, xy, plot=False, id=None):

        '''
        Deprecated. Use binarize_domain_size instead.
        '''

        zi, ri, pts = draw_crossection(domain, xy[0], xy[1])
        sinfit = fit_sin(ri*self.res, zi)
        self.domain_binaries[key].append(sinfit["period"])

        peaks, _ = sp.signal.find_peaks(np.abs(zi), height = np.abs(np.mean(zi)), width= 2, distance= 2 )
        self.domain_peakcounts[key].append( (ri[peaks[-1]] - ri[peaks[0]]) *self.res/(len(peaks)-1))

        if plot:
            print(f"Plotting {id}_{key}")
            plt.plot(ri*self.res, zi)
            plt.plot(ri*self.res, sinfit["fitfunc"](ri*self.res))
            plt.scatter(ri[peaks]*self.res, zi[peaks], marker="x")
            savename = f"{id}_{key}.pdf"
            plt.savefig(savename)
            plt.close()


        return
    
    def binarize_domain_size(self, domain, key, xy, plot=False, id=None, lw=4, order=5, reduce_func=np.median, filter_width= 2):
        '''
        Retrieves intensity profile from pfm image, and finds the domain size by peak counting before and after max-filtering-binarization-max-filtering.

        Parameters
        ----------
        domain : np.array
            The pfm image.
        key : str
            The key of the line to draw the cross section from.
        xy : list
            The coordinates of the line to draw the cross section from. x and y coordinates are separated.
        plot : bool, optional
            Whether to plot the cross section. The default is False.
        id : int, optional
            The id of the grain. The default is None.
        lw : int, optional
            The width of the line to sample intensities. Perpendicular to direction. The default is 4.
        order : int, optional
            The order of the spline interpolation. The default is 5, corresponding to Bi-quintic spline interpolation.
        reduce_func : function, optional
            The function to reduce the line intensity to a single value. The default is np.median.
        filter_width : int, optional
            The width of the max filter. The default is 2.
            
        '''

        intensity, x = skm_crossection(domain, xy[0], xy[1], lw=lw, order=order, reduce_func=reduce_func)

        peaks, _ = sp.signal.find_peaks(intensity, height = np.mean(intensity), width= 2, distance= 2 )

        self.domain_peakcounts[key].append( (x[peaks[-1]] - x[peaks[0]]) *self.res/(len(peaks)-1))

        intensity = ndi.maximum_filter1d(intensity, size=filter_width)

        binary_profile = intensity > np.mean(intensity)

        binary_profile = ndi.maximum_filter1d(binary_profile, size=filter_width)

        binary_peaks,_ = sp.signal.find_peaks(binary_profile)
        if len(binary_peaks) <=1:
            binary_peaks, _ = sp.signal.find_peaks(~binary_profile)
        
        try:
            self.domain_binaries[key].append( (x[binary_peaks[-1]] - x[binary_peaks[0]]) *self.res/(len(binary_peaks)-1)) 

        except:
            print("Failed")
            return

        if plot:
            print(f"Plotting {id}_{key}")
            plt.figure(figsize=(10,6))
            plt.plot(x*self.res, intensity)
            plt.scatter(x[peaks]*self.res, intensity[peaks], marker="x")
            plt.plot(x*self.res, binary_profile + np.max(intensity))
            plt.scatter(x[binary_peaks]*self.res,  binary_profile[binary_peaks] + np.max(intensity), marker="x") #TODO: Fix this. Hva farn.
            savename = f"{id}_{key}.pdf"
            plt.savefig(savename)
            plt.close()
        
        return
    
    def binarize_grad_domain_size(self, domain, key, xy, plot=False, id=None, lw=4, order=5, reduce_func=np.median, filter_width= 2):
        '''
        Retrieves intensity profile from pfm image, and finds the domain size by peak counting before and after max-filtering-binarization-max-filtering.

        Parameters
        ----------
        domain : np.array
            The pfm image.
        key : str
            The key of the line to draw the cross section from.
        xy : list
            The coordinates of the line to draw the cross section from. x and y coordinates are separated.
        plot : bool, optional
            Whether to plot the cross section. The default is False.
        id : int, optional
            The id of the grain. The default is None.
        lw : int, optional
            The width of the line to sample intensities. Perpendicular to direction. The default is 4.
        order : int, optional
            The order of the spline interpolation. The default is 5, corresponding to Bi-quintic spline interpolation.
        reduce_func : function, optional
            The function to reduce the line intensity to a single value. The default is np.median.
        filter_width : int, optional
            The width of the max filter. The default is 2.
            
        '''

        intensity, x = skm_crossection(domain, xy[0], xy[1], lw=lw, order=order, reduce_func=reduce_func)


        intensity = ndi.maximum_filter1d(intensity, size=filter_width)

        binary_profile = intensity > np.mean(intensity)

        binary_profile = ndi.maximum_filter1d(binary_profile, size=filter_width)

        binary_peaks,_ = sp.signal.find_peaks(binary_profile)

        binary_grad = np.gradient(binary_profile.astype(float))

        p_peaks, _ = sp.signal.find_peaks(binary_grad)
        n_peaks, _ = sp.signal.find_peaks(-binary_grad)

        normal_count = (x[binary_peaks][-1] - x[binary_peaks][0])*self.res/(len(binary_peaks)-1)/2

        self.domain_binaries[key].append(normal_count)

        try:

            maxp = np.max(p_peaks)
            maxn = np.max(n_peaks)
            minp = np.min(p_peaks)
            minn = np.min(n_peaks)

            maxs = np.max([maxp, maxn])
            mins = np.min([minp, minn])

            grad_overall = np.abs(maxs-mins)*self.res/(len(p_peaks) + len(n_peaks) -1)
            self.sizes_old.append(grad_overall)
        except:
            pass



        len_p = len(p_peaks)
        len_n = len(n_peaks)
        len_min = min(len_p, len_n)

        domain1 = np.abs([p_peaks[i] - n_peaks[i] for i in range(len_min)])
        domain2 = np.abs([p_peaks[i+ int(len_p>len_n)] - n_peaks[i+ int(len_p < len_n)] for i in range(len_min-1, -1, -1)])

        m1 = np.mean(domain1)*self.res
        m2 = np.mean(domain2)*self.res

        std1 = np.std(domain1)*self.res
        std2 = np.std(domain2)*self.res

        cm, cs = calc_combined_mean_std([m1, m2], [std1, std2], [len(domain1), len(domain2)])

        self.domain_grad_binaries_mean[key].append(cm)
        self.domain_grad_binaries_std[key].append(cs)
        self.domain_grad_binaries_ws[key].append(len(domain1) + len(domain2))

        combined = np.concatenate((domain1, domain2))
        self.sizes.extend(combined)

        if plot:
            print(f"Plotting {id}_{key}")
            plt.figure(figsize=(10,6))
            plt.plot(x*self.res, intensity)
            plt.plot(x*self.res, binary_profile + np.max(intensity))
            plt.scatter(x[p_peaks]*self.res, 0*x[p_peaks]+ 0.5 + np.max(intensity), marker="x")
            plt.scatter(x[n_peaks]*self.res,  0*x[n_peaks]+ 0.5 + np.max(intensity), marker="x") 
            savename = f"{id}_{key}_grad.pdf"
            plt.savefig(savename)
            plt.close()

        return


def automatic_grain_segmentation(image: np.array, show=True):
    '''
    Still working progress for automatic grain size estimation.
    '''

    # inverted = np.invert(image) TODO: Figure this out
    renorm = (image - np.min(image))/(np.max(image) - np.min(image)) * 255
    blurred = cle.gaussian_blur(renorm, sigma_x=1, sigma_y=1) #TODO: Find nice params
    inverted = np.gradient(blurred)
    binary = cle.binary_not(cle.threshold_otsu(inverted))
    labels = cle.voronoi_labeling(binary)

    if show:
        cle.imshow(image)
        cle.imshow(blurred)
        cle.imshow(inverted)
        cle.imshow(binary)
        cle.imshow(labels, labels=True)

    return labels

def open_grain_segmentation(image:np.array, show=True):
    """
    Inputs manually thresholded and inverted image, and performs binary mask creation and labelling.
    """
    # renorm = (image - np.min(image))/(np.max(image) - np.min(image)) * 255
    # binary = cle.binary_not(renorm)
    labels = cle.voronoi_labeling(image)

    if show:
        # cle.imshow(renorm)
        # cle.imshow(binary)
        cle.imshow(labels, labels=True)
    
    return labels


def draw_crossection(z, xs, ys):
    """
    Works, gives nice interpolation, but skm_crossection allows for line width.

    Draws a single intensity profile

    Parameters
    ----------
    z : np.array
        The 2D image.
    xs : list
        The x coordinates of the line; start and end.
    ys : list
        The y coordinates of the line; start and end.

    Returns
    -------
    zi : np.array
        The retrieved intensity profile along the designated line.
    ri : np.array
        The distance along the line.
    line : np.array
        The coordinates of the line.
    """

    x, y = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
    # Coordinates of the line we'd like to sample along
    line = np.array([xs, ys])

    # Convert the line to pixel/index coordinates
    x_world, y_world = np.array(xs), np.array(ys)
    col = z.shape[1] * (x_world - x.min()) / x.ptp()
    row = z.shape[0] * (y_world - y.min()) / y.ptp() 

    # Interpolate the line at "num" points...
    num = 1000
    row, col = [np.linspace(item[0], item[1], num) for item in [row, col]]

    # Extract the values along the line, using cubic interpolation
    zi = sp.ndimage.map_coordinates(z, np.vstack((row, col)))

    ri = np.sqrt((row - row[0])**2 + (col - col[0])**2)

    return zi, ri, line


def skm_crossection(z, xs, ys, lw=4, order=5, reduce_func=np.median):
    '''
    Draws a cross section of a 2D image using skimage.measure.profile_line

    Parameters
    ----------
    z : np.array
        The 2D image.
    xs : list
        The x coordinates of the line.
    ys : list
        The y coordinates of the line.
    lw : int, optional
        The width of the line to sample intensities. Perpendicular to direction. The default is 4.
    order : int, optional
        The order of the spline interpolation. The default is 5, corresponding to Bi-quintic spline interpolation.
    reduce_func : function, optional
        The function to reduce the line intensity to a single value. The default is np.median.

    Returns
    -------
    intensity : np.array
        The retrieved intensity profile along the designated line.
    x : np.array
        The x coordinates of the line.

    '''

    start = np.array([ys[0], xs[0]]) #TODO: Check.
    end = np.array([ys[1], xs[1]])

    intensity = skm.profile_line(z, start, end, linewidth=lw, order=order, reduce_func=reduce_func)
    x = np.arange(len(intensity))

    return intensity, x

def calc_combined_mean_std(means,stds, weights):
    '''
    Calculates the combined mean and standard deviation of an array of values.
    https://www.geeksforgeeks.org/combined-standard-deviation-meaning-formula-and-example/

    Parameters
    ----------
    means : np.array
        The array of means to combine.
    stds : np.array
        The array of standard deviations to combine.
    weights : np.array
        The weights corresponding to the sample size of each array.

    Returns
    -------
    combined_mean : float
        The combined mean.
    combined_std : float
        The combined standard deviation.

    '''

    means = np.array(means)
    stds = np.array(stds)
    weights = np.array(weights)

    combined_mean = np.sum(means*weights)/np.sum(weights)

    nom = np.sum(weights*(stds**2)) + np.sum(weights*(means - combined_mean)**2)
    den = np.sum(weights)

    combined_std = np.sqrt(nom/den)

    return combined_mean, combined_std



def plane_level(height: np.array):
    '''
    Old plane level. See transformations.py for updated functions. 
    '''
    XX, YY = np.meshgrid(np.arange(height.shape[0]), np.arange(height.shape[1]))
    data = np.c_[XX.ravel(), YY.ravel(), height.ravel()]

    order = 1    # 1: linear, 2: quadratic
    if order == 1:
        # best-fit linear plane
        # A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
        A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
        C,_,_,_ = sp.linalg.lstsq(A, data[:,2])    # coefficients
        
        # evaluate it on grid
        Z = C[0]*XX + C[1]*YY + C[2]

    return Z, C


def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  
        return A * np.sin(w*t + p) + c
    
    
    popt, pcov = sp.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}


def calc_grain_size(obj, pts, res):
    """
    Calculates the grain size of a single grain.
    """

    mask = np.zeros_like(obj)

    polygon = cv2.fillPoly(mask, [pts], 1).astype(bool)

    area = np.sum(polygon)
    diameter = np.sqrt(area/np.pi) * 2 * res

    return diameter, area, mask

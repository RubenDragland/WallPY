
import matplotlib.pyplot as plt
from rcParams import *
import sys
import attributes as attr



##TODO: Needs a kwargs update. Often not initialied. 
class Ax:
    """
    A class for handling an ax object in a matplotlib figure.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The figure to add the ax object to.
    **kwargs : dict, optional
        The keyword arguments for the ax object. The default is:

            {
            "xlabel": "",
            "ylabel": "",
            "legend": False,
            "labels": [],
            "scalebar": False,
            "colorbar": False,
            "values": None,
            "vmin": 0,
            "vmax": None,
            "vmax_std": None,
            "origin": "lower",
            }

    Methods
    -------
    set_labels()
        Sets the x- and y-labels of the ax object.
    update_kwargs(**kwargs)
        Updates the keyword arguments for the ax object.
    plot_cAFM(datafile, key:str="CR", **kwargs)
        Plots a cAFM scan.
    plot_inset(datafile, key:str="CR", **inset_kwargs)
        Plots an inset of a cAFM scan.
    """

    default_kwargs = {
        "xlabel": "",
        "ylabel": "",
        "legend": False,
        "labels": [],
        "scalebar": False,
        "colorbar": False,
        "values": None,
        "vmin": 0,
        "vmax": None,
        "vmax_std": None,
        "origin": "lower",
        #TODO: Add axis off? 
        }


    def __init__(self, fig, **kwargs):
        #TODO: possibly update a bit because only use fig.create_subplot to create these objects. 

        if "ax" in kwargs:
            self.ax = kwargs["ax"]
        else:
            self.ax = fig.add_subplot() #TODO: Will have to depend on figure etc. 


        self.fig = fig
        self.kwargs = kwargs

        for key, value in self.default_kwargs.items():
            if key not in self.kwargs:
                self.kwargs[key] = value
        
        self.set_labels()
        return
    
    def set_labels(self):
        self.ax.set_xlabel(self.kwargs["xlabel"])
        self.ax.set_ylabel(self.kwargs["ylabel"])
        return
    
    def update_kwargs(self, **kwargs):

        for key, value in kwargs.items():
            self.kwargs[key] = value
        return
    
    def plot_standard(self, data, xres, **kwargs ):
        """
        Plots a standard image using imshow.
        #TODO: Fix kwargs...
        """
        std_kwargs = {
            "cmap": "magma", 
            "vmin": None,
            "vmax": None,
            "raw": False,
            "normal": False,
            "binomial": False,
            "quantile": 0.5,
        }

        for key, value in kwargs.items():
            if key in std_kwargs:
                std_kwargs[key] = value

        #TODO: Do not remember the logic behind the above, but now it is possible to override. Check some day. 
        for key, value in kwargs.items():
            if key not in std_kwargs:
                std_kwargs[key] = value
        
        #TODO: Automation of vmin and vmax. 
        """
        If Gaussian use HWHM in both directions, accounting for asymmetry.
        If proper single crystal, do two curve fittings and use peaks-ish.
        """

        if not(std_kwargs["raw"]) and std_kwargs["vmin"] is None and std_kwargs["vmax"] is None:

            if std_kwargs["normal"]:
                
                hist, edges = np.histogram(data, bins=100)

                pvalue = np.max(hist)
                peak = edges[np.argmax(hist)]

                lower = edges[np.where(hist > pvalue*std_kwargs["quantile"])[0][0]]
                upper = edges[np.where(hist > pvalue*std_kwargs["quantile"])[0][-1]]

                std_kwargs["vmin"] = lower
                std_kwargs["vmax"] = upper
                #TODO: Something like this. 

                print(lower, upper)

            elif std_kwargs["binomial"]:

                hist, edges = np.histogram(data, bins=100)

                low_edges = edges[np.where(edges < np.mean(data))]
                high_edges = edges[np.where(edges > np.mean(data))]
                below_mean = hist[np.where(edges < np.mean(data))]
                above_mean = hist[np.where(edges > np.mean(data))]

                minvalue = low_edges[np.argmax(below_mean)]
                maxvalue = high_edges[np.argmax(above_mean)]

                std_kwargs["vmin"] = minvalue
                std_kwargs["vmax"] = maxvalue

                print(minvalue, maxvalue)

                #Something like this. Does not fully function. 
            else:
                # Some bullshit but fast way
                std_kwargs["vmin"] = np.min(data)
                std_kwargs["vmax"] = np.max(data)

        
        im = self.ax.imshow(data, cmap=std_kwargs["cmap"], vmin=std_kwargs["vmin"], vmax=std_kwargs["vmax"], origin=self.kwargs["origin"])

        if std_kwargs["colorbar"]:
            attr.add_colorbar(self.ax, im)
        
        if std_kwargs["scalebar"]: #TODO: Some mishaps between ax and this plotting function. Fix. Possibly combine from all over the place
            attr.add_scalebar(self.ax, xres)

        self.ax.set_xticks([])
        self.ax.set_yticks([])



        return

    
    def plot_cAFM(self, datafile, key:str="CR", **kwargs): #TODO: Fix imports
        """
        Plots a cAFM scan.
        """

        self.update_kwargs(**kwargs)

        #TODO: More universal. 

        if self.kwargs["values"] is not None:
            values = self.kwargs["values"]*1e12
        else:
            values = datafile[key]*1e12
        
        if self.kwargs["vmax_std"] is not None:
            vmax_std = self.kwargs["vmax_std"]
            vmax = np.std(values)*vmax_std
        else:
            vmax = None
        
        im = self.ax.imshow(values, vmax=vmax, origin=self.kwargs["origin"])      
        self.ax.axis("off")
        self.fig.colorbar(im, ax=self.ax, label="Current (pA)") #TODO: Fix the label. Or this is a bit hard-coded. See if more general is possible eventually.
        self.set_labels()
        # attr.add_scalebar(self.ax, datafile.x_res) #TODO: Fix this.
        return
    

    
    def plot_inset(self, datafile, key:str="CR", **inset_kwargs):

        default_kwargs = {
                "x1": 0,
                "y1": 0,
                "x2": 10,
                "y2": 10,
                "left_x": 0,
                "left_y": 0,
                "width": 0.5,
                "height": 0.5,
                "indicate": True,
                "values": None,
                "label": "value"
            }
        
        for kwarg_key, kwarg_value in default_kwargs.items():
            if kwarg_key not in inset_kwargs:
                inset_kwargs[kwarg_key] = kwarg_value

        if inset_kwargs["values"] is not None:
            values = inset_kwargs["values"] #*1e12
        else:
            values = datafile[key] #*1e12
        
        if self.kwargs["vmax_std"] is not None:
            vmax_std = self.kwargs["vmax_std"]
            vmax = np.std(values)*vmax_std
        else:
            vmax = None
        
        im = self.ax.imshow(values, vmax=vmax, origin="lower")      
        self.ax.axis("off")
        self.fig.colorbar(im, ax=self.ax, label=inset_kwargs["label"]) #TODO: Fix the label in kwargs or something.
        # attr.add_scalebar(self.ax, datafile.x_res) #TODO: Fix this.


        axins = self.ax.inset_axes([inset_kwargs["left_x"], inset_kwargs["left_y"], inset_kwargs["width"], inset_kwargs["height"]])
        axins.imshow(values, vmax=vmax, origin="lower")
        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_xlim(inset_kwargs["x1"], inset_kwargs["x2"])
        axins.set_ylim(inset_kwargs["y1"], inset_kwargs["y2"])
        if inset_kwargs["indicate"]:
            rp, lines = self.ax.indicate_inset_zoom(axins, edgecolor="black", lw=1)
            for l in lines:
                l.set_linestyle("--")
                l.set_color("black")
                l.set_linewidth(1)


        
        return


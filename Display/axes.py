
import matplotlib.pyplot as plt
from rcParams import *
import sys
import attributes as attr

# sys.path.append(sys.path[0] + "\\..\\IO")
# from ..IO.classes import CypherBatch, CypherFile



class Ax:

    default_kwargs = {
        "xlabel": "",
        "ylabel": "",
        "legend": False,
        "labels": [],
        "scalebar": False,
        "colorbar": False,
        "data": None,
        "vmin": 0,
        "vmax": None,
    } #TODO: Find the necessary kwargs

    def __init__(self, fig, **kwargs):
        self.ax = fig.add_subplot()
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
    
    def plot_cAFM(self, cypherfile, key:str="CR", **kwargs): #TODO: Fix imports
        """
        Plots a cAFM scan.
        """

        self.update_kwargs(**kwargs)

        #Figure out something universal for this.

        if self.kwargs["data"] is not None:
            values = self.kwargs["data"]*1e12
        else:
            values = cypherfile[key]*1e12
        
        if self.kwargs["vmax_std"] is not None:
            vmax_std = self.kwargs["vmax_std"]
            vmax = np.std(values)*10
        else:
            vmax = None
        
        im = self.ax.imshow(values, vmax=vmax)      
        self.ax.axis("off")
        self.fig.colorbar(im, ax=self.ax, label="Current (pA)")
        self.set_labels()
        attr.add_scalebar(self.ax, cypherfile.x_res)
        return
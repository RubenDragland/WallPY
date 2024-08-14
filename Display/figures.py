
import matplotlib.pyplot as plt
from rcParams import *
from axes import Ax
# from ..IO.classes import CypherBatch, CypherFile
import os
import time
import attributes as attr


import math


def plot_overview(gwyfile, size=10):
    cal_size = len(gwyfile.channel_names)
    cal_grid = math.ceil(math.sqrt(cal_size))
    fig, axs = plt.subplots(cal_grid, cal_grid, figsize=(size, size))
    for i, (name,ax) in enumerate(zip(gwyfile.channel_names, axs.flatten())):
        map = "gray" if "Height" in name else "magma"
        ax.imshow(gwyfile[i], cmap=map)
        ax.set_title(name)
    plt.show()
    return

class FigureSinglePlot:

    """
    A class for handling a figure with a single matplotlib ax.

    Attributes
    ----------
    datafile : object
        The container of the data to plot.
    **kwargs : dict, optional
        The keyword arguments for the figure. The default is:
            {
            "show": False,
            "path": "",
            "filename": f"FigureSinglePlot_{int(time.time())}",
            "extension": ".pdf",
            "figsize": (DEFAULT_FIGSIZE[0], DEFAULT_FIGSIZE[1]),
            "dpi": 300,
            }
    
    Methods
    -------
    create_figure()
        Creates the figure and initializes the ax object.
    __call__()
        Finalizes and saves the figure to the specified path.    
    """


    default_kwargs = {
        "show": False,
        "path": "",
        "filename": f"FigureSinglePlot_{int(time.time())}",
        "extension": ".pdf",
        "figsize": (DEFAULT_FIGSIZE[0], DEFAULT_FIGSIZE[1]),
        "dpi": 300,
        "transparent": False,
        "standard_scale": True, #Meaning scaling rcParams to the figure size.
    } #TODO: Find the necessary kwargs

    def __init__(self, datafile, **kwargs): #TODO: DO not use datafile. 

        self.datafile = datafile
        self.kwargs = kwargs

        for key, value in self.default_kwargs.items():
            if key not in self.kwargs:
                self.kwargs[key] = value
        
        self.create_figure()     

        return

    def __call__(self):

        self.fig.savefig(os.path.join(self.kwargs["path"],self.kwargs["filename"]+self.kwargs["extension"]) ,dpi=self.kwargs["dpi"], transparent=self.kwargs["transparent"]) 

        if self.kwargs["show"]: #TODO: Create defaults
            plt.show()
        
        return
    
    def create_figure(self): #-> tuple(plt.figure, Ax):

        self.fig = plt.figure(figsize=(self.kwargs["figsize"]))

        if self.kwargs["standard_scale"]:
            adapt_mpl_fig(self.fig) #TODO: Introduced to get normal scale between figsize and rcParams.

        self.Ax = Ax(self.fig, **self.kwargs)

        return self.fig, self.Ax 
    
        
class FigureSubplots(FigureSinglePlot):

    """
    A class for handling a figure with multiple matplotlib ax objects.

    Attributes
    ----------
    databatch : object
        The container of the data to plot.
    **kwargs : dict, optional
        The keyword arguments for the figure. The default is:
            {
            "show": False,
            "path": "",
            "filename": f"FigureSubplots_{int(time.time())}",
            "extension": ".pdf",
            "figsize": (DEFAULT_FIGSIZE[0], DEFAULT_FIGSIZE[1]),
            "dpi": 300,
            "nrows": 1,
            "ncols": 2,
            "standard_size": True,
            }

    Methods
    -------
    create_figure()
        Creates the figure and initializes the ax objects.
    create_subplot()
        Creates a subplot and returns the ax object.
    label_subplots()
        Adds alphabetic labels to the ax objects.
    __call__()
        Finalizes and saves the figure to the specified path.
    """

    add_on_kwargs = {
        "filename": f"FigureSubplots_{int(time.time())}",
        "nrows": 1,
        "ncols": 2,
        "standard_size": True,
    } #TODO: Find the necessary kwargs

    def __init__(self, databatch, **kwargs ):
        #TODO: Not necessary to include data file. 

        self.kwargs = kwargs

        for key, value in self.add_on_kwargs.items():
            if key not in self.kwargs:
                self.kwargs[key] = value

        super().__init__(databatch, **kwargs)

        self.Axes = [] #List of Ax objects.
        return
    
    def create_figure(self):
        # TODO: Option to set arb. figsize
        if self.kwargs["standard_size"]:
            self.fig = plt.figure(figsize=(self.kwargs["figsize"][0]*self.kwargs["ncols"], self.kwargs["figsize"][1]*self.kwargs["nrows"]))
            self.gs = self.fig.add_gridspec(self.kwargs["nrows"], self.kwargs["ncols"])
        else:
            self.fig = plt.figure(figsize=(self.kwargs["figsize"]))
            self.gs = self.fig.add_gridspec(self.kwargs["nrows"], self.kwargs["ncols"])

        if self.kwargs["standard_scale"]:
            adapt_mpl_fig(self.fig) #TODO: Introduced to get normal scale between figsize and rcParams.

        return
    
    def create_subplot(self, row=0, col=0, row_span=None, col_span=None, **kwargs):
        #TODO: Tungvint
        #TODO: Docstring
        #TODO: Better communicate that span is end point.

        """
        Create a subplot in the figure based on gridspec.

        Parameters
        ----------
        row : int, optional
            The row index of the subplot. The default is 0.
        col : int, optional
            The column index of the subplot. The default is 0.
        row_span : int, optional
            The end row index of the subplot. The default is None.
        col_span : int, optional
            The end column index of the subplot. The default is None.
        **kwargs : dict, optional
            The keyword arguments for the subplot. The default is:
                {
                "sharex": None,
                "sharey": None,
                "letter": "a",
                "projection": None,
                }

        """

        subplot_kwargs = {
            "sharex": None,
            "sharey": None,
            "letter": "a",
            "projection": None,
        }

        for key, value in subplot_kwargs.items():
            if key not in kwargs:
                kwargs[key] = value


        if row_span is None and col_span is None:
            ax = self.fig.add_subplot(self.gs[row,col], projection=kwargs["projection"] )# sharex=kwargs["sharex"], sharey=kwargs["sharey"]) #TODO: Fix share. and Share colorbar etc. 
            self.Axes.append(Ax(self.fig, ax=ax, **kwargs))
        elif row_span is None:
            ax = self.fig.add_subplot(self.gs[row,slice(col, col_span)], projection=kwargs["projection"])# sharex=kwargs["sharex"], sharey=kwargs["sharey"])
            self.Axes.append(Ax(self.fig, ax=ax, **kwargs))
        elif col_span is None:
            ax = self.fig.add_subplot(self.gs[slice(row, row_span),col], projection=kwargs["projection"]) # sharex=kwargs["sharex"], sharey=kwargs["sharey"])
            self.Axes.append(Ax(self.fig, ax=ax, **kwargs))
        else:
            ax = self.fig.add_subplot(self.gs[slice(row,row_span), slice(col, col_span) ], projection=kwargs["projection"])# sharex=kwargs["sharex"], sharey=kwargs["sharey"])
            self.Axes.append(Ax(self.fig, ax=ax, **kwargs))

        return self.Axes[-1]
    
    def label_subplots(self, **kwargs):
        """
        Automatic labelling of the subplots.
        """
        for i, Ax in enumerate(self.Axes):
            attr.add_alphabetic_label(Ax.ax, chr(i+97) , **kwargs)
        return
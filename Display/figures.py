
import matplotlib.pyplot as plt
from rcParams import *
from axes import Ax
# from ..IO.classes import CypherBatch, CypherFile
import os
import time
import attributes as attr

class FigureSinglePlot:


    default_kwargs = {
        "show": False,
        "path": "",
        "filename": f"FigureSinglePlot_{int(time.time())}",
        "extension": ".pdf",
        "figsize": (DEFAULT_FIGSIZE[0], DEFAULT_FIGSIZE[1]),
        "dpi": 300,
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

        self.fig.savefig(os.path.join(self.kwargs["path"],self.kwargs["filename"]+self.kwargs["extension"]) ,dpi=self.kwargs["dpi"]) 

        if self.kwargs["show"]: #TODO: Create defaults
            plt.show()
        
        return
    
    def create_figure(self): #-> tuple(plt.figure, Ax):

        self.fig = plt.figure(figsize=(self.kwargs["figsize"]))

        self.Ax = Ax(self.fig, **self.kwargs)

        return self.fig, self.Ax 
    
        
class FigureSubplots(FigureSinglePlot):

    add_on_kwargs = {
        "filename": f"FigureSubplots_{int(time.time())}",
        "nrows": 1,
        "ncols": 2,
        "standard_size": True,
    } #TODO: Find the necessary kwargs

    def __init__(self, databatch, **kwargs ):

        self.kwargs = kwargs

        for key, value in self.add_on_kwargs.items():
            if key not in self.kwargs:
                self.kwargs[key] = value

        super().__init__(databatch, **kwargs)

        self.Axes = [] #List of Ax objects.
        return
    
    def create_figure(self):

        if self.kwargs["standard_size"]:
            self.fig = plt.figure(figsize=(self.kwargs["figsize"][0]*self.kwargs["ncols"], self.kwargs["figsize"][1]*self.kwargs["nrows"]))
            self.gs = self.fig.add_gridspec(self.kwargs["nrows"], self.kwargs["ncols"])
        else:
            self.fig = plt.figure(figsize=(self.kwargs["figsize"]))
            self.gs = self.fig.add_gridspec(self.kwargs["nrows"], self.kwargs["ncols"])

        return
    
    def create_subplot(self, row=0, col=0, row_span=None, col_span=None, **kwargs):
        #TODO: Tungvint

        subplot_kwargs = {
            "sharex": None,
            "sharey": None,
            "letter": "a",
        }

        for key, value in subplot_kwargs.items():
            if key not in kwargs:
                kwargs[key] = value


        if row_span is None and col_span is None:
            ax = self.fig.add_subplot(self.gs[row,col], )# sharex=kwargs["sharex"], sharey=kwargs["sharey"]) #TODO: Fix share. and Share colorbar etc. 
            self.Axes.append(Ax(self.fig, ax=ax, **kwargs))
        elif row_span is None:
            ax = self.fig.add_subplot(self.gs[row,slice(col, col_span)],)# sharex=kwargs["sharex"], sharey=kwargs["sharey"])
            self.Axes.append(Ax(self.fig, ax=ax, **kwargs))
        elif col_span is None:
            ax = self.fig.add_subplot(self.gs[slice(row, row_span),col], ) # sharex=kwargs["sharex"], sharey=kwargs["sharey"])
            self.Axes.append(Ax(self.fig, ax=ax, **kwargs))
        else:
            ax = self.fig.add_subplot(self.gs[slice(row,row_span), slice(col, col_span) ],)# sharex=kwargs["sharex"], sharey=kwargs["sharey"])
            self.Axes.append(Ax(self.fig, ax=ax, **kwargs))

        return self.Axes[-1]
    
    def label_subplots(self, **kwargs):
        for i, Ax in enumerate(self.Axes):
            attr.add_alphabetic_label(Ax.ax, chr(i+97) , **kwargs)
        return

import matplotlib.pyplot as plt
from rcParams import *
from axes import Ax
# from ..IO.classes import CypherBatch, CypherFile
import os

class FigureSubplots:

    def __init__(self, cypherbatch, **kwargs ):

        self.cypherbatch = cypherbatch
        self.kwargs = kwargs





class FigureSinglePlot:


    default_kwargs = {
        "show": False,
        "path": "",
        "filename": "",
        "extension": ".pdf",
        "figsize": (DEFAULT_FIGSIZE[0], DEFAULT_FIGSIZE[1]),
        "dpi": 300,
    } #TODO: Find the necessary kwargs

    def __init__(self, cypherfile, **kwargs):

        self.cypherfile = cypherfile
        self.kwargs = kwargs

        for key, value in self.default_kwargs.items():
            if key not in self.kwargs:
                self.kwargs[key] = value
        
        self.create_figure()     

        return

    def __call__(self):

        self.fig.savefig(os.path.join(self.kwargs["path"],self.kwargs["filename"]+self.kwargs["extension"]) ,dpi=self.kwargs["dpi"]) #TODO: Nothing saved.

        if self.kwargs["show"]: #TODO: Create defaults
            plt.show()
        
        return
    
    def create_figure(self) :#-> tuple(plt.figure, Ax):

        self.fig = plt.figure(figsize=(self.kwargs["figsize"]))

        self.Ax = Ax(self.fig, **self.kwargs)

        return self.fig, self.Ax #Returns if overwriting the figure is necessary.
    
        

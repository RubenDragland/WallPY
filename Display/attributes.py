import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.patheffects as PathEffects
from matplotlib_scalebar.scalebar import ScaleBar

from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.patches as mpatches

import numpy as np



def add_scalebar(ax, res, **kwargs):
    """
    Adds a scalebar to the lower right corner of the ax object provided.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The ax object to add the scalebar to.
    res : float
        The resolution of the image in meters per pixel.
    kwargs : dict, optional
        The keyword arguments for the scalebar. The default is:
            {
            "fixed_value": None,
            "units": f"um",
            "color": "white",
            "location": 4,
            "frameon": True,
            "box_color" :"black",
            "box_alpha" : 0.45,
            "scale_loc": "bottom",
            "length_fraction":0.35,
            "width_fraction": 0.01,
            "scale_formatter":None, depends on units: (lambda value, unit:   f"{value} "  +r"$\\upmu$m"),
            "font_properties": {"size": 12}, TODO: But adjust when finding correct scaling
            }
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The ax object with the scalebar added.
    """

    mu_formatter = lambda value, unit:   f"{value} "  +r"$\\upmu$m"
    scale_kwargs ={
            "fixed_value": None,
            "units": f"um",
            "color": "white",
            "location": 4,
            "frameon": True,
            "box_color" :"black",
            "box_alpha" : 0.45,
            "scale_loc": "bottom",
            "length_fraction":0.35,
            "width_fraction": 0.01,
            "scale_formatter":None,
            "font_properties": {"size": 12},

        }

    if "scalebar_kwargs" not in kwargs:

        for key, value in kwargs.items():
            if key in scale_kwargs:
                scale_kwargs[key] = value
   
    else:

        new_kwargs = kwargs["scalebar_kwargs"]
        for key, value in new_kwargs.items():
            if key in scale_kwargs:
                scale_kwargs[key] = value

    size = res*1e6
    if size * scale_kwargs["length_fraction"] * ax.get_images()[0].get_array().shape[1] > 1:
        scale_kwargs["scale_formatter"] = mu_formatter


    bar = ScaleBar(dx = size, **scale_kwargs)
    ax.add_artist(bar)
    return ax


def add_polarization_direction(ax, **kwargs):
    """
    Adds a polarization direction to the plot at the specified position.
    The function makes sure the arrow is a sufficient size based on the size of the plot; i.e. hardcoded sizes for now.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The ax object to add the polarization direction to.
    polarization_kwargs : dict, optional
        The keyword arguments for the polarization direction. The default is:
            {
            "type": "out",
            "color": "white",
            "lw": 1,
            "alpha": 1,
            "pos": (100, 100),
            "angle": 0
            }
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The ax object with the polarization direction added.
    
    """

    polarization_kwargs={
            "type": "out",
            "color": "white",
            "lw": 1,
            "alpha": 1,
            "pos": (100, 100),
            "angle": 0,
            "ratio_size": 0.05, 
    }


    if "polarization_kwargs" not in kwargs:

        for key, value in kwargs.items():
            if key in polarization_kwargs:
                polarization_kwargs[key] = value
   
    else:

        new_kwargs = kwargs["polarization_kwargs"]
        for key, value in new_kwargs.items():
            if key in polarization_kwargs:
                polarization_kwargs[key] = value

    polarization_kwargs["size"] = polarization_kwargs["ratio_size"]*ax.get_images()[0].get_array().shape[1] #TODO: Possibly update when automating size of things based on figure size.

    def arrow_right(ax, pos_tip, **arrow_kwargs):
        '''
        Adds an arrow pointing to the right.
        Choose angle to point in another direction.
        '''

        r= arrow_kwargs["size"]
        width = r # TODO: Adjustable sizes possibly?
        dx = r* np.cos(np.deg2rad(arrow_kwargs["angle"]))
        dy = r * np.sin(np.deg2rad(arrow_kwargs["angle"]))
        arrow = mpatches.Arrow(pos_tip[1]-dx, pos_tip[0]-dy, dx, dy,width = width, color=arrow_kwargs["color"], alpha=arrow_kwargs["alpha"], lw=arrow_kwargs["lw"])

        ax.add_patch(arrow)

        return
    
    def pol_in(ax, pos_tip, **in_kwargs):
        '''
        Adds a circle with a cross indicating the polarization direction inwards.
        '''

        r= in_kwargs["size"]
        
        circle = mpatches.Circle(pos_tip, radius=r, edgecolor=in_kwargs["color"], alpha=in_kwargs["alpha"], lw=in_kwargs["lw"], facecolor="none") #TODO: Fix kwargs.
        ax.add_patch(circle)

        cross_path_data = [
    (pos_tip[0] - r*1/np.sqrt(2), pos_tip[1] - r*1/np.sqrt(2)), (pos_tip[0] + r*1/np.sqrt(2), pos_tip[1] + r*1/np.sqrt(2)),
    (pos_tip[0] - r*1/np.sqrt(2), pos_tip[1] + r*1/np.sqrt(2)), (pos_tip[0] + r*1/np.sqrt(2), pos_tip[1] - r*1/np.sqrt(2))
]

        cross_path = Path(cross_path_data, [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO])

        cross = PathPatch(cross_path, color= in_kwargs["color"], alpha=in_kwargs["alpha"], lw=in_kwargs["lw"])
        ax.add_patch(cross)

        return
    
    def pol_out(ax, pos_tip, **out_kwargs):
        '''
        Adds a circle with a dot indicating the polarization direction outwards.
        '''

        r = out_kwargs["size"]

        circle = mpatches.Circle(pos_tip, radius=r, edgecolor=out_kwargs["color"], alpha=out_kwargs["alpha"], lw=out_kwargs["lw"], facecolor="none") #TODO: Fix kwargs.
        ax.add_patch(circle)

        dot = mpatches.Circle(pos_tip, radius=r/5, edgecolor=out_kwargs["color"], alpha=out_kwargs["alpha"], lw=out_kwargs["lw"], facecolor=out_kwargs["color"]) #TODO: Fix kwargs.

        ax.add_patch(dot)

        return
    
    symbols = {
        "arr": arrow_right,
        "in": pol_in,
        "out": pol_out,

    }

    symbols[polarization_kwargs["type"]](ax, polarization_kwargs["pos"], ax.get_images()[0].get_array().shape, **polarization_kwargs)

    return ax


def add_alphabetic_label(ax, letter, **kwargs):
    """
    Adds an alphabetic label to the plot at the specified position.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The ax object to add the alphabetic label to.
    letter : str
        The letter to add to the plot.
    kwargs : dict, optional
        The keyword arguments for the alphabetic label. The default is:
            {
            "color": "white",
            "location": 1,
            "frameon": True,
            "box_color" :"black",
            "box_alpha" : 0.5,
            "formatter":0,
            "pad": 0,
            "location": "upper left",
            "horizontalalignment": "left",
            "verticalalignment": "top",
            "position": (0,1)
            }
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The ax object with the alphabetic label added.
    """

    formatter ={
        0: lambda letter,:   f"{letter}",
        1: lambda letter:   f"{letter})",
        2: lambda letter:   f"({letter})",
    }

    fig = ax.get_figure()
    import matplotlib.transforms as mtransforms
    trans = mtransforms.ScaledTranslation(0.05,-0.17,fig.dpi_scale_trans)

    alpha_kwargs ={
            "color": "white",
            "location": 1,
            "frameon": True,
            "box_color" :"black",
            "box_alpha" : 0.5,
            "formatter":0,
            "pad": 0,
            "location": "upper left",
            "horizontalalignment": "left",
            "verticalalignment": "top",
            "position": (0,1),
            "size": 12,
        }
        

    if "alpha_kwargs" not in kwargs:

        for key, value in kwargs.items():
            if key in alpha_kwargs:
                alpha_kwargs[key] = value

    
    else:

        new_kwargs = kwargs["alpha_kwargs"]

        for key, value in new_kwargs.items():
            if key in alpha_kwargs:
                alpha_kwargs[key] = value

    text = ax.text(alpha_kwargs["position"][0],alpha_kwargs["position"][1], formatter[alpha_kwargs["formatter"]](letter), horizontalalignment=alpha_kwargs["horizontalalignment"], verticalalignment=alpha_kwargs["verticalalignment"], size= alpha_kwargs["size"], color=alpha_kwargs["color"], transform=ax.transAxes, bbox=dict(facecolor=alpha_kwargs["box_color"], alpha=alpha_kwargs["box_alpha"], edgecolor="none", pad=alpha_kwargs["pad"]))
    bbox = text.get_window_extent().transformed(ax.transAxes.inverted())

    #TODO: Do not dare to touch this too much. 
    
    return ax


#TODO: Believe it works nicely? Improve some things?
def add_colorbar(ax, mappable, **kwargs):
    colorbar_kwargs = {
        # "cmap": "magma", Not needed or chosen here at all. 
        "label": None,
        "orientation": "vertical",
        "fraction": 0.046,
        "pad": 0.04,
    }

    if "colorbar_kwargs" not in kwargs:

        for key, value in kwargs.items():
            if key in colorbar_kwargs:
                colorbar_kwargs[key] = value
    else:
        colorbar_kwargs = kwargs["colorbar_kwargs"]

    cbar = plt.colorbar(**colorbar_kwargs, ax=ax, mappable=mappable)
    return ax, cbar
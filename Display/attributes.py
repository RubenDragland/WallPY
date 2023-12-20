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
            "units": f"um",
            "color": "white",
            "location": 4,
            "frameon": True,
            "box_color" :"black",
            "box_alpha" : 0.25,
            "scale_loc": "bottom",
            "length_fraction":0.5,
            "width_fraction": 0.01,
            "scale_formatter":scale_formatter
            }
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The ax object with the scalebar added.
    """

    scale_formatter = lambda value, unit:   f"{value} "  +r"$\\upmu$m"
    scale_kwargs ={
            # "dx": size,
            "fixed_value": None,
            "units": f"um", #TODO: Fix the label.
            "color": "white",
            "location": 4,
            "frameon": True,
            "box_color" :"black",
            "box_alpha" : 0.45,
            # "size_vertical": 8,
            "scale_loc": "bottom",
            "length_fraction":0.35,
            "width_fraction": 0.01,
            "scale_formatter":None,
            # "label_top": True,
            "font_properties": {"size": 12},

        }

    if "scalebar_kwargs" not in kwargs:

        for key, value in kwargs.items():
            if key in scale_kwargs:
                scale_kwargs[key] = value

    
    else: #TODO: Finish and prob do not use scalebar package. Use instead normal artist. 

        new_kwargs = kwargs["alpha_kwargs"]

        for key, value in new_kwargs.items():
            if key in scale_kwargs:
                scale_kwargs[key] = value

    size = res*1e6
    bar = ScaleBar(dx = size, **scale_kwargs)
    ax.add_artist(bar)
    return ax


def add_polarization_direction(ax, **polarization_kwargs):
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

    #TODO: Can be redone like the other functions. 
    if "type" not in polarization_kwargs:
        polarization_kwargs["type"] = "out"
    if "color" not in polarization_kwargs:
        polarization_kwargs["color"] = "white"
    if "lw" not in polarization_kwargs:
        polarization_kwargs["lw"] = 1
    if "alpha" not in polarization_kwargs:
        polarization_kwargs["alpha"] = 1
    if "pos" not in polarization_kwargs:
        polarization_kwargs["pos"] = (100, 100) #TODO: Reconsider to choose between pos and position.
    if "angle" not in polarization_kwargs:
        polarization_kwargs["angle"] = 0

    
    def arrow_right(ax, pos_tip, img_dim, **arrow_kwargs):

        # arrow_kwargs = {
        #     "color": "white",
        #     "lw": 1,
        #     "alpha": 1,
        #     "angle": 90,
        #     "width": 0.05,
        # } # TODO: Fix with kwargs

        width = img_dim[1]*0.05 # TODO: Adjustable sizes possibly?
        dx = img_dim[1]*0.05 * np.cos(np.deg2rad(arrow_kwargs["angle"]))
        dy = img_dim[0]*0.05 * np.sin(np.deg2rad(arrow_kwargs["angle"]))
        arrow = mpatches.Arrow(pos_tip[1]-dx, pos_tip[0]-dy, dx, dy,width = width, color=arrow_kwargs["color"], alpha=arrow_kwargs["alpha"], lw=arrow_kwargs["lw"])

        ax.add_patch(arrow)

        return
    
    def pol_in(ax, pos_tip, img_dim, **kwargs):
        '''
        Adds a circle with a cross indicating the polarization direction inwards.
        '''

        r= 0.05*img_dim[1]
        
        circle = mpatches.Circle(pos_tip, radius=r, edgecolor=kwargs["color"], alpha=kwargs["alpha"], lw=kwargs["lw"], facecolor="none") #TODO: Fix kwargs.
        ax.add_patch(circle)

        cross_path_data = [
    (pos_tip[0] - r*1/np.sqrt(2), pos_tip[1] - r*1/np.sqrt(2)), (pos_tip[0] + r*1/np.sqrt(2), pos_tip[1] + r*1/np.sqrt(2)),
    (pos_tip[0] - r*1/np.sqrt(2), pos_tip[1] + r*1/np.sqrt(2)), (pos_tip[0] + r*1/np.sqrt(2), pos_tip[1] - r*1/np.sqrt(2))
]

        cross_path = Path(cross_path_data, [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO])

        cross = PathPatch(cross_path, color= kwargs["color"], alpha=kwargs["alpha"], lw=kwargs["lw"])

        ax.add_patch(cross)

        return
    
    def pol_out(ax, pos_tip, img_dim, **kwargs):
        '''
        Adds a circle with a dot indicating the polarization direction outwards.
        '''

        r = 0.05*img_dim[1]

        circle = mpatches.Circle(pos_tip, radius=r, edgecolor="white", alpha=1, lw=1, facecolor="none") #TODO: Fix kwargs.
        ax.add_patch(circle)

        dot = mpatches.Circle(pos_tip, radius=r/5, edgecolor="white", alpha=1, lw=1, facecolor="white") #TODO: Fix kwargs.

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

    
    else: #TODO: Finish and prob do not use scalebar package. Use instead normal artist. 

        new_kwargs = kwargs["alpha_kwargs"]

        for key, value in new_kwargs.items():
            if key in alpha_kwargs:
                alpha_kwargs[key] = value

    text = ax.text(alpha_kwargs["position"][0],alpha_kwargs["position"][1], formatter[alpha_kwargs["formatter"]](letter), horizontalalignment=alpha_kwargs["horizontalalignment"], verticalalignment=alpha_kwargs["verticalalignment"], size= alpha_kwargs["size"], color=alpha_kwargs["color"], transform=ax.transAxes, bbox=dict(facecolor=alpha_kwargs["box_color"], alpha=alpha_kwargs["box_alpha"], edgecolor="none", pad=alpha_kwargs["pad"]))
    bbox = text.get_window_extent().transformed(ax.transAxes.inverted())
    # print(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
    # fs = text.get_fontsize()
    # print(fs)
    # ax.add_artist(text)
    # width = bbox.width
    # height = bbox.height

    # print(width, height)

    # extents = text.get_bbox_patch().get_extents()
    # x0 = extents.x0
    # y0 = extents.y0
    # x1 = extents.x1
    # y1 = extents.y1

    # print(x0, y0, x1, y1)

    # text.set_position((width*abs(x0)*abs(x1-x0),  (1-width*abs(y0)*(y1-y0) )))
    
    return ax


#TODO: Redo and improve.
def add_colorbar(ax, mappable, **kwargs):
    colorbar_kwargs = {
        "cmap": "magma",
        # "vmin": 0,
        # "vmax": 1,
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
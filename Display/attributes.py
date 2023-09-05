import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.patheffects as PathEffects
from matplotlib_scalebar.scalebar import ScaleBar

from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.patches as mpatches

import numpy as np



def add_scalebar(ax, res, **kwargs):

    if "scalabar_kwargs" not in kwargs:

        # unit_rep = "\\mathrm{%s}"%"\\mu m" Something something r"$\mathrm{%s}$"%"$\mu$m" r"$\si{\micro\meter}$"#
        scale_formatter = lambda value, unit:   f"{value} "  +r"$\mathrm{\mu}$m"
        scale_kwargs ={
            # "dx": size,
            "units": f"um", #TODO: Fix the label.
            "color": "white",
            "location": 4,
            "frameon": True,
            "box_color" :"black",
            "box_alpha" : 0.25,
            # "size_vertical": 8,
            "scale_loc": "bottom",
            "length_fraction":0.5,
            "width_fraction": 0.01,
            "scale_formatter":scale_formatter
            # "label_top": True,

        }
    else:

        scale_kwargs =kwargs["scalebar_kwargs"]
    size = res*1e6
    bar = ScaleBar(dx = size, **scale_kwargs)
    ax.add_artist(bar)
    return ax


def add_polarization_direction(ax, **polarization_kwargs):
    """
    Adds a polarization direction to the plot at the specified position.
    The function makes sure the arrow is a sufficient size based on the size of the plot.
    
    """
    # if "polarization_kwargs" not in kwargs: #TODO: Clean up. Lot of kwargs that don't do anything.
    #     polarization_kwargs= {
    #         "type": "out",
    #         "color": "white",
    #         "lw": 2,
    #         "alpha": 1,
    #         "pos": (100, 200), # TODO: Fix xy. No, keep in image coordinates for now. 

    #     }
    # else:
    #     polarization_kwargs = kwargs["polarization_kwargs"]


    if "type" not in polarization_kwargs:
        polarization_kwargs["type"] = "out"
    if "color" not in polarization_kwargs:
        polarization_kwargs["color"] = "white"
    if "lw" not in polarization_kwargs:
        polarization_kwargs["lw"] = 1
    if "alpha" not in polarization_kwargs:
        polarization_kwargs["alpha"] = 1
    if "pos" not in polarization_kwargs:
        polarization_kwargs["pos"] = (100, 100)
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

        width = img_dim[1]*0.05 #arrow_kwargs["width"]
        dx = img_dim[1]*0.05 * np.cos(np.deg2rad(arrow_kwargs["angle"]))
        dy = img_dim[0]*0.05 * np.sin(np.deg2rad(arrow_kwargs["angle"]))
        arrow = mpatches.Arrow(pos_tip[1]-dx, pos_tip[0]-dy, dx, dy,width = width, color=arrow_kwargs["color"], alpha=arrow_kwargs["alpha"], lw=arrow_kwargs["lw"])

        ax.add_patch(arrow)

        return
    
    def pol_in(ax, pos_tip, img_dim, **kwargs):

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


#TODO: Redo and improve.
def add_anchored_scalebar(ax, res, **kwargs):
    if "scalebar_kwargs" not in kwargs:
        size = int(1000 * res*1e6)

        scale_kwargs = {
            "size": size,
            "label": f"{int(size)} um", #TODO: Fix the label.
            "color": "white",
            "alpha": 0.5,
            "loc": 4,
            "frameon": False,
            "size_vertical": 8,
            "label_top": True,
            # "barcolor": "white",
            # "font_properties": {"size": 12}
        }
    else:
        scale_kwargs = kwargs["scalebar_kwargs"]

    scalebar0 = AnchoredSizeBar(ax.transData, **scale_kwargs)
    scalebar0.txt_label._text.set_path_effects(
        [PathEffects.withStroke(linewidth=2, foreground="black", capstyle="round")]
    )
    ax.add_artist(scalebar0)
    return ax


#TODO: Redo and improve.
def add_cmap(ax, **kwargs):
    if "cmap_kwargs" not in kwargs:
        cmap_kwargs = {
            "cmap": "inferno",
            "vmin": 0,
            "vmax": 1,
            "orientation": "vertical",
            "fraction": 0.046,
            "pad": 0.04,
        }
    else:
        cmap_kwargs = kwargs["cmap_kwargs"]

    cbar = plt.colorbar(**cmap_kwargs)
    return ax, cbar
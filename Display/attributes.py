import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.patheffects as PathEffects
from matplotlib_scalebar.scalebar import ScaleBar



def add_scalebar(ax, res, **kwargs):

    if "scalabar_kwargs" not in kwargs:

        # unit_rep = "\\mathrm{%s}"%"\\mu m" Something something
        scale_formatter = lambda value, unit: f"{value} $\mu$m"
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
            "cmap": "viridis",
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
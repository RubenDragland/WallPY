import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.patheffects as PathEffects


#TODO: Redo and improve.
def add_scalebar(ax, **kwargs):
    if "scalebar_kwargs" not in kwargs:
        size = 25 / (0.2 * 930) * 1350

        scale_kwargs = {
            "size": size,
            "label": f"25.0 mm",
            "color": "white",
            "loc": 4,
            "frameon": False,
            "size_vertical": 8,
            "label_top": False,
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
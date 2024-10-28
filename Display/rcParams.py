from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.cm import get_cmap
from cycler import cycler


# TODO: Change and adapt
try:
    plt.style.use(["tableau-colorblind10", "seaborn-paper"])
except:
    try:
        plt.style.use(["tableau-colorblind10", "seaborn-v0_8-paper"])
    except:
        "No style found."

    
# mpl.rcParams["axes.prop_cycle"] = cycler(
#     color=[
#         "#3D65A5",
#         "#E57A77",
#         "#7CA1CC",
#         "#F05039",
#         "#1F449C",
#         "#A8B6CC",
#         "#EEBAB4",
#         "#E57A77",
#         "#7CA1CC",
#         "#F05039",
#         "#1F449C",
#         "#A8B6CC",
#         "#EEBAB4",
#         "#3D65A5",
#     ]  # ["#F05039", "#E57A77", "#EEBAB4", "#1F449C", "#3D65A5", "#7CA1CC", "#A8B6CC"]
# ) + cycler(
#     linestyle=["-", "--", "-.", ":", "-", "--", "-."] * 2)

magma_colors = plt.cm.magma(np.linspace(0.15, 0.80, 4))
magma_colors = magma_colors[[0, -1, 2, 1]]

inferno_colors = plt.cm.inferno(np.linspace(0.15, 0.80, 4))
inferno_colors = inferno_colors[[-1, 1, 2, 0]]

cividis_colors = plt.cm.cividis(np.linspace(0.15, 0.80, 4))
cividis_colors = cividis_colors[[0, -1, 2, 1]]

colors = np.vstack((magma_colors, cividis_colors, inferno_colors, ))


mpl.rcParams["axes.prop_cycle"] = cycler(
    color= colors) + cycler(
    linestyle=["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--", "-.", ":"] )

#TODO: Make a function that sets appropiate rcParams for the figure given figsize. 
#TODO: Implement functionality first; find optimal settings later. 
DEFAULT_FIGSIZE = (5.69, 3.9) # TODO: Change and adapt
TOMMER = np.sqrt( DEFAULT_FIGSIZE[0]**2 + DEFAULT_FIGSIZE[1]**2)
w = 1
mpl.rcParams["axes.linewidth"] = w
mpl.rcParams["xtick.major.width"] = w
mpl.rcParams["xtick.minor.width"] = w / 2
mpl.rcParams["ytick.major.width"] = w
mpl.rcParams["ytick.minor.width"] = w / 2

mpl.rcParams["lines.markersize"] = 6 * w
mpl.rcParams["lines.linewidth"] = 1.5 * w
mpl.rcParams["font.size"] = 12 *w
mpl.rcParams["legend.fontsize"] = 14 *w
mpl.rcParams["figure.titlesize"] = 20 *w
mpl.rcParams["axes.titlesize"] = 16 *w
mpl.rcParams["xtick.labelsize"] = 12 * w
mpl.rcParams["ytick.labelsize"] = 12 * w
mpl.rcParams["axes.labelsize"] = 12 * w
mpl.rcParams["figure.figsize"] = DEFAULT_FIGSIZE  # (8, 6)
mpl.rcParams["figure.constrained_layout.use"] = True
plt.rcParams['image.cmap'] = 'magma'

#TODO: Implement a way to do the following cmaps:
#NOTE: cAFM: inferno
#NOTE: KPFM: magma
#NOTE: PFM: cividis
#NOTE: height: gray
#NOTE: EFM: plasma
#NOTE: MFM: viridis


size_settings = {
        "axes.linewidth": 1,
        "xtick.major.width": 1,
        "xtick.minor.width": 0.5,
        "ytick.major.width": 1,
        "ytick.minor.width": 0.5,
        "lines.markersize": 6,
        "lines.linewidth": 2,
        "font.size": 12,
        "legend.fontsize": 14,
        "figure.titlesize": 20,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.labelsize": 12,
    }


def adapt_mpl_fig(fig):
    '''
    Adapt the matplotlib rcParams to the figure introduced.
    Idea to just do this locally based in the figure?
    '''
    figsize = fig.get_size_inches()
    tommer = np.sqrt(figsize[0]**2 + figsize[1]**2)

    for key, value in size_settings.items():
        mpl.rcParams[key] = value * tommer/TOMMER
    
    return

def update_mpl_figsize(figx, figy):
    '''
    Update the matplotlib rcParams to the fig size introduced
    '''
    tommer = np.sqrt(figx**2 + figy**2)

    for key, value in size_settings.items():
        mpl.rcParams[key] = value * tommer/TOMMER
    
    return




mpl.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Palatino Linotype"], #["Tex Gyre Pagella"],
                "mathtext.default": "regular",
                "mathtext.fontset": "cm",
                'text.latex.preamble': r'\usepackage{siunitx} \usepackage{textalpha} \usepackage{upgreek}' #TODO: Might not be empty
            }
        )



def choose_formatter(incscape=True):
    if incscape:
        mpl.rcParams["svg.fonttype"] = "none"
        plt.rcParams["svg.fonttype"] = "none"
        plt.rcParams["axes.unicode_minus"] = False
        return
    else:
        mpl.rcParams["axes.formatter.use_mathtext"] = True
        mpl.rcParams["text.usetex"] = True  # Use Latex
        mpl.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Baskerville"]#["Palatino"],
            }
        )
    return

#TODO: For matplotlib axes size adjustments: see from mpl_toolkits.axes_grid1 import Divider, Size
def set_axis_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


def hex_to_rgb(value):
    """
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values"""
    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    """
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values"""
    return [v / 256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    """creates and returns a color map that can be used in heat map figures.
    If float_list is not provided, colour map graduates linearly between each color in hex_list.
    If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

    Parameters
    ----------
    hex_list: list of hex code strings
    float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

    Returns
    ----------
    colour map"""
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [
            [float_list[i], rgb_list[i][num], rgb_list[i][num]]
            for i in range(len(float_list))
        ]
        cdict[col] = col_list
    cmp = LinearSegmentedColormap("XRDCT_palette_cmp", segmentdata=cdict, N=256)
    return cmp
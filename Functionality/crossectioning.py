import numpy as np
import json
import matplotlib.pyplot as plt
import skimage.measure as skm
import scipy as sp
import os


def retrieve_crossection_coordinates(fig, ax, h5_file, name=f"cross_pts"):
    '''
    Interactively retrieve points from a plot and save them to a json file.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to interact with.
    ax : matplotlib.axes.Axes
        The axes to interact with.
    h5_file : object
        Converted raw data file to h5 format; GwyFile, CypherFile or similar.
        The file object containing metadata.
    name : str, optional
        The name of the json file to save the points to. The default is "cross_pts".

    Returns
    -------
    pts : list
        The list of points.
    pts_dict : dict
        The dictionary of points as saved in the json file.
    '''
    pts = [[]]
    cross_number = 0

    def onclick(event, pts):

        if event.inaxes == ax:
            x = event.xdata
            y = event.ydata

            if event.button == 1:
                pts[-1].append((x, y))
                ax.plot(x, y, "ro")
            elif event.button == 3:
                pts.append([])


            fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, pts))
    plt.show(block=True)

    pts_dict = {}
    for i, cross in enumerate(pts):
        pts_dict[i] = cross

    path = f"{h5_file.opath}_{name}.json"
    json.dump(pts_dict, open(path, "w")) #TODO: Connect json to file object somehow. And decide on path. TODO: Find out if object updates. And good practice to add to object in random functions?
    return pts, pts_dict, path
    # pts = np.array(fig.ginput(n=0, timeout=0, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=2)) #NOTE: Consider this


def read_json_pts(afm_file, name, key='0', path=None): 
    '''
    Read a json file containing points.

    Parameters
    ----------
    afm_file : object
        Converted raw data file to h5 format; GwyFile, CypherFile or similar.
        The file object containing metadata.
    name : str
        The name of the crossection add-on name.
        Typically describes the crossection.
    key : str, optional
        The dictionary key of the desired crossection.
        The default is '0'.
    path : str, optional
        The path to the json file. The default is None, in which case the opath of the afm_file is used.
        Default is None; afm_file.opath is used.

    Returns
    -------
    (x,y) : tuple
        The x and y coordinates of the crossection, respectively.
    '''

    name = name[:-5] if name.endswith(".json") else name
    if path is None:
        path = f"{afm_file.opath}_{name}.json"
    else:
        path = os.path.join(path, f"{name}.json")

    with open(path, "r") as f:
        pts_dict = json.load(f)
    
    pts = pts_dict[key]
    x, y = np.array(pts).T
    return x, y


def retrieve_crossection(z, xs, ys, lw=4, order=5, reduce_func=np.median):
    '''
    Draws a cross section of a 2D image using skimage.measure.profile_line

    Parameters
    ----------
    z : np.array
        The 2D image.
    xs : list
        The x coordinates of the line.
    ys : list
        The y coordinates of the line.
    lw : int, optional
        The width of the line to sample intensities. Perpendicular to direction. The default is 4.
    order : int, optional
        The order of the spline interpolation. The default is 5, corresponding to Bi-quintic spline interpolation.
    reduce_func : function, optional
        The function to reduce the line intensity to a single value. The default is np.median.

    Returns
    -------
    x : np.array
        The x coordinates of the line.
    intensity : np.array
        The retrieved intensity profile along the designated line.
    '''
    intensities = []
    # start = np.array([ys[0], xs[0]]) #TODO: Check. NOTE: Correct
    pts = np.array([[ys[i], xs[i]] for i in range(len(xs))])
    pts = np.array([ys, xs]).T


    for i, p in enumerate(pts[:-1]):
        intensity = skm.profile_line(z, p, pts[i+1], linewidth=lw, order=order, reduce_func=reduce_func)
        intensities.extend(intensity)
    x = np.arange(len(intensities))
    intensities = np.array(intensities)

    return x, intensities






            





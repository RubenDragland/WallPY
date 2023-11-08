
import numpy as np
import scipy as sp

def plane_level(height: np.array):
    """
    Employs a least squares fit to find the plane level of a height map.

    Parameters
    ----------
    height : np.array
        The height map.

    Returns
    -------
    Z : np.array
        The plane level.
    C : np.array
        The coefficients of the plane level.
    """
    XX, YY = np.meshgrid(np.arange(height.shape[0]), np.arange(height.shape[1]))
    data = np.c_[XX.ravel(), YY.ravel(), height.ravel()]

    order = 1    # 1: linear, 2: quadratic
    if order == 1:
        # best-fit linear plane
        # A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
        A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
        C,_,_,_ = sp.linalg.lstsq(A, data[:,2])    # coefficients
        
        # evaluate it on grid
        Z = C[0]*XX + C[1]*YY + C[2]

    return Z, C

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

# Least squares: https://mathworld.wolfram.com/LeastSquaresFittingPolynomial.html Check matte 3 or 4 for other methods.
# Possibly do iterative fitting: https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
# https://math.stackexchange.com/questions/1234240/equation-that-defines-multi-dimensional-polynomial

# TODO: This one: https://stackoverflow.com/questions/33964913/equivalent-of-polyfit-for-a-2d-polynomial-in-python 
#TODO: Does not work for rectangular areas. Must be something wrong.

def plane_level_heavyside(height:np.array, xmin:int, xmax:int):
    #TODO: Make more universal. Enable ymin and ymax as well. And default max or min to 0, shape, resp. 
    #TODO: Possibly issue with this one, still.
    """
    Employs a least squares fit to find the plane level of a height map, but retrieves the plane from a known flat region. 

    Parameters
    ----------
    height : np.array
        The height map.
    area_min : int
        The minimum area to consider.
    area_max : int
        The maximum area to consider.

    Returns
    -------
    Z : np.array
        The plane level.
    C : np.array
        The coefficients of the plane level.
    """
    XX, YY = np.meshgrid(np.arange(height.shape[0]), np.arange(height.shape[1]))

    XXROI, YYROI = np.meshgrid(np.arange(xmin, xmax), np.arange(height.shape[1]))
    data = np.c_[XXROI.ravel(), YYROI.ravel(), height[:,xmin:xmax].ravel()]

    order = 1    # 1: linear, 2: quadratic
    if order == 1:
        # best-fit linear plane
        # A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
        A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
        C,_,_,_ = sp.linalg.lstsq(A, data[:,2])    # coefficients
        
        # evaluate it on grid
        Z = C[0]*XX + C[1]*YY + C[2]

    return Z, C

def poly2D_least_squares(height: np.array, order: int, xmin:int=None, xmax:int=None, ymin=None, ymax=None):
    """
    Employs a least squares fit to find the polynomial of a set of points.

    Parameters
    ----------
    height : np.array
        The height map.
    order : int
        The order of the polynomial.

    Returns
    -------
    Z : np.array
        The polynomial.
    C : np.array
        The coefficients of the polynomial.
    """

    def find_lims(xmin, xmax, ymin, ymax, height):
        if xmin is None:
            xmin = 0
        if xmax is None:
            xmax = height.shape[0]
        if ymin is None:
            ymin = 0
        if ymax is None:
            ymax = height.shape[1]
        return xmin, xmax, ymin, ymax
    
    def get_basis(x, y, order):
        basis = []
        for i in range(order+1):
            for j in range(order-i+1):
                basis.append(x**j * y**i)
        return basis


    xmin, xmax, ymin, ymax = find_lims(xmin, xmax, ymin, ymax, height)
    x, y = np.arange(xmin, xmax), np.arange(ymin, ymax)
    XROI, YROI = np.meshgrid(x, y)
    basis = get_basis(XROI.ravel(), YROI.ravel(), order)

    A = np.vstack(basis).T
    b = height[xmin:xmax, ymin:ymax].ravel()

    # C, r, rank, s = sp.linalg.lstsq(A, b)
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None) #TODO: Find the best one

    X, Y = np.meshgrid(np.arange(height.shape[0]), np.arange(height.shape[1]))
    Z = np.sum(c[:, None, None] * np.array(get_basis(X, Y, order))
                .reshape(len(basis), *X.shape), axis=0)
    
    return Z#, c #TODO: Should 2th degree polynomial also be retrieved from same area?

def poly2D_3point_level(height: np.array, order: int, xpoints: list, ypoints:list):
    #TODO: Consider: Needs N points for degree N. 
    """
    Employs a least squares fit to find the polynomial of a set of points.

    Parameters
    ----------
    height : np.array
        The height map.
    order : int
        The order of the polynomial.
    points : list
        The points to consider.

    Returns
    -------
    Z : np.array
        The polynomial.
    C : np.array
        The coefficients of the polynomial.
    """
    assert len(xpoints) >= order, "The number of points must be equal to the order of the polynomial."
    assert len(ypoints) >= order, "The number of points must be equal to the order of the polynomial."
    X, Y = np.meshgrid(np.arange(height.shape[0]), np.arange(height.shape[1]))

    def get_basis(x, y, order):
        basis = []
        for i in range(order+1):
            for j in range(order-i+1):
                basis.append(x**j * y**i)
        return basis
    
    # XPoints, YPoints = np.meshgrid(xpoints, ypoints)
    xpoints = np.array(xpoints)
    ypoints = np.array(ypoints)

    basis = get_basis(xpoints.ravel(), ypoints.ravel(), order)



    A = np.vstack(basis).T
    ind = np.vstack([xpoints, ypoints]).T
    b = height[ypoints, xpoints].ravel() #? TODO: Changing these two worked. why?

    # C, r, rank, s = sp.linalg.lstsq(A, b)
    C, _, _, _ = np.linalg.lstsq(A, b, rcond=None) #TODO: Find the best one

    Z = np.sum(C[:, None, None] * np.array(get_basis(X, Y, order))
                .reshape(len(basis), *X.shape), axis=0)
    return Z

def scar_line_detection(data:np.array, rot:float=0, scan_axis:int=0, threshold:float=None, line_width:int=1):
    return

def median_conv_scan_line_removal(data:np.array, rotation:float=0, threshold:float=None, line_width:int=1):
    #TODO: Implement horizontal and/or vertical lines.
    return

def fourier_fringe_removal(data:np.array, threshold:float=None, line_width:int=1):
    return
    
def align_rows_median_or_poly():
    #TODO: Implement like gwyddion aligning scan rows. 
    #Really just an individual line adjustment algorithm.
    return
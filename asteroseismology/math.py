"""
Author: Carlos Casimiro Camino Mesa
Date: 11-11-2025

Last Update: 22-01-2026

Module to implement customized math functions
"""

import numpy as np
import numpy.typing as npt

def skewness(x:npt.ArrayLike, mu:float = None, w:npt.ArrayLike=None) -> float:
    """
    Function that yields the skewness of a distribution

    Parameters
    ----------
    x: ArrrayLike
        Values of the distribution.
    
    mu: float (optional)
        If specified, center from which skewness is measured. Else, is the mean of the distribution.

    w: ArrayLike (optional)
        If specified the assigned weights for each x_i in x when sum(w_i*(x_i-mu))

    Ouput
    -----
    skewness: float\n
        if > 0 -> asssymetry to the right of mu\n
        if < 0 -> assymetry to the left of mu\n
        if ~ 0 -> centered around mu
    """

    # 1. Convert the arrays to numpy arrays and checking possible errors
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("Distribution x must be 1D")
    
    if np.isnan(x).any():
        raise ValueError("Distribution x contains NaN values")
    
    if w is None:
        w = np.ones_like(x)
    else:
        w = np.asarray(w)
        if len(x) != len(w):
            raise ValueError("x and w must have the same length")
        elif np.isnan(w).any():
            raise ValueError("Distribution x contains NaN values")
        elif np.sum(w) != 1.0:
            raise ValueError("Weights need to be normalized. sum(w)=1")

    
    # 2. Compute the mean if none and number of measures
    if mu is None:
        mu = np.average(x, weights=w)
        

    # 3. Computing the skewness
    dif = x-mu
    m_3 = np.sum(w*dif**3)
    m_2 = np.sum(w*dif**2)
    if m_2 == 0:
        return np.nan
    skew = m_3/m_2**1.5
    return skew

def weighted_median(x:npt.ArrayLike, w:npt.ArrayLike=None, top:float=0.5, alpha:int=1) -> float:
    """
    Function that evaluates the weighted median of a distribution x. 
    It returns the element in x in which the cumulative sum of the weights (w) is higher than the upper limit (top) defined.

    Parameters
    ----------

    x: ArrayLike
        Distribution
    
    w: ArrayLike
        Distribution of weights. Must be in the same order as x. Default value is all 1, computing regular median.

    top: float
        Upper limit in percentage (by default is 50% = 0.5) to be reached by the cummulative sum of weights.

    alpha: float
        Exponent of the weights (w**alpha) to overponderate big weights over small weights. By default is 1.

    Output
    ------

    cum_med: float
        x_i whose cummulative w is bigger or equal to top.

    """

    x = np.asarray(x)

    if w is None:
        w = np.ones_like(x)
    else:
        w= np.asarray(w)

    if len(x) != len(w):
        raise ValueError("x and w must be of the same length.")
    
    idx_sorted = np.argsort(x)
    x_sorted = x[idx_sorted]
    w_sorted = w[idx_sorted]**alpha

    cum = np.cumsum(w_sorted)

    return x_sorted[np.searchsorted(cum, cum[-1]*top)]
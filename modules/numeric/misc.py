import numpy as np
from dataclasses import dataclass
from typing import Any
from math import factorial
from dataclasses import dataclass

_NUMERIC = float|np.ndarray[float]

def diff_coeffs(m: int, stencil: tuple[int], h: float=1, symbolic=False) -> np.ndarray:
    '''
    Finite difference coefficients: considering that a function $f(x)$ can be differentiated as
    $$
    f^{(m)}(x) \approx \frac{1}{h^{m}}\sum _{i=0}^{p-1} c_i f( x+\alpha_i h),
    $$
    where $m >0$, the constants $c_i$ are called finite difference coefficients 
    and $\alpha_i$ intagers that locates nodes igually spaced by $h$.

    Parameters
    ----------
    m: int
        Derivative order.
    stencil: ndarray[int]
        An 1-dimensional array containing the nodes location 
        that will be used on the differentiation.
    h: float, optional
        Space between the mesh nodes. Default is 1.
    
    Attributes
    ----------
    ndarray[float]
        Finite difference coefficients.
    '''
    return factorial(m)*np.linalg.inv(np.vander(stencil, increasing=True))[m]/h**m

def argnearest(arr: np.ndarray, value: Any):
    return np.abs(arr - value).argmin()
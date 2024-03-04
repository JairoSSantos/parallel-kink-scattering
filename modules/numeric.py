import numpy as np
import sympy as sp
from math import factorial
from scipy.ndimage import convolve1d
from dataclasses import dataclass
from functools import partial
from typing import Callable, Any
from abc import ABC, abstractmethod

def diff_coeffs(m: int, offsets: tuple[int], h: float=1) -> np.ndarray:
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
    offsets: ndarray[int]
        An 1-dimensional array containing the nodes location 
        that will be used on the differentiation.
    h: float, optional
        Space between the mesh nodes. Default is 1.
    
    Attributes
    ----------
    ndarray[float]
        Finite difference coefficients.
    '''
    offsets = np.r_[offsets]
    A = sp.Matrix(offsets[:, np.newaxis]**np.arange(len(offsets)))
    return factorial(m)*np.r_[A.inv()][m].astype(float)/h**m

def argnearest(arr: np.ndarray, value: Any):
    return np.abs(arr - value).argmin()

def rk4_solve(F: Callable[[float, float|np.ndarray], float|np.ndarray], y0: float|np.ndarray, dt: float, t_final: float, t0: float=0, 
               callback: Callable[[float, float|np.ndarray], float|np.ndarray]=None,
               stop_condition: Callable[[float, float|np.ndarray], float|np.ndarray]=None):
    '''
    Runge-Kutta integration for a generalized ODEs system.

    Parameters
    ----------
    F: callable
        The function $F(t, y)$ that will be integrated: $\dfrac{dy}{dt} = F(t, y)$
    y0: float or ndarray
        The initial configuration $y_0=y(t_0)$.
    dt: float
        Mesh spacing of the independent variable.
    t_final:
        Final value for the independent variable.
    t0: float, optioal
        Initial independent variable value $t_0$. Default if `t0=0`.
    callback: callable, optional
        A `func(t, y)` that will be called at each method iteration 
        and have to recieve `t`, the independent variable, and `y`, the result of the last method iteration.
        The function output have to be the modified `y`: `func(t, y) -> y`.
    stop_conditions: tuple[callable], optional
        These will be called at each method iteration and have to recieve two parameters: 
        `func(t, y)`, where `t` is the independent variable and `y` the result of the last method iteration. 
        Furthermore, the functions return type have to be boolean so that any `True` return makes the iteration stops.
    '''
    Y = [y0]
    T = np.arange(t0, t_final, dt)
    for i, t in enumerate(T[1:]):
        y = Y[-1]

        k1 = F(t, y)
        k2 = F(t + dt/2, y + k1*dt/2)
        k3 = F(t + dt/2, y + k2*dt/2)
        k4 = F(t + dt, y + k3*dt)
        Y.append(y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4))

        if callback != None: callback(t, Y)

        if stop_condition != None and stop_condition(t, Y[-1]): break

    return T[:i+2], np.r_[Y]

def delta(lamb: float|int) -> float:
    return np.sqrt(2/lamb)

def kink(x0: float|int, v: float|int, lamb: float|int):
    gamma = 1/np.sqrt(1 - v**2)
    c1 = gamma/delta(lamb)
    x, t =sp.symbols('x, t')
    z = c1*(x - x0 - v*t)
    return sp.tanh(z)

class Lattice:
    '''
    A generalized lattice object.

    Parameters
    ----------
    **axes: The axes names with that's range array, `x = np.arange(initial_value, final_value, spacing)`.

    Attributes
    ----------
    axes: dict
        A dict containing the axes ranges
    shape: tuple[int]
        The lattice shape.
    grid: ndarray
        A n-dimensional array with the lattice nodes locations.
    '''
    def __init__(self, **axes):
        self.axes = axes
    
    def __getattr__(self, kw):
        return self.axes[kw]
    
    def __getitem__(self, axis):
        return self.axes[axis]
    
    @property
    def shape(self) -> tuple:
        return tuple(map(len, self.axes.values()))
    
    @property
    def grid(self) -> np.ndarray:
        return np.stack(np.meshgrid(*tuple(self.axes.values())), axis=-1)
    
    def at(self, **locs) -> list:
        '''
        Indices that provides the node(s) location, given position values.

        Parameters
        ----------
        **locs: Position values.

        Returns
        -------
        list
            A list with the nodes locations as index format, 
            each item in the list corresponding to a lattice axe.
        '''
        kws = tuple(locs.keys())
        return [
            argnearest(axis, locs[kw]) if kw in kws else Ellipsis 
            for kw, axis in self.axes.items()
        ]
    
    def window(self, **lims) -> list:
        '''
        Indices that provides a range on the lattice, given values for the axes limits.

        Parameters
        ----------
        **locs: limits values, like `(min, max)`.

        Returns
        -------
        list
            A list with the lattice ranges as index format,
            each item in the list corresponding to a lattice axe.
        '''
        kws = tuple(lims.keys())
        return [
            slice(argnearest(axis, lims[kw][0]), argnearest(axis, lims[kw][1]))
            if kw in kws else Ellipsis 
            for kw, axis in self.axes.items()
        ]

@dataclass
class Kink:
    '''
    An object for kinks properties.

    Parameters
    ----------
    x0: float
        The kink location.
    v: float
        The kink velocity.
    lamb: float
        The $\lambda$ factor, that is correlated to the kink thickness.
    '''
    x0: float
    v: float
    lamb: float

    def __post_init__(self):
        gamma = 1/sqrt(1 - self.v**2)
        self._const = gamma/Kink.delta(self.lamb)

    def z(self, x: float|np.ndarray, t: float) -> float|np.ndarray:
        return self._const*(x - self.x0 - self.v*t)
    
    def call(self, x: float|np.ndarray, t: float) -> float|np.ndarray:
        return np.tanh(self.z(x, t))
    
    def diff_dt(self, x: float|np.ndarray, t: float) -> float|np.ndarray:
        return -self._const*self.v/np.cosh(self.z(x, t))**2
    
    def delta(lamb: float):
        return sqrt(2/lamb)

@dataclass
class KinkCollider:
    '''
    A class to speed up kink scattering processes.

    Parameters
    ----------
    x_lattice: Lattice
        The spatial lattice objetc, ensure that the lattice have an axis with "x" name.
    x0s: tuple[float]
            Initial positions of the kinks.
    dt: float
        Temporal grid spacing.
    
    Attributes
    ----------
    diff_dx: ndarray
        A matrix that provides the second derivative on the space axis, given the `x_lattice` informations.
    '''
    x_min: float
    x_max: float
    Nx: int
    x0s: tuple[float]
    dt: float
    dx: int=None
    order: int=4

    def __post_init__(self):
        self.x = np.linspace(self.x_min, self.x_max, self.Nx)
        if self.dx == None: self.dx = (self.x_max - self.x_min)/(self.N - 1)
        p = self.order + 1 # in general p = order + 2, but for central differences p = order + 1
        self._j = p//2
        self.D2x = diff_coeffs(2, np.arange(p) - self._j, h=self.dx)
    
    def get_system(self, lamb: float) -> np.ndarray:
        '''
        ODEs system.
        '''
        def F(t: float, Y: np.ndarray):
            y, dy_dt = Y

            # y[0], y[-1] = y1, y2
            y_reflected = np.r_[y[1:self._j+1][::-1], y, y[-self._j-1:-1][::-1]]

            return np.stack((
                dy_dt, # = dy(t)
                np.convolve(y_reflected, self.D2x, mode='valid') + lamb*y*(1 - y**2) # = ddy(t)
            ))
        return F
    
    def collide(self, vs: tuple[float], lamb: float, t_final: float, gnd: int=-1, **rk4_kwargs) -> np.ndarray:
        '''
        Run a collision.

        Parameters
        ----------
        vs: tuple[float]
            Initial velocitys.
        lamb: float
            The $\lambda$ factor.
        t_final: float
            Final value for the time.
        gnd: int
            Field state at left of the spatial range in the initial instant.
        '''
        # y0 = np.zeros((2, len(self.x)))
        # y0[0].fill(gnd)
        # for i, j in enumerate(np.argsort(self.x0s)):
        #     q = gnd*(-1)**(i + 1)
        #     k = Kink(self.x0s[j], vs[j], lamb)
        #     y0[0] += q*k(self.x, t=0)
        #     y0[1] += q*k.diff_dt(self.x, t=0)

        field = kink(self.x0s[0], vs[0], lamb) - kink(self.x0s[1], vs[1], lamb) - 1
        x, t =sp.symbols('x, t')
        phi = sp.lambdify(x, field.subs({t: 0}), 'numpy')
        phi_dt = sp.lambdify(x, field.diff(t).subs({t: 0}), 'numpy')
        y0 = np.stack((
            phi(self.x), 
            phi_dt(self.x)
        ))
        
        t, Y = rk4_solve(self.get_system(lamb=lamb), y0, self.dt, t_final, **rk4_kwargs)

        return Lattice(x=self.x, t=t), Y[:, 0], Y[:, 1]
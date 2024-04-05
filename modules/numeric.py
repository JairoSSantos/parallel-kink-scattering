import numpy as np
import sympy as sp
from math import factorial, sqrt, acosh, asinh, atanh
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
    x, t = sp.symbols('x, t')
    z = c1*(x - x0 - v*t)
    return Field((x, t), sp.tanh(z))

def kink_boundary(x0: float, v: float, lamb: float, H: float):
    gamma = 1/np.sqrt(1 - v**2)
    c1 = gamma/delta(lamb)

    x, t = sp.symbols('x, t')
    z = c1*(x - x0 - v*t)
    phi = sp.tanh(z)

    if H > 0:
        X0 = acosh(1/sqrt(abs(H)))
        phi_mod = sp.tanh(x - X0)
    else:
        X1 = asinh(1/sqrt(abs(H)))
        phi_mod = 1/sp.tanh(x - X1)

    return Field((x, t), phi_mod - phi + 1)

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

class Field:
    def __init__(self, x, y):
        self.y = y
        self.x = x
        self._func = sp.lambdify(self.x, self.y, 'numpy')
    
    def __call__(self, *x):
        return self._func(*x)
    
    def __add__(self, obj):
        if type(obj) == Field: obj = obj.y
        return Field(self.x, self.y + obj)
    
    def __sub__(self, obj):
        if type(obj) == Field: obj = obj.y
        return Field(self.x, self.y - obj)
    
    def __mul__(self, obj):
        if type(obj) == Field: obj = obj.y
        return Field(self.x, self.y*obj)
    
    def __truediv__(self, obj):
        if type(obj) == Field: obj = obj.y
        return Field(self.x, self.y/obj)
    
    def diff(self, var_index: int):
        return Field(self.x, self.y.diff(self.x[var_index]))

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
    H: float=0

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
            y, dy = Y
            # y_reflected = np.r_[y[1:self._j+1][::-1], y, y[-self._j-1:-1][::-1]]
            # d2x_y = np.convolve(y_reflected, self.D2x, mode='valid')

            # y_reflected = np.r_[y, y[-self._j-1:-1][::-1]]
            # d2x_y = np.r_[
            #     -(85*y[0] - 108*y[1] + 27*y[2] - 4*y[3] - 66*self.dx*self.H)/(18*self.dx**2), # erro na interpolação !!
            #     (29*y[0] - 54*y[1] + 27*y[2] - 2*y[3] - 6*self.dx*self.H)/(18*self.dx**2),
            #     np.convolve(y_reflected, self.D2x, mode='valid'),
            # ]

            y_reflected = np.r_[y, y[-self._j-1:-1][::-1]]
            d2x_y = np.r_[
                (35*self.H - 104*y[1] + 114*y[2] - 56*y[3] + 11*y[4])/(12*self.dx**2),
                (11*self.H - 20*y[1] + 6*y[2] + 4*y[3] - y[4])/(12*self.dx**2),
                np.convolve(y_reflected, self.D2x, mode='valid'),
            ]

            return np.stack((
                dy, # = dy(t)
                d2x_y + lamb*y*(1 - y**2) # = ddy(t)
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
        field = kink(self.x0s[0], vs[0], lamb) - kink(self.x0s[1], vs[1], lamb) - 1
        phi = lambda x: field(x, 0)
        phi_dt = lambda x: field.diff(1)(x, 0)
        y0 = np.stack((
            phi(self.x), 
            phi_dt(self.x)
        ))
        
        t, Y = rk4_solve(self.get_system(lamb=lamb), y0, self.dt, t_final, **rk4_kwargs)

        return Lattice(x=self.x, t=t), Y[:, 0], Y[:, 1]
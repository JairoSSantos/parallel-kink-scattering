import numpy as np
import sympy as sp
from math import factorial, sqrt, acosh, asinh, atanh
from dataclasses import dataclass
from typing import Callable, Any

from .boundaries import *

NUMERIC = float|np.ndarray[float]

def diff_coeffs(m: int, offsets: tuple[int], h: float=1, symbolic=False) -> np.ndarray:
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
    weights = factorial(m)*np.r_[A.inv()][m]/h**m
    return weights if symbolic else weights.astype(float)

def argnearest(arr: np.ndarray, value: Any):
    return np.abs(arr - value).argmin()

def rk4_solve(
        F: Callable[[float, NUMERIC], NUMERIC], 
        y0: NUMERIC, 
        dt: float, 
        t_final: float, 
        t0: float=0,
        F_params: dict={},
        callback: Callable[[float, NUMERIC], NUMERIC]=None,
        stop_condition: Callable[[float, NUMERIC], NUMERIC]=None):
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

        k1 = F(t, y, **F_params)
        k2 = F(t + dt/2, y + k1*dt/2, **F_params)
        k3 = F(t + dt/2, y + k2*dt/2, **F_params)
        k4 = F(t + dt, y + k3*dt, **F_params)
        Y.append(y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4))

        if callback != None: callback(t, Y)

        if stop_condition != None and stop_condition(t, Y[-1]): break

    return T[:i+2], np.r_[Y]

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

class Booster:
    def __init__(self, 
        x_lattice: tuple[float, float, int], 
        dt: float, 
        diff_order: int,
        y0: Callable[[NUMERIC], NUMERIC],
        pot_diff: Callable[[NUMERIC], NUMERIC],
        boundaries: Boundary|tuple[Boundary]=None
    ):
        assert diff_order%2 == 0

        self.x = np.linspace(*x_lattice)
        self.dt = dt
        self.y0 = y0
        self.pot_diff = pot_diff

        p = diff_order + 1
        self.DDx = diff_coeffs(m=2, offsets=np.arange(p) - p//2, h=(x_lattice[1] - x_lattice[0])/(x_lattice[2] - 1))
        self.dx = (x_lattice[1] - x_lattice[0])/(x_lattice[2] - 1)

        if boundaries == None:
            self.lb = self.rb = Reflective(nodes=p//2)
        else:
            try:
                self.lb, self.rb = boundaries
            except TypeError:
                self.lb = self.rb = boundaries
    
    def system(self, t, Y, **pot_params):

        y, dydt = Y
        y_boundaries = self.rb(self.lb(y)[::-1])[::-1]
        return np.stack((
            dydt,
            np.convolve(y_boundaries, self.DDx, mode='valid') - self.pot_diff(y, **pot_params)
        ))

        # y, dydt = Y
        # y_boundaries = self.rb(self.lb(y)[::-1])[::-1]
        # return np.stack((
        #     dydt,
        #     np.r_[
        #         np.convolve(y_boundaries, self.DDx, mode='valid')[:-2],
        #         [(29*y[-1] - 54*y[-2] + 27*y[-3] - 2*y[-4] - 6*self.dx*self.rb.param)/(18*self.dx**2),
        #         -(85*y[-1] - 108*y[-2] + 27*y[-3] - 4*y[-4] - 66*self.dx*self.rb.param)/(18*self.dx**2)]
        #     ] - self.pot_diff(y, **pot_params)
        # ))
    
    def run(self, T: float, y0_params: dict={}, pot_params: dict={}, **rk_params):
        t, Y = rk4_solve(
            F= self.system,
            y0= self.y0(self.x, **y0_params),
            dt= self.dt,
            t_final= T,
            F_params= pot_params,
            **rk_params
        )
        return Lattice(x=self.x, t=t), Y[:, 0], Y[:, 1]
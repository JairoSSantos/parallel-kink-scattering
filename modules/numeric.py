import numpy as np
import sympy as sp
from math import factorial, sqrt, acosh, asinh, atanh
from scipy.ndimage import convolve1d
from dataclasses import dataclass
from functools import partial
from typing import Callable, Any
from abc import ABC, abstractmethod

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
    
    def extent(self, axis_order: tuple=[]):
        out = []
        for kw in (axis_order if len(axis_order) > 0 else self.axes.keys()):
            for f in (np.min, np.max):
                out.append(f(self.axes[kw]))
        return out

# class Field:
#     def __init__(self, x, y):
#         self.y = y
#         self.x = x
#         self._func = sp.lambdify(self.x, self.y, 'numpy')
    
#     def __call__(self, *x):
#         return self._func(*x)
    
#     def __add__(self, obj):
#         if type(obj) == Field: obj = obj.y
#         return Field(self.x, self.y + obj)
    
#     def __sub__(self, obj):
#         if type(obj) == Field: obj = obj.y
#         return Field(self.x, self.y - obj)
    
#     def __mul__(self, obj):
#         if type(obj) == Field: obj = obj.y
#         return Field(self.x, self.y*obj)
    
#     def __truediv__(self, obj):
#         if type(obj) == Field: obj = obj.y
#         return Field(self.x, self.y/obj)
    
#     def diff(self, var_index: int):
#         return Field(self.x, self.y.diff(self.x[var_index]))

def kink(x: NUMERIC, t: float, v: float, c: float=1) -> NUMERIC:
    gamma = 1/sqrt(1 - v**2/c**2)
    return np.tanh(gamma*(x - v*t))

def kink_dt(x: NUMERIC, t: float, v: float, c: float=1) -> NUMERIC:
    gamma = 1/sqrt(1 - v**2/c**2)
    return -gamma*v/np.cosh(gamma*(x - v*t))**2

@dataclass
class Phi4:
    scale: float=2
    vacuum: float=1

    def __call__(self, y: NUMERIC) -> NUMERIC:
        return (self.scale/4)*(y**2 - self.vacuum**2)**2
    
    def diff(self, y: NUMERIC) -> NUMERIC:
        return self.scale*y*(y**2 - self.vacuum**2)

class Boundary:
    def __init__(self, nodes: int, order: int, derivative: int, param: float=None):
        K = np.arange(order)
        M = np.r_[
            [np.where(K == derivative, factorial(0), 0)],
            np.vander(K[1:], order, increasing=True)
        ]
        S = np.vander(-np.r_[1:nodes+1], order, increasing=True)
        self.G = S @ np.linalg.inv(M)
        self.param = param
        self.order = order
        self.derivative = derivative
    
    def set_param(self, value):
        self.param = value
    
    def __call__(self, Y):
        y = np.r_[self.param, Y[1:self.order]]
        y_ghost = (self.G @ y)[::-1]
        if self.derivative > 0:
            return np.r_[y_ghost, Y]
        else:
            return np.r_[y_ghost, self.param, Y[1:]]

class Dirichlet(Boundary):
    def __init__(self, nodes: int, order: int, param: float=None):
        super().__init__(nodes=nodes, order=order, derivative=0, param=param)

class Neumann(Boundary):
    def __init__(self, nodes: int, order: int, param: float=None):
        super().__init__(nodes=nodes, order=order, derivative=1, param=param)

class Reflective:
    def __init__(self, nodes: int):
        self.nodes = nodes
    
    def __call__(self, Y):
        return np.r_[Y[1:self.nodes+1][::-1], Y]

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

# @dataclass
# class KinkCollider:
#     '''
#     A class to speed up kink scattering processes.

#     Parameters
#     ----------
#     x_lattice: Lattice
#         The spatial lattice objetc, ensure that the lattice have an axis with "x" name.
#     x0s: tuple[float]
#             Initial positions of the kinks.
#     dt: float
#         Temporal grid spacing.
    
#     Attributes
#     ----------
#     diff_dx: ndarray
#         A matrix that provides the second derivative on the space axis, given the `x_lattice` informations.
#     '''
#     x_min: float
#     x_max: float
#     Nx: int
#     x0s: tuple[float]
#     dt: float
#     dx: int=None
#     order: int=4
#     H: float=0

#     def __post_init__(self):
#         self.x = np.linspace(self.x_min, self.x_max, self.Nx)
#         if self.dx == None: self.dx = (self.x_max - self.x_min)/(self.N - 1)
#         p = self.order + 1 # in general p = order + 2, but for central differences p = order + 1
#         self._j = p//2
#         self.D2x = diff_coeffs(2, np.arange(p) - self._j, h=self.dx)
    
#     def get_system(self, lamb: float) -> np.ndarray:
#         '''
#         ODEs system.
#         '''
#         def F(t: float, Y: np.ndarray):
#             y, dy = Y
#             # y_reflected = np.r_[y[1:self._j+1][::-1], y, y[-self._j-1:-1][::-1]]
#             # d2x_y = np.convolve(y_reflected, self.D2x, mode='valid')

#             # y_reflected = np.r_[y, y[-self._j-1:-1][::-1]]
#             # d2x_y = np.r_[
#             #     -(85*y[0] - 108*y[1] + 27*y[2] - 4*y[3] - 66*self.dx*self.H)/(18*self.dx**2), # erro na interpolação !!
#             #     (29*y[0] - 54*y[1] + 27*y[2] - 2*y[3] - 6*self.dx*self.H)/(18*self.dx**2),
#             #     np.convolve(y_reflected, self.D2x, mode='valid'),
#             # ]

#             # y_reflected = np.r_[y, y[-self._j-1:-1][::-1]]
#             # d2x_y = np.r_[
#             #     (35*self.H - 104*y[1]+114*y[2]-56*y[3]+11*y[4])/(12*self.dx**2),
#             #     (10*self.H - 15*y[1]-4*y[2]+14*y[3]-6*y[4]+1*y[5])/(12*self.dx**2),
#             #     np.convolve(y_reflected, self.D2x, mode='valid'),
#             # ]

#             # y_reflected = np.r_[
#             #     [(-4*self.H + 8*y[1] + y[2])/3, self.H/6 + y[1], self.H],
#             #     y[1:],
#             #     y[-self._j-1:-1][::-1]
#             # ]
#             # d2x_y = np.convolve(y_reflected, self.D2x, mode='valid')

#             y_reflected = np.r_[
#                 [6*self.H - 8*y[1] + 3*y[2], 3*self.H - 3*y[1] + 1*y[2], self.H],
#                 y[1:],
#                 y[-self._j-1:-1][::-1]
#             ]
#             d2x_y = np.convolve(y_reflected, self.D2x, mode='valid')

#             return np.stack((
#                 dy, # = dy(t)
#                 d2x_y + lamb*y*(1 - y**2) # = ddy(t)
#             ))
#         return F
    
#     def collide(self, vs: tuple[float], lamb: float, t_final: float, gnd: int=-1, **rk4_kwargs) -> np.ndarray:
#         '''
#         Run a collision.

#         Parameters
#         ----------
#         vs: tuple[float]
#             Initial velocitys.
#         lamb: float
#             The $\lambda$ factor.
#         t_final: float
#             Final value for the time.
#         gnd: int
#             Field state at left of the spatial range in the initial instant.
#         '''
#         field = kink(self.x0s[0], vs[0], lamb) - kink(self.x0s[1], vs[1], lamb) - 1
#         phi = lambda x: field(x, 0)
#         phi_dt = lambda x: field.diff(1)(x, 0)
#         y0 = np.stack((
#             phi(self.x), 
#             phi_dt(self.x)
#         ))
        
#         t, Y = rk4_solve(self.get_system(lamb=lamb), y0, self.dt, t_final, **rk4_kwargs)

#         return Lattice(x=self.x, t=t), Y[:, 0], Y[:, 1]
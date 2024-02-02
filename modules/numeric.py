import numpy as np
from sympy import Matrix
from math import factorial, sqrt
from scipy.ndimage import convolve1d
from dataclasses import dataclass
from functools import partial
from typing import Callable, Any

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
    A = Matrix(offsets[:, np.newaxis]**np.arange(len(offsets)))
    return factorial(m)*np.r_[A.inv()][m].astype(float)/h**m

# def diff_matrix(m: int, n: int, order: int, h: float) -> np.ndarray:
#     '''
#     Discrete derivative.

#     Parameters
#     ----------
#     m: int
#         Derivative order.
#     n: int
#         Amount of the mesh nodes.
#     order: int
#         Error order of the approximation.
#     h: float
#         Space between the mesh nodes.
    
#     Returns
#     ----------
#     ndarray[float]
#         Differentiation matrix.
#     '''
#     p = order + m
#     phalf = p//2
#     offsets = np.arange(p)
#     central = diff_coeffs(m, offsets - phalf)
#     forward = diff_coeffs(m, offsets)
#     backward = diff_coeffs(m, -offsets[::-1])
#     M = np.zeros((n, n))
#     for i in range(n):
#         if i < phalf: M[i, i:i+p] = forward
#         elif i > n-ceil(p/2): M[i, i-p+1:i+1] = backward
#         else: M[i, i-phalf:i+ceil(p/2)] = central
#     return M/h**m

def argnearest(arr: np.ndarray, value: Any):
    return np.abs(arr - value).argmin()

def rk4_solve(F: Callable[[float, float|np.ndarray], float|np.ndarray], y0: float|np.ndarray, dt: float, t_final: float, t0: float=0, 
               callback: Callable[[float, float|np.ndarray], float|np.ndarray]=(lambda t, y: y),
               stop_conditions: tuple[Callable[[float, float|np.ndarray], float|np.ndarray]]=[]):
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
    for t in T:
        y = Y[-1]

        stop = False
        for func in stop_conditions: 
            if func(t, Y[-1]):
                T = np.arange(t0, t, dt)
                stop = True
                break
        if stop: break

        k1 = F(t, y)
        k2 = F(t + dt/2, y + k1*dt/2)
        k3 = F(t + dt/2, y + k2*dt/2)
        k4 = F(t + dt, y + k3*dt)
        Y.append(callback(t, y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)))

    return T, np.r_[Y]

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
    
    def __call__(self, x: float|np.ndarray, t: float) -> float|np.ndarray:
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
    x_range: tuple[float]
    x0s: tuple[float]
    dt: float
    order: int=4

    def __post_init__(self):
        self.x = np.arange(*self.x_range)
        p = self.order + 2
        self.D2x = diff_coeffs(2, np.arange(p) - p//2, h=self.x_range[-1])
    
    def F(self, t: float, Y: np.ndarray, lamb: float) -> np.ndarray:
        '''
        ODEs system 
        $$
            \left\{\begin{array}{ l }
            \ddot{\Phi } =\Phi ''+\lambda \Phi \left( \Phi ^{2} -1\right)\\
            d\Phi =\dot{\Phi }  dt
            \end{array}\right..
        $$
        '''
        y, dy_dt = Y
        return np.stack((
            dy_dt, # = dy(t)
            convolve1d(y, self.D2x, mode='reflect') + lamb*y*(1 - y**2) # = ddy(t)
        ))
    
    def collide(self, vs: tuple[float], lamb: float, t_final: float, gnd: int=-1, **rk4_kwargs) -> np.ndarray:
        '''
        Run a collision.

        Parameters
        ----------
        vs: tuple[float]
            Inivial velocitys of the kinks.
        lamb: float
            The $\lambda$ factor.
        t_final: float
            Final value for the time.
        gnd: int
            Field state at left of the spatial range in the initial instant.
        '''
        y0 = np.zeros((2, len(self.x)))
        y0[0].fill(gnd)
        for i, j in enumerate(np.argsort(self.x0s)):
            q = gnd*(-1)**(i + 1)
            k = Kink(self.x0s[j], vs[j], lamb)
            y0[0] += q*k(self.x, t=0)
            y0[1] += q*k.diff_dt(self.x, t=0)
        
        t, y = rk4_solve(partial(self.F, lamb=lamb), y0, self.dt, t_final, **rk4_kwargs)

        return Lattice(x=self.x, t=t), y

    def neumann_boundary(n: int , y: float|tuple[float], dy: float|tuple[float]=0):
        if y != None:
            try: len(y)
            except TypeError: y = (y, y)
        if dy != None:
            try: len(dy)
            except TypeError: dy = (dy, dy)

        def boundary(t: float, Y: np.ndarray):
            if dy != None:
                Y[1, n], Y[1, -n-1] = dy
            if y != None:
                Y[0, n], Y[0, -n-1] = y
            Y[0, -n:] = Y[0, -2*n-1:-n-1][::-1]
            Y[0, :n] = Y[0, n+1:2*n+1][::-1]
            return Y
        return boundary
    
    def overflowed(t: float, Y: np.ndarray) -> bool:
        return np.any(np.isnan(Y))
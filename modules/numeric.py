import numpy as np
from math import factorial
from dataclasses import dataclass
from functools import partial
from typing import Callable

def coefficients(m: int, alpha: np.ndarray[int]) -> np.ndarray:
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
    alpha: ndarray[int]
        An 1-dimensional array containing the nodes location 
        that will be used on the differentiation.
    
    Attributes
    ----------
    ndarray[float]
        Finite difference coefficients.
    '''
    p = len(alpha)
    A = np.zeros((p, p))
    A[0].fill(1)
    for k in range(1, p):
        A[k] = alpha**k
    return factorial(m)*np.linalg.inv(A)[:, m]

@dataclass
class Diff:
    '''
    Discrete derivative.

    Parameters
    ----------
    m: int
        Derivative order.
    n: int
        Amount of the mesh nodes.
    p: int
        Amount of nodes that will be used to approximate the derivative value.
    h: float
        Space between the mesh nodes.
    
    Attributes
    ----------
    M: ndarray
        Transformation matrix of the approximate derivative.
    '''
    m: int
    n: int
    p: int
    h: float

    def __post_init__(self) -> None:
        assert self.n >= 2*self.p
        P = np.arange(self.p)[np.newaxis].repeat(self.p, axis=0)
        l = int(self.p/2)
        d = int(l*2 != self.p)
        C = np.stack([coefficients(self.m, alpha) for alpha in P - P.T])
        M = np.zeros((self.n - 2*l, self.n))
        for i in range(self.n - 2*l):
            M[i, i:i+self.p] = C[l]
        self.M = np.c_[ 
            '0',
            np.pad(C[:l], [(0, 0), (0, self.n - self.p)]),
            M,
            np.pad(C[l+d:], [(0, 0), (self.n - self.p, 0)]),
        ]/(self.h**self.m)
    
    def __call__(self, y: np.ndarray) -> np.ndarray:
        '''
        calculates the respective derivative of the `y` applying the transformation matrix `M`.

        Parameters
        ----------
        y: ndarray
            An 1-dimensional array containing sample of a scalar function.
        '''
        return self.M.dot(y)

class RKSolver:
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
    t0: float, optioal
        Initial independent variable value $t_0$. Default if `t0=0`.
    
    Attributes
    ----------
    y: ndarray
        Result of the aready calculated integration.
    t: ndarray
        Range of the independent variable for the aready calculated solution.
    '''
    def __init__(self, F: Callable[[float, float|np.ndarray], float|np.ndarray], y0: float|np.ndarray, dt: float, t0: float=0) -> None:
        self.dt = dt
        self.F = F
        self._y = [y0]
        self._t = [t0]

    def step(self) -> None:
        '''
        Solve just one step $y(t + dt)$.
        '''
        t = self._t[-1]
        y = self._y[-1]
        dt = self.dt

        k1 = self.F(t, y)
        k2 = self.F(t + dt/2, y + k1*dt/2)
        k3 = self.F(t + dt/2, y + k2*dt/2)
        k4 = self.F(t + dt, y + k3*dt)

        self._y.append(
            y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        )
        self._t.append(t + dt)

    def run_util(self, T: float) -> None:
        '''
        Run the `step` method util a final value for the independent variable: `t[-1] = T`.

        Parameters
        ----------
        T: float
            Final value for the independent variable.
        '''
        while self._t[-1] < T:
            self.step()

    @property
    def y(self) -> np.ndarray:
        return np.stack(self._y)

    @property
    def t(self) -> np.ndarray:
        return np.stack(self._t)

class Lattice:
    '''
    A generalized lattice object.

    Parameters
    ----------
    **axes: The axes names with that's range informations, like `x = (initial_value, final_value, spacing)`.

    Attributes
    ----------
    axes: dict
        An array containing the input informations.
    ranges: dict[str, ndarray]
        Nodes values obtained from the `axes` informations.
    shape: tuple[int]
        The lattice shape.
    grid: ndarray
        A `len(axes)`-dimansional array with the lattice nodes locations.
    '''
    def __init__(self, **axes):
        self.axes = axes
        self.ranges = {kw:np.arange(xl, xr, dx) for kw, (xl, xr, dx) in axes.items()}
    
    def __getattr__(self, kw):
        return self.ranges[kw]
    
    def __getitem__(self, axis):
        return self.ranges[axis]
    
    @property
    def shape(self) -> tuple:
        return tuple(map(len, self.ranges.values()))
    
    @property
    def grid(self) -> np.ndarray:
        return np.stack(np.meshgrid(*tuple(self.ranges.values())), axis=-1)
    
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
            np.abs(X - locs[kw]).argmin() if kw in kws else Ellipsis 
            for kw, X in self.ranges.items()
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
            slice(np.abs(X - lims[kw][0]).argmin(), 
                  np.abs(X - lims[kw][1]).argmin())
            if kw in kws else Ellipsis 
            for kw, X in self.ranges.items()
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
        gamma = 1/(1 - self.v**2)**0.5
        delta = (2/self.lamb)**0.5
        self._c = gamma/delta

    def z(self, x: float|np.ndarray, t: float) -> float|np.ndarray:
        return self._c*(x - self.x0 - self.v*t)
    
    def __call__(self, x: float|np.ndarray, t: float) -> float|np.ndarray:
        return np.tanh(self.z(x, t))
    
    def dt(self, x: float|np.ndarray, t: float) -> float|np.ndarray:
        return -self._c*self.v/np.cosh(self.z(x, t))**2
    
    def initial_config(x: np.ndarray, lamb: float, x0s: tuple[float], vs: tuple[float], gnd: int=-1) -> np.ndarray:
        '''
        Statical function to get initial configuration for kink scattering simulations.

        Parameters
        ----------
        x: ndarray
            Spatial array values.
        lamb: float
            The $\lambda$ factor.
        x0s: tuple[float]
            Initial positions of the kinks.
        vs: tuple[float]
            Inivial velocitys of the kinks.
        gnd: int
            Field state at left of the spatial range in the initial instant.
        '''
        y0 = np.zeros((2, x.shape[0]))
        y0[0].fill(gnd)
        for i, j in enumerate(np.argsort(x0s)):
            q = gnd*(-1)**(i + 1)
            k = Kink(x0s[j], vs[j], lamb)
            y0[0] += q*k(x, t=0)
            y0[1] += q*k.dt(x, t=0)
        return y0

@dataclass
class KinkCollider:
    '''
    A class to speed up kink scattering processes.

    Parameters
    ----------
    x_lattice: Lattice
        The spatial lattice objetc, ensure that the lattice have an axis with "x" name.
    dt: float
        Temporal grid spacing.
    
    Attributes
    ----------
    diff_dx: ndarray
        A matrix that provides the second derivative on the space axis, given the `x_lattice` informations.
    '''
    x_lattice: Lattice # (x0, xf, dx)
    dt: float

    def __post_init__(self):
        assert 'x' in tuple(self.x_lattice.ranges.keys()), 'Set the axis name for "x"'
        x0, xf, dx = self.x_lattice.axes['x']
        self.diff_dx = Diff(2, int(abs((xf - x0)/dx)), 5, dx)
    
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
            self.diff_dx(y) + lamb*y*(1 - y**2) # = ddy(t)
        ))
    
    def collide(self, x0s: tuple[float], vs: tuple[float], lamb: float, t_final: float) -> np.ndarray:
        '''
        Run a collision.

        Parameters
        ----------
        x0s: tuple[float]
            Initial positions of the kinks.
        vs: tuple[float]
            Inivial velocitys of the kinks.
        lamb: float
            The $\lambda$ factor.
        t_final: float
            Final value for the time.
        '''
        y0 = Kink.initial_config(self.x_lattice.x, lamb, x0s, vs, gnd=-1)
        solver = RKSolver(partial(self.F, lamb=lamb), y0, self.dt)
        solver.run_util(t_final)
        return solver.y[:, 0]
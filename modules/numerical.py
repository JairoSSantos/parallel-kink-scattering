import numpy as np
from math import factorial
from dataclasses import dataclass
from functools import partial

def coefficients(m, alpha):
    alpha = np.stack(alpha)
    p = len(alpha)
    A = np.zeros((p, p))
    A[0].fill(1)
    for k in range(1, p):
        A[k] = alpha**k
    return factorial(m)*np.linalg.inv(A)[:, m]

@dataclass
class Diff:
    m: int
    n: int
    p: int
    h: float

    def __post_init__(self):
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
    
    def __call__(self, y):
        return self.M.dot(y)

class RKSolver:
    def __init__(self, F, y0, dt, t0=0):
        self.dt = dt
        self.F = F
        self._y = [y0]
        self._t = [t0]

    def step(self):
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

    def run_util(self, T):
        while self._t[-1] < T:
            self.step()

    @property
    def y(self):
        return np.stack(self._y)

    @property
    def t(self):
        return np.stack(self._t)

class Lattice:
    def __init__(self, **axes):
        self.axes = axes
        self.ranges = {kw:np.arange(xl, xr, dx) for kw, (xl, xr, dx) in axes.items()}
    
    def __getattr__(self, kw):
        return self.ranges[kw]
    
    def __getitem__(self, axis):
        return self.ranges[axis]
    
    @property
    def shape(self):
        return tuple(map(len, self.ranges.values()))
    
    @property
    def grid(self):
        return np.stack(np.meshgrid(*tuple(self.ranges.values())), axis=-1)
    
    def at(self, **locs):
        kws = tuple(locs.keys())
        return [
            np.abs(X - locs[kw]).argmin() if kw in kws else Ellipsis 
            for kw, X in self.ranges.items()
        ]
    
    def window(self, **lims):
        kws = tuple(lims.keys())
        return [
            slice(np.abs(X - lims[kw][0]).argmin(), 
                  np.abs(X - lims[kw][1]).argmin())
            if kw in kws else Ellipsis 
            for kw, X in self.ranges.items()
        ]


@dataclass
class Kink:
    x0: float
    v: float
    lamb: float

    def __post_init__(self):
        gamma = 1/(1 - self.v**2)**0.5
        delta = (2/self.lamb)**0.5
        self._c = gamma/delta

    def z(self, x, t):
        return self._c*(x - self.x0 - self.v*t)
    
    def __call__(self, x, t):
        return np.tanh(self.z(x, t))
    
    def dt(self, x, t):
        return -self._c*self.v/np.cosh(self.z(x, t))**2
    
    def initial_config(x, lamb, x0s, vs, gnd=-1):
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
    x_lattice: Lattice # (x0, xf, dx)
    dt: float

    def __post_init__(self):
        x0, xf, dx = self.x_lattice.axes['x']
        self.diff_dx = Diff(2, int(abs((xf - x0)/dx)), 5, dx)
    
    def F(self, t, Y, lamb):
        y, dy_dt = Y
        return np.stack((
            dy_dt, # = y'(t)
            self.diff_dx(y) + lamb*y*(1 - y**2) # = y''(t)
        ))
    
    def collide(self, x0s, vs: tuple[float], lamb: float, t_final: float):
        y0 = Kink.initial_config(self.x_lattice.x, lamb, x0s, vs, gnd=-1)
        solver = RKSolver(partial(self.F, lamb=lamb), y0, self.dt)
        solver.run_util(t_final)
        return solver.y[:, 0]
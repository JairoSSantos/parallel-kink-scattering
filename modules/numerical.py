import numpy as np
from math import factorial
from dataclasses import dataclass

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

class KinkAntikink(RKSolver):
    def __init__(self, L, N, dt_dx, x0, v, lamb, x_order=4):

        self.N = N
        self.dx = 2*L/N
        self.dt = dt_dx*self.dx
        self.x = np.arange(-L, L, self.dx)
        self.diff = Diff(2, N, x_order+1, self.dx)

        self.lamb = lamb
        k1 = Kink(-x0, v, lamb)
        k2 = Kink(x0, -v, lamb)

        f = k1(x, t=0) - k2(x, t=0) - 1
        g = k1.dt(x, t=0) - k2.dt(x, t=0)

        self._y = [np.stack((f, g))]
        self._t = [0]
    
    def h(self, y):
        return self.lamb*y*(1 - y**2)

    def F(self, t, Y):
        y, dy_dt = Y
        return np.stack((
            dy_dt, # = y'
            self.diff(y) + self.h(y) # = y''
        ))
    
    @property
    def y(self):
        return np.stack(self._y)[:, 0]
    
class Lattice:
    def __init__(self, *setup):
        self.ranges = [np.arange(xl, xr, dx) for xl, xr, dx in setup]
    
    def __getitem__(self, axis):
        return self.ranges[axis]
    
    @property
    def shape(self):
        return tuple(map(len, self.ranges))
    
    @property
    def grid(self):
        return np.stack(np.meshgrid(*self.ranges), axis=-1)
    
    def loc(self, x, axis=0):
        return np.abs(self.ranges[axis] - x).argmin()
    
    def at(self, *locs):
        return tuple((
            self.loc(x, axis=i) if not x in (Ellipsis, None) else x 
            for i, x in enumerate(locs)
        ))
    
    def window(self, *lims):
        return tuple((
            slice(self.loc(lim[0], axis=i), self.loc(lim[1], axis=i)) if not lim in (Ellipsis, None) else lim 
            for i, lim in enumerate(lims)
        ))
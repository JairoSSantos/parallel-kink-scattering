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


class Grid:
    def __init__(self, xl, xr, dx=None, N=None):
        assert (dx != None or N != None)
        self.xl, self.xr = xl, xr
        if dx != None:
            self.dx = dx
            self.N = int((xr - xl)/dx)
        elif N != None:
            self.dx = (xr - xl)/N
            self.N = N
        self.x = np.arange(self.xl, self.xr, self.dx)
    
    def window(self, left=None, right=None):
        if left == None: left = self.xl
        if right == None: right = self.xr
        return (self.x >= left) & (self.x <= right)
    
    def at(self, x):
        return np.abs(self.x - x).argmin()

class HyperProblem:
    def __init__(self, grid, h, f, g):
        self.grid = grid
        self.h = h
        self.y0 = np.stack((
            f(grid.x),
            g(grid.x)
        ))
        self.diff = Diff(2, self.grid.N, 5, self.grid.dx)
        
    def d2y_dx2(self, t, y):
        return self.diff(y) + self.h(self.grid.x, t, y)

    def F(self, t, Y):
        y, dy_dt = Y
        return np.stack((dy_dt, self.d2y_dx2(t, y)))

class RKSolver:
    def __init__(self, hyper_problem, t0, dt):
        self.dt = dt
        self.F = hyper_problem.F
        self._y = [hyper_problem.y0]
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
    def __init__(self, grid, x0, v, lamb, dt, t0=0):
        self.lamb = lamb
        self._k1 = Kink(-x0, v, lamb)
        self._k2 = Kink(x0, -v, lamb)
        hyper_problem = HyperProblem(grid, self.h, self.f, self.g)
        self._yl, self._yr = hyper_problem.y0[0, 0], hyper_problem.y0[0, -1]
        super().__init__(hyper_problem, t0, dt)
    
    def f(self, x):
        return self._k1(x=x, t=0) - self._k2(x=x, t=0) - 1
    
    def g(self, x):
        return self._k1.dt(x=x, t=0) - self._k2.dt(x=x, t=0)
    
    def h(self, x, t, y):
        return self.lamb*y*(1 - y**2)
    
    @property
    def y(self):
        return np.stack(self._y)[:, 0]
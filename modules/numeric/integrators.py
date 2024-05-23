import numpy as np
from abc import ABC, abstractmethod
from typing import Callable

_NUMERIC = float|np.ndarray[float]

a = 2**(1/3)
b = 2**(1/5)
x1 = 1/(2 - a)
x0 = -a*x1
y1 = 1/(2 - b)
y0 = -b*y1

_SYMPLECTIC = {}

# ===== 4th-order symplectic coefficients (F. Neff, Lie algebras and canonical integration)
c1 = c4 = x1/2
c2 = c3 = (x0 + x1)/2
d1 = d3 = x1
d2 = x0
d4 = 0

_SYMPLECTIC['4th-order'] = (
    (c1, c2, c3, c4),
    (d1, d2, d3, d4)
)

# ===== 6th-order symplectic coefficients (H. Yoshida, Construction of higher order symplectic integrators)
d1 = d3 = d7 = d9 = x1*y1
d2 = d8 = x0*y1
d4 = d6 = x1*y0
d5 = x0*y0
d10 = 0
c1 = d1/2

ds = (d1, d2, d3, d4, d5, d6, d7, d8, d9, d10)
_SYMPLECTIC['6th-order'] = (
    ds,
    (c1, *((ds[i] + ds[i+1])/2 for i in range(9)))
)

class Integrator(ABC):
    def __init__(self, 
                 func: Callable[[float, _NUMERIC], _NUMERIC], 
                 dt: float, 
                 event: Callable[[float, _NUMERIC], None]=(lambda t, Y: None)):
        self.func = func
        self.dt = dt
        self.event = event

    @abstractmethod
    def step(self, t: float, Y: _NUMERIC) -> _NUMERIC:
        pass

    def run(self, y0: _NUMERIC, t_final: float, t0: float=0):
        t = [t0]
        Y = [y0]
        while t[-1] < t_final:
            Y.append(self.step(t[-1], Y[-1]))
            t.append(t[-1] + self.dt)
            self.event(t[-1], Y[-1])
        return np.stack(t), np.stack(Y)

class RungeKutta4th(Integrator):
    def step(self, t, Y):
        k1 = self.func(t, Y)
        k2 = self.func(t + self.dt/2, Y + k1*self.dt/2)
        k3 = self.func(t + self.dt/2, Y + k2*self.dt/2)
        k4 = self.func(t + self.dt, Y + k3*self.dt)
        return Y + (self.dt/6)*(k1 + 2*k2 + 2*k3 + k4)

class Symplectic(Integrator):
    def __init__(self, *args, integrator: str='4th-order', **kwargs):
        super().__init__(*args, **kwargs)
        self.coeffs = np.stack(_SYMPLECTIC[integrator], axis=1) # [(c1, d1), (c2, d2), ...]

    def step(self, t, Y):
        y, dy = Y
        for c, d in self.coeffs:
            y = y + self.dt*c*dy
            dy = dy + self.dt*d*self.func(t, (y, dy))[-1]
        return np.r_[[y], [dy]]

class Symplectic4th(Symplectic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, integrator='4th-order', **kwargs)

class Symplectic6th(Symplectic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, integrator='6th-order', **kwargs)
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
                 dt: float, 
                 func: Callable[[float, _NUMERIC], _NUMERIC]):
        
        self.dt = dt
        self.func = func

    @abstractmethod
    def step(self, t: float, u: _NUMERIC) -> _NUMERIC:
        pass

    def run(self, t_final: float, Y0: _NUMERIC, t0: float=0, 
            stack: bool=True, event: Callable[[float, _NUMERIC], None]=None):
        event_true = (event != None)
        if stack:
            t, Y = [t0], [Y0]
            while True:
                new_Y = self.step(t[-1], Y[-1])
                if event_true: event(t[-1], Y[-1])
                Y.append(new_Y)
                t.append(t[-1] + self.dt)
                if t[-1] > t_final: break
            return np.stack(t), np.stack(Y)
        else:
            while True:
                Y0 = self.step(t0, Y0)
                t0 += self.dt
                if event_true: event(t0, Y0)
                if t0 > t_final: break
            return t0, Y0

class RungeKutta4th(Integrator):

    def step(self, t, u):
        k1 = self.func(t, *u)
        k2 = self.func(t + self.dt/2, *(u + k1*self.dt/2))
        k3 = self.func(t + self.dt/2, *(u + k2*self.dt/2))
        k4 = self.func(t + self.dt, *(u + k3*self.dt))
        return u + (self.dt/6)*(k1 + 2*k2 + 2*k3 + k4)

class Symplectic(Integrator):
    def __init__(self, *args, integrator: str='4th-order', **kwargs):
        super().__init__(*args, **kwargs)
        self.coeffs = np.stack(_SYMPLECTIC[integrator], axis=1) # [(c1, d1), (c2, d2), ...]

    def step(self, t, u):
        y, y_dt = u
        for c, d in self.coeffs:
            y = y + self.dt*c*y_dt
            _, y_dt2 = self.func(t, y, y_dt)
            y_dt = y_dt + self.dt*d*y_dt2
        return np.stack((y, y_dt))

class Symplectic4th(Symplectic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, integrator='4th-order', **kwargs)

class Symplectic6th(Symplectic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, integrator='6th-order', **kwargs)
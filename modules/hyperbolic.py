import numpy as np
from dataclasses import dataclass
from collections.abc import Callable

@dataclass
class Wave1dProblem:
    dt: float
    dx: float
    L:  float
    f:  Callable[[float], float]
    g:  Callable[[float], float]
    x0: float=0
    u:  float=1
    left_free:  bool=False
    right_free: bool=False
    h:  Callable[[float, float, float], float] = None

    def __post_init__(self):
        self.N = int((self.L - self.x0)/self.dx)
        self.x = np.linspace(self.x0, self.L, self.N)
        self._t = [0, self.dt]
        f = self.f(self.x)
        self._Y = [f[1:-1]]
        self._Y.append(
            f[1:-1] + self.dt*self.g(self.x[1:-1]) + (self.dt**2/2)*(self.get_h(0) + self.u**2*second_4(f, self.dx))
        )
        self._f0 = self.f(self.x0)
        self._fL = self.f(self.L)

    def _get_alpha(self, i: int=-1):
        return self._f0 if not self.left_free else self._Y[i][0]
    
    def _get_beta(self, i: int=-1):
        return self._fL if not self.right_free else self._Y[i][-1]
    
    @property
    def y(self):
        return np.r_[self._get_alpha(-1), self._Y[-1], self._get_beta(-1)]
    
    @property
    def t(self):
        return np.stack(self._t)
    
    @property
    def Y(self):
        return np.stack([
            np.r_[
                self._get_alpha(i), 
                self._Y[i], 
                self._get_beta(i),
            ] 
            for i in range(len(self._Y))
        ])
    
    @property
    def b(self):
        return np.r_[self._get_alpha(-1), np.zeros(self.N-4), self._get_beta(-1)]
    
    def get_h(self, i: int):
        if self.h == None:
            return 0
        else:
            return self.h(self._t[i], self.x[1:-1], self._Y[i])
    
    def get_grid_extents(self):
        return self.x.min(), self.x.max(), self.t.min(), self.t.max()
    
    def run(self, steps: int):
        for _ in range(steps):
            self.step()

class Wave1d(Wave1dProblem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lamb = (self.u*self.dt/self.dx)**2
        self.LAMB = 2*(1 - self.lamb)*np.identity(self.N-2) + self.lamb*(np.diagflat(np.ones(self.N-3), 1) + np.diagflat(np.ones(self.N-3), -1))
    
    def step(self):
        self._Y.append(self.LAMB.dot(self._Y[-1]) - self._Y[-2] + self.lamb*self.b + self.dt**2*self.get_h(-1))
        self._t.append(self._t[-1] + self.dt)
import numpy as np
from math import sqrt
from dataclasses import dataclass

NUMERIC = float|np.ndarray[float]

@dataclass
class Phi4:
    def __init__(self, lamb: float=2, eta: float=1):
        self.lamb, self.eta = lamb, eta
    
    @property
    def scale_factor(self):
        return self.eta*sqrt(self.lamb/2)

    def __call__(self, y: NUMERIC) -> NUMERIC:
        return self.lamb*(y**2 - self.eta**2)**2/4
    
    def diff(self, y: NUMERIC) -> NUMERIC:
        return self.lamb*y*(y**2 - self.eta**2)

    def kink(self, x: NUMERIC, t: float, v: float) -> NUMERIC:
        gamma = 1/sqrt(1 - v**2)
        z = gamma*(x - v*t)
        return self.eta*np.tanh(self.scale_factor*z)

    def kink_dt(self, x: NUMERIC, t: float, v: float) -> NUMERIC:
        gamma = 1/sqrt(1 - v**2)
        z = gamma*(x - v*t)
        return -self.eta*self.scale_factor*gamma*v/np.cosh(self.scale_factor*z)**2
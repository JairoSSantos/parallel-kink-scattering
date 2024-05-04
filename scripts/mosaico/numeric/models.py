import numpy as np
from math import sqrt
from dataclasses import dataclass

NUMERIC = float|np.ndarray[float]

@dataclass
class Phi4:
    def __init__(self, scale: float=2, vacuum: float=1):
        self.scale, self.vacuum = scale, vacuum
    
    @property
    def scale_factor(self):
        return self.vacuum*sqrt(self.scale/2)

    def __call__(self, y: NUMERIC) -> NUMERIC:
        return (self.scale/4)*(y**2 - self.vacuum**2)**2
    
    def diff(self, y: NUMERIC) -> NUMERIC:
        return self.scale*y*(y**2 - self.vacuum**2)

    def kink(self, x: NUMERIC, t: float, v: float) -> NUMERIC:
        gamma = 1/sqrt(1 - v**2)
        z = gamma*(x - v*t)
        return self.vacuum*np.tanh(self.scale_factor*z)

    def kink_dt(self, x: NUMERIC, t: float, v: float) -> NUMERIC:
        gamma = 1/sqrt(1 - v**2)
        z = gamma*(x - v*t)
        return -self.vacuum*self.scale_factor*gamma*v/np.cosh(gamma*(x - v*t))**2
import numpy as np
from math import sqrt
from dataclasses import dataclass

NUMERIC = float|np.ndarray[float]

def kink(x: NUMERIC, t: float, v: float, c: float=1) -> NUMERIC:
    gamma = 1/sqrt(1 - v**2/c**2)
    return np.tanh(gamma*(x - v*t))

def kink_dt(x: NUMERIC, t: float, v: float, c: float=1) -> NUMERIC:
    gamma = 1/sqrt(1 - v**2/c**2)
    return -gamma*v/np.cosh(gamma*(x - v*t))**2

# def delta(lamb: float|int) -> float:
#     return np.sqrt(2/lamb)

# def kink(x0: float|int, v: float|int, lamb: float|int):
#     gamma = 1/np.sqrt(1 - v**2)
#     c1 = gamma/delta(lamb)
#     x, t = sp.symbols('x, t')
#     z = c1*(x - x0 - v*t)
#     return Field((x, t), sp.tanh(z))

# def kink_boundary(x0: float, v: float, lamb: float, H: float):
#     gamma = 1/np.sqrt(1 - v**2)
#     c1 = gamma/delta(lamb)

#     x, t = sp.symbols('x, t')
#     z = c1*(x - x0 - v*t)
#     phi = sp.tanh(z)

#     if H > 0:
#         X0 = acosh(1/sqrt(abs(H)))
#         phi_mod = sp.tanh(x - X0)
#     else:
#         X1 = asinh(1/sqrt(abs(H)))
#         phi_mod = 1/sp.tanh(x - X1)

#     return Field((x, t), phi_mod - phi + 1)

@dataclass
class Phi4:
    scale: float=2
    vacuum: float=1

    def __call__(self, y: NUMERIC) -> NUMERIC:
        return (self.scale/4)*(y**2 - self.vacuum**2)**2
    
    def diff(self, y: NUMERIC) -> NUMERIC:
        return self.scale*y*(y**2 - self.vacuum**2)
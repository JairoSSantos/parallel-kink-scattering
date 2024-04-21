import numpy as np
from math import factorial

class Boundary:
    def __init__(self, m: int, order: int, btype: int, param: float=None, h: float=1):
        self.m = m
        nodes = (order + m)//2 - 1
        self.w = 5
        self.param = param

        K = np.arange(self.w)
        M = np.r_[
            [np.where(K == btype, factorial(btype), 0)],
            np.vander(K[1:], self.w, increasing=True)
        ]
        M_inv = np.linalg.inv(M)
        self.C = np.stack([
            [sum([self._ell(i, j)*M_inv[j, k] for j in range(m, self.w)]) 
             for k in range(self.w)]
            for i in range(nodes)
        ])/h**m
    
    def _ell(self, i, j):
        return i**(j - self.m)*factorial(j)/factorial(j - self.m)
    
    def set_param(self, value):
        self.param = value
    
    def __call__(self, Y):
        return self.C @ np.r_[self.param, Y[1:self.w]]

class Dirichlet(Boundary):
    def __init__(self, m: int, order: int, param: float=None, h: float=1):
        super().__init__(btype=0, m=m, order=order, param=param, h=h)

class Neumann(Boundary):
    def __init__(self, m: int, order: int, param: float=None, h: float=1):
        super().__init__(btype=1, m=m, order=order, param=param, h=h)

class Reflective(Boundary):
    def __init__(self, m: int, order: int, h: float=1):
        self.nodes = (order + m)//2 - 1
        self.C = factorial(m)*np.linalg.inv(np.vander(np.r_[-self.nodes:self.nodes+1], increasing=True))[m]/h**m
    
    def __call__(self, Y):
        return np.convolve(np.r_[Y[1:self.nodes+1][::-1], Y[:2*self.nodes]], self.C, mode='valid')
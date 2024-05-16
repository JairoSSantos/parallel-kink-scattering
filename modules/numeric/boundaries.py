import numpy as np
from math import factorial
from typing import Callable

class Boundary:
    def __init__(self, order: int, btype: int, param: float=None, h: float=1):
        self.m = 2
        nodes = (order + self.m)//2 - 1
        self.w = order + 1
        self.param = param
        self.btype = btype

        K = np.arange(self.w)
        M = np.r_[
            [np.where(K == btype, factorial(btype), 0)],
            np.vander(K[1:] if btype == 0 else K[:-1], self.w, increasing=True)
        ]
        M_inv = np.linalg.inv(M)
        self.C = np.stack([
            [sum([self._ell(i, j)*M_inv[j, k] for j in range(self.m, self.w)]) 
             for k in range(self.w)]
            for i in range(nodes)
        ])/h**self.m
    
    def _ell(self, i, j):
        return i**(j - self.m)*factorial(j)/factorial(j - self.m)
    
    def __call__(self, Y):
        return self.C @ np.r_[self.param, Y[1:self.w] if self.btype == 0 else Y[:self.w-1]]

class Dirichlet(Boundary):
    def __init__(self, f: Callable, order: int, param: float=None, h: float=1):
        self.f = f
        super().__init__(btype=0, order=order, param=param, h=h)
    
    def __call__(self, Y):
        return np.r_[self.f(Y[0]), Boundary.__call__(self, Y)[1:]]

# class Dirichlet(Boundary):
#     def __init__(self, order: int, param: float=None, h: float=1):
#         super().__init__(btype=0, order=order, param=param, h=h)

class Neumann(Boundary):
    def __init__(self, order: int, param: float=None, h: float=1):
        super().__init__(btype=1, order=order, param=param, h=h)

class Reflective(Boundary):
    def __init__(self, order: int, h: float=1):
        m = 2
        self.nodes = (order + m)//2 - 1
        self.C = factorial(m)*np.linalg.inv(np.vander(np.r_[-self.nodes:self.nodes+1], increasing=True))[m]/h**m
    
    def __call__(self, Y):
        return np.convolve(np.r_[Y[1:self.nodes+1][::-1], Y[:2*self.nodes]], self.C, mode='valid')
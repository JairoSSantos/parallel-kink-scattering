import numpy as np
from math import factorial

class Boundary:
    def __init__(self, nodes: int, order: int, derivative: int, param: float=None):
        K = np.arange(order)
        M = np.r_[
            [np.where(K == derivative, factorial(derivative), 0)],
            np.vander(K[1:], order, increasing=True)
        ]
        S = np.vander(-np.r_[1:nodes+1], order, increasing=True)
        self.G = S @ np.linalg.inv(M)
        self.param = param
        self.order = order
        self.derivative = derivative
    
    def set_param(self, value):
        self.param = value
    
    def __call__(self, Y):
        y = np.r_[self.param, Y[1:self.order]]
        y_ghost = (self.G @ y)[::-1]
        if self.derivative > 0:
            return np.r_[y_ghost, Y]
        else:
            return np.r_[y_ghost, self.param, Y[1:]]

class Dirichlet(Boundary):
    def __init__(self, nodes: int, order: int, param: float=None):
        super().__init__(nodes=nodes, order=order, derivative=0, param=param)

class Neumann(Boundary):
    def __init__(self, nodes: int, order: int, param: float=None):
        super().__init__(nodes=nodes, order=order, derivative=1, param=param)

class Reflective(Boundary):
    def __init__(self, nodes: int):
        self.nodes = nodes
    
    def __call__(self, Y):
        return np.r_[Y[1:self.nodes+1][::-1], Y]
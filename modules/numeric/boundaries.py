import numpy as np
from math import factorial

class Boundary:
    def __init__(self, m: int, order: int, param: float=None):
        self.order = order
        self.w = order + 1
        self.param = param

        K = np.arange(self.w)
        M = np.r_[
            [np.where(K == m, factorial(m), 0)],
            np.vander(K[1:] if m == 0 else K[:-1], self.w, increasing=True)
            # np.vander(K[:-1], self.w, increasing=True)
        ]
        self.M_inv = np.linalg.inv(M)
        self.slice = slice(1, self.w) if m == 0 else slice(0, self.w - 1)
        # self.slice = slice(0, self.w - 1)

    def _ell(self, n, i, j):
        return i**(j - n)*factorial(j)/factorial(j - n)
    
    def _c_tilde(self, n, i, k):
        return sum([self._ell(n, i, j)*self.M_inv[j, k] for j in range(n, self.w)])

    def get_diff(self, n: int, h: float=1):
        nodes = (self.order + n)//2 - 1
        return np.array([[self._c_tilde(n, i, k) for k in range(self.w)] for i in range(nodes)])/h**n
    
    def get_ghost_operator(self, nodes):
        return np.vander(-np.arange(1, nodes+1), self.w, increasing=True) @ self.M_inv
    
    def take(self, y):
        return np.r_[self.param, y[self.slice]]

class Dirichlet(Boundary):
    def __init__(self, order: int, param: float=None):
        super().__init__(m=0, order=order, param=param)

class Neumann(Boundary):
    def __init__(self, order: int, param: float=None):
        super().__init__(m=1, order=order, param=param)

class Reflective(Neumann):
    def __init__(self, order: int):
        super().__init__(order=order, param=0)

class Ghost:
    def __init__(self, nodes: int, lb: Boundary, rb: Boundary):
        self.lb = lb
        self.rb = rb
        self.lb_operator = lb.get_ghost_operator(nodes)
        self.rb_operator = rb.get_ghost_operator(nodes)
    
    def __call__(self, y):
        return np.r_[((self.lb_operator @ self.lb.take(y))[::-1], y, (self.rb_operator @ self.rb.take(y[::-1])))]
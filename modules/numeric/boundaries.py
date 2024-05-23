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
    
    def ghost(self, p):
        return np.vander(-np.arange(1, p+1), self.w, increasing=True) @ self.M_inv
    
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

# class Boundary:
#     def __init__(self, m, n, order: int, param: float=None):
#         w = order + 1
#         self.param = param

#         nodes = (order + n)//2 - 1
#         K = np.arange(w)
#         M = np.r_[
#             [np.where(K == m, factorial(m), 0)],
#             np.vander(K[1:] if m == 0 else K[:-1], w, increasing=True)
#         ]
#         M_inv = np.linalg.inv(M)

#         def ell(i, j):
#             return i**(j - n)*factorial(j)/factorial(j - n)
        
#         def c_tilde(i, k):
#             return sum([ell(i, j)*M_inv[j, k] for j in range(n, w)])

#         self.C = np.stack([[c_tilde(i, k) for k in range(w)] for i in range(nodes)])/h**n
#         self.slice = slice(1, w) if m == 0 else slice(0, w-1)
    
#     def __call__(self, Y):
#         return self.C @ np.r_[self.param, Y[self.slice]]

# class Dirichlet(Boundary):
#     def __init__(self, f: Callable, n: int, order: int, param: float=None, h: float=1):
#         self.f = f
#         super().__init__(m=0, n=n, order=order, param=param, h=h)
    
#     def __call__(self, Y):
#         return np.r_[self.f(Y[0]), Boundary.__call__(self, Y)[1:]]

# class Neumann(Boundary):
#     def __init__(self, n: int, order: int, param: float=None, h: float=1):
#         super().__init__(m=1, n=n, order=order, param=param, h=h)

# class Reflective(Neumann):
#     def __init__(self, n, order: int, h: float=1):
#         super().__init__(n=n, order=order, param=0, h=h)

# class Reflective(Boundary):
#     def __init__(self, order: int, h: float=1):
#         m = 2
#         self.nodes = (order + m)//2 - 1
#         self.C = factorial(m)*np.linalg.inv(np.vander(np.r_[-self.nodes:self.nodes+1], increasing=True))[m]/h**m
    
#     def __call__(self, Y):
#         return np.convolve(np.r_[Y[1:self.nodes+1][::-1], Y[:2*self.nodes]], self.C, mode='valid')
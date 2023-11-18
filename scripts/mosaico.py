import numpy as np
import pandas as pd
from multiprocessing import Pool
from os import cpu_count
import time
from pathlib import Path
data_dir = Path('../data/mosaico')

def coefficients_matrix(N: int):
    M = np.zeros((N-2, N))
    for i in range(1, N-3):
        M[i, i-1:i+4] = (-1/12, 4/3, -5/2, 4/3, -1/12)
    M[0, :3] = M[-1, -3:] = (1, -2, 1)
    return M

class Grid:
    def __init__(self, xl, xr, dx=None, N=None):
        assert (dx != None or N != None)
        self.xl, self.xr = xl, xr
        if dx != None:
            self.dx = dx
            self.N = int((xr - xl)/dx)
        elif N != None:
            self.dx = (xr - xl)/N
            self.N = N
        self.x = np.arange(self.xl, self.xr, self.dx)

class HyperProblem:
    def __init__(self, grid, h, f, g):
        self.grid = grid
        self.h = h
        self.Y0 = np.stack((
            f(grid.x[1:-1]),
            g(grid.x[1:-1])
        ))
        M = coefficients_matrix(grid.N)/grid.dx**2
        self.M = M[:, 1:-1]
        self.M0 = np.zeros(grid.N - 2)
        self.M0[:2] = M[:2, 0]*f(grid.xl)
        self.M0[-2:] = M[-2:, -1]*f(grid.xr)
        
    def d2y_dx2(self, t, y):
        return self.M0 + self.M.dot(y) + self.h(self.grid.x[1:-1], t, y)

    def F(self, t, Y):
        y, dy_dt = Y
        return np.stack((dy_dt, self.d2y_dx2(t, y)))

class RKSolver:
    def __init__(self, t0, dt, hproblem):
        self.dt = dt
        self.F = hproblem.F
        self._Y = [hproblem.Y0]
        self._t = [t0]

    def step(self):
        t = self._t[-1]
        y = self._Y[-1]
        dt = self.dt

        k1 = self.F(t, y)
        k2 = self.F(t + dt/2, y + k1*dt/2)
        k3 = self.F(t + dt/2, y + k2*dt/2)
        k4 = self.F(t + dt, y + k3*dt)

        self._Y.append(
            y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        )
        self._t.append(t + dt)

    def run_util(self, T):
        while self._t[-1] < T:
            self.step()

    @property
    def Y(self):
        return np.stack(self._Y)

    @property
    def t(self):
        return np.stack(self._t)

def kink_antikink(x0, v, lamb):
    try: len(v)
    except TypeError: 
        v1 = v
        v2 = -v
    else: v1, v2 = v
    finally:
        g1 = 1/(1 - v1**2)**0.5
        g2 = 1/(1 - v2**2)**0.5
        delta = (2/lamb)**0.5
        c1 = g1/delta
        c2 = g2/delta
        return (
            lambda x, t, y: lamb*y*(1 - y**2),
            lambda x: np.tanh((x + x0)*c1) - np.tanh((x - x0)*c2) - 1,
            lambda x: - c1*v1/np.cosh((x + x0)*c1)**2 + c2*v2/np.cosh((x - x0)*c2)**2
        )

if __name__ == '__main__':
    x0 = 5
    grid = Grid(-40, 40, N=1000)

    def solve_and_save(vl):
        v, lamb = vl
        tic = time.time()
        solver = RKSolver(0, grid.dx*0.7, HyperProblem(grid, *kink_antikink(x0, v, lamb)))
        solver.run_util(x0/v + grid.xr)
        tac = time.time()
        data = pd.DataFrame(solver.Y[:, 0], columns=grid.x[1:-1], index=solver.t)
        data.to_csv(data_dir/f'{v}-{lamb}-{tac-tic}.csv')

    v = np.linspace(0.1, 0.3, 100)
    lamb = np.linspace(1, 50, 100)
    v, lamb = np.meshgrid(v, lamb)
    v_lamb = np.stack((v.flatten(), lamb.flatten()), axis=1)
    
    with Pool(processes=cpu_count()) as pool:
        pool.map(solve_and_save, v_lamb)
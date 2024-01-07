import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
from time import time
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%d/%m/%Y - %H:%M:%S'))
logger.addHandler(ch)

from math import factorial
from dataclasses import dataclass

def coefficients(m, alpha):
    alpha = np.stack(alpha)
    p = len(alpha)
    A = np.zeros((p, p))
    A[0].fill(1)
    for k in range(1, p):
        A[k] = alpha**k
    return factorial(m)*np.linalg.inv(A)[:, m]

@dataclass
class Diff:
    m: int
    n: int
    p: int
    h: float

    def __post_init__(self):
        assert self.n >= 2*self.p
        P = np.arange(self.p)[np.newaxis].repeat(self.p, axis=0)
        l = int(self.p/2)
        d = int(l*2 != self.p)
        C = np.stack([coefficients(self.m, alpha) for alpha in P - P.T])
        M = np.zeros((self.n - 2*l, self.n))
        for i in range(self.n - 2*l):
            M[i, i:i+self.p] = C[l]
        self.M = np.c_[ 
            '0',
            np.pad(C[:l], [(0, 0), (0, self.n - self.p)]),
            M,
            np.pad(C[l+d:], [(0, 0), (self.n - self.p, 0)]),
        ]/(self.h**self.m)
    
    def __call__(self, y):
        return self.M.dot(y)

class RKSolver:
    def __init__(self, F, y0, dt, t0=0):
        self.dt = dt
        self.F = F
        self._y = [y0]
        self._t = [t0]

    def step(self):
        t = self._t[-1]
        y = self._y[-1]
        dt = self.dt

        k1 = self.F(t, y)
        k2 = self.F(t + dt/2, y + k1*dt/2)
        k3 = self.F(t + dt/2, y + k2*dt/2)
        k4 = self.F(t + dt, y + k3*dt)

        self._y.append(
            y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        )
        self._t.append(t + dt)

    def run_util(self, T):
        while self._t[-1] < T:
            self.step()

    @property
    def y(self):
        return np.stack(self._y)

    @property
    def t(self):
        return np.stack(self._t)

class Lattice:
    def __init__(self, **axes):
        self.axes = axes
        self.ranges = {kw:np.arange(xl, xr, dx) for kw, (xl, xr, dx) in axes.items()}
    
    def __getattr__(self, kw):
        return self.ranges[kw]
    
    def __getitem__(self, axis):
        return self.ranges[axis]
    
    @property
    def shape(self):
        return tuple(map(len, self.ranges.values()))
    
    @property
    def grid(self):
        return np.stack(np.meshgrid(*tuple(self.ranges.values())), axis=-1)
    
    def at(self, **locs):
        kws = tuple(locs.keys())
        return [
            np.abs(X - locs[kw]).argmin() if kw in kws else Ellipsis 
            for kw, X in self.ranges.items()
        ]
    
    def window(self, **lims):
        kws = tuple(lims.keys())
        return [
            slice(np.abs(X - lims[kw][0]).argmin(), 
                  np.abs(X - lims[kw][1]).argmin())
            if kw in kws else Ellipsis 
            for kw, X in self.ranges.items()
        ]


@dataclass
class Kink:
    x0: float
    v: float
    lamb: float

    def __post_init__(self):
        gamma = 1/(1 - self.v**2)**0.5
        delta = (2/self.lamb)**0.5
        self._c = gamma/delta

    def z(self, x, t):
        return self._c*(x - self.x0 - self.v*t)
    
    def __call__(self, x, t):
        return np.tanh(self.z(x, t))
    
    def dt(self, x, t):
        return -self._c*self.v/np.cosh(self.z(x, t))**2
    
    def initial_config(x, lamb, x0s, vs, gnd=-1):
        y0 = np.zeros((2, x.shape[0]))
        y0[0].fill(gnd)
        for i, j in enumerate(np.argsort(x0s)):
            q = gnd*(-1)**(i + 1)
            k = Kink(x0s[j], vs[j], lamb)
            y0[0] += q*k(x, t=0)
            y0[1] += q*k.dt(x, t=0)
        return y0

@dataclass
class KinkCollider:
    x_lattice: Lattice # (x0, xf, dx)
    dt: float

    def __post_init__(self):
        x0, xf, dx = self.x_lattice.axes['x']
        self.diff_dx = Diff(2, int(abs((xf - x0)/dx)), 5, dx)
    
    def F(self, t, Y, lamb):
        y, dy_dt = Y
        return np.stack((
            dy_dt, # = y'(t)
            self.diff_dx(y) + lamb*y*(1 - y**2) # = y''(t)
        ))
    
    def collide(self, x0s, vs: tuple[float], lamb: float, t_final: float):
        y0 = Kink.initial_config(self.x_lattice.x, lamb, x0s, vs, gnd=-1)
        solver = RKSolver(partial(self.F, lamb=lamb), y0, self.dt)
        solver.run_util(t_final)
        return solver.y[:, 0]

from pathlib import Path

class Config:
    L = 50
    N = int((5/4)*1000)
    dx = 2*L/N
    dt = 0.7*dx

    x_lattice = Lattice(x=(-L, L, dx))
    cm_index = x_lattice.at(x=0)
    collider = KinkCollider(
        x_lattice = x_lattice,
        dt = dt
    )

    v_min = 0.05
    v_max = 0.5
    v_lims = (v_min, v_max)

    lamb_min = 0.075
    lamb_max = 50
    lamb_lims = (lamb_min, lamb_max)

    separation_by_delta = 10

    save_dir = Path('../data/dataset')
    save_dir.mkdir(exist_ok= True)
    info_path = save_dir/'info.csv'

def get_info():
    if Config.info_path.exists():
        info = pd.read_csv(Config.info_path).to_dict('list')
    else:
        info = {'lamb': [], 'exec_time': []}
    return info

def it_was_not_calculated(v, lamb):
    return not len(tuple(Config.save_dir.glob(f'{v}-{lamb}-*.csv'))) > 0

def init_collisions_queue(vs, lambs):
    collisions_queue = mp.Queue()

    info = get_info()
    for lamb in lambs:
        if lamb in info['lamb']:
            vs = tuple(filter(partial(it_was_not_calculated, lamb=lamb), vs))
        if len(vs):
            collisions_queue.put({
                'vs': vs,
                'lamb': lamb
            })

    return collisions_queue

def collider_task(collider, collisions_queue, output_queue):
    while not collisions_queue.empty():
        point = collisions_queue.get()
        logger.debug('(%s) ~~ Iniciando colisões para lamb=%s'%(mp.current_process().name, point['lamb']))
        local_t0 = time()
        delta = np.sqrt(2/point['lamb'])
        x0 = Config.separation_by_delta*delta/2

        collisions = []
        for v in point['vs']:
            _t0 = time()
            y = collider.collide(
                x0s= (-x0, x0),
                vs= (v, -v),
                lamb= point['lamb'],
                t_final= 2*x0/v + Config.L
            )
            _tf = time()
            collisions.append({
                'v': v,
                'y': y,
                'delay': _tf - _t0
            })

        local_tf = time()
        output_queue.put({
            'lamb': point['lamb'],
            'exec_time': local_tf - local_t0,
            'collisions': collisions
        })

def manager_task(collisions_queue, output_queue):
    info = get_info()
    total = collisions_queue.qsize()
    received = 0
    while received < total:
        output = output_queue.get()
        received += 1
        logger.debug('({}) Salvando os resultados para lambda={} ({}%)'.format(
            mp.current_process().name, 
            output['lamb'], 
            received/total * 100
        ))

        for collision in output['collisions']:
            pd.DataFrame(collision['y']).to_csv(
                Config.save_dir/'{v}-{lamb}-{delay}.csv'.format(
                    lamb= output['lamb'], 
                    v= collision['v'], 
                    delay= collision['delay']
                ),
                index= False,
                header= False
            )
        
        if output['lamb'] in info['lamb']:
            info['exec_time'][info['lamb'].index(output['lamb'])] = output['exec_time']
        else:
            info['lamb'].append(output['lamb'])
            info['exec_time'].append(output['exec_time'])
        pd.DataFrame(info).to_csv(Config.info_path, index=False)

def init_session(n_processes, collisions_queue):
    output_queue = mp.Queue()

    logger.debug(f'({mp.current_process().name}) Iniciando {n_processes} + 1 processos...')
    
    manager = mp.Process(target=manager_task, args=(collisions_queue, output_queue))
    manager.start()

    processes = []
    for _ in range(n_processes):
        process = mp.Process(target=collider_task, args=(Config.collider, collisions_queue, output_queue))
        process.start()
        processes.append(process)
    
    for process in processes:
        process.join()
    manager.join()

if __name__ == '__main__':
    P = mp.cpu_count() - 1
    V = np.linspace(*Config.v_lims, P*20)
    LAMB = np.linspace(*Config.lamb_lims, P*20)
    init_session(P, init_collisions_queue(V, LAMB))
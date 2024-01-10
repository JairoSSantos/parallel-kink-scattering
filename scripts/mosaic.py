import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
from time import time
import logging

logger = logging.getLogger()

formatter = logging.Formatter('~[%(asctime)s - %(processName)s] %(message)s', datefmt='%d/%m/%Y - %H:%M:%S')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)

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

def get_summary():
    return pd.DataFrame([
        tuple(map(float, filename.stem.split('-'))) 
        for filename in Config.save_dir.glob('*')
    ], columns=('v', 'lamb', 'exec_time', 'delay'))

# def it_was_not_calculated(v, lamb):
    # return not (len(tuple(Config.save_dir.glob(f'{v}-{lamb}-*.csv'))) > 0)
    # summary = get_summary()
    # return np.any(((summary.v == v) & (summary.lamb == lamb)).values)

def init_collisions_queue(vs, lambs):
    collisions_queue = mp.Queue()

    n_collisions = 0
    summary = get_summary()
    for lamb in lambs:
        to_calc_vs = vs[~np.isin(vs, summary[summary.lamb == lamb].v.values)]
        # vs = tuple(filter(partial(it_was_not_calculated, lamb=lamb), vs))

        len_to_calc_vs = len(to_calc_vs)
        if len_to_calc_vs > 0:
            n_collisions += len_to_calc_vs
            collisions_queue.put({
                'vs': to_calc_vs,
                'lamb': lamb
            })

    return n_collisions, collisions_queue

def collider_task(collider, collisions_queue):
    while not collisions_queue.empty():
        point = collisions_queue.get()
        logger.debug('Iniciando %s colisões para lamb=%s'%(len(point['vs']), point['lamb']))
        local_t0 = time()
        delta = np.sqrt(2/point['lamb'])
        x0 = Config.separation_by_delta*delta/2

        for v in point['vs']:
            _t0 = time()
            y = collider.collide(
                x0s= (-x0, x0),
                vs= (v, -v),
                lamb= point['lamb'],
                t_final= 2*x0/v + Config.L
            )
            _tf = time()
            exec_time, delay = _tf - local_t0, _tf - _t0
            kink = []
            for row in y:
                try: kink.append(Config.x_lattice.x[row >= 0].max())
                except ValueError: kink.append(None)
            pd.DataFrame({
                'y_cm': y[:, Config.cm_index[0]],
                'kink': kink
            }).to_csv(
                Config.save_dir/('%s-%s-%s-%s.csv'%(v, point['lamb'],  exec_time, delay)),
                index= False,
                header= False
            )
            logger.debug('Simulação finalizada: lamb=%s; v=%s; delay=%s; exec_time=%s'%(
                point['lamb'],
                v,
                delay,
                exec_time
            ))

def manager_task(n_collisions, collisions_queue, output_queue):
    received = 0
    while received < n_collisions:
        output = output_queue.get()
        received += 1
        logger.debug('Salvando os resultados para lambda={} e v={} ({}%)'.format(
            output['lamb'],
            output['v'],
            received/n_collisions * 100
        ))

        pd.DataFrame(output['y']).to_csv(
            Config.save_dir/('%s-%s-%s-%s.csv'%(output['v'], output['lamb'], output['exec_time'], output['delay'])),
            index= False,
            header= False
        )

def init_session(n_processes, n_collisions, collisions_queue):
    logger.debug(f'Iniciando {n_processes} processos para simular {n_collisions} colisões')
    processes = []
    for _ in range(n_processes):
        process = mp.Process(target=collider_task, args=(Config.collider, collisions_queue))
        process.start()
        processes.append(process)
    
    for process in processes:
        process.join()
    logger.debug(f'Encerrando sessão')

if __name__ == '__main__':
    import sys

    cpu_count = mp.cpu_count()
    params = {
        '-p': cpu_count,
        '--k-v': cpu_count,
        '--k-lamb': cpu_count,
    }
    for arg in sys.argv:
        try:
            name, value = arg.split('=')
        except ValueError: pass
        else:
            match name:
                case '-k': params['--k-v'] = params['--k-lamb'] = int(value)
                case '--dataset-name': 
                    Config.save_dir = Path('../data')/value
                    Config.save_dir.mkdir(exist_ok= True)
                case other: 
                    try: params[other] = value
                    except KeyError: pass
    
    init_session(
        int(params['-p']), 
        *init_collisions_queue(
            np.linspace(*Config.v_lims, int(params['--k-v'])), 
            np.linspace(*Config.lamb_lims, int(params['--k-lamb']))
        )
    )
import sys
sys.path.insert(1, '../../')

import numpy as np
import pandas as pd
from modules.numeric import *
from multiprocessing import Pool, Value
from ctypes import c_int
from pathlib import Path
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

L = 100
N = 1024
DX = L/(N - 1)
DT = 4e-2
X0 = 10

V_IN = np.linspace(0, 1, 752)[1:-1]
HS = (1, 0.75, 0.5, 0.2, 0, -0.2, -0.5, -0.75, 0.9)
TOTAL = len(V_IN)*len(HS)
PHI4 = Phi4()

SAVEPATH = Path('../../data/dirichlet/velocities')
with open(SAVEPATH/'v_in.npy', 'wb') as file:
    np.save(file, V_IN)

LOGGER = get_logger()

def chi(H, mu, nu):
    return np.arctanh((mu*H)**nu)

def varphi(x, mu, nu):
    return -mu*np.tanh(x)**nu

def phi_dirichlet(x, H):
    assert H > -1
    mu = 1
    nu = 1 if H < 1 else -1
    return np.r_[[1]*len(x)] if H == 1 else varphi(x - chi(H, mu, nu), mu, nu)

def y0(x, v, H):
    return np.stack((
        phi_dirichlet(x, H) + PHI4.kink(x + X0, 0, v) - 1,
        PHI4.kink_dt(x + X0, 0, v)
    ))

def track_kink(y, pf=0.55):
    '''
    output: [
        [(t1_kink1, x1_kink1), (t2_kink1, x2_kink1), ...]
        [(t1_kink2, x1_kink2), (t2_kink2, x2_kink2), ...]
        ....
    ]
    '''
    trackers = []
    for t, u in enumerate(-np.abs(y)):
        peaks, _ = find_peaks(u, prominence=pf*u.ptp())
        xs = list(peaks)
        
        added = []
        n_trackers = len(trackers)
        for x in xs:
            if len(added) < n_trackers:
                min_dist = np.inf
                j = 0
                for i in range(n_trackers):
                    if not i in added:
                        dist = abs(trackers[i][-1][1] - x)
                        if dist < min_dist:
                            min_dist = dist
                            j = i
                trackers[j].append((t, x))
                added.append(j)
            else:
                trackers.append([(t, x)])
            
    return trackers

def linear(x, a, b):
    return a*x + b

def velocity(t, x):
    try:
        (v, _), _ = curve_fit(linear, t, x)
    except:
        v = np.nan
    return v

def scan(H):
    global counter

    collider = Collider(
        x_lattice= (-L, 0, N),
        dt= DT, 
        order= 4,
        y0= y0,
        pot_diff= PHI4.diff,
        boundaries= ('reflective', 'dirichlet'),
        integrator= 'sy6'
    )
    collider.rb.param = H
    
    v_out = []
    for v in V_IN:
        LOGGER.debug(f'({(100*counter.value/TOTAL):.2f}%) Rodando a colisão para H={H} e v={v}...')
        with counter.get_lock(): counter.value += 1
        lat, Y = collider.run(X0/v + L, v=v, H=H)
        ub = Y[:, 0, -2]
        vs = [len(find_peaks(-ub, prominence=ub.ptp()/10)[0])]
        k = int(len(lat.t)*0.05)
        for tracker in track_kink(Y[:, 0]):
            tk, xk = np.transpose(tracker)
            t, x = lat.t[tk], lat.x[xk]
            vs.append(velocity(t[-k:], x[-k:]))
        v_out.append(vs)
    LOGGER.debug(f'Salvando resultados para H={H}...')
    with open(SAVEPATH/f'H={H}.npy', 'wb') as file:
        np.save(file, pd.DataFrame(v_out).values)

if __name__ == '__main__':
    global counter
    counter = Value(c_int, 0)

    LOGGER.debug(f'Iniciando simulações...')
    with Pool(len(HS)) as pool:
        pool.map(scan, HS)
    LOGGER.debug(f'Mosaico finalizado')
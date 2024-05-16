import sys
sys.path.insert(1, '../')

import numpy as np
from modules.numeric import *
from multiprocessing import Pool
import logging
from pathlib import Path

phi4 = Phi4()
L = 100
N = 1024
DX = L/(N - 1)
DT = 4e-2
X0 = -10

logger = logging.getLogger()
formatter = logging.Formatter('~[%(asctime)s - %(processName)s] %(message)s', datefmt='%d/%m/%Y - %H:%M:%S')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)

savedir = Path('../data/session-dirichlet')

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
        phi_dirichlet(x, H) + phi4.kink(x - X0, 0, v) - 1,
        phi4.kink_dt(x - X0, 0, v)
    ))

def scan(H):
    booster = Booster(
        x_lattice= (-L, 0, N), 
        dt= DT, 
        order= 4,
        y0= y0,
        pot_diff= phi4.diff,
        boundaries= ('reflective', 'dirichlet'),
        integrator='sy6',
    )
    booster.rb.param = H
    for v in np.linspace(0, 1, 750):
        path = savedir/f'({H}),({v}).npy'
        if not path.exists():
            logger.debug(f'Runnung the simulation for H={H} e v={v}...')
            _, Y = booster.run(100, v=v, H=H)
            with open(path, 'wb') as file:
                np.save(file, Y[:, 0])

if __name__ == '__main__':
    Hs = (-0.9, -0.5, 0, 0.5, 1, 1.5)
    with Pool(len(Hs)) as pool:
        pool.map(scan, Hs)
import sys
sys.path.insert(1, '../../')

import numpy as np
import pandas as pd
from modules.numeric import *
from pathlib import Path
from multiprocessing import Pool, Value, Queue, Process, Lock
from os import cpu_count
from ctypes import c_int
from scipy import signal

phi4 = Phi4()
X0 = 10
L = 200
N = 2048
DX = L/(N - 1)
DT = 4e-2
DATAPATH = Path('../../data/dirichlet')
LOGGER = get_logger()

def phi_dirichlet(x, H, mu=1):
    nu = np.sign(1 - abs(H))
    return mu*np.tanh(mu*np.arctanh(H**nu) - x)**nu

def init(x, v, H):
    return np.stack((
        phi_dirichlet(x, H) + phi4.kink(x + X0, 0, v) - 1,
        phi4.kink_dt(x + X0, 0, v)
    ))

def diff_4th(y):
    return (-25/12)*y[-1] + 4*y[-2] -3*y[-3] + (4/3)*y[-4] - (1/4)*y[-5]

def get_derivative(u, tmin=0, tmax=np.infty):
    def wrapper(t, Y):
        if tmin <= t <= tmax:
            y, _ = Y
            u.append(diff_4th(y)/DX)
    return wrapper

def save_progress():
    global H_VALUES, V_VALUES, COUNTER, TOTAL, RESULT, COMPLETE

    if len(COMPLETE) > 0:
        mosaic = [
            pd.read_csv(DATAPATH/'mosaic.csv', index_col=0).values,
            pd.read_csv(DATAPATH/'mosaic_freq.csv', index_col=0).values
        ]
    else:
        mosaic = [
            np.full((len(H_VALUES), len(V_VALUES)), np.nan),
            np.full((len(H_VALUES), len(V_VALUES)), np.nan)
        ]

    while COUNTER.value < TOTAL:
        H, U, W = RESULT.get()
        index = np.argwhere(H_VALUES == H)
        mosaic[0][index] = U
        mosaic[1][index] = W
        pd.DataFrame(mosaic[0], columns=V_VALUES, index=H_VALUES).to_csv(DATAPATH/'mosaic.csv')
        pd.DataFrame(mosaic[1], columns=V_VALUES, index=H_VALUES).to_csv(DATAPATH/'mosaic_freq.csv')
        LOGGER.debug('Progresso salvo!')

def task(H):
    global H_VALUES, V_VALUES, COUNTER, TOTAL, RESULT, COMPLETE

    if H in COMPLETE:
        LOGGER.debug(f'Colisões para H={H} encontradas no arquivo de salvamento...')
    else:
        dirichlet = Dirichlet(order=4, param=H)
        reflective = Reflective(order=4)

        collider = Wave(
            x_grid= (-L, 0, N), 
            dt= DT, 
            order= 4,
            y0= init,
            F= phi4.diff,
            boundaries= (reflective, dirichlet),
            integrator= 'rk4',
        )
        U = []
        W = []
        for v in V_VALUES:
            LOGGER.debug(f'Rodando a colisão para H={H} e v={v}...')

            u = []
            collider.run(400, v=v, H=H, stack=False, event=get_derivative(u))
            with COUNTER.get_lock(): COUNTER.value += 1
            
            index = int((X0/v + 100)/DT)
            if index >= len(u): index = -1
            U.append(u[index])

            f, Pxx = signal.periodogram(u, 1/DT)
            W.append(f[np.argmax(Pxx)])

        LOGGER.debug(f'({(100*COUNTER.value/TOTAL):.2f}%) Enviando os resultados de H={H}...')
        RESULT.put((H, U, W))

if __name__ == '__main__':
    global H_VALUES, V_VALUES, COUNTER, TOTAL, RESULT

    Nv = 500
    H_VALUES = np.linspace(-1, 2, Nv+1)[1:]
    V_VALUES = np.linspace(0, 1, Nv+2)[1:-1]
    RESULT = Queue()

    COMPLETE = [H_VALUES[index] for index, u in enumerate(pd.read_csv(DATAPATH/'mosaic.csv', index_col=0).values) if np.nansum(u) != 0]
    # COMPLETE = tuple()

    COUNTER = Value(c_int, 0)
    TOTAL = Nv**2

    saving = Process(target=save_progress)
    saving.start()

    LOGGER.debug(f'Iniciando simulações...')
    with Pool(int(cpu_count()*0.6)) as pool:
        pool.map(task, H_VALUES)

    saving.join()
    
    LOGGER.debug(f'Mosaico finalizado')
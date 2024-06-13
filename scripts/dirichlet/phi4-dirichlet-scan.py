import sys
sys.path.insert(1, '../../')

import numpy as np
import time
from multiprocessing import Pool, Value, current_process
from os import cpu_count

from modules.numeric import *
from findiff import FinDiff

# ===== Definições gerais
SAVEPATH = 'phi4-dirichlet-scan.npy'

PHI4 = Phi4()
X0 = 10
L = 200
N = 2048
DX = L/(N - 1)
DT = 4e-2
DIFF = FinDiff((0, DX, 1), acc=4)

def chi(H, mu, nu):
    return mu*np.arctanh(H**nu)

def varphi(x, mu, nu):
    return mu*np.tanh(-x)**nu

def phi_dirichlet(x, H, mu=1):
    assert H > -1
    nu = np.sign(1 - abs(H))
    return mu*np.tanh(chi(H, mu, nu) - x)**nu

def init(x, v, H):
    return np.stack((
        phi_dirichlet(x, H) + PHI4.kink(x + X0, 0, v) - 1,
        PHI4.kink_dt(x + X0, 0, v)
    ))

# V = np.linspace(0, 1, 503)[1:-1] # velicidades iniciais
# Hs = np.linspace(-1, 2, 501) # parâmetro de borda
# TOTAL = len(V)*len(Hs) # total de pontos

# Objeto `logger` para visualizar o andamento do código em tempo real
# logger = get_logger()

# ===== Função para salvar mosaico
# def save_progress():
#     global mosaic
#     with open(SAVEPATH, 'wb') as file:
#         np.save(file, mosaic.to_numpy())

# # ===== Função realizada paralelamente pelos clusters
# def scan(H):
#     global mosaic, counter

#     j = np.argwhere(H == Hs)

#     dirichlet = Dirichlet(order=4, param=H)
#     reflective = Reflective(order=4)

#     collider = Wave(
#         x_grid= (-L, 0, N), 
#         dt= DT, 
#         order= 4,
#         y0= init,
#         F= PHI4.diff,
#         boundaries= (reflective, dirichlet),
#         integrator= 'rk4',
#     )
    
#     for i, v in enumerate(V):
#         logger.debug(f'Executando simulação para H={H} e v={v}...')
#         with counter.get_lock(): counter.value += 1
#         _, Y = collider.run(X0/v + L, v=v, H=H)
#         mosaic_array = mosaic.to_numpy()
#         mosaic_array[i, j] = DIFF(Y[-1, 0])[-1]
    
#     logger.debug(f'({(100*counter.value/TOTAL):.2f}%) Salvando mosaico...')
#     save_progress()

def main_task(H_QUEUE, V_PARAM, RESULT_QUEUE, PROGRESS):
    dirichlet = Dirichlet(order=4)
    reflective = Reflective(order=4)

    collider = Wave(
        x_grid= (-L, 0, N), 
        dt= DT, 
        order= 4,
        y0= init,
        F= PHI4.diff,
        boundaries= (reflective, dirichlet),
        integrator= 'rk4',
    )

    while not H_QUEUE.empty():
        H = H_QUEUE.get()
        dirichlet.param = H
        result = []
        for v in V_PARAM:
            PROGRESS[current_process().pid] += 1
            _, (y, _) = collider.run(X0/v + L, stack=False, v=v, H=H)
            result.append(DIFF(y)[-1])
        RESULT_QUEUE.put((v, H, result))

def save_task(H_QUEUE, V_PARAM, RESULT_QUEUE, PROGRESS):
    

if __name__ == '__main__':
    global mosaic, counter

    # mosaico compartilhado entre processos
    mosaic = ArrayBuilder(np.float64, (len(V), len(Hs)))
    save_progress()

    # contador compartilhado entre processos
    counter = Value('i', 0)

    logger.debug(f'Iniciando simulações...')
    with Pool(int(cpu_count()*0.8)) as pool:
        pool.map(scan, Hs)

    save_progress()
    
    logger.debug(f'Mosaico finalizado')
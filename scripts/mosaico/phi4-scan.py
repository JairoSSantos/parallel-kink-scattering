# ===== Adicionando bibliotecas e módulos
import numpy as np
import logging
from multiprocessing import Pool, Value
from os import cpu_count
from numeric import * # ferramentas
from ctypes import c_int

# ===== Definições gerais

SAVEPATH = 'phi4-scan.npy'

# Parâmetros fixos das simulações
L = 40
N = 1024
CM = N//2-1 # índice do centro de massa
DX = 2*L/(N - 1)
DT = 4e-2
X0 = 10
V = np.linspace(0.1, 0.5, 302)[1:-1] # velicidades iniciais
LAMB = np.linspace(1, 10, 300) # parâmetro de escala
TOTAL = len(V)*len(LAMB) # total de pontos

# Objeto `logger` para visualizar o andamento do código em tempo real
logger = logging.getLogger()
formatter = logging.Formatter('[%(asctime)s - %(processName)s] %(message)s', datefmt='%d/%m/%Y - %H:%M:%S')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)

# ===== Função para salvar mosaico
def save_progress():
    global mosaic
    with open(SAVEPATH, 'wb') as file:
        np.save(file, mosaic.to_numpy())

# ===== Função realizada paralelamente pelos clusters
def scan(scale):
    global mosaic, counter

    j = np.argwhere(LAMB == scale)
    phi4 = Phi4(scale=scale)

    # ===== Condição inicial
    def y0(x, v):
        return np.stack((
            phi4.kink(x + X0, 0, v) - phi4.kink(x - X0, 0, -v) - 1,
            phi4.kink_dt(x + X0, 0, v) - phi4.kink_dt(x - X0, 0, -v)
        ))

    # ===== Objeto que realizara as simulações
    collider = Collider(
        x_lattice= (-L, L, N),
        dt= DT, 
        order= 4,
        y0= y0,
        pot_diff= phi4.diff
    )

    for i, v in enumerate(V):
        logger.debug(f'Rodando a colisão para scale={scale} e v={v}...')
        with counter.get_lock(): counter.value += 1
        _, Y = collider.run(X0/v + 2*L, v=v)
        mosaic_array = mosaic.to_numpy()
        mosaic_array[i, j] = Y[-1, 0, CM]
    
    logger.debug(f'({(100*counter.value/TOTAL):.2f}%) Salvando mosaico...')
    save_progress()

# ===== Iniciando programa
if __name__ == '__main__':
    global mosaic, counter

    # mosaico compartilhado entre processos
    mosaic = ArrayBuilder(np.float64, (len(V), len(LAMB)))
    save_progress()

    # contador compartilhado entre processos
    counter = Value(c_int, 0)

    logger.debug(f'Iniciando simulações...')
    with Pool(int(cpu_count()*0.8)) as pool:
        pool.map(scan, LAMB)

    save_progress()
    
    logger.debug(f'Mosaico finalizado')
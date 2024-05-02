# ===== Adicionando bibliotecas e módulos
import numpy as np
import logging
from multiprocessing import Pool
from os import cpu_count
from pathlib import Path
from numeric import * # ferramentas

# ===== Definições gerais

# Parâmetros fixos das simulações
L = 50
N = 1024
CM = N//2-1 # índice do centro de massa
DX = 2*L/(N - 1)
DT = 4e-2
X0 = -10
V = np.linspace(0, 1, 302)[1:-1] # velicidades iniciais
LAMB = np.linspace(2, 50, 300) # parâmetro de escala

# Objeto `logger` para visualizar o andamento do código em tempo real
logger = logging.getLogger()
formatter = logging.Formatter('~[%(asctime)s - %(processName)s] %(message)s', datefmt='%d/%m/%Y - %H:%M:%S')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)

# Pasta onde as simulações serão salvas
savedir = Path('../phi4-scan')
savedir.mkdir(exist_ok=True) # verificar se a pasta existe

# ===== Função realizada paralelamente pelos clusters
def scan(scale):
    phi4 = Phi4(scale=scale)

    # ===== Condição inicial
    def y0(x, v):
        return np.stack((
            phi4.kink(x + X0, 0, v) - phi4.kink(x - X0, 0, v) - 1,
            phi4.kink_dt(x + X0, 0, v) - phi4.kink_dt(x - X0, 0, v) # primeira derivada temporal
        ))

    # ===== Objeto que realizara as simulações
    collider = Collider(
        x_lattice= (-L, L, N),
        dt= DT, 
        order= 4,
        y0= y0,
        pot_diff= phi4.diff
    )

    for v in V:
        logger.debug(f'Rodando a colisão para scale={scale} e v={v}...')
        path = savedir/f'scale={scale}, v={v}.npy'
        _, Y = collider.run(100, v=v)
        with open(path, 'rb') as file:
            np.save(file, Y[:, 0, CM])

# ===== Iniciando programa
if __name__ == '__main__':
    with Pool(int(cpu_count()*0.8)) as pool:
        pool.map(scan, LAMB)
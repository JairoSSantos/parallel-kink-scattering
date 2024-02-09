import sys
sys.path.insert(1, '../')

import numpy as np
from multiprocessing import Pool, Value
from pathlib import Path
from scipy.signal import find_peaks
from modules.numeric import argnearest

save_dir = Path('../data/session-experiment-lamb=2')

L = 40
N = (L/40)*1000
dx = 2*L/(N - 1)
x_range = (-L, L + dx, dx)
x = np.arange(*x_range)
cm_index = argnearest(x, 0)

counter = Value('i', 0)
files = tuple(save_dir.glob('*.npy'))
total = len(files)

def get_trail(Y):
    trail = []
    for y in Y[:, 0]:
        plateau = y >= 0
        if np.any(plateau): trail.append(x[plateau].max())
        else: trail.append(np.nan)
    return np.r_[trail]

def task(file: Path):
    Y = np.load(file)
    y_cm = Y[:, 0, cm_index]
    trail = get_trail(Y)
    np.savetxt(file.with_suffix('.csv'), np.stack((y_cm, trail)), delimiter=',')

if __name__ == '__main__':
    with Pool(processes=20) as pool:
        pool.map(task, files)
import numpy as np
import pandas as pd
import time
from random import random
from math import sqrt
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path
data_dir = Path('../data')

def timeit(function):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        y = function(*args, **kwargs)
        tf = time.time()
        return tf - t0, y

    return wrapper

def exception_printer(function):
    def wrapper(*args, **kwargs) -> tuple:
        y = None, None
        try: 
            y = function(*args, **kwargs)
        except Exception as error:
            print(error)
        return y

    return wrapper

def N_in(N):
    # return sum((sqrt(random()**2 + random()**2) for _ in range(N)))
    return np.sum(
        np.sqrt(
            np.sum(
                np.random.random((N, 2))**2,
                axis=1
            )
        ) <= 1
    )

@exception_printer
@timeit
def approx_pi(N):
    return 4*N_in(N)/N

@exception_printer
@timeit
def approx_pi_concurrent(N: int, n_workers: int) -> float:
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        output = executor.map(N_in, [int(N/n_workers)]*n_workers)
    return 4*sum(output)/N

@exception_printer
@timeit
def approx_pi_multiprocess(N: int, n_workers: int) -> float:
    with Pool(n_workers) as pool:
        output = pool.map(N_in, [int(N/n_workers)]*n_workers)
    return 4*sum(output)/N

if __name__ == '__main__':
    columns=('method', 'workers', 'N', 'delay', 'pi')
    df = []
    w = 30
    for N in np.linspace(1e5, 1e8, 10).astype(int):
        for _ in range(3):
            for workers in np.arange(1, w + 1):
                print(f'=> Running N={N}, workers={workers}')
                df.append(('threading', workers, N, *approx_pi_concurrent(N, workers)))
                df.append(('multiprocessing', workers, N, *approx_pi_multiprocess(N, workers)))
                if workers == 1:
                    df.append(('sequential', 1, N, *approx_pi(N)))

    df = pd.DataFrame(df, columns=columns)
    df['delay_per_point'] = df.delay/df.N
    df.to_csv(data_dir/'approx_pi_numpy.csv')
    df.head()

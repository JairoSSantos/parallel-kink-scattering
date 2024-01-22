# nohup python run_session.py ../data/mosaic --v_num=599 --lamb_num=599 --decrease=True --shuffle=True > mosaic.out &

import numpy as np
import pandas as pd
import multiprocessing as mp
import logging
import json
from time import time
from datetime import datetime
from dataclasses import dataclass, asdict, field
from pathlib import Path
from os import cpu_count
from .numeric import *

INFO_FILE = 'session-info.json'
FPARAMS = ('v', 'lamb', 'exec_time', 'delay')

def read_path_info(path: Path) -> dict[str, float]:
    return dict(zip(FPARAMS, map(float, path.stem.split('-'))))

def get_session_info(session_path):
    summary = []
    info = []
    for session in session_path.glob('session-*'):
        summary.append(pd.DataFrame([{'path':path, **read_path_info(path)} for path in session.glob('*.csv')]))
        with open(session/INFO_FILE, 'r') as json_file:
            info.append(pd.Series(json.loads(json_file.read())))
    return summary, info

def triangular(n):
    return n*(n + 1)/2

@dataclass
class CollisionSession:
    session_path: Path

    L:  int=50
    N:  int=int((5/4)*1000)
    dtdx: float=0.7

    sep_by_delta: float=10

    v_min: float=0.05
    v_max: float=0.5
    v_num: int=field(default=cpu_count())

    lamb_min: float=0.075
    lamb_max: float=50
    lamb_num: int=field(default=cpu_count())

    n_processes: int=field(default=cpu_count())

    decrease: bool=False
    shuffle: bool=False

    # n_fixer: int=100

    def __post_init__(self):
        dx = 2*self.L/self.N
        self.dt = self.dtdx*dx
        self.x = np.arange(-self.L, self.L, dx)
        self.collider = KinkCollider(self.x, (None, None), self.dt)
        self.cm_index = argnearest(self.x, 0)

        name = datetime.now().strftime('session-%Y-%m-%d-%H-%M-%S')
        self.save_dir = self.session_path/name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._create_logger()
        self.logger.debug(f'Creating {name}...')
        info = asdict(self)
        info['session_path'] = str(self.session_path)
        info['name'] = name
        with open(self.save_dir/INFO_FILE, 'w') as json_file:
            json.dump(info, json_file)
    
    def _create_logger(self):
        self.logger = logging.getLogger()
        formatter = logging.Formatter('~[%(asctime)s - %(processName)s] %(message)s', datefmt='%d/%m/%Y - %H:%M:%S')
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.setLevel(logging.DEBUG)

    def collider_task(self):
        while not self.collisions_queue.empty():
            v, lamb = self.collisions_queue.get()
            # vs, lamb = point['vs'], point['lamb']
            # self.logger.debug(f'Iniciando {len(vs)} colisões para lamb={lamb}')

            delta = Kink.delta(lamb)
            x0 = self.sep_by_delta*delta/2

            t0 = time()
            self.collider.x0s = (-x0, x0)
            _, Y = self.collider.collide(
                vs= (v, -v),
                lamb= lamb,
                t_final= x0/v + self.L,
                # callbacks=[KinkCollider.fixed_boundary(self.n_fixer)],
                stop_conditions=[KinkCollider.overflowed]
            )
            tf = time()
            exec_time, delay = tf - self.exec_init, tf - t0
            
            trail = []
            for y in Y[:, 0]:
                plateau = y >= 0
                if np.any(plateau): trail.append(self.x[plateau].max())
                else: trail.append(None)

            pd.DataFrame({
                'y_cm': Y[:, 0, self.cm_index],
                'trail': trail
            }).to_csv(
                self.save_dir/('-'.join(map(str, (v, lamb, exec_time, delay, self.n_processes.value))) + '.csv'),
                index= False,
                header= False
            )

            with self.total_counter.get_lock():
                self.total_counter.value += 1
                self.logger.debug(f'Finishing simulation ({(self.total_counter.value/self.n_collisions*100):.2f}%): lamb={lamb}; v={v}; delay={delay}; exec_time={exec_time}')
            
            if self.decrease:
                with self.batch_counter.get_lock():
                    self.batch_counter.value += 1
                    if self.batch_counter.value >= self.batch_size:
                        self.batch_counter.value = 0
                        self.n_processes.value -= 1
                        self.logger.debug(f'Joining...')
                        break

    def run(self):
        self.collisions_queue = mp.Queue()
        self.n_collisions = 0

        self.logger.debug(f'Checking saved sessions...')
        vs = np.linspace(self.v_min, self.v_max, self.v_num)
        lambs = np.linspace(self.lamb_min, self.lamb_max, self.lamb_num)
        mesh = np.stack(np.meshgrid(vs, lambs), axis=-1).reshape((-1, 2))
        summaries, _ = get_session_info(self.session_path)
        for i in (np.random.permutation if self.shuffle else np.arange)(len(mesh)):
            v, lamb = mesh[i]
            calculated = False
            for summary in summaries:
                if len(summary) > 0 and np.any(summary[summary.lamb == lamb].v == v):
                    calculated = True
                    break
            if calculated: 
                continue
            else:
                self.n_collisions += 1
                self.collisions_queue.put((v, lamb))
        self.logger.debug(f'Setting a queue with {self.n_collisions} simulations...')

        self.n_processes = mp.Value('i', self.n_processes)
        self.batch_size = int(self.n_collisions/triangular(self.n_processes.value)) if self.decrease else self.n_collisions
        if self.decrease != 0:
            self.logger.debug(f'Setting the processes amount decrease with {self.batch_size} simulations batch size...')
        self.batch_counter = mp.Value('i', 0)
        self.total_counter = mp.Value('i', 0)

        self.exec_init = time()
        self.logger.debug(f'Starting session with {self.n_collisions} simulations and {self.n_processes.value}{" initial" if self.decrease else ""} processes...')
        processes = []
        for _ in range(self.n_processes.value):
            process = mp.Process(target=self.collider_task)
            process.start()
            processes.append(process)
        
        for process in processes:
            process.join()
        self.logger.debug(f'Finishing session...')
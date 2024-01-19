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

@dataclass
class CollisionSession:
    session_path: str

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

    n_fixer: int=100

    def  __post_init__(self):
        dx = 2*self.L/self.N
        self.dt = self.dtdx*dx
        self.x = np.arange(-self.L, self.L, dx)
        self.collider = KinkCollider(self.x, (None, None), self.dt)
        self.cm_index = argnearest(self.x, 0)

        name = datetime.now().strftime('session-%Y-%m-%d-%H-%M-%S')
        self.save_dir = Path(self.session_path)/name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        info = asdict(self)
        info['name'] = name
        with open(self.save_dir/'session-info.json', 'w') as json_file:
            json.dump(info, json_file)

        self.collisions_queue = mp.Queue()
        self.n_collisions = 0

        vs = np.linspace(self.v_min, self.v_max, self.v_num)
        lambs = np.linspace(self.lamb_min, self.lamb_max, self.lamb_num)
        summary = self.get_summary()
        for lamb in lambs:
            to_calc_vs = vs[~np.isin(vs, summary[summary.lamb == lamb].v.values)]

            len_to_calc_vs = len(to_calc_vs)
            if len_to_calc_vs > 0:
                self.n_collisions += len_to_calc_vs
                self.collisions_queue.put({
                    'vs': to_calc_vs,
                    'lamb': lamb
                })
            
        self.logger = logging.getLogger()
        formatter = logging.Formatter('~[%(asctime)s - %(processName)s] %(message)s', datefmt='%d/%m/%Y - %H:%M:%S')
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.setLevel(logging.DEBUG)

    def get_summary(self):
        return pd.DataFrame([
            tuple(map(float, filename.stem.split('-'))) 
            for filename in self.save_dir.glob('*.csv')
        ], columns=('v', 'lamb', 'exec_time', 'delay'))

    def collider_task(self):
        while not self.collisions_queue.empty():
            point = self.collisions_queue.get()
            vs, lamb = point['vs'], point['lamb']
            self.logger.debug(f'Iniciando {len(vs)} colisões para lamb={lamb}')
    
            delta = Kink.delta(lamb)
            x0 = self.sep_by_delta*delta/2

            local_t0 = time()
            for v in vs:
                _t0 = time()
                self.collider.x0s = (-x0, x0)
                _, Y = self.collider.collide(
                    vs= (v, -v),
                    lamb= lamb,
                    t_final= x0/v + self.L,
                    callbacks=[KinkCollider.fixed_boundary(self.n_fixer)],
                    stop_conditions=[KinkCollider.overflowed]
                )
                _tf = time()
                exec_time, delay = _tf - local_t0, _tf - _t0
                
                trail = []
                for y in Y[:, 0]:
                    plateau = y >= 0
                    if np.any(plateau): trail.append(self.x[plateau].max())
                    else: trail.append(None)

                pd.DataFrame({
                    'y_cm': Y[:, 0, self.cm_index],
                    'trail': trail
                }).to_csv(
                    self.save_dir/('%s-%s-%s-%s.csv'%(v, lamb, exec_time, delay)),
                    index= False,
                    header= False
                )
                self.logger.debug(f'Simulação finalizada: lamb={lamb}; v={v}; delay={delay}; exec_time={exec_time}')

    def run(self):
        self.logger.debug(f'Iniciando {self.n_processes} processos para simular {self.n_collisions} colisões')
        processes = []
        for _ in range(self.n_processes):
            process = mp.Process(target=self.collider_task)
            process.start()
            processes.append(process)
        
        for process in processes:
            process.join()
        self.logger.debug(f'Encerrando sessão')
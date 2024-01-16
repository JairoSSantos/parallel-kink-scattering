import numpy as np
import pandas as pd
import multiprocessing as mp
from time import time
from dataclasses import dataclass
from .numeric import *
from .default import SessionConfig

@dataclass
class CollisionSession(SessionConfig):
    def  __post_init__(self):
        self.save_dir.mkdir(exist_ok=True)
        self.collisions_queue = mp.Queue()
        self.n_collisions = 0

        summary = self.get_summary()
        for lamb in self.lambs:
            to_calc_vs = self.vs[~np.isin(self.vs, summary[summary.lamb == lamb].v.values)]

            len_to_calc_vs = len(to_calc_vs)
            if len_to_calc_vs > 0:
                n_collisions += len_to_calc_vs
                self.collisions_queue.put({
                    'vs': to_calc_vs,
                    'lamb': lamb
                })

    def get_summary(self):
        return pd.DataFrame([
            tuple(map(float, filename.stem.split('-'))) 
            for filename in self.save_dir.glob('*')
        ], columns=('v', 'lamb', 'exec_time', 'delay'))

    def collider_task(self):
        while not self.collisions_queue.empty():
            point = self.collisions_queue.get()
            logger.debug('Iniciando %s colisões para lamb=%s'%(len(point['vs']), point['lamb']))

            delta = np.sqrt(2/point['lamb'])
            x0 = self.separation_by_delta*delta/2

            local_t0 = time()

            for v in point['vs']:
                _t0 = time()
                y = self.collider.collide(
                    x0s= (-x0, x0),
                    vs= (v, -v),
                    lamb= point['lamb'],
                    t_final= 2*x0/v + self.L
                )
                _tf = time()
                exec_time, delay = _tf - local_t0, _tf - _t0
                
                kink = []
                for row in y:
                    try: kink.append(self.x_lattice.x[row >= 0].max())
                    except ValueError: kink.append(None)
                pd.DataFrame({
                    'y_cm': y[:, self.cm_index[0]],
                    'kink': kink
                }).to_csv(
                    self.save_dir/('%s-%s-%s-%s.csv'%(v, point['lamb'],  exec_time, delay)),
                    index= False,
                    header= False
                )
                logger.debug('Simulação finalizada: lamb=%s; v=%s; delay=%s; exec_time=%s'%(
                    point['lamb'],
                    v,
                    delay,
                    exec_time
                ))

    def run(self):
        logger.debug(f'Iniciando {self.n_processes} processos para simular {self.n_collisions} colisões')
        processes = []
        for _ in range(self.n_processes):
            process = mp.Process(target=self.collider_task)
            process.start()
            processes.append(process)
        
        for process in processes:
            process.join()
        logger.debug(f'Encerrando sessão')
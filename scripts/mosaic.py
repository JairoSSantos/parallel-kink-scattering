import numpy as np
import pandas as pd
import multiprocessing as mp
from time import time
import logging

logger = logging.getLogger()

formatter = logging.Formatter('~[%(asctime)s - %(processName)s] %(message)s', datefmt='%d/%m/%Y - %H:%M:%S')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)

from dataclasses import dataclass
from ..modules.numeric import *
from ..modules.default import SessionConfig

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

    def collider_task(self, collisions_queue):
        while not collisions_queue.empty():
            point = collisions_queue.get()
            logger.debug('Iniciando %s colisões para lamb=%s'%(len(point['vs']), point['lamb']))
            
            local_t0 = time()

            delta = np.sqrt(2/point['lamb'])
            x0 = self.separation_by_delta*delta/2

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

    def init(self):
        n_collisions, collisions_queue = self.get_collision_queue()
        logger.debug(f'Iniciando {self.n_processes} processos para simular {n_collisions} colisões')
        processes = []
        for _ in range(self.n_processes):
            process = mp.Process(target=collider_task, args=(Config.collider, collisions_queue))
            process.start()
            processes.append(process)
        
        for process in processes:
            process.join()
        logger.debug(f'Encerrando sessão')

def manager_task(n_collisions, output_queue):
    received = 0
    while received < n_collisions:
        output = output_queue.get()
        received += 1
        logger.debug('Salvando os resultados para lambda={} e v={} ({}%)'.format(
            output['lamb'],
            output['v'],
            received/n_collisions * 100
        ))

        pd.DataFrame(output['y']).to_csv(
            Config.save_dir/('%s-%s-%s-%s.csv'%(output['v'], output['lamb'], output['exec_time'], output['delay'])),
            index= False,
            header= False
        )

if __name__ == '__main__':
    import sys

    cpu_count = mp.cpu_count()
    params = {
        '-p': cpu_count,
        '--k-v': cpu_count,
        '--k-lamb': cpu_count,
    }
    for arg in sys.argv:
        try:
            name, value = arg.split('=')
        except ValueError: pass
        else:
            match name:
                case '-k': params['--k-v'] = params['--k-lamb'] = int(value)
                case '--dataset-name': 
                    Config.save_dir = Path('../data')/value
                    Config.save_dir.mkdir(exist_ok= True)
                case other: 
                    try: params[other] = value
                    except KeyError: pass
    
    init_session(
        int(params['-p']), 
        *init_collisions_queue(
            np.linspace(*Config.v_lims, int(params['--k-v'])), 
            np.linspace(*Config.lamb_lims, int(params['--k-lamb']))
        )
    )
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
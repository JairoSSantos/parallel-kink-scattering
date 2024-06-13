import numpy as np
import logging
from multiprocessing import RawArray

def get_logger():
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s - %(processName)s] %(message)s', datefmt='%d/%m/%Y - %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)
    return logger

class ArrayBuilder:
    '''
    Sharing numpy array between parallel processes.
    '''
    def __init__(self, dtype: np.dtype, shape: tuple):
        self.dtype = dtype
        self.shape = shape
        self._shared_array = RawArray(np.ctypeslib.as_ctypes_type(dtype), int(np.prod(shape)))
    
    def to_numpy(self):
        return np.frombuffer(self._shared_array, dtype=self.dtype).reshape(self.shape)
    
import numpy as np
from multiprocessing import RawArray

class ArrayBuilder:
    '''
    Objeto responsável por compartilhar um array numpy entre processos.
    '''
    def __init__(self, dtype: np.dtype, shape: tuple):
        self.dtype = dtype
        self.shape = shape
        self._shared_array = RawArray(np.ctypeslib.as_ctypes_type(dtype), int(np.prod(shape)))
    
    def to_numpy(self):
        return np.frombuffer(self._shared_array, dtype=self.dtype).reshape(self.shape)
    
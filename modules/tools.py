import numpy as np
from numpy import ndarray

def memory_economize(t: float, Y: ndarray):
    if len(Y) > 2:
        del Y[0]
    return Y

def catch_cm(cm_index: int, Ycm: list):
    def wrapped(t, Y):
        y, _ = Y[-1]
        Ycm.append(y[cm_index])
        return Y
    return wrapped

def composite(*fs):
    def wrapped(t: float, Y: ndarray):
        for f in fs:
            Y = f(t, Y)
        return Y
    return wrapped

class Tracker:
    def __init__(self, x: ndarray, x0: float):
        self.x = x
        self.x0 = x0
        self.trail = []
        self.is_bion = True

    def __call__(self, t: float, Y: ndarray):
        y, _ = Y[-1]
        try:
            k = self.x[y >= 0].max()
        except:
            k = np.nan
        self.trail.append(k)
        return Y
    
class BouncesCounter:
    def __init__(self, cm_index: int):
        self.cm_index = cm_index
        self.bounces = 0
        self._last_cm = None
        self.locs = []
        self._iter = 0

    def __call__(self, t, Y):
        y_cm = Y[-1][0, self.cm_index]
        if self._last_cm != None and y_cm <= 0 and self._last_cm > 0:
            self.bounces += 1
            self.locs.append(self._iter)
        self._last_cm = y_cm
        self._iter += 1
        return Y
    
    def blocker(self, nmax):
        def condition(t, Y):
            return self.bounces >= nmax
        return condition
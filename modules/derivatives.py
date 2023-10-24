import numpy as np
from typing import Callable

def second_2(y:np.ndarray, dt: float, axis: int=0):
    '''
    Segunda derivada de segunda ordem.

    Parameters
    ----------
    y : np.ndrray
        Função a ser derivada.
    dt: float
        Espaçamento entre os pontos do domínio.
    axis: int
        Eixo no qual será calculada a derivada.
    
    Returns
    -------
    Segunda derivada : np.ndrray, shape[axis] = y.shape[axis] - 2.
    '''
    indices = np.r_[:y.shape[axis]]
    a, b, c = (
        np.take_along_axis(y, indices[:-2], axis=axis),
        np.take_along_axis(y, indices[1:-1], axis=axis),
        np.take_along_axis(y, indices[2:], axis=axis),
    )
    return ( a - 2*b + c )/dt**2

def second_4(y:np.ndarray, dt: float, axis: int=0):
    '''
    Segunda derivada de quarta ordem.

    Parameters
    ----------
    y : Array-like
        Função a ser derivada.
    dt: float
        Espaçamento entre os pontos do domínio.
    axis: int
        Eixo no qual será calculada a derivada.
    
    Returns
    -------
    Segunda derivada : np.ndrray, shape[axis] = y.shape[axis] - 2.
    '''
    indices = np.r_[:y.shape[axis]]
    a, b, c, d, e = (
        np.take_along_axis(y, indices[:-4], axis=axis),
        np.take_along_axis(y, indices[1:-3], axis=axis),
        np.take_along_axis(y, indices[2:-2], axis=axis),
        np.take_along_axis(y, indices[3:-1], axis=axis),
        np.take_along_axis(y, indices[4:], axis=axis)
    )
    return np.r_[
        second_2(y=np.take_along_axis(y, indices[:3], axis=axis), dt=dt, axis=axis), # borda inferior
        ( -30*c + 16*(b + d) - (a + e) )/(12*dt**2),
        second_2(y=np.take_along_axis(y, indices[-3:], axis=axis), dt=dt, axis=axis) # bordas superior
    ]
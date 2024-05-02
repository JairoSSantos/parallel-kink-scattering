from .misc import *
from .integrators import *
from .boundaries import *
from .models import *

_NUMERIC = float|np.ndarray[float]
_BOUNDARIES = {
    'dirichlet': Dirichlet,
    'neumann': Neumann,
    'reflective': Reflective
}
_INTEGRATORS = {
    'rk4': RungeKutta4th,
    'sy4': Symplectic4th,
    'sy6': Symplectic6th
}

class Lattice:
    '''
    A generalized lattice object.

    Parameters
    ----------
    **axes: The axes names with that's range array, `x = np.arange(initial_value, final_value, spacing)`.

    Attributes
    ----------
    axes: dict
        A dict containing the axes ranges
    shape: tuple[int]
        The lattice shape.
    grid: ndarray
        A n-dimensional array with the lattice nodes locations.
    '''
    def __init__(self, **axes):
        self.axes = axes
    
    def __getattr__(self, kw):
        return self.axes[kw]
    
    def __getitem__(self, axis):
        return self.axes[axis]
    
    @property
    def shape(self) -> tuple:
        return tuple(map(len, self.axes.values()))
    
    @property
    def grid(self) -> np.ndarray:
        return np.stack(np.meshgrid(*tuple(self.axes.values())), axis=-1)
    
    def at(self, **locs) -> list:
        '''
        Indices that provides the node(s) location, given position values.

        Parameters
        ----------
        **locs: Position values.

        Returns
        -------
        list
            A list with the nodes locations as index format, 
            each item in the list corresponding to a lattice axe.
        '''
        kws = tuple(locs.keys())
        return [
            argnearest(axis, locs[kw]) if kw in kws else Ellipsis 
            for kw, axis in self.axes.items()
        ]
    
    def window(self, **lims) -> list:
        '''
        Indices that provides a range on the lattice, given values for the axes limits.

        Parameters
        ----------
        **locs: limits values, like `(min, max)`.

        Returns
        -------
        list
            A list with the lattice ranges as index format,
            each item in the list corresponding to a lattice axe.
        '''
        kws = tuple(lims.keys())
        return [
            slice(argnearest(axis, lims[kw][0]), argnearest(axis, lims[kw][1]))
            if kw in kws else Ellipsis 
            for kw, axis in self.axes.items()
        ]

class Collider:
    def __init__(self, 
                 x_lattice: tuple[float, float, int], 
                 dt: float, 
                 order: int,
                 y0: Callable[[_NUMERIC], _NUMERIC],
                 pot_diff: Callable[[_NUMERIC], _NUMERIC],
                 boundaries: tuple[Boundary|str]=None,
                 integrator: Integrator|str='rk4',
                 event: Callable[[float, _NUMERIC], None]=(lambda t, Y: None)):
        
        assert order%2 == 0

        self.x = np.linspace(*x_lattice)
        # self.dt = dt
        self.y0 = y0
        self.pot_diff = pot_diff

        nodes = order//2
        h = (x_lattice[1] - x_lattice[0])/(x_lattice[2] - 1)
        self.DDx = diff_coeffs(m=2, stencil=range(-nodes, nodes+1), h=h)
        self.dx = (x_lattice[1] - x_lattice[0])/(x_lattice[2] - 1)

        if boundaries == None:
            self._boundaries = [Reflective(m=2, order=order, h=h)]*2
        else:
            self._boundaries = []
            for boundary in boundaries:
                match type(boundary).__name__:
                    case 'str': self._boundaries.append(_BOUNDARIES[boundary](m=2, order=order, h=h))
                    case 'Boundary': self._boundaries.append(boundary)
                    case other: raise f'Type "{other}" is not recognized as a boundary.'
        
        integrator_params = {
            'fun': self.fun,
            'dt': dt,
            'event': event
        }
        match type(integrator).__name__:
            case 'str':
                self.integrator = _INTEGRATORS[integrator](**integrator_params)
            case 'Integrator':
                self.integrator = integrator
            case other: raise f'Type "{other}" is not recognized as a integrator.'
    
    @property
    def lb(self):
        return self._boundaries[0]
    
    @property
    def rb(self):
        return self._boundaries[1]
    
    def fun(self, t, Y):
        y, dydt = Y
        y_d2x = np.r_[
            self.lb(y),
            np.convolve(y, self.DDx, mode='valid'),
            self.rb(y[::-1])[::-1]
        ]
        return np.stack((
            dydt,
            y_d2x - self.pot_diff(y)
        ))

    def run(self, t_final: float, t0: float=0, **y0_params):
        t, Y = self.integrator.run(self.y0(self.x, **y0_params), t_final=t_final, t0=t0)
        return Lattice(x=self.x, t=t), Y
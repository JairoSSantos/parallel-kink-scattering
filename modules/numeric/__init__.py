from .misc import *
from .integrators import *
from .boundaries import *
from .models import *
from .parallel import *

_NUMERIC = float|np.ndarray[float]
_INTEGRATORS = {
    'rk4': RungeKutta4th,
    'sy4': Symplectic4th,
    'sy6': Symplectic6th
}
_FIXED = (Dirichlet, Reflective) # fixed value boundaries

class Grid:
    def __init__(self, **axes):
        self.axes = axes
    
    def __getattr__(self, kw):
        return self.axes[kw]
    
    def __getitem__(self, axis):
        return self.axes[axis]
    
    @property
    def shape(self) -> tuple[int]:
        return tuple(map(len, self.axes.values()))
    
    @property
    def grid(self) -> np.ndarray[float]:
        return np.stack(np.meshgrid(*tuple(self.axes.values())), axis=-1)
    
    def at(self, **locs) -> tuple[int]:
        kws = tuple(locs.keys())
        return tuple([
            argnearest(axis, locs[kw]) if kw in kws else Ellipsis 
            for kw, axis in self.axes.items()
        ])
    
    def window(self, **lims) -> tuple[slice]:
        kws = tuple(lims.keys())
        return tuple([
            slice(argnearest(axis, lims[kw][0]), argnearest(axis, lims[kw][1]))
            if kw in kws else Ellipsis 
            for kw, axis in self.axes.items()
        ])
    
    def extent(self, *axis) -> np.ndarray[float]:
        if not len(axis): 
            axis = self.axes.keys()
        return np.concatenate([(self.axes[ax].min(), self.axes[ax].max()) for ax in axis])
    
class Wave:
    def __init__(self, 
                 x_grid: tuple[float, float, int], 
                 dt: float, 
                 order: int,
                 y0: Callable[[_NUMERIC], _NUMERIC],
                 F: Callable[[_NUMERIC], _NUMERIC],
                 boundaries: tuple[Boundary|str]=None,
                 integrator: Integrator|str='rk4'):
        
        assert order%2 == 0

        self.x = np.linspace(*x_grid)
        self.y0 = y0
        self.F = F

        self.nodes = order//2
        dx = (x_grid[1] - x_grid[0])/(x_grid[2] - 1)
        self.central = diff_coeffs(m=2, stencil=range(-self.nodes, self.nodes + 1), h=dx)

        self.update_boundaries(*boundaries)
        
        integrator_params = {
            'dt': dt,
            'func': self.func
        }
        error = Exception(f'Integrator not recognized. You can try one of these: ' + ', '.join(_INTEGRATORS.keys()))
        if type(integrator).__name__ == 'str':
            try:
                self.integrator = _INTEGRATORS[integrator](**integrator_params)
            except KeyError:
                raise error
        elif 'Integrator' in (type(integrator).__name__ *type(integrator).__bases__):
            self.integrator = integrator
        else:
            raise error
    
    def update_boundaries(self, left=None, right=None) -> None:
        self.ghost = Ghost(self.nodes, left, right)
        self.glue = [
            int(not type(left) in _FIXED), 
            int(not type(right) in _FIXED)
        ]
    
    def func(self, t, y, y_dt) -> np.ndarray[float]:
        y_dx2 = np.convolve(self.ghost(y), self.central, mode='valid')

        y_dt2 = y_dx2 - self.F(y)
        y_dt2[0] *= self.glue[0]
        y_dt2[-1] *= self.glue[1]
        return np.stack((
            y_dt,
            y_dt2
        ))

    def run(self, t_final: float, t0: float=0, stack: bool=True, **y0_params) -> tuple[Grid, np.ndarray[float]]:
        Y0 = self.y0(self.x, **y0_params)
        t, Y = self.integrator.run(t_final, Y0, t0=t0, stack=stack)
        return Grid(t=t, x=self.x), Y
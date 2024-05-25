from .misc import *
from .integrators import *
from .boundaries import *
from .models import *
from .parallel import *
from findiff import FinDiff

_NUMERIC = float|np.ndarray[float]
_INTEGRATORS = {
    'rk4': RungeKutta4th,
    'sy4': Symplectic4th,
    'sy6': Symplectic6th
}

class Grid:
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
        kws = tuple(locs.keys())
        return [
            argnearest(axis, locs[kw]) if kw in kws else Ellipsis 
            for kw, axis in self.axes.items()
        ]
    
    def window(self, **lims) -> list:
        kws = tuple(lims.keys())
        return [
            slice(argnearest(axis, lims[kw][0]), argnearest(axis, lims[kw][1]))
            if kw in kws else Ellipsis 
            for kw, axis in self.axes.items()
        ]
    
    def extent(self, *axis):
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
                 integrator: Integrator|str='rk4',
                 event: Callable[[float, _NUMERIC], None]=(lambda t, Y: None)):
        
        assert order%2 == 0

        self.x = np.linspace(*x_grid)
        self.y0 = y0
        self.F = F

        nodes = order//2
        dx = (x_grid[1] - x_grid[0])/(x_grid[2] - 1)
        self.central = diff_coeffs(m=2, stencil=range(-nodes, nodes + 1), h=dx)

        # if boundaries == None:
        #     self.lb = self.rb = Reflective(order=order)
        # else:
        #     self.lb, self.rb = boundaries
        # self.lb_diff = self.lb.get_diff(2, h=dx)
        # self.rb_diff = self.rb.get_diff(2, h=dx)

        # self.glue = (
        #     int(not type(self.lb) in fixed_boundaries),
        #     int(not type(self.rb) in fixed_boundaries)
        # )

        self.ghost = Ghost(nodes, *boundaries)
        fixed_boundaries = (Dirichlet, Reflective)
        self.glue = [int(not type(b) in fixed_boundaries) for b in boundaries]
        
        integrator_params = {
            'func': self.func,
            'dt': dt,
            'event': event
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
    
    def func(self, t, y, y_dt):
        y_dx2 = np.convolve(self.ghost(y), self.central, mode='valid')

        y_dt2 = y_dx2 - self.F(y)
        y_dt2[0] *= self.glue[0]
        y_dt2[-1] *= self.glue[1]
        return np.stack((
            y_dt,
            y_dt2
        ))

    def run(self, t_final: float, t0: float=0, **y0_params):
        Y0 = self.y0(self.x, **y0_params)
        t, Y = self.integrator.run(t_final, Y0, t0=t0)
        return Grid(x=self.x, t=t), Y
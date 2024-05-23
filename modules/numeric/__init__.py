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

        self.nodes = order//2
        dx = (x_grid[1] - x_grid[0])/(x_grid[2] - 1)
        self.central = diff_coeffs(m=2, stencil=range(-self.nodes, self.nodes+1), h=dx)

        if boundaries == None:
            self.lb = self.rb = Reflective(order=order)
        else:
            self.lb, self.rb = boundaries
        self.lb_diff = self.lb.get_diff(2, h=dx)
        self.rb_diff = self.rb.get_diff(2, h=dx)
        self.dirichlet_filter = (
            int(not type(self.lb) is Dirichlet),
            int(not type(self.rb) is Dirichlet)
        )
        self.Gl = self.lb.ghost(self.nodes)
        self.Gr = self.rb.ghost(self.nodes)
        
        integrator_params = {
            'func': self.func,
            'dt': dt,
            'event': event
        }
        match type(integrator).__name__:
            case 'str':
                self.integrator = _INTEGRATORS[integrator](**integrator_params)
            case 'Integrator':
                self.integrator = integrator
            case other: raise f'Type "{other}" is not recognized as a integrator.'
    
    def func(self, t, Y):
        y, Dt_y = Y

        yb = np.r_[(self.Gl @ self.lb.take(y))[::-1], y, (self.Gr @ self.rb.take(y[::-1]))]
        D2x_y = np.convolve(yb, self.central, mode='valid')
        
        # D2x_y = np.r_[
        #     self.lb_diff @ np.r_[self.lb.param, y[self.lb.slice]],
        #     np.convolve(y, self.central, mode='valid'),
        #     (self.rb_diff @ np.r_[self.rb.param, y[::-1][self.rb.slice]])[::-1]
        # ]

        D2t_y = D2x_y - self.F(y)
        D2t_y[0] *= self.dirichlet_filter[0]
        D2t_y[-1] *= self.dirichlet_filter[1]
        return np.stack((
            Dt_y,
            D2t_y
        ))

    def run(self, t_final: float, t0: float=0, **y0_params):
        t, Y = self.integrator.run(self.y0(self.x, **y0_params), t_final=t_final, t0=t0)
        return Grid(x=self.x, t=t), Y
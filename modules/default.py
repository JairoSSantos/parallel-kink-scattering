from dataclasses import dataclass, field
from pathlib import Path
from .numeric import *

@dataclass
class Config:
    L:  int = 50
    N:  int = int((5/4)*1000)
    dx: float = 2*L/N
    dt: float = 0.7*dx

    x_lattice: Lattice = Lattice(x=(-L, L, dx))
    cm_index:  list = field(default_factory=x_lattice.at(x=0))

    separation_by_delta: float = 10

@dataclass
class SessionConfig(Config):
    collider: KinkCollider = KinkCollider(x_lattice = Config.x_lattice, dt = Config.dt)

    v_min: float = 0.05
    v_max: float = 0.5

    lamb_min: float = 0.075
    lamb_max: float = 50

    dataset_name: str = 'dataset'
    save_dir: Path = Path('../data')/dataset_name

    @property
    def v_lims(self):
        return self.v_min, self.v_max
    
    @property
    def lamb_lims(self):
        return self.lamb_min, self.lamb_max
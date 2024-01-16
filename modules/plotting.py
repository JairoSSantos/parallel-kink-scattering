import numpy as np
from matplotlib.colors import *

def linear_cmap(pcolors: tuple[str, float]):
    n_colors = len(pcolors)
    def wrapped(n):
        colors = []
        for i in range(n_colors):
            cf, pf = pcolors[i]
            if i == 0:
                colors += [to_rgb(cf)]*int(n*pf)
            else:
                c0, p0 = pcolors[i-1]
                colors += np.stack([
                    np.linspace(v0, vf, int((pf-p0)*n)) 
                    for v0, vf in zip(to_rgb(c0), to_rgb(cf))
                ], axis=-1).tolist()
        colors += [to_rgb(cf)]*int(n*(1-pf))
        return colors
    return wrapped

def get_cmap():
    return ListedColormap(linear_cmap([
        ('white', 0),
        ('khaki', 0.1),
        ('darkorange', 0.3),
        ('firebrick', 0.4),
        ('indigo', 0.6),
        ('skyblue', 0.7),
        ('slateblue', 0.8),
        ('slategray', 1),
    ])(500)).reversed()
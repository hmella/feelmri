import numpy as np
from scipy.signal.windows import tukey


def Riesz(size, width=0.6, lift=0.7):
    decay = (1.0-width)/2
    s = size
    s20 = np.round(decay*s).astype(int)
    s1 = np.linspace(0, s20/2, s20)
    w1 = 1.0 - np.power(np.abs(s1/(s20/2)),2)*(1.0-lift)

    # Filters
    H0 = np.ones([s,])
    H0[0:s20] *= np.flip(w1)
    H0[s-s20:s] *= w1

    return H0

def Tukey(size, width=0.6, lift=0.7):
    alpha = 1.0-width
    H0 = tukey(size, alpha=alpha)*(1.0-lift) + lift
    return H0

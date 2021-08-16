
# import tensorflow as tf
# tf.random.set_seed(0)

import numpy as np
from timeit import default_timer as Timer

from modules.PhaseScreenGenerator import PhaseScreenGenerator
from modules.func_utils import *
from modules.propagation import *

'''
    Propagate beam through turbulence - focus through lens to output plane
'''

N = 2048
r0 = 0.3
L0 = 1e3
l0 = 0.01
D = 1
delta = D/N

def test1():
    nscr = 5
    Dz = 1e3
    wvl = 1e-6

    start = Timer()

    z = np.array(range(nscr)) * Dz / (nscr-1)
    PSG = PhaseScreenGenerator(r0, N, delta, L0, l0)

    x1, y1 = np.meshgrid(
        np.array(list(range(-N//2,N//2))) * delta, 
        np.array(list(range(-N//2,N//2))) * delta
    )

    sg = np.exp(-(x1/(0.47*N*delta))**16) * np.exp(-(y1/(0.47*N*delta))**16)
    sg = np.tile(sg, [nscr, 1, 1])

    # TODO: Gaussian Beam
    Uin = circ(x1, y1, D)

    print(Timer()-start)
    start = Timer()

    phz = np.empty([nscr, N, N])
    for i in range(nscr):
        phz[i] = PSG.next()
    
    print(Timer()-start)
    start = Timer()
    
    Uout, xn, yn = ang_spec_multi_prop_no_scale(Uin, wvl, delta, z, sg*np.exp(1j*phz))
    # Uout = Uout * np.exp(-1j*np.pi/(wvl*Dz) * (xn**2+yn**2))
    print(Timer()-start)

    fl = 5e3
    k = 2*PI/wvl
    lens = np.exp(1j * -k/(2*fl) * (x1**2 + y1**2))
    Uout = Uout * lens

    Uout,x1,y1 = fresnel_prop_no_scale(Uout, wvl, delta, fl)

    plt.figure(1)
    plt.imshow(np.abs(Uout)**2)
    # plt.set_cmap('gray')    
    plt.colorbar()

    plt.figure(2)
    plt.imshow(np.angle(Uout))
    # plt.set_cmap('gray')    
    plt.colorbar()

    plt.show()


if __name__ == '__main__':

    test1()



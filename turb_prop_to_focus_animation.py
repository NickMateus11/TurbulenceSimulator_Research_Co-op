
# import tensorflow as tf
# tf.random.set_seed(0)

import numpy as np
from timeit import default_timer as Timer
from matplotlib import pyplot as plt, animation  
from opensimplex import OpenSimplex

from PhaseScreenGenerator import PhaseScreenGenerator
from func_utils import *
from propagation import *

'''
    Animate propagations through turbulence - phase screens are shifted after each propagation
    Output is lens 'focused' beam at observation plane
'''

def frozen_flow_multi(phzs, vels, delta):
    # horizontal flow only
    N = len(phzs[0,0])
    shifted_phzs = np.empty_like(phzs)
    for i in range(N-1,-1,-1):
        for j in range(len(vels)):
            idx = i-int(vels[j]/delta)
            shifted_phzs[j,:,i] = phzs[j,:, idx%N if idx>0 else idx]
    return shifted_phzs

def test1():
    N = 1024
    M = 1024
    nscr = 5
    n_props = 60
    r0 = 0.35
    L0 = 1e3
    l0 = 0.01
    wvl = 1e-6
    Dz = 1e3
    D = 1
    delta = D/N
    fl = 5e3
    k = 2*PI/wvl

    noise = OpenSimplex()
    max_wind = 0.025
    # TODO: model through wind gradient instead of noise
    wind_vels = np.array([noise.noise2d(x/(nscr/10),0) for x in range(nscr)]) * max_wind
    # wind_vels = np.random.rand(nscr) * (2*max_wind) - max_wind

    x, y = np.meshgrid(
        np.array(list(range(-N//2,N//2))) * delta, 
        np.array(list(range(-N//2,N//2))) * delta
    )
    mask = circ(x, y, D/2)
    sg = np.exp(-(x/(0.47*N*delta))**16) * np.exp(-(y/(0.47*N*delta))**16)
    sg = np.tile(sg, [nscr, 1, 1])

    z = np.array(range(nscr)) * Dz / (nscr-1)

    Uin = circ(x, y, D)
    lens = np.exp(1j * -k/(2*fl) * (x**2 + y**2))

    PSG = PhaseScreenGenerator(r0, N, delta, L0, l0)
    phase_screens = PSG.next(nscr)
    Uouts = np.empty((n_props,N,N), np.complex)

    start = Timer()
    fig2 = plt.figure()
    # plt.set_cmap('gray')
    ax2 = plt.axes()
    Uout, xn, yn = ang_spec_multi_prop_no_scale(Uin, wvl, delta, z, sg*np.exp(1j*phase_screens))
    Uout = Uout * lens
    Uout,x1,y1 = fresnel_prop_no_scale(Uout, wvl, delta, fl)
    Uouts[0] = Uout
    im_Uout = ax2.imshow(np.abs(Uouts[0])**2 * mask)

    for n in range(1,n_props):
        print(n)
        phase_screens = frozen_flow_multi(phase_screens, wind_vels, delta)
        Uout, xn, yn = ang_spec_multi_prop_no_scale(Uin, wvl, delta, z, sg*np.exp(1j*phase_screens))
        Uout = Uout * lens
        Uout,x1,y1 = fresnel_prop_no_scale(Uout, wvl, delta, fl)
        Uouts[n] = Uout
    print(Timer()-start)
    np.save('speckle', Uouts)

    def animate(i):
        im_Uout.set_data(np.abs(Uouts[i])**2 * mask)

    ani = animation.FuncAnimation(fig2, animate, frames=n_props, interval=250)
    plt.show()

if __name__ == '__main__':
    test1()



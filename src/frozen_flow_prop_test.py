
# import tensorflow as tf
# tf.random.set_seed(0)

import numpy as np
from timeit import default_timer as Timer
from matplotlib import pyplot as plt, animation  
from opensimplex import OpenSimplex
import os

from modules.PhaseScreenGenerator import PhaseScreenGenerator
from modules.func_utils import *
from modules.propagation import *


'''
    Animate propagations through turbulence - phase screens are shifted after each propagation
    Output is field intensity at observation plane
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
    nscr = 9
    n_props = 20
    r0 = 0.3
    L0 = 1000
    l0 = 0.01
    wvl = 1e-6
    Dz = 1e3
    D = 2
    delta = D/N
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

    PSG = PhaseScreenGenerator(r0, N, delta, L0, l0)
    phase_screens = PSG.next(nscr)
    Uouts = np.empty((n_props,N,N), np.complex)

    # fig1 = plt.figure()
    # ax1 = plt.axes()
    # im_phz_scr = ax1.imshow(phase_screens[0])

    start = Timer()
    fig2 = plt.figure()
    plt.set_cmap('gray')
    ax2 = plt.axes()
    Uout, xn, yn = ang_spec_multi_prop_no_scale(Uin, wvl, delta, z, sg*np.exp(1j*phase_screens))
    Uouts[0] = Uout
    im_Uout = ax2.imshow(np.abs(Uouts[0])**2 * mask)

    # fig3 = plt.figure()
    # plt.set_cmap('gray')
    # ax3 = plt.axes()
    # im_Uout_phz = ax3.imshow(np.angle(Uout))

    for n in range(1,n_props):
        print(n)
        phase_screens = frozen_flow_multi(phase_screens, wind_vels, delta)
        Uout, xn, yn = ang_spec_multi_prop_no_scale(Uin, wvl, delta, z, sg*np.exp(1j*phase_screens))
        Uouts[n] = Uout
    print(Timer()-start)

    filename = 'speckle.npy'
    if 'src' in os.getcwd().split('\\'):
        np.save(f'../saved_data/{filename}', Uouts)
    else:
        np.save(f'./saved_data/{filename}', Uouts)

    def animate(i):
        im_Uout.set_data(np.abs(Uouts[i])**2 * mask)

    ani = animation.FuncAnimation(fig2, animate, frames=n_props, interval=250)
    plt.show()

if __name__ == '__main__':
    test1()




# import tensorflow as tf
# tf.random.set_seed(0)

import numpy as np
from timeit import default_timer as Timer
from matplotlib import pyplot as plt, animation  

from PhaseScreenGenerator import PhaseScreenGenerator
from func_utils import *
from propagation import *

'''
    Animate a phase screen being shifted using frozen flow model - no interpolation for non-integer grid size shifting
    Both simple pixel shifting and FT (multiply by complex expontial in spatial-frequency) are attempted
'''

def frozen_flow(phz, vel, delta):
    # horizontal flow only
    N = len(phz[0])
    shifted_phz = np.empty_like(phz)
    for i in range(N-1,-1,-1):
        shifted_phz[:,i] = phz[:, i-int(vel/delta)]
    return shifted_phz

def frozen_flow_ft(phz, vel, delta):
    N = len(phz)
    delta_f = 1/(N*delta)
    fx, _ = np.meshgrid(
        delta_f * np.linspace(-N//2,N//2-1,N),
        delta_f * np.linspace(-N//2,N//2-1,N),
    )

    ft_phz_scr = ft2(phz, delta)
    shifted_ft_phz_scr = np.exp(-1j * 2*np.pi * fx * vel) * ft_phz_scr

    return np.real(ift2(shifted_ft_phz_scr, delta_f))

def test1():
    N = 2048
    M = 512
    r0 = 0.1
    L0 = 100
    l0 = 0.01
    D = 2
    delta = D/N
    speed = 0.02

    x, y = np.meshgrid(
        np.array(list(range(-N//2,N//2))) * delta, 
        np.array(list(range(-M//2,M//2))) * delta
    )
    mask = circ(x,y,D/(N/M))

    PSG = PhaseScreenGenerator(r0, (N,M), delta, L0, l0)
    phase_screen = PSG.next().numpy()
    # normalize to 0-1
    phase_screen = (phase_screen-np.min(phase_screen)) / (np.max(phase_screen)-np.min(phase_screen))

    fig = plt.figure()
    ax = plt.axes()
    im = ax.imshow(phase_screen * mask)
    # im = ax.imshow(phase_screen)
    def animate(i):
        nonlocal phase_screen
        phase_screen = frozen_flow(phase_screen, speed, delta)
        # phase_screen = frozen_flow_ft(phase_screen, speed, delta)
        im.set_data(phase_screen * mask)
        # im.set_data(phase_screen)
    ani = animation.FuncAnimation(fig, animate, interval=200)
    plt.show()

if __name__ == '__main__':
    test1()



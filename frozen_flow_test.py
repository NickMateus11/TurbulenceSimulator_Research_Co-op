
# import tensorflow as tf
# tf.random.set_seed(0)

import numpy as np
from timeit import default_timer as Timer
from matplotlib import pyplot as plt, animation  

from PhaseScreenGenerator import PhaseScreenGenerator
from func_utils import *
from propagation import *


def frozen_flow(phz, vel, delta):
    # horizontal flow only
    N = len(phz)
    shifted_phz = np.empty_like(phz)
    for i in range(N-1,-1,-1):
        shifted_phz[:,i] = phz[:, i-int(vel/delta)]
    return shifted_phz

def test1():
    N = 2048
    r0 = 0.1
    L0 = 100
    l0 = 0.01
    D = 2
    delta = D/N
    speed = 0.01

    x, y = np.meshgrid(
        np.array(list(range(-N//2,N//2))) * delta, 
        np.array(list(range(-N//2,N//2))) * delta
    )
    mask = circ(x,y,D/2)

    PSG = PhaseScreenGenerator(r0, N, delta, L0, l0)
    phase_screen = PSG.next().numpy()
    # normalize to 0-1
    phase_screen = (phase_screen-np.min(phase_screen)) / (np.max(phase_screen)-np.min(phase_screen))

    fig = plt.figure()
    ax = plt.axes()
    im = ax.imshow(phase_screen * mask)
    def animate(i):
        nonlocal phase_screen
        phase_screen = frozen_flow(phase_screen, speed, delta)
        im.set_data(phase_screen * mask)
    ani = animation.FuncAnimation(fig, animate)
    plt.show()

if __name__ == '__main__':
    test1()



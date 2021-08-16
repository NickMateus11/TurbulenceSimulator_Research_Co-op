from modules.func_utils import circ
import numpy as np
from matplotlib import pyplot as plt, animation
import os

'''
    Load numpy array from file (N x N x m) - ie: m frames of N x N data
    Animate the data using matplotlib animation module
'''

def main():
    filename = 'speckle_focused.npy'
    if 'src' in os.getcwd().split('\\'):
        Uouts = np.load(f'../saved_data/{filename}')
    else:
        Uouts = np.load(f'./saved_data/{filename}')
    n = len(Uouts)
    N = len(Uouts[0])

    x, y = np.meshgrid(
        np.array(list(range(-N//2,N//2))), 
        np.array(list(range(-N//2,N//2)))
    )
    mask = circ(x, y, N/2)

    fig = plt.figure()
    ax = plt.axes()
    # plt.set_cmap('gray')
    im = ax.imshow(np.abs(Uouts[0])**2 * mask)

    def func(i):
        im.set_data(np.abs(Uouts[i])**2 * mask)

    ani = animation.FuncAnimation(fig, func, frames=n, interval=50)

    fig.colorbar(im)
    plt.xlim([N//2-N//16, N//2+N//16])
    plt.ylim([N//2-N//16, N//2+N//16])

    plt.show()

if __name__ == '__main__':
    main()
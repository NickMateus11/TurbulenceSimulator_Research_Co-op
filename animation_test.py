from func_utils import circ
import numpy as np
from matplotlib import pyplot as plt, animation

Uouts = np.load('speckle.npy')
# Uouts = np.load('speckle_focused.npy')
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
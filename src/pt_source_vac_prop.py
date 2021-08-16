import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from modules.func_utils import cart2pol, circ
from modules.propagation import ang_spec_multi_prop


def main():
     N = 512
     nscr = 11
     delta1 = 10e-3
     deltan = 10e-3

     wvl = 1e-6
     k = 2*np.pi/wvl
     Dz = 50e3
     R = Dz
     D2 = 0.5
     D1 = wvl*Dz / (4*D2)

     x = np.array(list(range(-N//2,N//2))) * delta1
     y = np.array(list(range(-N//2,N//2))) * delta1
     x1, y1 = np.meshgrid(x, y)
     theta, r1 = cart2pol(x1, y1)
     r1 = tf.cast(r1, tf.complex64)

     pt = np.exp(-1j*k/(2*R) * r1**2) / D1**2 \
          * np.sinc(x1/D1) * np.sinc(y1/D1)   \
          * np.exp(-(r1/(4*D1))**2)

     z = np.array(list(range(1,nscr))) * Dz / (nscr-1)

     sg = np.exp(-(x1/(0.47*N*delta1))**16) \
          * np.exp(-(y1/(0.47*N*delta1))**16)
     t = np.tile(sg, [nscr, 1, 1])
     Uvac, xn, yn = ang_spec_multi_prop(pt, wvl, delta1, deltan, z, t)
     Uvac = Uvac * np.exp(-1j*np.pi / (wvl*R) * (xn**2+yn**2))

     # mesh(xn, yn, np.abs(Uvac)**2)
     plt.figure(1)
     plt.imshow(np.abs(Uvac)**2)

     plt.set_cmap('gray')    
     plt.show()


if __name__ == '__main__':

    main()

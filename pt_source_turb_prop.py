import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from func_utils import circ, cart2pol, corr2_ft
from propagation import ang_spec_multi_prop
from phase_screen import ft_sub_harm_phase_screen


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

l0 = 0.01
L0 = 1e3
# l0 = 0
# L0 = np.inf

z = np.array(list(range(1, nscr))) * Dz / (nscr-1)
zt = np.array([0, *z])

Delta_z = zt[1:] - zt[:nscr-1]
alpha = zt / zt[-1]
delta = (1-alpha) * delta1 + alpha * deltan

x = np.array(list(range(-N//2,N//2))) * delta1
y = np.array(list(range(-N//2,N//2))) * delta1
x1, y1 = np.meshgrid(x, y)

theta, r1 = cart2pol(x1, y1)
r1 = tf.cast(r1, tf.complex64)

# pt = np.exp(-1j*k/(2*R) * r1**2) / D1**2 \
#      * np.sinc(x1/D1) * np.sinc(y1/D1)   \
#      * np.exp(-(r1/(4*D1))**2)

pt = circ(x1, y1, D1)

sg_1 = np.exp(-(x1/(0.47*N*delta1))**16) \
     * np.exp(-(y1/(0.47*N*delta1))**16)
sg = np.tile(sg_1, [nscr, 1, 1])

nx, ny = np.meshgrid(
    np.linspace(-N//2,N//2-1,N),
    np.linspace(-N//2,N//2-1,N),
)  
xn = nx * delta[-1]
yn = ny * delta[-1]

nreals = 1

Uout = np.zeros([N,N])
# mask = circ(xn/D2, yn/D2, 1)
# MCF2 = np.zeros([N,N])

r0scrn = np.array([75.7858, 0.3869, 0.3510, 0.3259, 0.3062, 0.2856, 0.2462, 0.2111, 0.2622, 0.2659, 75.7807])
# r0scrn = np.array([71.6222, 50.0032])

phz = np.empty((nscr, N, N))
for idxreal in range(nreals):
    # print(idxreal)
    for idxscr in range(nscr):
        phz_scr = ft_sub_harm_phase_screen(r0scrn[idxscr], N, delta[idxscr], L0, l0)
        phz[idxscr] = phz_scr
    
    Uout, xn, yn = ang_spec_multi_prop(pt, wvl, delta1, deltan, z, sg*np.exp(1j*phz))
    Uout = Uout * np.exp(-1j*np.pi/(wvl*R) * (xn**2+yn**2))
    # MCF2 = MCF2 + corr2_ft(Uout, Uout, mask, deltan)

# MCDOC2 = np.abs(MCF2) / (MCF2[N//2+1][N//2+1])

plt.figure(1)
plt.imshow(np.abs(Uout)**2)
plt.set_cmap('gray')    
plt.colorbar()

plt.figure(2)
plt.imshow(np.angle(Uout))
plt.set_cmap('gray')    
plt.colorbar()

plt.show()
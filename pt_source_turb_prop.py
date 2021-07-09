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

pt = np.exp(-1j*k/(2*R) * r1**2) / D1**2 \
     * np.sinc(x1/D1) * np.sinc(y1/D1)   \
     * np.exp(-(r1/(4*D1))**2)

sg_1 = np.exp(-(x1/(0.47*N*delta1))**16) \
     * np.exp(-(y1/(0.47*N*delta1))**16)
sg = np.tile(sg_1, [nscr, 1, 1])

nx, ny = np.meshgrid(
    np.linspace(-N//2,N//2-1,N),
    np.linspace(-N//2,N//2-1,N),
)  
xn = nx * delta[-1]
yn = ny * delta[-1]

phz = np.zeros([nscr,N,N])
nreals = 1

Uout = np.zeros([N,N])
mask = circ(xn/D2, yn/D2, 1)
MCF2 = np.zeros([N,N])

r0scrn = np.array([75.7858, 0.3869, 0.3510, 0.3259, 0.3062, 0.2856, 0.2462, 0.2111, 0.2622, 0.2659, 75.7807])
# r0scrn = np.array([71.6222, 50.0032])
# phz = np.array([[
#    -0.0084,   -0.0085,   -0.0077,   -0.0046,   -0.0027,   -0.0021,   -0.0009,    0.0012,
#    -0.0081,   -0.0065,   -0.0047,   -0.0016,   -0.0002,    0.0006,    0.0016,    0.0028,
#    -0.0064,   -0.0047,   -0.0026,   -0.0007,    0.0004,    0.0026,    0.0044,    0.0048,
#    -0.0040,   -0.0027,   -0.0010,    0.0008,    0.0027,    0.0052,    0.0067,    0.0067,
#    -0.0035,   -0.0023,    0.0000,    0.0015,    0.0037,    0.0058,    0.0067,    0.0080,
#    -0.0050,   -0.0037,   -0.0013,    0.0011,    0.0037,    0.0066,    0.0079,    0.0083,
#    -0.0056,   -0.0041,   -0.0026,   -0.0010,    0.0023,    0.0050,    0.0064,    0.0072,
#    -0.0061,   -0.0062,   -0.0050,   -0.0030,   -0.0003,    0.0028,    0.0037,    0.0059,],
#    [0.0005,   -0.0003,   -0.0004,    0.0004,    0.0012,    0.0003,   -0.0011,   -0.0022,
#     0.0033,    0.0040,    0.0032,    0.0026,    0.0031,    0.0031,    0.0025,    0.0012,
#     0.0027,    0.0039,    0.0037,    0.0019,    0.0022,    0.0026,    0.0021,    0.0009,
#     0.0010,    0.0016,    0.0003,    0.0005,    0.0022,    0.0022,    0.0011,   -0.0021,
#    -0.0021,   -0.0006,   -0.0011,   -0.0016,    0.0006,    0.0011,    0.0009,   -0.0032,
#    -0.0038,   -0.0025,   -0.0023,   -0.0022,   -0.0014,    0.0002,   -0.0007,   -0.0037,
#    -0.0042,   -0.0031,   -0.0018,   -0.0010,    0.0010,    0.0013,   -0.0016,   -0.0043,
#    -0.0056,   -0.0033,   -0.0021,    0.0002,    0.0014,    0.0014,   -0.0008,   -0.0033,]])


phz = np.empty((nscr, N, N))
for idxreal in range(nreals):
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
plt.show()
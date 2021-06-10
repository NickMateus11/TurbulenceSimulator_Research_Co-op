from matplotlib import pyplot as plt
from gaussian_beam import GaussianBeam
from propagation import fresnel_prop
from func_utils import plot_wave_slice


N = 512
wvl = 600e-9
w0 = 1e-3
dz = 5
L = 5*w0
delta = L/N

G = GaussianBeam(wvl, w0, 0, delta, N)

# original
G.plot_slice()

# propogated wave
g_prop,x2,y2 = fresnel_prop(G.wave, G.wvl, G.delta, G.delta, dz)
plot_wave_slice(g_prop, x2)
GaussianBeam.plot_waist_location(g_prop, x2)

# analytical prop
G.propagate(dz)
G.plot_slice()

plt.show()
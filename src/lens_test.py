from matplotlib import pyplot as plt
import numpy as np
from modules.gaussian_beam import GaussianBeam
from modules.propagation import fresnel_prop, fresnel_prop_no_scale
from modules.func_utils import plot_wave_slice


def main():
    PI = np.pi

    N = 512
    wvl = 600e-9
    w0 = 1e-3
    dz = 5
    L = 5*w0
    delta = L/N
    x,y = np.meshgrid(
        delta * np.linspace(-N/2,N/2-1,N),
        delta * np.linspace(-N/2,N/2-1,N),
    )

    fl = 3
    k = 2*PI/wvl
    lens = np.exp(1j * -k/(2*fl) * (x**2 + y**2))

    G = GaussianBeam(wvl, w0, 0, delta, N)

    GaussianBeam.plot_waist_over_distance(G.wave, G.x, wvl, delta, 5)
    g_prop,x1,y1 = fresnel_prop_no_scale(G.wave, wvl, delta, 5)

    g_prop *= lens
    GaussianBeam.plot_waist_over_distance(g_prop, G.x, wvl, delta, 6, offset=5, new_fig=False)
    g_prop,x1,y1 = fresnel_prop_no_scale(g_prop, wvl, delta, 6)

    plt.show()

if __name__ == '__main__':
    main()


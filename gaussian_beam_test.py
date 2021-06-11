from matplotlib import pyplot as plt
import numpy as np
from gaussian_beam import GaussianBeam
from propagation import fresnel_prop, fresnel_prop_no_scale
from func_utils import plot_wave_slice


N = 512
wvl = 600e-9
w0 = 1e-3
dz = 5
L = 5*w0
delta = L/N


def simple_prop_test():
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

def continuous_prop_test():
    # analytical prop
    z_span = 10
    G = GaussianBeam(wvl, w0, 0, delta, N)
    
    waists = []
    waists.append(G.w_z)
    n = 20

    dz = z_span / n
    for _ in range(n//2-1):
        z = z+dz
        G.propagate(dz)
        waists.append(G.w_z)
    
    waists = waists[::-1] + waists
    upper_waists =  np.array(waists)
    lower_waists = -np.array(waists)
    xx = np.linspace(-z_span/2, z_span/2, n)
    plt.plot(xx, upper_waists, 'b')
    plt.plot(xx, lower_waists, 'b')
    y_scale = waists[-1] * 2
    plt.ylim(-y_scale, y_scale)
    
    # fresnel prop
    G = GaussianBeam(wvl, w0, 0, delta, N)    
    waists = []
    waists.append(w0)
    n = 20

    dz = z_span / n
    g_prop = G.wave
    for _ in range(n//2-1):
        g_prop, x1, y1 = fresnel_prop_no_scale(g_prop, wvl, delta, dz)
        waists.append(GaussianBeam.get_waist_from_field(g_prop, x1))
    waists = waists[::-1] + waists
    upper_waists =  np.array(waists)
    lower_waists = -np.array(waists)
    xx = np.linspace(-z_span/2, z_span/2, n)

    plt.figure()
    plt.plot(xx, upper_waists, 'b')
    plt.plot(xx, lower_waists, 'b')
    y_scale = waists[-1] * 2
    plt.ylim(-y_scale, y_scale)
    
    plt.show()

def main():
    simple_prop_test()
    continuous_prop_test()

if __name__ == '__main__':
    main()
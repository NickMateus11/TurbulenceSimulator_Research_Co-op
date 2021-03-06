from matplotlib import pyplot as plt
import numpy as np
from modules.gaussian_beam import GaussianBeam
from modules.propagation import fresnel_prop, fresnel_prop_no_scale
from modules.func_utils import plot_wave_slice


N = 512
wvl = 600e-9
w0 = 1e-3
dz = 5
L = 5*w0
delta = L/N


def simple_prop_test():
    '''
    Shows Gaussian beam slice before and after propagation. Take note of the beam waste as it increases after propagation.
    Comparison of theory and fresnel (numerical) propagation results.
    '''
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

def beam_waist_prop_test():
    '''
    Plots beam waist as a function of prop distance. Notice the saddle shape.
    '''
    # analytical prop
    z_span = 10
    G = GaussianBeam(wvl, w0, 0, delta, N)
    
    waists = []
    waists.append(G.w_z)
    n = 20

    dz = z_span / n
    for _ in range(n//2-1):
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
    n = 20
    
    GaussianBeam.plot_waist_over_distance(G.wave, G.x, wvl, delta, z_span/2, n)
    GaussianBeam.plot_waist_over_distance(G.wave, G.x, wvl, delta, -z_span/2, n, new_fig=False)
    
    plt.show()

def main():
    simple_prop_test()
    beam_waist_prop_test()

if __name__ == '__main__':
    main()
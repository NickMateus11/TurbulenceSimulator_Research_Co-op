import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt, animation  
from opensimplex import OpenSimplex
from timeit import default_timer as Timer
import os

from modules.PhaseScreenGenerator import PhaseScreenGenerator
from modules.phase_screen_generation import ft_sub_harm_phase_screen
from modules.propagation import *
from modules.func_utils import *


def pt_source_vac_prop():
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

def pt_source_turb_prop():
    N = 512
    nscr = 5
    delta1 = 10e-3
    deltan = 10e-3

    wvl = 1e-6
    k = 2*np.pi/wvl
    Dz = 50e3
    R = Dz
    D2 = 0.2
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

    r0scrn = np.array([0.3869, 0.3869, 0.3510, 0.3259, 0.3062, 0.2856, 0.2462, 0.2111, 0.2622, 0.2659, 0.3869])

    phz = np.empty((nscr, N, N))
    for idxreal in range(nreals):
        for idxscr in range(nscr):
            phz_scr = ft_sub_harm_phase_screen(r0scrn[idxscr], N, delta[idxscr], L0, l0)
            phz[idxscr] = phz_scr
        
        Uout, xn, yn = ang_spec_multi_prop_no_scale(pt, wvl, delta1, z, sg*np.exp(1j*phz))
        Uout = Uout * np.exp(-1j*np.pi/(wvl*R) * (xn**2+yn**2))
        # MCF2 = MCF2 + corr2_ft(Uout, Uout, mask, deltan)

    # MCDOC2 = np.abs(MCF2) / (MCF2[N//2+1][N//2+1])

    plt.figure(1)
    plt.imshow(np.abs(Uout)**2)
    # plt.set_cmap('gray')    
    plt.colorbar()

    plt.figure(2)
    plt.imshow(np.angle(Uout))
    # plt.set_cmap('gray')    
    plt.colorbar()

    plt.show()

def turb_prop_to_focus():
    '''
    Propagate beam through turbulence - focus through lens to output plane
    '''
    N = 2048
    r0 = 0.3
    L0 = 1e3
    l0 = 0.01
    D = 1
    delta = D/N
    nscr = 5
    Dz = 1e3
    wvl = 1e-6

    start = Timer()

    z = np.array(range(nscr)) * Dz / (nscr-1)
    PSG = PhaseScreenGenerator(r0, N, delta, L0, l0)

    x1, y1 = np.meshgrid(
        np.array(list(range(-N//2,N//2))) * delta, 
        np.array(list(range(-N//2,N//2))) * delta
    )

    sg = np.exp(-(x1/(0.47*N*delta))**16) * np.exp(-(y1/(0.47*N*delta))**16)
    sg = np.tile(sg, [nscr, 1, 1])

    # TODO: Gaussian Beam
    Uin = circ(x1, y1, D)

    print(Timer()-start)
    start = Timer()

    phz = np.empty([nscr, N, N])
    for i in range(nscr):
        phz[i] = PSG.next()
    
    print(Timer()-start)
    start = Timer()
    
    Uout, xn, yn = ang_spec_multi_prop_no_scale(Uin, wvl, delta, z, sg*np.exp(1j*phz))
    # Uout = Uout * np.exp(-1j*np.pi/(wvl*Dz) * (xn**2+yn**2))
    print(Timer()-start)

    fl = 5e3
    k = 2*PI/wvl
    lens = np.exp(1j * -k/(2*fl) * (x1**2 + y1**2))
    Uout = Uout * lens

    Uout,x1,y1 = fresnel_prop_no_scale(Uout, wvl, delta, fl)

    plt.figure(1)
    plt.imshow(np.abs(Uout)**2)
    # plt.set_cmap('gray')    
    plt.colorbar()
    plt.xlim([N//2-N//16, N//2+N//16])
    plt.ylim([N//2-N//16, N//2+N//16])

    # plt.figure(2)
    # plt.imshow(np.angle(Uout))
    # # plt.set_cmap('gray')    
    # plt.colorbar()

    plt.show()

def turb_prop_to_focus_animation():
    '''
    Animate propagations through turbulence - phase screens are shifted after each propagation
    Output is lens 'focused' beam at observation plane
    '''
    N = 1024
    M = 1024
    nscr = 5
    n_props = 30
    r0 = 0.35
    L0 = 1e3
    l0 = 0.01
    wvl = 1e-6
    Dz = 1e3
    D = 1
    delta = D/N
    fl = 5e3
    k = 2*PI/wvl

    noise = OpenSimplex()
    max_wind = 0.025
    # TODO: model through wind gradient instead of noise
    wind_vels = np.array([noise.noise2d(x/(nscr/10),0) for x in range(nscr)]) * max_wind
    # wind_vels = np.random.rand(nscr) * (2*max_wind) - max_wind

    x, y = np.meshgrid(
        np.array(list(range(-N//2,N//2))) * delta, 
        np.array(list(range(-N//2,N//2))) * delta
    )
    mask = circ(x, y, D/2)
    sg = np.exp(-(x/(0.47*N*delta))**16) * np.exp(-(y/(0.47*N*delta))**16)
    sg = np.tile(sg, [nscr, 1, 1])

    z = np.array(range(nscr)) * Dz / (nscr-1)

    Uin = circ(x, y, D)
    lens = np.exp(1j * -k/(2*fl) * (x**2 + y**2))

    PSG = PhaseScreenGenerator(r0, N, delta, L0, l0)
    phase_screens = PSG.next(nscr)
    Uouts = np.empty((n_props,N,N), np.complex)

    start = Timer()
    fig2 = plt.figure()
    # plt.set_cmap('gray')
    ax2 = plt.axes()
    Uout, xn, yn = ang_spec_multi_prop_no_scale(Uin, wvl, delta, z, sg*np.exp(1j*phase_screens))
    Uout = Uout * lens
    Uout,x1,y1 = fresnel_prop_no_scale(Uout, wvl, delta, fl)
    Uouts[0] = Uout
    im_Uout = ax2.imshow(np.abs(Uouts[0])**2 * mask)

    for n in range(1,n_props):
        print(n)
        phase_screens = PSG.frozen_flow_multi(phase_screens, wind_vels, delta)
        Uout, xn, yn = ang_spec_multi_prop_no_scale(Uin, wvl, delta, z, sg*np.exp(1j*phase_screens))
        Uout = Uout * lens
        Uout,x1,y1 = fresnel_prop_no_scale(Uout, wvl, delta, fl)
        Uouts[n] = Uout
    print(Timer()-start)

    filename = 'speckle.npy'
    if 'src' in os.getcwd().split('\\'):
        np.save(f'../saved_data/{filename}', Uouts)
    else:
        np.save(f'./saved_data/{filename}', Uouts)

    def animate(i):
        im_Uout.set_data(np.abs(Uouts[i])**2 * mask)

    ani = animation.FuncAnimation(fig2, animate, frames=n_props, interval=250)
    
    plt.xlim([N//2-N//16, N//2+N//16])
    plt.ylim([N//2-N//16, N//2+N//16])
    plt.show()

def main():

    # pt_source_vac_prop()

    # pt_source_turb_prop()

    turb_prop_to_focus()

    # turb_prop_to_focus_animation()

if __name__ == '__main__':
    main()

import multiprocessing
from matplotlib import pyplot as plt
import numpy as np
import time
from numpy.ma.extras import average

from gaussian_beam import GaussianBeam
from func_utils import str_fnc2_ft, circ, plot_slice, mesh
from propagation import fresnel_prop_no_scale
from phase_screen import ft_sub_harm_phase_screen, ft_phase_screen


def phase_screen_gen():
    D = 2
    r0 = 0.1
    N = 256
    L0 = 100
    l0 = 0.01

    delta = D/N
    x = np.linspace(-N/2,N/2-1, N) * delta
    y = np.linspace(-N/2,N/2-1, N) * delta

    [phz_lo, phz_hi] = ft_sub_harm_phase_screen(r0, N, delta, L0, l0)
    phz = phz_lo + phz_hi # (SH + FT)

    plt.figure(1)
    plt.imshow(phz, extent=[x[0], x[-1], y[-1], y[0]])

    plt.set_cmap('gray')    
    plt.colorbar()
    plt.show()


def phase_screen_calcs(args):
    N, mask, L0, l0, r0, delta = args
    [phz_lo, phz_hi] = ft_sub_harm_phase_screen(r0, N, delta, L0, l0)
    phz1 = phz_lo + phz_hi
    D1 = str_fnc2_ft(phz1, mask, delta)
    phz2 = ft_phase_screen(r0, N, delta, L0, l0)
    D2 = str_fnc2_ft(phz2, mask, delta)
    return D1, D2


def phase_screen_statistics():
    D = 2
    r0 = 0.1
    N = 256
    L0 = 100
    l0 = 0.01

    delta = D/N
    xx = np.linspace(-N/2,N/2-1, N) * delta
    yy = np.linspace(-N/2,N/2-1, N) * delta
    [x, y] = np.meshgrid(xx, yy)

    mask  = circ(x, y, D*0.9)
    # mask = np.ones( (N,N) )
    # mesh(x,y, mask)

    N_phase_screens = 40

    avgD1 = np.zeros( (N,N) )
    avgD2 = np.zeros( (N,N) )

    start = time.time()
    averages = [phase_screen_calcs( [N, mask, L0, l0, r0, delta] ) for _ in range(N_phase_screens)]
    print(time.time() - start)

    start = time.time()
    with multiprocessing.Pool() as pool:
        averages = pool.map(phase_screen_calcs, [(N, mask, L0, l0, r0, delta)]*N_phase_screens)
        # averages = pool.starmap(phase_screen_calcs, [(N, mask, L0, l0, r0, delta)]*N_phase_screens)
    print(time.time() - start)    

    for i in range(N_phase_screens):
        # a1, a2 = averages.get(timeout=20)
        a1, a2 = averages[i]
        avgD1 += a1/N_phase_screens
        avgD2 += a2/N_phase_screens
        
    theo_struc = 6.88 * (np.abs(xx)/r0) ** (5/3)

    plt.figure()
    plt.plot(xx/l0, theo_struc)
    plot_slice(avgD1, x/l0, new_fig=False)
    plot_slice(avgD2, x/l0, new_fig=False)

    plt.legend(["Theory","SH","FT"])
    plt.xlim(0, 0.8*max(xx/l0))
    plt.ylim(0, 250)

    plt.show()    


def phase_screen_prop():
    D = 2
    r0 = 0.1
    N = 256
    L0 = 100
    l0 = 0.01


    delta = D/N
    x = np.linspace(-N/2,N/2-1, N) * delta
    y = np.linspace(-N/2,N/2-1, N) * delta

    [phz_lo, phz_hi] = ft_sub_harm_phase_screen(r0, N, delta, L0, l0)
    phz = phz_lo + phz_hi # (SH + FT)
    
    # mask  = circ(x, y, D*0.9)
    # phz *= mask

    # N = 512
    wvl = 600e-9
    w0 = 1e-3
    dz = 5
    L = 5*w0
    delta = L/N

    [x1,y1] = np.meshgrid(
        delta * np.linspace(-N//2,N//2-1,N),
        delta * np.linspace(-N//2,N//2-1,N),
    )

    G = GaussianBeam(wvl, w0, 0, delta, N)
    g_prop = G.wave * phz
    # g_prop2,x2,y2 = fresnel_prop_no_scale(g_prop, wvl, delta, 5)
    # mesh(x2,y2,np.abs(g_prop2)**2)
    mesh(x1,y1,np.abs(g_prop)**2)

    plt.show()


def main():
    # phase_screen_gen()
    phase_screen_statistics()
    # phase_screen_prop()


if __name__ == '__main__':
    main()
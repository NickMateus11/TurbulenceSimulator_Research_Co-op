from matplotlib import pyplot as plt
import numpy as np

from func_utils import str_fnc2_ft, mesh, plot_wave_slice
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

    # [phz_lo, phz_hi] = ft_sub_harm_phase_screen(r0, N, delta, L0, l0)
    # phz = phz_lo + phz_hi # (SH + FT)

    N_phase_screens = 1
    mask  = np.ones ( (N,N) )
    avgD1 = np.zeros( (N,N) )
    avgD2 = np.zeros( (N,N) )

    for _ in range(N_phase_screens):
        [phz_lo, phz_hi] = ft_sub_harm_phase_screen(r0, N, delta, L0, l0)
        phz1 = phz_lo + phz_hi
        D1 = str_fnc2_ft(phz1, mask, delta)
        avgD1 += D1/N_phase_screens

        phz2 = ft_phase_screen(r0, N, delta, L0, l0)
        D2 = str_fnc2_ft(phz2, mask, delta)
        avgD2 += D2/N_phase_screens

    theo_struc = 6.88 * (np.abs(xx)/r0) ** (5/3)

    plt.plot(xx/l0, theo_struc)
    plot_wave_slice(avgD1, x/l0, new_fig=False)
    plot_wave_slice(avgD2, x/l0, new_fig=False)

    plt.legend(["Theory","SH","FT"])
    # plt.xlim(0, 12)
    # plt.ylim(0, 1e3)

    plt.show()    


def main():
    # phase_screen_gen()
    phase_screen_statistics()


if __name__ == '__main__':
    main()
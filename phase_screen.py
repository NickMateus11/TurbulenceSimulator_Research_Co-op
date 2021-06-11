
import numpy as np
from func_utils import ft2, ift2, cart2pol
from matplotlib import pyplot as plt


PI = np.pi


def compute_strfunc(phz, mask, delta):
    N = np.size(phz,0)
    phz *= mask

    P = ft2(phz, delta)
    S = ft2(phz**2, delta)
    W = ft2(mask, delta)

    delta_f = 1/(N*delta)
    w2 = ift2(W*np.conj(W), delta_f)

    D = 2 * ift2(np.real(S*np.conj(W)) - np.abs(P)**2, delta_f) / w2 * mask
    return D


def ft_phase_screen(r0, N, delta, L0, l0, method='modified von karman'):
    del_f = 1/(N*delta)

    fx = np.linspace(-N/2,N/2-1, N) * del_f
    fy = np.linspace(-N/2,N/2-1, N) * del_f
    [fx, fy] = np.meshgrid(fx, fy)

    [th, f] = cart2pol(fx, fy)
    fm = 5.92 / (l0*2*PI)
    f0 = 1/L0

    if method == 'von karman':
        PSD_phi = 0.023 * r0**(-5/3) / (f**2 + f0**2)**(11/6)
    elif method =='modified von karman':
        PSD_phi = 0.023 * r0**(-5/3) * np.exp(-(f/fm)**2) / (f**2 + f0**2)**(11/6)
    else: # kolmogorov
        PSD_phi = 0.023 * r0**(-5/3)

    PSD_phi[N//2, N//2] = 0

    cn = (np.random.randn(N,N) + 1j*np.random.randn(N,N)) * np.sqrt(PSD_phi) * del_f
    phz = np.real(ift2(cn, 1))

    return phz


def ft_sub_harm_phase_screen(r0, N, delta, L0, l0, method='modified von karman'):
    N_p = 3
    
    D = N*delta
    phz_hi = ft_phase_screen(r0, N, delta, L0, l0, method)

    x = np.linspace(-N/2,N/2-1, N) * delta
    y = np.linspace(-N/2,N/2-1, N) * delta
    [x, y] = np.meshgrid(x, y)

    phz_lo = np.zeros_like(phz_hi)
    for p in range(N_p):
        del_f = 1 / (3**(p+1)*D)
        fx = np.linspace(-del_f, del_f, N_p)
        fy = np.linspace(-del_f, del_f, N_p)
        [fx, fy] = np.meshgrid(fx, fy)
        [th, f] = cart2pol(fx, fy)
        fm = 5.92 / (l0*2*PI)
        f0 = 1/L0

        # conversion from r0 to Cn^2 -> 0.033 = 0.023 * 1.46 (ch9 eqn 9.40)
        if method == 'von karman':
            PSD_phi = 0.023 * r0**(-5/3) / (f**2 + f0**2)**(11/6)
        elif method =='modified von karman':
            PSD_phi = 0.023 * r0**(-5/3) * np.exp(-(f/fm)**2) / (f**2 + f0**2)**(11/6)
        else: # kolmogorov
            PSD_phi = 0.023 * r0**(-5/3)

        PSD_phi[1, 1] = 0

        cn = (np.random.randn(N_p,N_p) + 1j*np.random.randn(N_p,N_p)) * np.sqrt(PSD_phi) * del_f
        SH = np.zeros(shape=(N,N))

        for i in range(N_p**2):
            r,c = [i%N_p, i//N_p]
            SH = SH + cn[r, c] * np.exp(1j*2*PI*(fx[r, c]*x + fy[r, c]*y))
        
        phz_lo = phz_lo + SH

    phz_lo = np.real(phz_lo) - np.mean(np.real(phz_lo).flatten())

    return phz_lo, phz_hi


def main():
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


if __name__ == '__main__':
    main()

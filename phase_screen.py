
import numpy as np
from func_utils import ft2, ift2, cart2pol
from matplotlib import pyplot as plt
import tensorflow as tf


PI = np.pi

# def read_file(filename):
#     with open(filename) as infile:
#         data = [x.split(',') for x in infile.read().strip().split()]
#         data = [[eval(x.replace('i','j')) for x in r] for r in data]
#         data = np.array(data)
#     return data

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

@tf.function
def ft_phase_screen(r0, N, delta, L0, l0, method='modified von karman'):
    del_f = 1/(N*delta)

    fx = tf.linspace(-N/2,N/2-1, N) * del_f
    fy = tf.linspace(-N/2,N/2-1, N) * del_f
    [fx, fy] = tf.meshgrid(fx, fy)

    [th, f] = cart2pol(fx, fy)
    fm = 5.92 / (l0*2*PI)
    f0 = 1/L0

    if method == 'von karman':
        PSD_phi = 0.023 * r0**(-5/3) / (f**2 + f0**2)**(11/6)
    elif method =='modified von karman':
        PSD_phi = 0.023 * r0**(-5/3) * tf.exp(-(f/fm)**2) / (f**2 + f0**2)**(11/6)
    else: # kolmogorov
        PSD_phi = 0.023 * r0**(-5/3)

    # convert to tf.Variable? to mutate
    # PSD_phi[N//2, N//2] = 0

    # cn = read_file("dist_mat1.txt") * np.sqrt(PSD_phi) * del_f
    cn = tf.complex(tf.random.normal( (N,N) ), tf.random.normal( (N,N) )) * tf.cast(tf.math.sqrt(PSD_phi), tf.complex64) * del_f
    phz = tf.math.real(ift2(cn, 1))

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

        # cn = read_file(f"dist_mat2_{p+1}.txt") * np.sqrt(PSD_phi) * del_f
        cn = (np.random.randn(N_p,N_p) + 1j*np.random.randn(N_p,N_p)) * np.sqrt(PSD_phi) * del_f
        SH = np.zeros(shape=(N,N))

        for i in range(N_p**2): #parallelize
            r,c = [i%N_p, i//N_p]
            SH = SH + cn[r, c] * np.exp(1j*2*PI*(fx[r, c]*x + fy[r, c]*y))
        
        phz_lo = phz_lo + SH

    phz_lo = np.real(phz_lo) - np.mean(np.real(phz_lo).flatten())

    return phz_lo, phz_hi



import numpy as np
from tensorflow.python.framework.ops import Tensor
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

def compute_strfunc(phz: tf.Tensor, mask, delta):
    N = (phz.shape)[0]
    phz = phz * mask

    P = ft2(phz, delta)
    S = ft2(phz**2, delta)
    W = ft2(mask, delta)

    delta_f = 1/(N*delta)
    w2 = ift2(W*tf.math.conj(W), delta_f)

    D = 2 * ift2(tf.cast(tf.math.real(S*tf.math.conj(W)) - tf.abs(P)**2, tf.complex64), delta_f) / w2 * mask
    return D

@tf.function
def ft_phase_screen(r0, N, delta, L0, l0, method='modified von karman'):
    del_f = 1/(N*delta)

    fx = tf.linspace(-N//2,N//2-1, N) * del_f
    fy = tf.linspace(-N//2,N//2-1, N) * del_f
    [fx, fy] = tf.meshgrid(fx, fy)

    [th, f] = cart2pol(fx, fy)
    fm = 5.92 / (l0*2*PI) if l0 != 0 else np.inf 
    f0 = 1/L0

    # PSD_phi = tf.Variable( tf.zeros_like(f, tf.complex64), dtype=tf.complex64 )
    if method == 'von karman':
        PSD_phi = 0.023 * r0**(-5/3) / (f**2 + f0**2)**(11/6)
    elif method =='modified von karman':
        PSD_phi = 0.023 * r0**(-5/3) * tf.exp(-(f/fm)**2) / (f**2 + f0**2)**(11/6)
    else: # kolmogorov
        PSD_phi = 0.023 * r0**(-5/3)

    # convert to tf.Variable? to mutate
    # PSD_phi[N//2, N//2] = 0
    tf.tensor_scatter_nd_update(PSD_phi, [[N//2,N//2]], [0])

    # cn = read_file("dist_mat1.txt") * np.sqrt(PSD_phi) * del_f
    cn = tf.complex(tf.random.normal( (N,N) ), tf.random.normal( (N,N) )) * tf.cast(tf.math.sqrt(PSD_phi) * del_f, tf.complex64)
    phz = tf.math.real(ift2(cn, 1))

    return phz

@tf.function
def ft_sub_harm_phase_screen(r0, N, delta, L0, l0, method='modified von karman'):
    N_p = 3
    
    D = N*delta
    phz_hi = ft_phase_screen(r0, N, delta, L0, l0, method)

    x = tf.linspace(-N//2,N//2-1, N) * delta
    y = tf.linspace(-N//2,N//2-1, N) * delta
    [x, y] = tf.meshgrid(x, y)

    phz_lo = tf.zeros_like(phz_hi, tf.complex64)
    for p in range(N_p):
        del_f = 1 / (3**(p+1)*D)
        fx = tf.linspace(-del_f, del_f, N_p)
        fy = tf.linspace(-del_f, del_f, N_p)
        [fx, fy] = tf.meshgrid(fx, fy)
        [th, f] = cart2pol(fx, fy)
        fm = 5.92 / (l0*2*PI) if l0 != 0 else np.inf
        f0 = 1/L0

        # PSD_phi = tf.Variable( tf.zeros_like(f, tf.complex64), dtype=tf.complex64 )
        # conversion from r0 to Cn^2 -> 0.033 = 0.023 * 1.46 (ch9 eqn 9.40)
        if method == 'von karman':
            PSD_phi = 0.023 * r0**(-5/3) / (f**2 + f0**2)**(11/6)
        elif method =='modified von karman':
            PSD_phi = 0.023 * r0**(-5/3) * tf.exp(-(f/fm)**2) / (f**2 + f0**2)**(11/6)
        else: # kolmogorov
            PSD_phi = 0.023 * r0**(-5/3)

        # convert to tf.Variable? to mutate
        # PSD_phi[1, 1] = 0 #  ****** (1,1) because 3x3? ******
        tf.tensor_scatter_nd_update(PSD_phi, [[1,1]], [0])

        # cn = read_file(f"dist_mat2_{p+1}.txt") * np.sqrt(PSD_phi) * del_f
        cn = tf.complex(tf.random.normal( (N_p,N_p) ), tf.random.normal( (N_p,N_p) )) * tf.cast(tf.math.sqrt(PSD_phi) * del_f , tf.complex64)       
        SH = tf.zeros( (N,N), tf.complex64 )

        for i in range(N_p**2): #parallelize
            r,c = [i%N_p, i//N_p]
            SH = SH + cn[r, c] * tf.cast(tf.exp(tf.complex(tf.constant(0, tf.float64), 2*PI*(fx[r, c]*x + fy[r, c]*y))), tf.complex64)
        
        phz_lo = phz_lo + SH

    phz_lo = tf.math.real(phz_lo) - tf.reduce_mean(tf.math.real(phz_lo))

    phz_screen = phz_lo + phz_hi
    return phz_screen


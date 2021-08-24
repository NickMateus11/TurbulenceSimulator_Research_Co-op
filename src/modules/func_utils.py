
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf


'''
    Generally useful functions related to FT computations, and matplotlib wrappers
'''

def str_fnc2_ft(ph, mask, delta):

    N = (ph.shape)[0]
    ph = ph * mask

    P = ft2(ph, delta)
    S = ft2(ph**2, delta)
    W = ft2(mask, delta)

    delta_f = 1/(N*delta)
    w2 = ift2(W*tf.math.conj(W), delta_f)

    D = 2 * ift2(tf.cast(tf.math.real(S*tf.math.conj(W)) - tf.abs(P)**2, tf.complex64), delta_f) / w2 * mask
    return tf.abs(D)

def corr2_ft(u1, u2, mask, delta):
    N = (u1.shape)[0]
    c = np.zeros( (N,N) )
    delta_f = 1/(N*delta)

    U1 = ft2(u1 * mask, delta)
    U2 = ft2(u2 * mask, delta)
    U12corr = ift2(tf.math.conj(U1) * U2, delta_f)

    maskcorr = ift2(abs(ft2(mask, delta))**2, delta_f) * delta**2
    idx = tf.cast(maskcorr, tf.bool)
    c[idx] = U12corr[idx] / maskcorr[idx] * mask[idx]
    return c

def cart2pol(x, y):
    phi = tf.math.atan2(y, x)
    rho = tf.sqrt(x**2 + y**2)
    return phi, rho

def ft2(g, delta):
    g = tf.cast(g, tf.complex64)
    return tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(g))) * delta**2

def ift2(g, delta_f):
    N = np.size(g,0) # assume square
    g = tf.cast(g, tf.complex64)
    return tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(g))) * (N * delta_f)**2

def rect(x,width):
    return np.heaviside(x+width/2,0.5) - np.heaviside(x-width/2,0.5)

def square(x,y,s):
    return rect(x,s)*rect(y,s)

def circ(x,y,D):
    r = np.sqrt(x**2 + y**2)
    return (r < D/2.0).astype(np.float32)

def mesh(x,y,z):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.plot_surface(x,y,z, cmap=cm.coolwarm)
    ax.plot_surface(x,y,z)
    return ax

def plot_slice(U, x, new_fig=True):
    if new_fig: 
        plt.figure()
    xx = x[len(x)//2]
    plt.plot(xx, U[len(U)//2])

def plot_wave_slice(U, x, new_fig=True):
    if new_fig: 
        plt.figure()
    I = np.abs(U[len(U)//2])**2
    xx = x[len(x)//2]
    plt.plot(xx, I)


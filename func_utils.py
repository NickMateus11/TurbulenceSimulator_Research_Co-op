
import numpy as np
from matplotlib import pyplot as plt, cm


def cart2pol(x, y):
    phi = np.arctan2(y, x)
    rho = np.sqrt(x**2 + y**2)
    return phi, rho

def ft2(g, delta):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(g))) * delta**2

def ift2(g, delta_f):
    N = np.size(g,0) # assume square
    return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(g))) * (N * delta_f)**2

def rect(x,width):
    return np.heaviside(x+width/2,0.5) - np.heaviside(x-width/2,0.5)

def square(x,y,s):
    return rect(x,s)*rect(y,s)

def circ(x,y,D):
    r = np.sqrt(x**2 + y**2)
    return (r < D/2).astype(np.float32)

def mesh(x,y,z):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.plot_surface(x,y,z, cmap=cm.coolwarm)
    ax.plot_surface(x,y,z)
    return ax

def plot_wave_slice(U, x):
    plt.figure()
    I = np.abs(U[len(U)//2])**2
    xx = x[len(x)//2]
    plt.plot(xx, I)

from matplotlib import pyplot as plt
import numpy as np
from func_utils import plot_wave_slice
from propagation import fresnel_prop_no_scale


PI = np.pi

class GaussianBeam():
    def __init__(self, wvl, w0, z, delta, N):
        self.wvl = wvl
        self.w0 = w0
        self.z = z
        self.delta = delta
        self.N = N

        self.k = 2*PI/self.wvl

        self.x, self.y = np.meshgrid(
            self.delta * np.linspace(-N//2,N//2-1,N),
            self.delta * np.linspace(-N//2,N//2-1,N),
        )
        self.zr = PI * self.w0**2 / self.wvl
        self.w_z = self.w0 * np.sqrt(1+(self.z/self.zr)**2)
        self.r_z = self.z * (1+(self.zr/self.z)**2) if not (self.z == 0) else np.inf

    def amplitude(self):
        return np.sqrt(2/PI) * (1/self.w_z) * np.exp(-(self.x**2 + self.y**2)/self.w_z**2)

    def phase(self):
        if not (self.r_z is np.inf):
            phase = -(self.k*self.z + self.k*(self.x**2 + self.y**2)/(2*self.r_z) - np.arctan2(self.z,self.zr))
        else:
            phase = np.zeros(shape=(self.N,self.N))
        return phase

    @property
    def wave(self):
        return self.amplitude() * np.exp(1j * self.phase())

    def propagate(self, dz):
        self.z += dz
        self.w_z = self.w0 * np.sqrt(1 + (self.wvl*self.z / (PI*self.w0**2))**2)

    def plot_slice(self):
        plot_wave_slice(self.wave, self.x)
        self.plot_waist_location(self.wave, self.x)

    @staticmethod
    def get_waist_from_field(U, x):
        I = np.abs(U[len(U)//2])**2
        xx = x[len(x)//2]
        idx = 0
        thresh = np.max(I)/np.exp(1)**2
        for i in range(len(I)):
            if I[i] < thresh:
                idx = i
            else: break     
        return np.abs(xx[idx])

    @staticmethod
    def plot_waist_location(U, x, override_title=True):
        maxI = np.max(np.abs(U[len(U)//2])**2)
        n = 50
        waist = GaussianBeam.get_waist_from_field(U, x)
        plt.plot(np.ones(n)* waist, np.linspace(0,maxI, n), 'r:')
        plt.plot(np.ones(n)*-waist, np.linspace(0,maxI, n), 'r:')

        if override_title:
            plt.title(f'w_z: {np.abs(waist)}')

    @staticmethod
    def plot_waist_over_distance_symmetric(U, x, wvl, delta, z_span, n=20):

        waists = []
        xx = np.linspace(-z_span/2, z_span/2, n)
        waists.append(GaussianBeam.get_waist_from_field(U, x))

        dz = z_span / n
        g_prop = U
        for _ in range(n//2-1):
            g_prop, x1, y1 = fresnel_prop_no_scale(g_prop, wvl, delta, dz)
            waists.append(GaussianBeam.get_waist_from_field(g_prop, x1))
        waists = waists[::-1] + waists
        upper_waists =  np.array(waists)
        lower_waists = -np.array(waists)

        plt.figure()
        plt.plot(xx, upper_waists, 'b')
        plt.plot(xx, lower_waists, 'b')
        y_scale = max(waists) * 2
        plt.ylim(-y_scale, y_scale)

    @staticmethod
    def plot_waist_over_distance(U, x, wvl, delta, z_span, offset=0, n=20):

        waists = []
        xx = np.linspace(offset, offset+z_span, n)
        waists.append(GaussianBeam.get_waist_from_field(U, x))

        dz = z_span / n
        g_prop = U
        for _ in range(n-1):
            g_prop, x1, y1 = fresnel_prop_no_scale(g_prop, wvl, delta, dz)
            waists.append(GaussianBeam.get_waist_from_field(g_prop, x1))
        upper_waists =  np.array(waists)
        lower_waists = -np.array(waists)

        if offset == 0:
            plt.figure()
        plt.plot(xx, upper_waists, 'b')
        plt.plot(xx, lower_waists, 'b')
        y_scale = max(waists) * 2
        plt.ylim(-y_scale, y_scale)


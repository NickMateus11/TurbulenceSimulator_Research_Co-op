import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from func_utils import *


class PhaseScreenGenerator():
    
    def __init__(self, r0, N, delta, L0, l0, N_p=3, method='modified von karman'):
        self.r0 = r0
        self.N = N
        self.delta = delta
        self.L0 = L0
        self.l0 = l0
        self.N_p = N_p
        self.method = method

        self.phaseScreens = []
        
        [self.x, self.y] = tf.meshgrid(
            tf.linspace(-N/2,N/2-1, N) * delta,
            tf.linspace(-N/2,N/2-1, N) * delta
        )

        self.ftPhaseScreenOneTimeSetup()
        self.ftSubHarmPhaseScreenOneTimeSetup()        

    def ftPhaseScreenOneTimeSetup(self):
        '''
        Defines and calculates (self.) del_f, PSD_phi
        '''
        self.del_f = 1/(self.N*self.delta)
        [fx, fy] = tf.meshgrid(
            tf.linspace(-self.N//2,self.N//2-1, self.N) * self.del_f, 
            tf.linspace(-self.N//2,self.N//2-1, self.N) * self.del_f
        )
        f = tf.sqrt(fx**2 + fy**2)
        fm = 5.92 / (self.l0*2*np.pi) if self.l0 != 0 else np.inf 
        f0 = 1/self.L0

        if self.method == 'von karman':
            self.PSD_phi = 0.023 * self.r0**(-5/3) / (f**2 + f0**2)**(11/6)
        elif self.method =='modified von karman':
            self.PSD_phi = 0.023 * self.r0**(-5/3) * tf.exp(-(f/fm)**2) / (f**2 + f0**2)**(11/6)
        else: # kolmogorov
            self.PSD_phi = 0.023 * self.r0**(-5/3) * tf.ones_like(f)

        tf.tensor_scatter_nd_update(self.PSD_phi, [[self.N//2,self.N//2]], [0])

    def ftSubHarmPhaseScreenOneTimeSetup(self):
        '''
        Defines and calculates (self.) del_f_SH, PSD_phi_SH, const_SH_phases
        '''
        for p in range(self.N_p):
            self.del_f_SH = 1 / (self.N_p**(p+1)*(self.N*self.delta))
            [fx_SH, fy_SH] = tf.meshgrid(
                tf.linspace(-self.del_f_SH,self. del_f_SH, self.N_p), 
                tf.linspace(-self.del_f_SH, self.del_f_SH, self.N_p)
            )
            f_SH = tf.sqrt(fx_SH**2 + fy_SH**2)
            fm_SH = 5.92 / (self.l0*2*np.pi) if self.l0 != 0 else np.inf
            f0_SH = 1/self.L0

            if self.method == 'von karman':
                self.PSD_phi_SH = 0.023 * self.r0**(-5/3) / (f_SH**2 + f0_SH**2)**(11/6)
            elif self.method =='modified von karman':
                self.PSD_phi_SH = 0.023 * self.r0**(-5/3) * tf.exp(-(f_SH/fm_SH)**2) / \
                    (f_SH**2 + f0_SH**2)**(11/6)
            else: # kolmogorov
                self.PSD_phi_SH = 0.023 * self.r0**(-5/3)
            
            tf.tensor_scatter_nd_update(self.PSD_phi_SH, [[self.N_p//2,self.N_p//2]], [0])
            
            self.const_SH_phases = tf.TensorArray(tf.complex64, self.N_p**2, clear_after_read=False)
            for i in range(self.N_p**2):
                r,c = [i%self.N_p, i//self.N_p]
                phz = tf.cast(tf.exp(tf.complex(tf.constant(0, tf.float32), \
                    2*np.pi*(fx_SH[r, c]*self.x + fy_SH[r, c]*self.y))), tf.complex64)
                self.const_SH_phases.write(i, phz)

    def show(self, idx=-1, immediate=True, newFig=True):
        if newFig:
            plt.figure()
        plt.imshow(self.phaseScreens[idx])
        plt.set_cmap('gray')    
        plt.colorbar()
        
        if immediate:
            plt.show()
    
    def next(self):
        self.phaseScreens.append(self.subHarmPhaseScreen())
        return self.phaseScreens[-1]

    def subHarmPhaseScreen(self):
        cn = tf.complex(tf.random.normal( (self.N,self.N) ), tf.random.normal( (self.N,self.N) )) * \
            tf.cast(tf.math.sqrt(self.PSD_phi) * self.del_f, tf.complex64)
        phz_hi = tf.math.real(ift2(cn, 1))
        
        phz_lo = tf.zeros_like(phz_hi, tf.complex64)
        for _ in range(self.N_p):
            cn = tf.complex(tf.random.normal( (self.N_p,self.N_p) ), tf.random.normal( (self.N_p,self.N_p) )) * \
                tf.cast(tf.math.sqrt(self.PSD_phi_SH) * self.del_f_SH , tf.complex64)       
            SH = tf.zeros( (self.N,self.N), tf.complex64 )
            for i in range(self.N_p**2): #parallelize
                r,c = [i%self.N_p, i//self.N_p]
                SH = SH + cn[r, c] * self.const_SH_phases.read(i)
            phz_lo = phz_lo + SH
        phz_lo = tf.math.real(phz_lo) - tf.reduce_mean(tf.math.real(phz_lo))

        return phz_lo + phz_hi

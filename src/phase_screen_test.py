from matplotlib import pyplot as plt
import numpy as np
from timeit import default_timer as Timer
import tensorflow as tf
import time

from modules.gaussian_beam import GaussianBeam
from modules.PhaseScreenGenerator import PhaseScreenGenerator
from modules.func_utils import str_fnc2_ft, circ, plot_slice, mesh
from modules.propagation import fresnel_prop_no_scale, ang_spec_multi_prop_no_scale
from modules.phase_screen_generation import ft_sub_harm_phase_screen, ft_phase_screen


def phase_screen_gen():
    '''
    Basic phase screen generation using explicit calls to the FT and SubHarm method functions
    '''
    D = 2
    r0 = 0.1
    N = 256
    L0 = 1e3
    l0 = 0.01

    delta = D/N
    x = np.linspace(-N/2,N/2-1, N) * delta
    y = np.linspace(-N/2,N/2-1, N) * delta

    phz1, phz_hi, phz_lo = ft_sub_harm_phase_screen(r0, N, delta, L0, l0, return_all=True)
    phz2 = ft_phase_screen(r0, N, delta, L0, l0)

    plt.figure()
    plt.set_cmap('gray')    
    plt.imshow(phz1)
    
    plt.figure()
    plt.set_cmap('gray') 
    plt.imshow(phz_hi)
    
    plt.figure()
    plt.set_cmap('gray') 
    plt.imshow(phz_lo)
    
    # plt.colorbar()

    plt.figure()
    plt.imshow(phz2, extent=[x[0], x[-1], y[-1], y[0]])
    plt.set_cmap('gray')    
    plt.colorbar()
    
    plt.show()


def phase_screen_statistics():
    '''
    Calculates the phase screen statistics for a FT and SubHarm method phase screen against theory.
    '''

    def phase_screen_calcs(args):
        N, mask, L0, l0, r0, delta = args
        mask = tf.cast(mask, tf.complex64)

        phz1 = ft_sub_harm_phase_screen(r0, N, delta, L0, l0)
        phz1 = tf.cast(phz1, tf.complex64)
        D1 = str_fnc2_ft(phz1, mask, delta)

        phz2 = ft_phase_screen(r0, N, delta, L0, l0)
        phz2 = tf.cast(phz2, tf.complex64)
        D2 = str_fnc2_ft(phz2, mask, delta)

        return D1, D2

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

    N_phase_screens = 1000

    avgD1 = np.zeros( (N,N) )
    avgD2 = np.zeros( (N,N) )

    start = time.time()
    averages = [phase_screen_calcs( [N, mask, L0, l0, r0, delta] ) for _ in range(N_phase_screens)]
    print(time.time() - start) 

    for i in range(N_phase_screens):
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

    plt.xlabel("Turbulence Strength")
    plt.ylabel("Avg Structure Function")

    plt.show()    


def PhaseScreenGenerator_generation_speed_TEST(n=100):
    '''
    Test to see how fast the PhaseScreenGenerator Class can create 'n' phase screens.
    '''
    N = 1024
    r0 = 0.3
    L0 = 100
    l0 = 0.01
    D = 2
    delta = D/N

    PSG = PhaseScreenGenerator(r0, N, delta, L0, l0)

    PSG.next()
    PSG.show()

    PSG.next()
    PSG.show()

    start = Timer()
    for _ in range(n):
        phz = PSG.next()
    print(Timer()-start)


def PhaseScreenGenerator_single_propagation_TEST():
    '''
    Phase screen generation by the PhaseScreenGenerator Class. Plane wave is propagated through 'nscr' phase screens.
    Output wave intensity and phase are plotted to view results.
    '''
    N = 1024
    r0 = 0.3
    L0 = 100
    l0 = 0.01
    D = 2
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
    Uin = circ(x1, y1, D*0.9)

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

    plt.figure(1)
    plt.imshow(np.abs(Uout)**2)
    # plt.set_cmap('gray')    
    plt.colorbar()

    plt.figure(2)
    plt.imshow(np.angle(Uout))
    # plt.set_cmap('gray')    
    plt.colorbar()

    plt.figure(3)
    plt.imshow(np.abs(Uin)**2)
    # plt.set_cmap('gray')    
    plt.clim(0,2)
    plt.colorbar()

    plt.show()


def rectangular_phase_screen_test():
    '''
    Generating a rectangular phase screen. -- NOTE -- This is not robust and needs to be statistically verified. 
    Current implementation is slow and not efficient. Find Phase Screen Generation papers in the ../documents folder
    '''
    N = 4096
    M = 1024
    r0 = 0.1
    L0 = 100
    l0 = 0.01
    D = 2
    delta = D/N
    PSG = PhaseScreenGenerator(r0, (N,M), delta, L0, l0)
    rect_phase_screen = PSG.next()

    plt.imshow(rect_phase_screen)
    plt.set_cmap('gray')
    plt.show()


# slow because converting tensors to variables?
# disables eager exec
# @tf.function
def tensorflow_test(n_scr):
    '''
    Preliminary tests done to experiment with Tensorflow optimization and parallelization techniques
    '''
    D = 2
    r0 = 0.1
    N = 1024
    L0 = 100
    l0 = 0.01

    delta = D/N
    x = np.linspace(-N/2,N/2-1, N) * delta
    y = np.linspace(-N/2,N/2-1, N) * delta

    # n_scr = 10
    # ft_phase_screen(r0, N, delta, L0, l0) # execute once for better benchmarking accuracy
    # ft_sub_harm_phase_screen(r0, N, delta, L0, l0)

    # out = tf.TensorArray(tf.float32, size=n_scr)
    out = np.empty(n_scr, tf.Tensor)
   
    # start = Timer()
    # i = tf.constant(0)
    # cond = lambda i, out: tf.less(i, n_scr)
    # def body(i, out: tf.TensorArray):
    #     curr_phz = ft_sub_harm_phase_screen(r0, N, delta, L0, l0)
    #     curr_i = i
    #     return (tf.add(i,1), out.write(curr_i, curr_phz))
    # i, out = tf.while_loop(cond, body, [i, out], parallel_iterations=4)
    # # tf.print(out.stack())
    # # print(Timer()-start)
    # # phz = out.stack()
    # # phz = out.read(0)
    # # with tf.compat.v1.Session() as sess:
    # #     phzs = sess.run(out.read(0))
    # tf.print(Timer()-start)

    start = Timer()
    for i in range(n_scr):
    #    out = out.write(i, ft_sub_harm_phase_screen(r0, N, delta, L0, l0))
       out[i] = ft_sub_harm_phase_screen(r0, N, delta, L0, l0)
    print(Timer()-start)
    # phz = out[0]
    # phz = out.stack() # careful if (n_scr * N^2) exeeds memory
    # print(Timer()-start)
    # tf.print(out.stack())

    return out, x, y


def main():
    # phase_screen_gen()

    # phase_screen_statistics()

    # phase_screen_prop()

    # tensorflow_test(100)

    PhaseScreenGenerator_generation_speed_TEST()

    # PhaseScreenGenerator_single_propagation_TEST()

    # rectangular_phase_screen_test()

if __name__ == '__main__':
    main()

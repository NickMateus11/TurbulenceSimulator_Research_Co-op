import numpy as np
from modules.propagation import fraunhofer_prop, fresnel_prop
from modules.func_utils import *


def main():
    N = 128
    L = 5e-3
    delta = L/N
    D = 1e-3
    wvl = 600e-9
    # k = 2*np.pi/wvl
    dz = 1

    x = np.linspace(-L/2,L/2-1*delta,N)
    y = np.linspace(-L/2,L/2-1*delta,N)
    [x1,y1] = np.meshgrid(x,y)
    Uin = circ(x1,y1,D)
    [Uout,x2,y2] = fraunhofer_prop(Uin,wvl,delta,dz)
    [Uout2,x3,y3] = fresnel_prop(Uin,wvl,delta,delta,dz)

    mesh(x1,y1,np.abs(Uin)**2)
    mesh(x2,y2,np.abs(Uout)**2)
    mesh(x3,y3,np.abs(Uout2)**2)

    plt.figure(4)
    plt.plot(x3[N//2],np.abs(Uout2[N//2])**2)
    print(x3[N//2])

    plt.show()


if __name__ == '__main__':

    main()

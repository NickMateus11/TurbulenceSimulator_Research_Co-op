import numpy as np
from func_utils import ft2, ift2


PI = np.pi


def fraunhofer_prop(Uin, wvl, delta, dz):
    N = np.size(Uin,0) # assume square
    k = 2*PI / wvl
    fX = np.linspace(-1/delta, 1/delta, N)
    fY = np.linspace(-1/delta, 1/delta, N)
    [x2, y2] = np.meshgrid(wvl*dz*fX, wvl*dz*fY)
    Uout = np.exp(1j*k/(2*dz)*(x2**2+y2**2)) / (1j*wvl*dz) * ft2(Uin, delta)
    return Uout,x2,y2
    

def fresnel_prop(Uin, wvl, delta1, delta2, dz):
    #using angular spectrum propagation technique
    N = np.size(Uin,0)
    k = 2*PI / wvl
    [x1,y1] = np.meshgrid(
        delta1 * np.linspace(-N//2,N//2-1,N),
        delta1 * np.linspace(-N//2,N//2-1,N),
    )
    delta_f1 = 1/(N*delta1) 
    [fX,fY] = np.meshgrid(
        delta_f1 * np.linspace(-N//2,N//2-1,N),
        delta_f1 * np.linspace(-N//2,N//2-1,N),
    )
    m = delta2/delta1
    [x2,y2] = np.meshgrid(
        delta2 * np.linspace(-N//2,N//2-1,N),
        delta2 * np.linspace(-N//2,N//2-1,N),
    )
    Q1 = np.exp(1j*k/2 * (1-m)/dz * (x1**2+y1**2))
    Q2 = np.exp(-1j*PI**2 * 2*dz/(m*k) * (fX**2+fY**2))
    Q3 = np.exp(1j*k/2 * (m-1)/(m*dz) * (x2**2+y2**2))
    Uout = Q3 * ift2(Q2*ft2(Q1*Uin/m, delta1), delta_f1)
    return Uout, x2, y2


def fresnel_prop_no_scale(Uin, wvl, delta1, dz):
    #simplified fresnel - no spatial scaling from in->out spacial coords
    #using angular spectrum propagation technique
    N = np.size(Uin,0)
    k = 2*PI / wvl
    [x1,y1] = np.meshgrid(
        delta1 * np.linspace(-N//2,N//2-1,N),
        delta1 * np.linspace(-N//2,N//2-1,N),
    )
    delta_f1 = 1/(N*delta1)
    [fX,fY] = np.meshgrid(
        delta_f1 * np.linspace(-N//2,N//2-1,N),
        delta_f1 * np.linspace(-N//2,N//2-1,N),
    )
    Q = np.exp(-1j*k*dz) * np.exp(-1j*PI*wvl*dz*(fX**2 + fY**2))
    Uout = ift2(Q * ft2(Uin, delta1), delta_f1)
    return Uout, x1, y1
    
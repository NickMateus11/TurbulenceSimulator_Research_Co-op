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


def ang_spec_multi_prop(Uin, wvl, delta1, deltan, z, t):
    # evaluate fresnel diffraction integral through ang spec method (ch9), used for multiple partial props
    N = np.size(Uin, 0)
    nx, ny = np.meshgrid(
        np.linspace(-N//2,N//2-1,N),
        np.linspace(-N//2,N//2-1,N),
    )    
    k = 2*PI/wvl
    nsq = nx**2 + ny**2
    w = 0.47*N
    sg = np.exp(-nsq**8 / w**16)

    z = np.array([0, *z])
    n = len(z)

    Delta_z = z[1:] - z[:n-1] 

    alpha = z / z[-1]
    delta = (1-alpha) * delta1 + alpha * deltan
    m = delta[1:] / delta[:n-1]
    x1 = nx * delta[0]
    y1 = ny * delta[0]
    r1sq = x1**2 + y1**2
    Q1 = np.exp(1j*k/2*(1-m[0]) / Delta_z[0] * r1sq)
    Uin = Uin * Q1 * t[0,:,:]

    for idx in range(n-1):
        deltaf = 1/ (N*delta[idx])
        fX = nx * deltaf
        fY = ny * deltaf 
        fsq = fX**2 + fY**2
        Z = Delta_z[idx]
        Q2 = np.exp(-1j*(PI**2)*2 * Z / (m[idx] * k) * fsq)
        Uin = sg * t[idx+1,:,:] * ift2(Q2 * ft2(Uin/m[idx], delta[idx]), deltaf)
    
    xn = nx * delta[-1]
    yn = ny * delta[-1]
    rnsq = xn**2 + yn**2
    Q3 = np.exp(1j*k/2*(m[-1]-1) / (m[-1]*Z) * rnsq)
    Uout = Q3 * Uin

    return Uout, xn, yn
    
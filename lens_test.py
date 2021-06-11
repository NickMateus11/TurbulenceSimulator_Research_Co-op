from matplotlib import pyplot as plt
import numpy as np
from gaussian_beam import GaussianBeam
from propagation import fresnel_prop, fresnel_prop_no_scale
from func_utils import plot_wave_slice


N = 512
wvl = 600e-9
w0 = 1e-3
dz = 5
L = 5*w0
delta = L/N


G = GaussianBeam(wvl, w0, 0, delta, N)
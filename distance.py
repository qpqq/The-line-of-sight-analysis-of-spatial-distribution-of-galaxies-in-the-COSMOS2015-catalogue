"""
Fractal Dimension Estimation
© Stanislav Shirokov, 2014-2020
"""

from math import sqrt
from scipy import integrate

omega_m = 0.3
omega_v = 0.7
c = 299792.458
H = 72


def r_comoving(z):
    return c / H * integrate.quad(lambda y: 1 / sqrt(omega_v + omega_m * (1 + y) ** 3), 0, z)[0]


def r_proper(z):
    return c * (z + 1) / H * integrate.quad(lambda y: 1 / sqrt(omega_v + omega_m * (1 + y) ** 3), 0, z)[0]

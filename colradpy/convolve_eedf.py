"""Rate coefficients of electron collisions from their cross-sections, averaged over a thermal electron
energy distribution: Maxwell-Boltzmann, or Maxwell-Juttner (relativistic), which it becomes when kT
is a sizeable fraction of the electron rest energy (2% at 10 keV).

The rate coefficient of a process with threshold E0 is

    q(T) = integral from E0 to infinity of sigma(E) v(E) f(E, T) dE,

with E the kinetic energy of the incident electron, v its speed and f the distribution. It is
integrated in u = (E - E0)/kT by Gauss-Laguerre quadrature, so the nodes start at each threshold and
follow the Boltzmann factor whatever E0/kT: the integrand is exp(-u) times a function of u that is
smooth for cross-sections that are. The reduced rate q(T) exp(E0/kT), from which ColRadPy makes
effective collision strengths and reduced ionization rates, does not underflow at low temperature.
"""

import numpy as np
from scipy import constants
from scipy.special import kve

#electron rest energy (eV)
ELECTRON_REST_ENERGY = constants.physical_constants['electron mass energy equivalent in MeV'][0]*1e6
#speed of light (cm s-1)
SPEED_OF_LIGHT = constants.c*100
#Gauss-Laguerre nodes of the integrals, enough for 1e-6 on smooth cross-sections and about 1e-3
#on cross-sections interpolated between points a factor of 2 apart
QUADRATURE_NODES = 64


def maxwellian(energy, temp, relativistic=True, reduced=False):
    """Thermal electron energy distribution (eV-1), normalized over the kinetic energy.

    :param energy: kinetic energies of the electrons (eV)
    :type energy: float array

    :param temp: temperature kT (eV), broadcast against energy
    :type temp: float or float array

    :param relativistic: Maxwell-Juttner distribution instead of Maxwell-Boltzmann
    :type relativistic: bool

    :param reduced: leave out the Boltzmann factor exp(-E/kT)
    :type reduced: bool

    :returns: float array --the distribution (eV-1)
    """
    energy = np.asarray(energy, dtype=float)
    temp = np.asarray(temp, dtype=float)
    boltzmann = 1. if reduced else np.exp(-energy/temp)
    if(not relativistic):
        return 2*np.sqrt(energy/np.pi)*temp**-1.5*boltzmann
    #gamma^2 beta exp(-gamma/theta)/(theta K2(1/theta)) per unit gamma, with the scaled Bessel
    #function kve(2, x) = K2(x) exp(x), which does not underflow at low temperature
    theta = temp/ELECTRON_REST_ENERGY
    gamma = 1 + energy/ELECTRON_REST_ENERGY
    beta = speed(energy)/SPEED_OF_LIGHT
    return gamma**2*beta/(theta*kve(2, 1/theta)*ELECTRON_REST_ENERGY)*boltzmann


def speed(energy, relativistic=True):
    """Speed (cm s-1) of electrons with the kinetic energies energy (eV)."""
    energy = np.asarray(energy, dtype=float)
    if(not relativistic):
        return SPEED_OF_LIGHT*np.sqrt(2*energy/ELECTRON_REST_ENERGY)
    #beta = sqrt(1 - 1/gamma^2), without the cancellation at low energy
    return SPEED_OF_LIGHT*np.sqrt(energy*(energy + 2*ELECTRON_REST_ENERGY))/(energy + ELECTRON_REST_ENERGY)


def rate_coefficients(cross_section, threshold, temps, relativistic=True, reduced=False,
                      nodes=QUADRATURE_NODES):
    """Rate coefficients (cm3 s-1) of a cross-section averaged over the thermal distribution.

    :param cross_section: cross-section (cm2) at an array of incident electron energies (eV), zero
                          below the threshold
    :type cross_section: callable

    :param threshold: threshold energy E0 (eV), 0 for recombination
    :type threshold: float

    :param temps: temperatures kT (eV)
    :type temps: float array

    :param relativistic: Maxwell-Juttner distribution and relativistic speeds
    :type relativistic: bool

    :param reduced: return q exp(E0/kT), which does not underflow at low temperature
    :type reduced: bool

    :param nodes: number of Gauss-Laguerre nodes
    :type nodes: int

    :returns: float array --the rate coefficients at temps
    """
    temps = np.atleast_1d(np.asarray(temps, dtype=float))
    u, weights = np.polynomial.laguerre.laggauss(nodes)
    energy = threshold + np.outer(temps, u)                         #(temperature, node)
    integrand = (np.asarray(cross_section(energy), dtype=float)*speed(energy, relativistic)*
                 maxwellian(energy, temps[:, None], relativistic, reduced=True))
    rates = temps*(integrand @ weights)
    if(not reduced):
        with np.errstate(under='ignore'):
            rates = rates*np.exp(-threshold/temps)
    return rates


def resonance_rate_coefficients(strength, energy, temps, relativistic=True):
    """Rate coefficients (cm3 s-1) of a resonant capture, a cross-section strength (cm2 eV) at one
    energy (eV) of the incident electron, such as dielectronic capture into an autoionizing level:
    strength v(E) f(E, T).

    :param strength: integral of the cross-section over the resonance (cm2 eV)
    :type strength: float

    :param energy: energy of the captured electron (eV)
    :type energy: float

    :param temps: temperatures kT (eV)
    :type temps: float array

    :param relativistic: Maxwell-Juttner distribution and relativistic speed
    :type relativistic: bool

    :returns: float array --the rate coefficients at temps
    """
    temps = np.atleast_1d(np.asarray(temps, dtype=float))
    with np.errstate(under='ignore'):
        return strength*speed(energy, relativistic)*maxwellian(energy, temps, relativistic)

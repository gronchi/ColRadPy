"""Boltzmann level populations and Saha ionization balance in LTE."""

import numpy as np
from scipy import constants
from scipy.special import expit


EV_PER_WAVENUMBER = constants.h * constants.c * 100 / constants.e


def _levels(w, energy):
    """Validate level angular momenta and energies (in cm^-1)."""
    w = np.asarray(w, dtype=float)
    energy = np.asarray(energy, dtype=float)
    if (w.ndim != 1 or energy.ndim != 1 or w.size == 0 or
            w.shape != energy.shape or not np.all(np.isfinite(w)) or
            not np.all(np.isfinite(energy)) or np.any(w < 0)):
        raise ValueError("w and energy must be nonempty, equal-length arrays of finite levels; w must be nonnegative")
    return w, energy


def _temperatures(temp):
    temp = np.atleast_1d(np.asarray(temp, dtype=float))
    if temp.ndim != 1 or temp.size == 0 or not np.all(np.isfinite(temp)) or np.any(temp <= 0):
        raise ValueError("temp must contain positive, finite temperatures in eV")
    return temp


def boltzmann_population(w, energy, temp, norm_tot=False):
    """Return LTE level populations with shape ``(levels, temperatures)``.

    ``w`` contains level angular momenta J, so each statistical weight is
    ``2*J+1``. ``energy`` is in cm^-1 and ``temp`` is in eV. Energies are
    measured from the first (ground) level. By default populations are
    relative to that level (its population is one); with ``norm_tot=True``
    each temperature column sums to one.
    """
    w, energy = _levels(w, energy)
    temp = _temperatures(temp)
    weights = 2 * w + 1
    populations = (weights / weights[0])[:, None] * np.exp(
        -(energy - energy[0])[:, None] * EV_PER_WAVENUMBER / temp[None, :]
    )
    if norm_tot:
        populations /= populations.sum(axis=0)
    return populations


def saha_population(ci, w_z, w_z1, energy_z, energy_z1, ftemp, ne,
                    norm_tot=False):
    """Return the LTE ratio ``N_(z+1)/N_z`` at each temperature.

    ``ci`` is the ionization energy between the two ground levels in cm^-1;
    level energies are in cm^-1, electron temperatures ``ftemp`` in eV, and
    electron densities ``ne`` in cm^-3 (scalar or one per temperature).
    ``w_z`` and ``w_z1`` contain level J values. Both stage partition
    functions include every supplied level. With ``norm_tot=True``, return
    the fraction ``N_(z+1)/(N_z + N_(z+1))`` instead of the stage ratio.
    """
    temp = _temperatures(ftemp)
    density = np.asarray(ne, dtype=float)
    if (density.ndim > 1 or (density.ndim == 1 and density.size != temp.size) or
            not np.all(np.isfinite(density)) or np.any(density <= 0)):
        raise ValueError("ne must be positive and finite, with one value per temperature or a scalar")
    ci = float(ci)
    if not np.isfinite(ci) or ci < 0:
        raise ValueError("ci must be a nonnegative, finite ionization energy in cm^-1")

    partition_z = boltzmann_population(w_z, energy_z, temp).sum(axis=0)
    partition_z1 = boltzmann_population(w_z1, energy_z1, temp).sum(axis=0)
    ground_weights = (2 * np.asarray(w_z1)[0] + 1) / (2 * np.asarray(w_z)[0] + 1)

    # 2 / lambda_e^3 is in m^-3; divide by 1e6 for cm^-3.
    log_ratio = (np.log(2 / 1e6) + 1.5 * np.log(2 * np.pi * constants.m_e *
                 constants.e * temp / constants.h**2) + np.log(ground_weights) +
                 np.log(partition_z1) - np.log(partition_z) -
                 ci * EV_PER_WAVENUMBER / temp - np.log(density))
    if norm_tot:
        return expit(log_ratio)
    with np.errstate(over="ignore", under="ignore"):
        return np.exp(log_ratio)

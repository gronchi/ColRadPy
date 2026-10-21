################################################################################
# file name         : colrad.py
# author            : Curt Johnson
# description       : This code solves the collsion radiative matrix
# version           : 0.3
# python version    : Python 3.6.5 ipython 6.4.0 anaconda
# dependencies      : read_adf04_py3_class.py, numpy, scipy interpolate,
#                   : matplotlib, re, sys, r8yip, r8necip, ecip_rates
#                   : burgess_tully_rates, collections
# license           : to be freed after code is stable and papers are published
#
# This code originally went through different file names and versions that
# incorprated different functionalitiy before it was under version control.
# These are included in the repos for historical and bug tracking reasons.
# The order went adf04_read.py, adf04_read1.py, adf04_dens_grid.py,
# adf04_dens_temp_grid_tmp.py, colrad.py
# 
# The code solves the colisional radiative problem see Stuart loch's 
# spectroscopy class notes or Curt Johnson's upcoming thesis for a derivation
# of the CR matrix.
#
# The read_adf04.py will read adf04 files and produces the dictionary with data
# Takes a density array, temperature array, metastable number, 
#
#
#
################################################################################

import numpy as np
# Prefer current NumPy APIs, with fallbacks for older releases.
try:
    from numpy import isin as _isin
except ImportError:
    from numpy import in1d as _isin

try:
    from numpy import trapezoid as _trapezoid
except ImportError:
    from numpy import trapz as _trapezoid

import re
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import constants
import sys
sys.path.append('./')
from colradpy.r8necip import *
from colradpy.read_adf04_py3_class import *
from colradpy.ecip_rates import *
from colradpy.burgess_tully_rates import *
from colradpy.split_multiplet import *
from colradpy.nist_read_txt import *
from colradpy.solve_matrix_exponential import *
from colradpy.colradpy_utility import *
from colradpy.write_adf15 import *
if( (sys.version_info[0] ==3) & (sys.version_info[1] >=10)):
    from collections.abc import Mapping
else:
    from collections import Mapping
    
from matplotlib import rc,rcParams
from fractions import Fraction
import os
import warnings
import pathlib
import subprocess

def get_git_hash(base_path=None):
    base_path = pathlib.Path(base_path or __file__).resolve().parent

    # Try using git directly if available (works even in subfolders)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=base_path,
            capture_output=True,
            text=True,
            check=True)
        return result.stdout.strip()
    except Exception:
        pass  # fall through if git not installed or .git missing
    # Try to read from .git manually (works in shallow clone)
    git_dir = base_path / ".git"
    if git_dir.exists():
        try:
            with open(git_dir / "HEAD", "r") as head:
                ref = head.readline().strip().split(" ")[-1]
            ref_file = git_dir / ref
            if ref_file.exists():
                return ref_file.read_text().strip()
        except Exception:
            pass
    # Try to infer from environment (e.g. CI/CD or baked-in version)
    env_hash = os.environ.get("GIT_HASH")
    if env_hash:
        return env_hash.strip()
    # Try to read from an embedded version file (zip or packaged)
    version_file = base_path / "VERSION"
    if version_file.exists():
        return version_file.read_text().strip()
    # Nothing worked
    return "unknown"

#Letters of the total orbital angular momentum L = 0, 1, 2, ... (J is skipped)
L_LETTERS = 'SPDFGHIKLMNOQRTUV'

#eV per cm-1 (h c / e, 1/8065.544), converts level energies and ionization potentials in the
#Boltzmann factors of excitation, ionization and three-body recombination
EV_PER_WAVENUMBER = constants.h*constants.c*100/constants.e

#Photon energy in J of a transition of 1 cm-1 (h c)
J_PER_WAVENUMBER = constants.h*constants.c*100

#Largest reduced ionization rate exp(dE/kT)*S (cm3 s-1) taken from an adf04 file; physical
#values are below about 1e-7 cm3 s-1, and larger ones are placeholders that some files hold
#where the rate underflowed at their lowest temperatures.
MAX_REDUCED_IONIZATION = 1e-6
#Largest recombination rate coefficient (cm3 s-1) taken from the R lines of an adf04 file;
#partial radiative and dielectronic coefficients stay below about 1e-9 cm3 s-1.
MAX_RECOMBINATION = 1e-7
#Steepest decrease d ln(rate)/d ln(T) of recombination rate coefficients: radiative recombination
#into levels bound by less than kT and dielectronic recombination above its resonances fall as
#T^-1.5, so points at the lowest temperatures of a file that fall faster towards the higher
#temperatures, with a margin, are artifacts (OpenADAS cop98#10_ls#ne1 'R  2  +3' gives
#2.56e-8 cm3 s-1, then 1e-30; cop98#10_ls#ne7 'R  3  +1' 5.90e-10, then 7.35e-14).
STEEPEST_RECOMBINATION = -2.
#Steepest decrease of reduced ionization rates exp(dE/kT)*S, which rise as T^1/2 near threshold
#and fall as about T^-1/2 far above it (OpenADAS cop98#10_ls#ne3 'S  1  +1' gives 1.51e-7,
#7.04e-8, then 8.50e-9 cm3 s-1 at its lowest temperatures).
STEEPEST_REDUCED_IONIZATION = -1.
#Exponents of the power laws (log-log interpolation) that the rates follow outside the
#temperatures of a file: recombination falls at most as T^-1.5 and does not grow; reduced
#ionization above them stays between T^-1 and T^1/2 (below them it follows its two lowest
#points, or is held constant when they fall)
RECOMBINATION_EXPONENTS = (-1.5, 0.)
REDUCED_IONIZATION_EXPONENTS_BELOW = (0., None)
REDUCED_IONIZATION_EXPONENTS_ABOVE = (-1., 0.5)


def low_temperature_artifacts(temps, rates, steepest):
    """Points at the lowest temperatures of each row of rates (one row for each transition,
       temperatures temps sorted) that fall towards a later point faster than T^steepest, which
       no rate of the kind does: they are left out from the lowest point up to the first one
       that does not. Points that are not positive are not compared.

    :param temps: temperatures, increasing
    :type temps: float arr

    :param rates: rates, one row for each transition
    :type rates: float arr

    :param steepest: steepest physical d ln(rate)/d ln(T), negative
    :type steepest: float

    :returns: bool arr -- True for the points that are left out
    """
    artifacts = np.zeros(np.shape(rates), dtype=bool)
    log_temps = np.log(temps)
    for i, row in enumerate(np.asarray(rates)):
        positive = np.where(row > 0)[0]
        for n, k in enumerate(positive[:-1]):
            later = positive[n+1:]
            slopes = (np.log(row[later]) - np.log(row[k]))/(log_temps[later] - log_temps[k])
            if(slopes.min() >= steepest):
                break
            artifacts[i, k] = True
    return artifacts


def parent_statistical_weight(term):
    """Statistical weight of a parent term from the first line of an adf04 file.
       LS parents are written (2S+1)L, e.g. 2S or 3P, with weight (2S+1)(2L+1).
       IC parents add J, e.g. 3P2.0 or 2P1.5 (or 3/2), with weight 2J+1.
       See the ADF04 specification (ADAS appxa-04).

    :param term: The parent term, without parentheses
    :type term: str

    :returns: float --statistical weight

    """
    match = re.fullmatch(r'(\d+)([A-Za-z])(\d+(?:\.\d*)?|\d+/\d+)?', str(term).strip())
    if(match is None or match.group(2).upper() not in L_LETTERS):
        raise ValueError('parent term ' + repr(term) + ' is not (2S+1)L or (2S+1)LJ')
    if(match.group(3) is None):
        return float(int(match.group(1))*(2*L_LETTERS.index(match.group(2).upper())+1))
    weight = 2*float(Fraction(match.group(3)) if '/' in match.group(3) else match.group(3))+1
    if(weight < 1 or weight != round(weight)):
        raise ValueError('parent term ' + repr(term) + ' has J that is not a whole or half integer')
    return weight


def nist_energies_dir():
    """Directory of the NIST level energy files, with a trailing separator: atomic/nist_energies
       inside the installed package, or at the top of a source checkout (and an editable install).

    :returns: str --the directory
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (here, os.path.dirname(here)):
        directory = os.path.join(base, 'atomic', 'nist_energies')
        if(os.path.isdir(directory)):
            return directory + os.sep
    raise FileNotFoundError('the NIST level energies (atomic/nist_energies) are neither in the colradpy '
                            'package nor next to it in ' + os.path.dirname(here))


def parent_levels(ion_pot, ion_term, energy, S, L, w):
    """Levels of the next ion that are the parents on the first line of an adf04 file. Each
       parent is the level with its term (2S+1, L and, for IC terms, J) whose energy is
       closest to the parent's energy above the lowest parent, ion_pot[p] - ion_pot[0].

    :param ion_pot: ionization potentials of the parents (cm-1)
    :type ion_pot: float array

    :param ion_term: terms of the parents, e.g. 2P or 3P2.0
    :type ion_term: str array

    :param energy: level energies of the next ion (cm-1)
    :type energy: float array

    :param S: 2S+1 of the levels of the next ion
    :type S: int array

    :param L: L of the levels of the next ion
    :type L: int array

    :param w: J (IC files) or the term's (statistical weight - 1)/2 (LS files) of the levels
    :type w: float array

    :returns: int array --the parent levels (0 based)
    """
    energy, S, L, w = map(np.asarray, (energy, S, L, w))
    levels = []
    for p in range(len(ion_pot)):
        term = str(ion_term[p]).strip()
        match = re.fullmatch(r'(\d+)([A-Za-z])(\d+(?:\.\d*)?|\d+/\d+)?', term)
        if(match is None or match.group(2).upper() not in L_LETTERS):
            raise ValueError('parent term ' + repr(term) + ' is not (2S+1)L or (2S+1)LJ')
        same = (S == int(match.group(1))) & (L == L_LETTERS.index(match.group(2).upper()))
        if(match.group(3) is not None):
            j = float(Fraction(match.group(3)) if '/' in match.group(3) else match.group(3))
            same &= np.isclose(w, j)
        target = ion_pot[p] - ion_pot[0]
        candidates = np.where(same)[0]
        if(candidates.size == 0):
            raise ValueError('no level of the next ion has the term ' + term + ' of parent ' + str(p+1))
        k = candidates[np.argmin(np.abs(energy[candidates] - target))]
        if(abs(energy[k] - target) > max(0.05*target, 1000.)):
            raise ValueError('parent ' + str(p+1) + ' (' + term + ', ' + str(target) + ' cm-1 above the ' +
                             'lowest parent) matches no level of the next ion; the closest ' + term +
                             ' level is at ' + str(energy[k]) + ' cm-1')
        levels.append(k)
    if(len(set(levels)) != len(levels)):
        raise ValueError('parents ' + str(list(ion_term)) + ' match the same levels ' + str(levels))
    return np.array(levels, dtype=int)


def strip_closed_shells(conf, closed_shells, nist_ground):
    """Removes the closed subshells that lead a configuration, e.g. 1s2.2s2.2p6.3s.3p,
       and are not in the NIST ground configuration, e.g. 3s2, because NIST leaves out
       those inner subshells.

    :param conf: configuration with subshells separated by '.'
    :type conf: str

    :param closed_shells: closed subshells such as 1s2
    :type closed_shells: array

    :param nist_ground: NIST ground configuration
    :type nist_ground: str

    :returns: str --configuration in the NIST form

    """
    ground_shells = str(nist_ground).split('.')
    shells = str(conf).split('.')
    while(len(shells) > 1 and shells[0] in closed_shells and shells[0] not in ground_shells):
        shells = shells[1:]
    return '.'.join(shells)


#Wavelengths below this (nm) are kept in vacuum, as in the NIST ASD. The Edlen formula
#is not valid in the vacuum ultraviolet and has poles at 87.7 and 160.3 nm.
MIN_AIR_WAVELENGTH = 200.


def air_refractive_index(wave_vac, pressure_pa=101325., temperature_c=15.,
                         water_vapor_pressure_pa=0.):
    """Refractive index of air from the 1993 revised Edlen formula, as in AtomSpect.
       See Steck, Rubidium 87 D Line Data, Eq. (3): https://steck.us/alkalidata/rubidium87numbers.pdf

       The defaults are dry standard air (101325 Pa and 15 C). The quoted 3-sigma
       uncertainty in the index is 3e-8 at 350-650 nm under laboratory conditions.

    :param wave_vac: The wavelengths in nm in vacuum
    :type wave_vac: array

    :param pressure_pa: Air pressure in Pa
    :type pressure_pa: float

    :param temperature_c: Air temperature in degrees Celsius
    :type temperature_c: float

    :param water_vapor_pressure_pa: Partial pressure of water vapor in Pa (not relative humidity)
    :type water_vapor_pressure_pa: float

    :returns: array --refractive index of air

    """
    kappa_squared = (1000. / np.asarray(wave_vac, dtype=float))**2
    dispersion = 8342.54 + 2406147. / (130. - kappa_squared) + 15998. / (38.9 - kappa_squared)
    pressure_temperature = (pressure_pa / 96095.43
                            * (1. + 1e-8 * (0.601 - 0.00972 * temperature_c) * pressure_pa)
                            / (1. + 0.0036610 * temperature_c))
    humidity = water_vapor_pressure_pa * (0.037345 - 0.000401 * kappa_squared)
    return 1. + (dispersion * pressure_temperature - humidity) * 1e-8


def vac_to_air(wave_vac, pressure_pa=101325., temperature_c=15.,
               water_vapor_pressure_pa=0., min_air_wavelength=MIN_AIR_WAVELENGTH):
    """This function converts the vacuum wavelength of spectral lines to
       the air wavelength, lambda_air = lambda_vac / n. The wavelength is the
       difference in energy levels of the upper and lower transition, which
       corresponds to the vacuum wavelength.

       Wavelengths below min_air_wavelength are returned unchanged (in vacuum).

    :param wave_vac: The wavelengths in nm in vacuum
    :type wave_vac: array

    :param min_air_wavelength: Shortest vacuum wavelength in nm that is converted to air
    :type min_air_wavelength: float

    The other parameters are those of air_refractive_index.

    :returns: array --wavelengths in nm, in air from min_air_wavelength up

    """
    wave_vac = np.asarray(wave_vac, dtype=float)
    in_air = wave_vac >= min_air_wavelength
    #evaluate the index away from its poles for wavelengths kept in vacuum
    index = air_refractive_index(np.where(in_air, wave_vac, 500.), pressure_pa,
                                 temperature_c, water_vapor_pressure_pa)
    return np.where(in_air, wave_vac / index, wave_vac)


def convert_to_air(lam):
    """Returns the refractive index of standard air (not a wavelength) at the vacuum
       wavelengths lam in nm. Kept for compatibility, use vac_to_air for wavelengths.

    :param lam: The wavelengths in nm in vacuum
    :type lam: array

    :returns: array --refractive index of air

    """
    return air_refractive_index(lam)
def myprint(d):
    for k, v in d.items():
        if isinstance(v, dict):
            myprint(v)
        else:
            if(type(v) == np.ndarray):
                if(v.dtype.kind=='U'):
                    print(k)
                    d[k] = np.array([s.encode() for s in np.copy(d[k])])
    return d


class colradpy():
    """The ColRadPy class, this class will store data and carry out calculation to solve
       the collisional radiative set of equations. A full tutorial is provided in the 
       documentation.
   
    Args:
      :param fil: the file path to the input file
      :type fil: string

      :param metas: array of the levels that metastable
      :type metas: integer array

      :param temp_grid: array of temperature for calculation (eV)
      :type metas: float array

      :param dens_grid: array of temperature for calculation (cm-3)
      :type metas: float array

      :param htemp_grid: Temperature grid of thermal CX hydrogen (eV)
      :type htemp_grid: float array

      :param hdens_grid: Density grid of thermal CX hydrogen (cm-3)
      :type hdens_grid: float array

      :param use_ionization: Flag to turn on ionization in calculation, default = True
      :type metas: bool

      :param suppliment_with_ecip: Flag to turn on ECIP supplimentation to ionization rates, default = Tue
      :type metas: bool

      :param use_recombination_three_body: Flag to turn 3 body recombination on,  default = True
      :type metas: bool

      :param use_recombination: Flag to turn recombination on, default = True
      :type metas: bool

      :param td_t: Time array for time dependent solution to CR equations
      :type metas: float array

      :param td_n0: Initial populations of levels at time t=0 for TD CR equations
      :type metas: float array

      :param td_source: source term for populations in the TD CR equations
      :type metas: float array

      :param default_pop_norm: Normalize to population within the charge state 
      :type default_pop_norm: bool

      :param temp_dens_pair: use temperature density pair array instead of temperature density grids
      :type temp_dens_pair: bool

      :param rate_interp_ion: Interpolation of the reduced ionization rates of the adf04 file over
                              temperature: a SciPy kind ('slinear', 'quadratic', 'cubic', ...) of
                              the rates on a linear temperature axis, 'log_' and a kind on a log
                              temperature axis, or 'loglog_' and a kind for the log of the rates on
                              a log temperature axis, between the positive values of each
                              transition and as a power law outside them (default
                              'loglog_slinear'). Rates of the other options are extrapolated
                              linearly above the file temperatures and as a power law below them.
      :type rate_interp_ion: string

      :param rate_interp_recomb: Interpolation of the recombination rates, as rate_interp_ion
                                 (default 'loglog_slinear'); the other options extrapolate them
                                 linearly on both sides, without negative values
      :type rate_interp_recomb: string

      :param rate_interp_col: Interpolation type for excitation, a kind or 'log_' and a kind
      :type rate_interp_col: string

      :param rate_interp_cx: Interpolation type for charge exchange, a kind or 'log_' and a kind
      :type rate_interp_cx: string

      :param use_cx: Use charge exchange in the calculation
      :type use_cx: bool

      :param scale_file_ioniz: Scale ionization in the file
      :type scale_file_ioniz: bool

      :param ne_tau: n_e*tau values for time dependent calc to be evaluated at can not be defined at the same time as td_t (cm-3 s)
      :type ne_tau: float array

    """
    
    def __init__(self,fil,metas=np.array([]),temp_grid=np.array([]),electron_den=np.array([]),
                 htemp_grid = np.array([]), hdens_grid = np.array([]),
                 use_ionization=True,suppliment_with_ecip=True,use_recombination_three_body=True,
                 use_recombination = True, td_t = np.array([]), td_n0=np.array([]),td_source=np.array([]),
                  default_pop_norm=True,temp_dens_pair=False,rate_interp_ion = 'loglog_slinear',
                 rate_interp_recomb='loglog_slinear',rate_interp_col='log_slinear',
                 rate_interp_cx='log_quadratic',use_cx=False,scale_file_ioniz=False,
                 ne_tau = -1):
        
        """The initializing method. Sets up the nested list for data storage and starts to populate with user data
           as well as reading in the adf04 file

        """

        self.data = {}
        self.processed = {} 
        self.data['user'] = {}
        self.data['user']['temp_grid'] = np.asarray(temp_grid) #eV
        self.data['user']['dens_grid'] = np.asarray(electron_den)#cm-3
        self.data['user']['htemp_grid'] = np.asarray(htemp_grid) #eV
        self.data['user']['hdens_grid'] = np.asarray(hdens_grid) #cm-3
        if((len(self.data['user']['dens_grid']) != len(self.data['user']['hdens_grid'])) & use_cx):
            print('Electron density and neutral density grids must be the same length')
            sys.exit()
        if((len(self.data['user']['temp_grid']) != len(self.data['user']['htemp_grid'])) & use_cx):
            print('Electron temperature and neutral temperature grids must be the same length')
            sys.exit()
        self.data['user']['use_ionization'] = use_ionization
        self.data['user']['suppliment_with_ecip'] = suppliment_with_ecip
        self.data['user']['use_recombination_three_body'] = use_recombination_three_body
        self.data['user']['use_recombination'] = use_recombination
        self.data['user']['use_cx'] = use_cx
        self.data['user']['scale_file_ioniz'] = scale_file_ioniz
        self.data['user']['metas'] = 'look at [atomic][metas] for values'
        self.data['user']['td_t'] = np.asarray(td_t)# seconds
        self.data['user']['td_n0'] = np.asarray(td_n0)
        self.data['user']['td_source'] = np.asarray(td_source)
        self.data['user']['default_pop_norm'] = default_pop_norm
        self.data['user']['temp_dens_pair'] = temp_dens_pair

        #choosing the collisional interp and extrap type
        #for low temperature, burgess tully is used for high temp extrap
        self.data['user']['rate_interp_kind_col'] = rate_interp_col
        self.data['user']['log_rate_col'],\
            self.data['user']['interp_kind_col'] = rate_interp_parse(rate_interp_col)
        #choosing interp and extrapolation type for ionization
        self.data['user']['rate_interp_kind_ion'] = rate_interp_ion
        self.data['user']['log_rate_ion'],\
            self.data['user']['interp_kind_ion'] = rate_interp_parse(rate_interp_ion)
        #choosing interp and extrapolation type for recombination
        self.data['user']['rate_interp_kind_recomb'] = rate_interp_recomb
        self.data['user']['log_rate_recomb'],\
            self.data['user']['interp_kind_recomb'] = rate_interp_parse(rate_interp_recomb)
        #choosing interp and extrap type for cx
        self.data['user']['rate_interp_kind_cx'] = rate_interp_cx
        self.data['user']['log_rate_cx'],\
            self.data['user']['interp_kind_cx'] = rate_interp_parse(rate_interp_cx)
        #the log of the rates is interpolated (on a log temperature axis) for ionization and
        #recombination
        self.data['user']['loglog_rate_ion'] = rate_interp_log_values(rate_interp_ion)
        self.data['user']['loglog_rate_recomb'] = rate_interp_log_values(rate_interp_recomb)
        for name, option in (('rate_interp_col', rate_interp_col), ('rate_interp_cx', rate_interp_cx)):
            if(rate_interp_log_values(option)):
                raise ValueError(name + " = '" + option + "': the log-log interpolation is available for "
                                 "ionization and recombination (rate_interp_ion, rate_interp_recomb)")
        
        
        '''
        if(self.data['user']['rate_interp'][0:3] =='log'):
            self.data['user']['log_rate'] = True

        self.data['user']['interp_kind'] = 'slinear'
        if('linear' in self.data['user']['rate_interp']):
            self.data['user']['interp_kind'] = 'linear'
        if('nearest' in self.data['user']['rate_interp']):
            self.data['user']['interp_kind'] = 'nearest'
        if('zero' in self.data['user']['rate_interp']):
            self.data['user']['interp_kind'] = 'zero'
        if('slinear' in self.data['user']['rate_interp']):
            self.data['user']['interp_kind'] = 'slinear'
        if('quadratic' in self.data['user']['rate_interp']):
            self.data['user']['interp_kind'] = 'quadratic'
        if('cubic' in self.data['user']['rate_interp']):
            self.data['user']['interp_kind'] = 'cubic'
        '''
            
        
        if(self.data['user']['temp_dens_pair']):
            if( len(self.data['user']['temp_grid']) != len(self.data['user']['dens_grid'])):
                print("Temperature and density were chosen to be pairs not grid. There for the length of temperature and density arrays must be the same")
                print('Exit here fix to run')
                sys.exit()
                
        self.populate_data(fil)
        self.data['atomic']['metas'] = np.asarray(metas)
        #the quasistatic model needs the ground level as a metastable, it cannot be an excited level
        if(self.data['atomic']['metas'].size > 0):
            levels = self.data['atomic']['metas']
            if(0 not in levels):
                raise ValueError('metas must include the ground level 0, got ' + str(levels.tolist()))
            if(len(np.unique(levels)) != len(levels) or levels.min() < 0 or
               levels.max() >= len(self.data['atomic']['energy'])):
                raise ValueError('metas must be different levels of the adf04 file (0 to ' +
                                 str(len(self.data['atomic']['energy'])-1) + '), got ' + str(levels.tolist()))


        #check for levels that have identical identifications which can cause a problem for splitting later
        #This happens when the parentage is not included in the configuration string leading to ambiguity
        #This implimentation assumes that the order of energy is correct in terms of parentage
        
        tot_id_arr = np.array([self.data['atomic']['config'], #make a 2d array to search over
                               self.data['atomic']['L'],
                               self.data['atomic']['S'],
                               self.data['atomic']['w']])
        
        uniq,uniq_idx, inv,counts= np.unique(tot_id_arr,#find the unique elements in the array
                                             axis=1,
                                             return_counts=True,
                                             return_index=True,
                                             return_inverse=True)
        new_arr = uniq[:,counts == 1] #indexes of array that are unique
        a_idx = np.arange(tot_id_arr.shape[1]) #array of every index
        
        self.data['atomic']['inds_same_id'] = a_idx[_isin(a_idx, uniq_idx[counts==1], invert=True)]
        
        self.data['atomic']['id_groups'] = []#we won't know how many levels are in a group

        #this groups all of the duplicated identifications
        #this is maybe a few line numpy call to do this but I can't figure it out
        #this should be too slow so YOLO
        for ij in self.data['atomic']['inds_same_id']:
            if len(self.data['atomic']['id_groups']) >0:
                if ij not in np.concatenate(self.data['atomic']['id_groups']):
                    #if the ij index is not already in a group create a new group
                    self.data['atomic']['id_groups'].append(np.where(
                          (self.data['atomic']['config'] == self.data['atomic']['config'][[ij]]) &
                          (self.data['atomic']['S'] == self.data['atomic']['S'][[ij]]) &
                          (self.data['atomic']['L'] == self.data['atomic']['L'][[ij]]) &        
                          (self.data['atomic']['w'] == self.data['atomic']['w'][[ij]]) )[0])
            else:#creating the first group
                self.data['atomic']['id_groups'].append(np.where(
                          (self.data['atomic']['config'] == self.data['atomic']['config'][[ij]]) &
                          (self.data['atomic']['S'] == self.data['atomic']['S'][[ij]]) &
                          (self.data['atomic']['L'] == self.data['atomic']['L'][[ij]]) &        
                          (self.data['atomic']['w'] == self.data['atomic']['w'][[ij]]) )[0])
                

        self.data["user"]["git_hash"] = get_git_hash()

                
    def update_dict(self,d, u):
        """This function will update dictionary that is stores all parameter for the class

        :param d: the dictionary that is to be updated
        :type d: dictionary

        :param w: the dictionary the use wants to update to
        :type w: dictionary
        """

        for k, v in u.items():
            if isinstance(v, Mapping):
                d[k] = self.update_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    
    def populate_data(self,fil):
        """This function will populate atomic data. Currently this only uses the
           adf04 file but there is nothing special about this format and should be
           put out to pasture at some point

        :param fil: path to the file that contains atomic data
        :type fil: file path

        :returns: array --wavelengths in nm in air

        """
        if(type(fil) == str or type(fil) == np.str_):

            self.data = self.update_dict(self.data,read_adf04(fil))
            self.data['user']['file_loc'] = str(fil)#make this a string changed for hdfdict
        else:
            if(self.data):
                self.data = self.update_dict(self.data,fil)
            else:
                self.data = fil

                
    def make_ecip(self):
        """This function calls the ecip_rates function and populates the ecip
           values in the dictionary. See documentation ecip_rates.py for a better desciption
           
        """

        self.data['rates']['ioniz']['ecip'] = np.zeros((len(self.data['atomic']['energy']),
                                                        len(self.data['atomic']['ion_pot']),
                                                        len(self.data['user']['temp_grid'])))
        #ECIP is made for the levels marked {p} in the level list that are below the ionization
        #potential of parent p (ecip_rates), so a level above the first ionization potential
        #still ionizes to a higher parent it is marked for

        if(np.where( self.data['atomic']['zpla'][:,0]>-1)[0].size == 0):
            print("adf04 file specifies no ionization rates but you requested ECIP, the adf04 file creator"+ \
                   "most likely forgot about specify this (looking at you R-matrix people). Blindly creating"+ \
                   "ECIP rates for every transition, you don't want this specify in the file which levels" +    \
                   "should and should not have ionization rates.")

            self.data['atomic']['zpla'][:] = 1
            self.data['atomic']['zpla1'][:] = 1.000
        
        #reduced rates exp(dE/kT)*S, which do not underflow at low temperature and are kept for
        #three body recombination
        self.data['rates']['ioniz']['ecip_reduced'] = ecip_rates(self.data['atomic']['energy'],
                                                  self.data['atomic']['ion_pot'],self.data['atomic']['zpla'],
                                                  self.data['atomic']['zpla1'],self.data['atomic']['charge_state'],
                                                  self.data['user']['temp_grid'], reduced=True)
        #the rates, zero for exponents above 180 as in r8ecip
        exponent = self.ecip_exponent()
        with np.errstate(over='ignore', invalid='ignore'):
            self.data['rates']['ioniz']['ecip'] = np.where(
                (exponent > 180.) | (self.data['rates']['ioniz']['ecip_reduced'] == 0), 0.,
                self.data['rates']['ioniz']['ecip_reduced']*np.exp(-exponent))

    def ecip_exponent(self):
        """Exponent dE/kT of the ECIP rates, as calculated in r8ecip, with shape
           (levels, parents, temperatures).
        """
        return np.einsum('ij,k->ijk', self.data['atomic']['ion_pot'][None,:] -
                         self.data['atomic']['energy'][:,None],
                         9.11269E-06*157890.0/(self.data['user']['temp_grid']*11604.5))


    def make_burgess_tully(self):
        """This function calls the burgess_tully_rates function and updates the 'rates' dictionary

           values in the dictionary. See documentation ecip_rates.py for a better desciption
           
        """



        if(np.size(self.data['rates']['inf_engy']) == 0):
            print('No infinite energy points were in the adf04 file. Burgess tully extrapoluation is applied from'+
                  ' the last two calculated points')
            self.data['rates']['inf_engy'] = np.zeros((len(self.data['rates']['excit']['col_transitions'])))


        
        self.data['rates'].update( burgess_tully_rates(self.data['user']['temp_grid'],self.data['input_file']['temp_grid'],
                                                       self.data['rates']['excit']['col_transitions'],self.data['rates']\
                                                       ['excit']['col_excit'],
                                                       self.data['atomic']['energy'],self.data['atomic']['w'],
                                                       self.data['rates']['a_val'],self.data['atomic']['S'],
                                                       self.data['atomic']['L'],self.data['rates']['inf_engy']))

    def _interp_rates(self, temps, rates, log_axis, kind):
        """Rates (one row for each transition) given at the temperatures temps (eV), interpolated
           on the user temperatures (on a log temperature axis if log_axis) and extrapolated
           linearly outside temps, without negative values. One temperature gives constant rates,
           and too few temperatures for kind fall back to slinear.
        """
        user_temps = self.data['user']['temp_grid']
        if(len(temps) == 1):
            return np.repeat(rates, len(user_temps), axis=1)
        if(len(temps) <= {'quadratic': 2, 'cubic': 3}.get(kind, 1)):
            kind = 'slinear'
        if(log_axis):
            return interp1d(np.log(temps), rates, axis=1, kind=kind,
                            fill_value='extrapolate')(np.log(user_temps)).clip(min=0)
        return interp1d(temps, rates, axis=1, kind=kind, fill_value='extrapolate')(user_temps).clip(min=0)

    def _interp_rates_loglog(self, temps, rates, kind, below, above):
        """Rates (one row for each transition) given at the increasing temperatures temps (eV),
           on the user temperatures: the log of the rates interpolated on a log temperature axis
           between the positive values of each row, and outside them a power law through the two
           nearest ones, whose exponent is clipped to below or above, (min, max) with None for no
           limit. Linear interpolation of the rates overestimates them between temperatures where
           they change by orders of magnitude (the arithmetic mean of the two values instead of
           their geometric mean), and its extrapolation crosses zero. Rows without a positive
           value are zero, rows with one are constant.
        """
        user = np.log(self.data['user']['temp_grid'])
        log_temps = np.log(temps)
        grid = np.zeros((len(rates), len(user)))
        for i, row in enumerate(np.asarray(rates)):
            positive = row > 0
            if(positive.sum() < 2):
                grid[i] = row[positive].sum()
                continue
            x, y = log_temps[positive], np.log(row[positive])
            k = kind if len(x) > {'quadratic': 2, 'cubic': 3}.get(kind, 1) else 'slinear'
            log_grid = np.empty(len(user))
            inside = (user >= x[0]) & (user <= x[-1])
            if(inside.any()):
                log_grid[inside] = interp1d(x, y, kind=k)(user[inside])
            for outside, (a, b), limits in ((user < x[0], (0, 1), below), (user > x[-1], (-1, -2), above)):
                if(outside.any()):
                    power = np.clip((y[b] - y[a])/(x[b] - x[a]), *limits)
                    log_grid[outside] = y[a] + power*(user[outside] - x[a])
            grid[i] = np.exp(log_grid)
        return grid

    def _rates_left_out(self, temps, rates, bound, steepest):
        """Placeholders (above bound, or not finite) and artifacts at the lowest temperatures
           (falling faster than T^steepest) among rates given at the increasing temperatures temps.
        """
        placeholders = ~(np.isfinite(rates) & (rates <= bound))
        artifacts = low_temperature_artifacts(temps, np.where(placeholders, 0., rates), steepest)
        return placeholders, artifacts

    def _warn_left_out(self, what, left_out, labels, temps):
        """Warn of the rates left out, left_out (rows, temperatures), with a label for each row."""
        rows = [labels[i] + ' at ' + ', '.join(format(t, '.3g') for t in temps[left_out[i]]) + ' eV'
                for i in np.where(left_out.any(axis=1))[0]]
        if(rows):
            warnings.warn(str(self.data['user'].get('file_loc', 'the adf04 file')) + ': ' + what +
                          ' are not physical and are left out: ' + '; '.join(rows))

    def _reduced_ionization_on_grid(self, temps, rates):
        """Reduced ionization rates (one row for each transition) given at the temperatures temps
           (eV), on the user temperatures. Below temps the rates follow a power law through the
           two lowest temperatures (near threshold they go as T^1/2) instead of the interpolator's
           extrapolation, which crosses zero; rates that do not decrease towards lower temperature
           are held constant.
        """
        user_temps = self.data['user']['temp_grid']
        order = np.argsort(temps)
        temps, rates = temps[order], rates[:, order]
        grid = self._interp_rates(temps, rates, self.data['user']['log_rate_ion'],
                                  self.data['user']['interp_kind_ion'])
        if(len(temps) == 1):
            return grid

        low = user_temps < temps[0]
        if(low.any()):
            with np.errstate(divide='ignore', invalid='ignore'):
                power = np.log(rates[:,1]/rates[:,0])/np.log(temps[1]/temps[0])
            power[~np.isfinite(power) | (power < 0)] = 0.
            grid[:,low] = rates[:,[0]]*(user_temps[low]/temps[0])**power[:,None]
        return grid

    def make_ioniz_from_reduced_ionizrates(self):
        """This function calls will make ionization rates to be used in the CR matrix from the
           reduced ionization rates that are provided in the adf04 file. This function
           must be called even if ionization is not provided so that ECIP rates can be 
           supplimented into the matrix
        """

        #ionization rates and reduced rates exp(dE/kT)*S, which do not underflow at low temperature
        #and are kept for three body recombination
        self.data['rates']['ioniz']['ionization'] = np.zeros((len(self.data['atomic']['energy']),
                                                              len(self.data['atomic']['ion_pot']),
                                                              len(self.data['user']['temp_grid'])))
        self.data['rates']['ioniz']['ionization_reduced'] = np.zeros_like(self.data['rates']['ioniz']['ionization'])

        #if there is no ionization in the file keep the zero matrix so that ECIP can be subbed in
        if(np.size(self.data['rates']['ioniz']['ion_excit']) < 1):
            print('No ionization in the input file ECIP can be made')
            return




        #Placeholders instead of reduced rates (above MAX_REDUCED_IONIZATION, e.g. 1 to 1e40 cm3 s-1
        #at the lowest temperatures of OpenADAS cop98#10_ls#ne4 to ne8), and smaller artifacts at the
        #lowest temperatures that fall faster than any reduced rate, are left out, and the rates at
        #those temperatures come from the other points of the transition
        file_temps = self.data['input_file']['temp_grid']/11604.5
        order = np.argsort(file_temps)
        file_temps = file_temps[order]
        ion_excit = self.data['rates']['ioniz']['ion_excit'][:, order]
        placeholders, artifacts = self._rates_left_out(file_temps, ion_excit, MAX_REDUCED_IONIZATION,
                                                       STEEPEST_REDUCED_IONIZATION)
        valid = ~placeholders & ~artifacts
        ion_excit_interp_grid = np.zeros((len(ion_excit), len(self.data['user']['temp_grid'])))
        if(self.data['user']['loglog_rate_ion']):
            ion_excit_interp_grid = self._interp_rates_loglog(file_temps, np.where(valid, ion_excit, 0.),
                                                              self.data['user']['interp_kind_ion'],
                                                              REDUCED_IONIZATION_EXPONENTS_BELOW,
                                                              REDUCED_IONIZATION_EXPONENTS_ABOVE)
        else:
            all_valid = valid.all(axis=1)
            if(all_valid.any()):
                ion_excit_interp_grid[all_valid] = self._reduced_ionization_on_grid(file_temps, ion_excit[all_valid])
            for i in np.where(~all_valid & valid.any(axis=1))[0]:
                ion_excit_interp_grid[i] = self._reduced_ionization_on_grid(file_temps[valid[i]],
                                                                          ion_excit[i:i+1, valid[i]])[0]
        labels = ['level ' + str(level) + ' to parent ' + str(parent)
                  for level, parent in self.data['rates']['ioniz']['ion_transitions']]
        self._warn_left_out('reduced ionization rates above ' + format(MAX_REDUCED_IONIZATION, 'g') + ' cm3 s-1',
                            placeholders, labels, file_temps)
        self._warn_left_out('reduced ionization rates at the lowest temperatures that fall faster than T^' +
                            format(STEEPEST_REDUCED_IONIZATION, 'g'), artifacts, labels, file_temps)

        #scale the rates by the {parent} values of the level list, whatever the interpolation;
        #levels without a value for the parent (-1) are not scaled
        if(self.data['user']['scale_file_ioniz']):
            ion_transitions = self.data['rates']['ioniz']['ion_transitions']
            scale = self.data['atomic']['zpla'][ion_transitions[:,0]-1, ion_transitions[:,1]-1]
            ion_excit_interp_grid = ion_excit_interp_grid*np.where(scale > -1, scale, 1.)[:,None]

        #at large dE/kT the exponential overflows and the rate is 0, as it should be
        with np.errstate(over='ignore'):
            for i in range(0,len(self.data['rates']['ioniz']['ion_transitions'])):
                self.data['rates']['ioniz']['ionization_reduced'][self.data['rates']['ioniz']\
                        ['ion_transitions'][i,0] -1,self.data['rates']['ioniz']\
                        ['ion_transitions'][i,1] -1, :] = ion_excit_interp_grid[i]
                for j in range(0,len(self.data['user']['temp_grid'])):

                    self.data['rates']['ioniz']['ionization'][self.data['rates']['ioniz']\
                        ['ion_transitions'][i,0] -1,self.data['rates']['ioniz']\
                        ['ion_transitions'][i,1] -1, j] = ion_excit_interp_grid[i,j]/\
                        np.exp((self.data['atomic']['ion_pot'][self.data['rates']['ioniz']\
                        ['ion_transitions'][i,1]-1] - self.data['atomic']['energy']\
                        [self.data['rates']['ioniz']['ion_transitions'][i,0] -1 ])\
                        *EV_PER_WAVENUMBER/self.data['user']['temp_grid'][j])
                
                
    def suppliment_with_ecip(self):
        """This function will suppliment the ionization that is to be used in the CR matrix
           with ECIP ionization if there is no ionization that is provided. While ECIP ionization
           is better than not including ionization using other ionization rates is generall better
           Ionization must be must be included to accurately model plasma even if it is approximate ECIP
        """

        if('ecip' not in self.data['rates']['ioniz']):
            print("ECIP was not previously calculated, calculating now"+\
                   " from inside 'suppliment_with_ecip'")
            self.make_ecip()

        if(('ionization' not in self.data['rates']['ioniz']) and (self.data['user']['use_ionization'] or
                                                                  self.data['user']['use_recombination_three_body'])):
            print('No ionization rates from files were calculated,'+\
                  ' try to calculate them')
            self.make_ioniz_from_reduced_ionizrates()

        #added for the case that ionization is turned off and
        if('ionization' not in self.data['rates']['ioniz']):
            self.data['rates']['ioniz']['ionization'] = np.zeros((len(self.data['atomic']['energy']),
                                                                  len(self.data['atomic']['ion_pot']),
                                                                  len(self.data['user']['temp_grid'])))
        if('ionization_reduced' not in self.data['rates']['ioniz']):
            self.data['rates']['ioniz']['ionization_reduced'] = \
                                        self.reduced_ionization(self.data['rates']['ioniz']['ionization'])

        if('ecip_reduced' in self.data['rates']['ioniz']):
            ecip_reduced = self.data['rates']['ioniz']['ecip_reduced']
        else:
            #ECIP rates made elsewhere, those r8ecip sets to zero above an exponent of 180 stay zero
            ecip = self.data['rates']['ioniz']['ecip']
            ecip_reduced = np.exp(np.log(ecip, out=np.full_like(ecip, -np.inf), where=ecip > 0) +
                                  self.ecip_exponent())
        for p in range(0,len(self.data['atomic']['ion_pot'])):
            #there was a stupid problem beacuse {X} only designates ecip not other ionization
            ion_inds = np.where( self.data['atomic']['zpla'][:,p] > -1)[0]
            ion_inds2 = np.where( self.data['atomic']['zpla1'] == p +1)[0]
            ion_inds = np.intersect1d(ion_inds,ion_inds2)
            #levels without ionization from the file. The reduced rates are used because the
            #rates themselves can underflow to zero at low temperature.
            ecip_inds = np.where(np.all(self.data['rates']['ioniz']['ionization_reduced'][:,p,:] == 0,
                                        axis=1))[0]
            to_use = np.intersect1d(ecip_inds,ion_inds)
            self.data['rates']['ioniz']['ionization'][to_use,p,:] =\
                                        self.data['rates']['ioniz']['ecip'][to_use,p,:]
            self.data['rates']['ioniz']['ionization_reduced'][to_use,p,:] = ecip_reduced[to_use,p,:]

    def reduced_ionization(self, ionization):
        """Reduced ionization rates exp(dE/kT)*S of ionization rates S with shape
           (levels, parents, temperatures). Rates that are zero stay zero.

        :param ionization: Ionization rates in cm3 s-1
        :type ionization: float array

        :returns: array --reduced ionization rates in cm3 s-1

        """
        exponent = np.einsum('ij,k->ijk', self.data['atomic']['ion_pot'][None,:] -
                             self.data['atomic']['energy'][:,None],
                             EV_PER_WAVENUMBER/self.data['user']['temp_grid'])
        return np.exp(np.log(ionization, out=np.full_like(ionization, -np.inf, dtype=float),
                             where=ionization > 0) + exponent)
    def make_cx_rates_from_file(self):
        """This function will make thermal charge exchange rates from the rates that are provided in the
           adf04 file.
        """


        if(self.data['user']['log_rate_cx']):
            cx_excit_interp = interp1d(np.log(self.data['input_file']['temp_grid']/11604.5),
                                           self.data['rates']['cx']['cx_excit'],
                                           axis=1,
                                           kind = self.data['user']['interp_kind_cx'],
                                           fill_value =  'extrapolate')

            self.data['rates']['cx']['cx_excit_interp_grid'] =\
                                        cx_excit_interp(np.log(self.data['user']['htemp_grid']))
        else:
            cx_excit_interp = interp1d(self.data['input_file']['temp_grid']/11604.5,
                                           self.data['rates']['cx']['cx_excit'],
                                           axis=1,
                                           kind = self.data['user']['interp_kind_cx'],
                                           fill_value =  'extrapolate')

            self.data['rates']['cx']['cx_excit_interp_grid'] =\
                                        cx_excit_interp(self.data['user']['htemp_grid'])

        #make sure that we can get negative rates
        self.data['rates']['cx']['cx_excit_interp_grid'][np.where(self.data['rates']['cx']['cx_excit_interp_grid'] <0)] = 0.0

        #one column for each parent, as for recombination
        self.data['rates']['cx']['cx'] = \
            np.zeros((len(self.data['atomic']['energy']),
                      len(self.data['atomic']['ion_pot']),
                      len(self.data['user']['htemp_grid'])))

        for q in range(0,len(self.data['rates']['cx']['cx_transitions'])):
            self.data['rates']['cx']['cx']\
                [self.data['rates']['cx']['cx_transitions'][q,0]-1,
                 self.data['rates']['cx']['cx_transitions'][q,1]-1,:] = \
                                    self.data['rates']['cx']['cx_excit_interp_grid'][q]



            
    def make_recombination_rates_from_file(self):
        """This function will make recombination rates from the rates that are provided in the
           adf04 file.
        """

        if(np.size(self.data['rates']['recomb']['recomb_excit'] > 0)):

            #Placeholders instead of recombination rates (above MAX_RECOMBINATION, e.g. 3.25e-5 cm3 s-1
            #followed by 1e-30 at the lowest temperatures of OpenADAS cop98#10_ls#ne1), and smaller
            #artifacts at the lowest temperatures that fall faster than any recombination rate, are
            #left out, and the rates at those temperatures come from the other points of the transition
            recomb_excit = self.data['rates']['recomb']['recomb_excit']
            file_temps = self.data['input_file']['temp_grid']/11604.5
            order = np.argsort(file_temps)
            file_temps, recomb_excit = file_temps[order], recomb_excit[:, order]
            placeholders, artifacts = self._rates_left_out(file_temps, recomb_excit, MAX_RECOMBINATION,
                                                           STEEPEST_RECOMBINATION)
            valid = ~placeholders & ~artifacts
            grid = np.zeros((len(recomb_excit), len(self.data['user']['temp_grid'])))
            if(self.data['user']['loglog_rate_recomb']):
                grid = self._interp_rates_loglog(file_temps, np.where(valid, recomb_excit, 0.),
                                                 self.data['user']['interp_kind_recomb'],
                                                 RECOMBINATION_EXPONENTS, RECOMBINATION_EXPONENTS)
            else:
                all_valid = valid.all(axis=1)
                if(all_valid.any()):
                    grid[all_valid] = self._interp_rates(file_temps, recomb_excit[all_valid],
                                                         self.data['user']['log_rate_recomb'],
                                                         self.data['user']['interp_kind_recomb'])
                for i in np.where(~all_valid & valid.any(axis=1))[0]:
                    grid[i] = self._interp_rates(file_temps[valid[i]], recomb_excit[i:i+1, valid[i]],
                                                 self.data['user']['log_rate_recomb'],
                                                 self.data['user']['interp_kind_recomb'])[0]
            labels = ['parent ' + str(parent) + ' to level ' + str(level)
                      for parent, level in self.data['rates']['recomb']['recomb_transitions']]
            self._warn_left_out('recombination rates above ' + format(MAX_RECOMBINATION, 'g') + ' cm3 s-1',
                                placeholders, labels, file_temps)
            self._warn_left_out('recombination rates at the lowest temperatures that fall faster than T^' +
                                format(STEEPEST_RECOMBINATION, 'g'), artifacts, labels, file_temps)
            self.data['rates']['recomb']['recomb_excit_interp_grid'] = grid


            '''
        #replace values lower than 1e-30 with a linear interpolation
        #because slinear gives the the wrong answer for some reason
        #maybe dont need this part now, 1dinterp was using kind='slinear'
        #changed now and hopefully wont get the errors
        a,b,c = np.unique(np.where(self.data['rates']['recomb']['recomb_excit']\
                                   <1.e-30)[0],return_inverse=True,return_counts=True)
        if(c.any()):
            if(any(self.data['user']['temp_grid'] < \
                   self.data['input_file']['temp_grid'][c.max()]/11604.5)):
                for v in range(0,len(a)):
                    tmp = np.where(self.data['user']['temp_grid'] <\
                                   self.data['input_file']['temp_grid'][c.max()]/11604.5)[0]
                    w = interp1d(self.data['input_file']['temp_grid'][0:c[v]+1]/11604.5,\
                                 self.data['rates']['recomb']['recomb_excit'][a[v],0:c[v]+1],\
                                 kind='slinear')

                    self.data['rates']['recomb']['recomb_excit_interp_grid'][a[v],0:c[v]+1] =\
                                                    w(self.data['user']['temp_grid'][0:tmp+1])
            '''
            self.data['rates']['recomb']['recombination'] = \
                np.zeros((len(self.data['atomic']['energy']),
                          len(self.data['atomic']['ion_pot']),
                          len(self.data['user']['temp_grid'])))

            for q in range(0,len(self.data['rates']['recomb']['recomb_transitions'])):
                self.data['rates']['recomb']['recombination']\
                    [self.data['rates']['recomb']['recomb_transitions'][q,1]-1,
                     self.data['rates']['recomb']['recomb_transitions'][q,0]-1,:] = \
                                        self.data['rates']['recomb']['recomb_excit_interp_grid'][q]

            
    def make_three_body_recombination(self):
        """This function will make three body recombination rates by using a detailed balance
           relation with ionization that used. Three body recombination becomes important at high densities
           and low temperatures
        """
        
        #from the los alimos lab manual
        #he['w'] = (he['S']*(he['L']*2+1)-1)/2
        self.data['rates']['recomb']['recomb_three_body'] = \
                        np.zeros((len(self.data['atomic']['energy']),
                                  len(self.data['atomic']['ion_pot']),
                                  len(self.data['user']['temp_grid'])))
        #detailed balance needs the ionization rates, also when ionization is not in the CR matrix
        if('ionization' not in self.data['rates']['ioniz']):
            self.make_ioniz_from_reduced_ionizrates()
            if(self.data['user']['suppliment_with_ecip']):
                self.suppliment_with_ecip()
        #statistical weight of each parent
        self.data['rates']['recomb']['w_ion'] = np.array([parent_statistical_weight(term)
                                                          for term in self.data['atomic']['ion_term']])
        #reduced ionization exp(dE/kT)*S, so the Boltzmann factors of detailed balance cancel
        #without the overflow and underflow of exp(dE/kT) and S at low temperature
        if('ionization_reduced' in self.data['rates']['ioniz']):
            ionization_reduced = self.data['rates']['ioniz']['ionization_reduced']
        else:
            ionization_reduced = self.reduced_ionization(self.data['rates']['ioniz']['ionization'])
        for p in range(0,len(self.data['atomic']['ion_pot'])):
            #this is from detailed balance
            self.data['rates']['recomb']['recomb_three_body'][:,p,:] = 1.656742E-22* \
                (self.data['atomic']['w'].reshape(len(self.data['atomic']['w']),1)*2+1)/ \
                self.data['rates']['recomb']['w_ion'][p]*ionization_reduced[:,p,:] / \
                (self.data['user']['temp_grid'])**(1.5)

        recomb_3b_ind = np.where(np.isnan(self.data['rates']['recomb']['recomb_three_body']))

        if(recomb_3b_ind[0].size > 0):
            print('There were NaNs in three body recombination making those zero')
            self.data['rates']['recomb']['recomb_three_body'][recomb_3b_ind] = 0.0
            
            #rates of this are in cm3

        
    def make_electron_excitation_rates(self):
        """This function will make both electron impact excitation and deexcitation rates
           from the values that stored in adf04 file on a user defined temperature grid
           If values are above the last calculate point in the adf04 then a burgess tully
           extrapolation will be used. Below the first calculated temperature point the
           burgess tully values are extrapolated in log towards their finite limit at T = 0,
           or the collision strength is held constant if the calculated temperatures are all
           far above the transition threshold.
        """
        if(self.data['user']['log_rate_col']):
            tmp = interp1d(np.log(self.data['input_file']['temp_grid']/11604.5),
                           self.data['rates']['excit']['col_excit'],axis=1,
                                           kind = self.data['user']['interp_kind_col'],
                                           fill_value =  'extrapolate')
        else:
            tmp = interp1d(self.data['input_file']['temp_grid']/11604.5,
                           self.data['rates']['excit']['col_excit'],axis=1,
                                           kind = self.data['user']['interp_kind_col'],
                                           fill_value =  'extrapolate')


            
        self.data['rates']['excit']['col_excit_interp'] = np.zeros((len(self.data['rates']['excit']['col_excit']),
                                                                         len(self.data['user']['temp_grid'])))

        if(self.data['user']['log_rate_col']):
            self.data['rates']['excit']['col_excit_interp'] = tmp(np.log(self.data['user']['temp_grid']))
        else:
            self.data['rates']['excit']['col_excit_interp'] = tmp(self.data['user']['temp_grid'])


        extrap_hi = np.max(self.data['input_file']['temp_grid'])/11604.5 < np.max(self.data['user']['temp_grid'])
        extrap_low = np.min(self.data['input_file']['temp_grid'])/11604.5 > np.min(self.data['user']['temp_grid'])
        if(extrap_hi):
            print('Atleast one user temp point above last calculated temperature using extrapolation be carefull')
        if(extrap_low):
            print('Atleast one user temp point below first calculated temperature using extrapolation be carefull')
        if(extrap_hi or extrap_low):
            self.make_burgess_tully()

        if(extrap_hi):

            for i in range(0,len(self.data['rates']['burg_tully']['extrap_temp_inds_hi'])):
                for j in range(0,4):
                    self.data['rates']['excit']['col_excit_interp']                        [self.data['rates']['burg_tully']['ind_arrs'][j],
                                      self.data['rates']['burg_tully']['extrap_temp_inds_hi'][i] ] =                                      self.data['rates']['burg_tully']['excit_extrap'][j][:,i]

                for j in range(0,4):
                    if(self.data['rates']['burg_tully']['zero_inds'][j].size >0):
                        self.data['rates']['excit']['col_excit_interp']                            [self.data['rates']['burg_tully']['ind_arrs'][j]                                           [self.data['rates']['burg_tully']['zero_inds'][j]],
                                        self.data['rates']['burg_tully']['extrap_temp_inds_hi'][i] ] =                                        self.data['rates']['burg_tully']['excit_extrap_lin'][j][:,i]

        if(extrap_low):
            for j in range(0,4):
                self.data['rates']['excit']['col_excit_interp'][np.ix_(
                    self.data['rates']['burg_tully']['ind_arrs'][j],
                    self.data['rates']['burg_tully']['extrap_temp_inds_low'])] =                     self.data['rates']['burg_tully']['excit_extrap_low_lin'][j]
            num_constant = sum(map(len, self.data['rates']['burg_tully']['low_constant_inds']))
            if(num_constant > 0):
                print(str(num_constant) + ' transitions keep the collision strength of the first ' +
                      'calculated temperature below it, it is far above their threshold')

        #transitions whose collision strengths are all zero in the file have zero rates (the
        #logs of their extrapolations are NaN)
        all_zero = (self.data['rates']['excit']['col_excit'] == 0).all(axis=1)
        self.data['rates']['excit']['col_excit_interp'][all_zero] = 0.

        tttmp = np.where(np.isnan(self.data['rates']['excit']['col_excit_interp']))
        if(tttmp[0].size>0):
            print('NaNs in electron excitations where set to zero')
            self.data['rates']['excit']['col_excit_interp'][tttmp] = 0.

        #negative collision strengths would give negative rates and populations
        tttmp = np.where(self.data['rates']['excit']['col_excit_interp'] < 0)
        if(tttmp[0].size>0):
            print('Negative electron excitations where set to zero')
            self.data['rates']['excit']['col_excit_interp'][tttmp] = 0.
                        
            '''
            if(self.data['user']['log_rate_col']):
                self.data['rates']['excit']['col_excit_interp']\
                 [:,self.data['rates']['burg_tully']['interp_temp_inds']] =\
                 tmp(np.log(self.data['user']['temp_grid'][self.data['rates']['burg_tully']['interp_temp_inds']]))
            else:
                self.data['rates']['excit']['col_excit_interp']\
                 [:,self.data['rates']['burg_tully']['interp_temp_inds']] =\
                 tmp(self.data['user']['temp_grid'][self.data['rates']['burg_tully']['interp_temp_inds']])
            '''


    def populate_cr_matrix(self):
        """This function will populate the collision radiative matrix with all the rates that
           user asks to be included into the calculation.

           Ionization is included from the ADF04 file, there is a user option to suppliment with ECIP
           rates that ColRadPy will make

           Recombination is included from the ADF04 file.
           Three-body recombination can also be made from ColRadPy through detailed balance
           Excitation rates are included from the ADF04 file.

           The dictionary ['cr_matrix'] is made in this definition mostly from rates in the ['rates'] dictionary.
        """
        
        #add in the excitation/de-excitation if they are not already calculated
        if('col_excit_interp' not in self.data['rates']['excit'].keys()):
            print('Electron collisional rates have not yet been made on '+\
                  'the user defined temperature grid. Doing that now')
            self.make_electron_excitation_rates()
        #add in ecip ionization if not already included
        if('ecip' not in self.data['rates']['ioniz'] and \
                                         self.data['user']['suppliment_with_ecip']):
            self.suppliment_with_ecip()
        #add in ionization if it was not already included
        if('ionization' not in self.data['rates']['ioniz'] and \
                                               self.data['user']['use_ionization']):
            self.make_ioniz_from_reduced_ionizrates()
        #add in threebody recombination if it was not already included
        if(self.data['user']['use_recombination_three_body'] and \
                                 'recomb_three_body' not in self.data['rates']['recomb']):
            print('Three body recombination was not previously calculated. Doing that now')
            self.make_three_body_recombination()
        #add in recombination if present and it was not already calculated
        if(self.data['user']['use_recombination'] and \
           (np.size(self.data['rates']['recomb']['recomb_excit'] > 0)) and \
                       'recomb_excit_interp_grid' not in self.data['rates']['recomb']):
            print('Recombination from file was not previously calculated. Doing that now')            
            self.make_recombination_rates_from_file()

        
            
        self.data['cr_matrix'] = {}
        self.data['cr_matrix']['q_ji'] = np.zeros((len(self.data['atomic']['energy']),
                                                   len(self.data['atomic']['energy']),
                                                   len(self.data['user']['temp_grid'])))
        self.data['cr_matrix']['q_ij'] = np.zeros((len(self.data['atomic']['energy']),
                                                   len(self.data['atomic']['energy']),
                                                   len(self.data['user']['temp_grid'])))
        self.data['cr_matrix']['A_ji'] = np.zeros((len(self.data['atomic']['energy']),
                                                   len(self.data['atomic']['energy'])))
        for i in range(0,len(self.data['rates']['excit']['col_transitions'])):
            # the if statement here is to account for files that have had energies
            # deleted from the begining of the adf04. It is assumed if those
            # entries are not at the top of the adf04 file then the user does not
            # want to included then this allows the matrix to be made ignoring
            # levels above the levels taken out in the file.
            if( self.data['rates']['excit']['col_transitions'][i,0] < len(self.data['atomic']['energy'])+1):
                for j in range(0,len(self.data['user']['temp_grid'])):

                    self.data['cr_matrix']['q_ji'][self.data['rates']['excit']['col_transitions'][i,0]-1,
                                                   self.data['rates']['excit']['col_transitions'][i,1]-1, j] =  \
                    1/(2*self.data['atomic']['w'][self.data['rates']['excit']['col_transitions'][i,0]-1] +1)* \
                    np.sqrt(13.6058/self.data['user']['temp_grid'][j]) * \
                    self.data['rates']['excit']['col_excit_interp'][i,j]*2.1716E-8

                    self.data['cr_matrix']['q_ij'][self.data['rates']['excit']['col_transitions'][i,0]-1,
                                                   self.data['rates']['excit']['col_transitions'][i,1]-1, j] =  \
                    (2*self.data['atomic']['w'][self.data['rates']['excit']['col_transitions'][i,0]-1]+1)/  \
                    (2*self.data['atomic']['w'][self.data['rates']['excit']['col_transitions'][i,1]-1]+1) * \
                    np.exp(-np.abs(self.data['atomic']['energy'][self.data['rates']['excit']['col_transitions'][i,0]-1] - \
                    self.data['atomic']['energy'][self.data['rates']['excit']['col_transitions'][i,1]-1])*  \
                    EV_PER_WAVENUMBER/self.data['user']['temp_grid'][j]) * \
                    self.data['cr_matrix']['q_ji'][self.data['rates']['excit']['col_transitions'][i,0]-1,
                                                   self.data['rates']['excit']['col_transitions'][i,1]-1,j]

                self.data['cr_matrix']['A_ji'][self.data['rates']['excit']['col_transitions'][i,0]-1,
                                               self.data['rates']['excit']['col_transitions'][i,1]-1] = \
                                                                                            self.data['rates']['a_val'][i]


        #sum up all the spontaneous emissions from a level for use in natural broadening
        self.data['cr_matrix']['A_ji_loss'] = np.sum(self.data['cr_matrix']['A_ji'],axis=1)
        
        #transpose the self.data['cr_matrix']['q_ij'] matrix so the indexes make sense
        self.data['cr_matrix']['q_ij'] = self.data['cr_matrix']['q_ij'].transpose(1,0,2)

        
        #one column for each parent of the adf04 file (its first line), which the ionization,
        #recombination and charge exchange rates are made for
        nsigmaplus = len(self.data['atomic']['ion_pot'])
        nsigmaplus_cx = 0
        if(self.data['user']['use_cx']):
            nsigmaplus_cx = nsigmaplus



        if(self.data['user']['temp_dens_pair']):

            self.data['cr_matrix']['cr'] = np.zeros((len(self.data['atomic']['energy'])+nsigmaplus+nsigmaplus_cx,
                                                     len(self.data['atomic']['energy'])+nsigmaplus+nsigmaplus_cx,
                                                     len(self.data['user']['temp_grid'])))



        else:
            self.data['cr_matrix']['cr'] = np.zeros((len(self.data['atomic']['energy'])+nsigmaplus+nsigmaplus_cx,
                                                     len(self.data['atomic']['energy'])+nsigmaplus+nsigmaplus_cx,
                                                     len(self.data['user']['temp_grid']),
                                                     len(self.data['user']['dens_grid'])))



        if(self.data['user']['temp_dens_pair']):

            #level i depopulating mechanisms
            #these are all the transitions from the level
            for i in range(0,len(self.data['atomic']['energy'])):

                self.data['cr_matrix']['cr'][i,i,:]  =  -1*np.sum(self.data['cr_matrix']['A_ji'][i,:])
                #these are all the excitation and dexcitation bringing you out of the level
                self.data['cr_matrix']['cr'][i,i,:] = self.data['cr_matrix']['cr'][i,i,:] - \
                np.sum(np.einsum('ij,j->ij',self.data['cr_matrix']['q_ji'][i,:,:],
                                 self.data['user']['dens_grid']),axis=0)-\
                np.sum(np.einsum('ij,j->ij',self.data['cr_matrix']['q_ij'][i,:,:],
                                 self.data['user']['dens_grid']),axis=0)
                


                if(self.data['user']['use_ionization']):

                    self.data['cr_matrix']['cr'][i,i,:] = self.data['cr_matrix']['cr'][i,i,:] - \
                                               np.sum(np.einsum('ij,j->ij',self.data['rates']['ioniz']['ionization'][i,:,:],
                                                                  self.data['user']['dens_grid']),axis=0)
                    



                self.data['cr_matrix']['cr'][i,0:len(self.data['atomic']['energy']),:] = \
                                        self.data['cr_matrix']['cr'][i,0:len(self.data['atomic']['energy']),:] + \
                                        self.data['cr_matrix']['A_ji'][:,i,None]

                #these are excitation and dexciation into the level i
                self.data['cr_matrix']['cr'][i,0:len(self.data['atomic']['energy']),:] = \
                                self.data['cr_matrix']['cr'][i,:len(self.data['atomic']['energy']),:] + \
                         np.einsum('ij,j->ij',self.data['cr_matrix']['q_ij'][:,i,:],
                                   self.data['user']['dens_grid']) + \
                         np.einsum('ij,j->ij',self.data['cr_matrix']['q_ji'][:,i,:],
                                   self.data['user']['dens_grid'])


            if(self.data['user']['use_recombination']):
                for p in range(0,nsigmaplus):
                    #these are the rates from the plus ion into the current ion levels
                    self.data['cr_matrix']['cr'][0:len(self.data['atomic']['energy']),
                                                 len(self.data['atomic']['energy'])+p,:] =\
                    self.data['cr_matrix']['cr'][0:len(self.data['atomic']['energy']),
                                                 len(self.data['atomic']['energy'])+p,:] + \
                    np.einsum('ij,j->ij',self.data['rates']['recomb']['recombination'][:,p,:],
                              self.data['user']['dens_grid'])

                    #this is the sum of all the pop lost to the current ion levels taken
                    #out of the plus ion
                    self.data['cr_matrix']['cr'][len(self.data['atomic']['energy'])+p,
                                                 len(self.data['atomic']['energy'])+p,:] = \
                    self.data['cr_matrix']['cr'][len(self.data['atomic']['energy'])+p,
                                                 len(self.data['atomic']['energy'])+p,:] - \
                    np.sum(np.einsum('ij,j->ij',self.data['rates']['recomb']['recombination'][:,p,:],
                                                 self.data['user']['dens_grid']),axis=0)
            #three body recombination out of plus ion to current ion
            if(self.data['user']['use_recombination_three_body']):
                for p in range(0,nsigmaplus):
                    self.data['cr_matrix']['cr'][0:len(self.data['atomic']['energy']),
                                                 len(self.data['atomic']['energy'])+p,:] = \
                    self.data['cr_matrix']['cr'][0:len(self.data['atomic']['energy']),
                                                 len(self.data['atomic']['energy'])+p,:] + \
                    np.einsum('ij,j->ij',self.data['rates']['recomb']['recomb_three_body'][:,p,:],
                                          self.data['user']['dens_grid']**2)

                    self.data['cr_matrix']['cr'][len(self.data['atomic']['energy'])+p,
                                                 len(self.data['atomic']['energy'])+p,:] = \
                    self.data['cr_matrix']['cr'][len(self.data['atomic']['energy'])+p,
                                                 len(self.data['atomic']['energy'])+p,:] - \
                    np.sum(np.einsum('ij,j->ij',self.data['rates']['recomb']['recomb_three_body'][:,p,:],
                                          self.data['user']['dens_grid']**2),axis=0)

            if(self.data['user']['use_ionization']):
                for p in range(0,nsigmaplus):
                    self.data['cr_matrix']['cr'][len(self.data['atomic']['energy'])+ p,
                                                     0:len(self.data['atomic']['energy']),:] = \
                                         self.data['cr_matrix']['cr'][len(self.data['atomic']['energy'])+p,
                                                             0:len(self.data['atomic']['energy']),:] + \
                                   np.einsum('ij,j->ij',self.data['rates']['ioniz']['ionization'][:,p,:],
                                                       self.data['user']['dens_grid'])
            if(self.data['user']['use_cx']):

                for p in range(0,nsigmaplus_cx):
                    self.data['cr_matrix']['cr'][0:len(self.data['atomic']['energy']),
                                                 len(self.data['atomic']['energy'])+nsigmaplus+p,:] =\
                    self.data['cr_matrix']['cr'][0:len(self.data['atomic']['energy']),
                                                 len(self.data['atomic']['energy'])+nsigmaplus+p,:] + \
                    np.einsum('ij,j->ij',self.data['rates']['cx']['cx'][:,p,:],
                              self.data['user']['hdens_grid'])
        else:

            for i in range(0,len(self.data['atomic']['energy'])):
                #level i depopulating mechanisms
                #these are all the transitions from the level

                self.data['cr_matrix']['cr'][i,i,:,:]  =  -1*np.sum(self.data['cr_matrix']['A_ji'][i,:])
                #these are all the excitation and dexcitation bringing you out of the level
                self.data['cr_matrix']['cr'][i,i,:,:] = self.data['cr_matrix']['cr'][i,i,:,:] - \
                np.sum(np.einsum('ij,k->ijk',self.data['cr_matrix']['q_ji'][i,:,:],
                                 self.data['user']['dens_grid']),axis=0)-\
                np.sum(np.einsum('ij,k->ijk',self.data['cr_matrix']['q_ij'][i,:,:],
                                 self.data['user']['dens_grid']),axis=0)


                #these are the ways to ionize out of ion
                if(self.data['user']['use_ionization']):

                    self.data['cr_matrix']['cr'][i,i,:,:] = self.data['cr_matrix']['cr'][i,i,:,:] - \
                                          np.sum(np.einsum('ij,k->ijk',self.data['rates']['ioniz']['ionization'][i,:,:],
                                                           self.data['user']['dens_grid']),axis=0)

                #level i populating mechanisms
                #these are the transition rates from higher levels into the level i
                self.data['cr_matrix']['cr'][i,0:len(self.data['atomic']['energy']),:,:] = \
                                        self.data['cr_matrix']['cr'][i,0:len(self.data['atomic']['energy']),:,:] + \
                                        self.data['cr_matrix']['A_ji'][:,i,None,None]

                #these are excitation and dexciation into the level i
                self.data['cr_matrix']['cr'][i,0:len(self.data['atomic']['energy']),:,:] = \
                                self.data['cr_matrix']['cr'][i,:len(self.data['atomic']['energy']),:,:] + \
                         np.einsum('ij,k->ijk',self.data['cr_matrix']['q_ij'][:,i,:],
                                   self.data['user']['dens_grid']) + \
                         np.einsum('ij,k->ijk',self.data['cr_matrix']['q_ji'][:,i,:],
                                   self.data['user']['dens_grid'])

            #recombination out of plus ion to current
            if(self.data['user']['use_recombination']):
                for p in range(0,nsigmaplus):
                    #these are the rates from the plus ion into the current ion levels
                    self.data['cr_matrix']['cr'][0:len(self.data['atomic']['energy']),
                                                 len(self.data['atomic']['energy'])+p,:,:] =\
                    self.data['cr_matrix']['cr'][0:len(self.data['atomic']['energy']),
                                                 len(self.data['atomic']['energy'])+p,:,:] + \
                    np.einsum('ij,k->ijk',self.data['rates']['recomb']['recombination'][:,p,:],
                              self.data['user']['dens_grid'])

                    #this is the sum of all the pop lost to the current ion levels taken
                    #out of the plus ion
                    self.data['cr_matrix']['cr'][len(self.data['atomic']['energy'])+p,
                                                 len(self.data['atomic']['energy'])+p,:,:] = \
                    self.data['cr_matrix']['cr'][len(self.data['atomic']['energy'])+p,
                                                 len(self.data['atomic']['energy'])+p,:,:] - \
                    np.sum(np.einsum('ij,k->ijk',self.data['rates']['recomb']['recombination'][:,p,:],
                                                 self.data['user']['dens_grid']),axis=0)
            #three body recombination out of plus ion to current ion
            if(self.data['user']['use_recombination_three_body']):
                for p in range(0,nsigmaplus):
                    self.data['cr_matrix']['cr'][0:len(self.data['atomic']['energy']),
                                                 len(self.data['atomic']['energy'])+p,:,:] = \
                    self.data['cr_matrix']['cr'][0:len(self.data['atomic']['energy']),
                                                 len(self.data['atomic']['energy'])+p,:,:] + \
                    np.einsum('ij,k->ijk',self.data['rates']['recomb']['recomb_three_body'][:,p,:],
                                          self.data['user']['dens_grid']**2)

                    self.data['cr_matrix']['cr'][len(self.data['atomic']['energy'])+p,
                                                 len(self.data['atomic']['energy'])+p,:,:] = \
                    self.data['cr_matrix']['cr'][len(self.data['atomic']['energy'])+p,
                                                 len(self.data['atomic']['energy'])+p,:,:] - \
                    np.sum(np.einsum('ij,k->ijk',self.data['rates']['recomb']['recomb_three_body'][:,p,:],
                                          self.data['user']['dens_grid']**2),axis=0)



            if(self.data['user']['use_cx']):

                for p in range(0,nsigmaplus_cx):
                    self.data['cr_matrix']['cr'][0:len(self.data['atomic']['energy']),
                                                 len(self.data['atomic']['energy'])+nsigmaplus+p,:] =\
                    self.data['cr_matrix']['cr'][0:len(self.data['atomic']['energy']),
                                                 len(self.data['atomic']['energy'])+nsigmaplus+p,:] + \
                    np.einsum('ij,k->ijk',self.data['rates']['cx']['cx'][:,p,:],
                              self.data['user']['hdens_grid'])

                    
            if(self.data['user']['use_ionization']):
                for p in range(0,nsigmaplus):
                    self.data['cr_matrix']['cr'][len(self.data['atomic']['energy'])+ p,
                                                     0:len(self.data['atomic']['energy']),:,:] = \
                                         self.data['cr_matrix']['cr'][len(self.data['atomic']['energy'])+p,
                                                             0:len(self.data['atomic']['energy']),:,:] + \
                                   np.einsum('ij,k->ijk',self.data['rates']['ioniz']['ionization'][:,p,:],
                                                       self.data['user']['dens_grid'])

        #the loss terms of the levels (radiative, collisional and ionization), which are the
        #diagonal of the CR matrix for the levels
        levels = np.arange(len(self.data['atomic']['energy']))
        self.data['cr_matrix']['cr_loss'] = np.zeros_like(self.data['cr_matrix']['cr'])
        self.data['cr_matrix']['cr_loss'][levels,levels] = self.data['cr_matrix']['cr'][levels,levels]




            

                
    def solve_quasi_static(self):
        """This function will solve the CR matrix using the quasistatic approximation, after solving this problem
           the generalized radiative coefficients GCRs will be calculated other claculated quanities such as 
           PEC and SXB are also calculated. This function is analgous to ADAS208.


           Creates the ['processed'] dictionary which will hold all of the quanties that require the CR
           matrix to be solved in order to be obtained. 
           
        """
        
        if('cr_matrix' not in self.data.keys()):
            self.populate_cr_matrix()
        levels_to_keep = np.setdiff1d( np.linspace(0,len(self.data['atomic']['energy'])-1,
                                              len(self.data['atomic']['energy']) ,dtype='int64'),
                                       self.data['atomic']['metas'])


        #[:,0:len(self.data['atomic']['metas']),:,:] 
        self.data['cr_matrix']['beta']= \
                                          -self.data['cr_matrix']['cr'][levels_to_keep][:,self.data['atomic']['metas']]
        cr_red = self.data['cr_matrix']['cr'][levels_to_keep][:,levels_to_keep]

        self.data['processed'] = {}
        self.data['cr_matrix']['cr_red'] = cr_red
        self.data['processed']['excited_levels'] = levels_to_keep

        num_parents = len(self.data['atomic']['ion_pot'])
        if(self.data['user']['use_recombination'] or self.data['user']['use_recombination_three_body']):
            recomb_driving_lvls = len(self.data['atomic']['energy']) + np.arange(num_parents)
            self.data['cr_matrix']['beta'] = np.append(self.data['cr_matrix']['beta'],
                                            -self.data['cr_matrix']['cr'][levels_to_keep][:,recomb_driving_lvls],axis=1)



        if(self.data['user']['use_cx']):
            #the charge exchange columns follow the parent columns of the CR matrix, which it
            #has whether or not recombination is used
            cx_driving_lvls = len(self.data['atomic']['energy']) + num_parents + np.arange(num_parents)
            self.data['cr_matrix']['beta'] = np.append(self.data['cr_matrix']['beta'],
                                            -self.data['cr_matrix']['cr'][levels_to_keep][:,cx_driving_lvls],axis=1)

            
            
        if(self.data['user']['temp_dens_pair']):
            self.data['cr_matrix']['cr_red_inv'] = np.zeros((len(cr_red),len(cr_red), len(self.data['user']['temp_grid'])))
            self.data['cr_matrix']['cr_red_inv'] = np.linalg.inv(cr_red.transpose(2,0,1)).transpose(1,2,0)

            if(self.data['user']['use_recombination'] or self.data['user']['use_recombination_three_body']):
                self.data['processed']['pops'] = np.zeros((len(cr_red),
                                                len(self.data['atomic']['metas'])+len(self.data['atomic']['ion_pot']),
                                                len(self.data['user']['temp_grid'])))
            else:
                self.data['processed']['pops'] = np.zeros((len(cr_red),
                                                len(self.data['atomic']['metas']),
                                                len(self.data['user']['temp_grid'])))

            self.data['processed']['pops'] = np.einsum('ijk,jnk->ink',
                                                   self.data['cr_matrix']['cr_red_inv'],
                                                   self.data['cr_matrix']['beta'])






                
        else:
            self.data['cr_matrix']['cr_red_inv'] = np.zeros((len(cr_red),len(cr_red), len(self.data['user']['temp_grid']),
                               len(self.data['user']['dens_grid'])))
        
            self.data['cr_matrix']['cr_red_inv'] = np.linalg.inv(cr_red.transpose(2,3,0,1)).transpose(2,3,0,1)


            '''This part of code was not needed because of using einsum
            if(self.data['user']['use_recombination'] or self.data['user']['use_recombination_three_body']):
                self.data['processed']['pops'] = np.zeros((len(cr_red),
                                                len(self.data['atomic']['metas'])+len(self.data['atomic']['ion_pot']),
                                                len(self.data['user']['temp_grid']),
                                                len(self.data['user']['dens_grid'])))
            else:
                self.data['processed']['pops'] = np.zeros((len(cr_red),
                                                len(self.data['atomic']['metas']),
                                                len(self.data['user']['temp_grid']),
                                                len(self.data['user']['dens_grid'])))
            '''
            self.data['processed']['pops'] = np.einsum('ijkl,jnkl->inkl',
                                                   self.data['cr_matrix']['cr_red_inv'],
                                                   self.data['cr_matrix']['beta'])
        #normalization of populations is done here. Quasistatic approximation assumes that
        #there is a small population in 'excited' states, this is sometimes not the case
        #so the populations must be renomalized the default when there is one metastable
        #present is to normalize, but there is a choice for the user to decide
        #default when there is more than one metastable is to not normalize



        if(len(self.data['atomic']['metas']) > 1):

            self.data['processed']['driving_populations_norm'] = False
        else:
            self.data['processed']['driving_populations_norm'] = True

        if(not self.data['user']['default_pop_norm']):
            self.data['processed']['driving_populations_norm'] =\
                          not self.data['processed']['driving_populations_norm']

        #Only the excitation columns are normalized, their populations are relative to the
        #metastable of this ion. Recombination and charge exchange populations are relative
        #to the parent populations, which are not part of this ion.
        nmeta = len(self.data['atomic']['metas'])
        if(self.data['processed']['driving_populations_norm']):
            self.data['processed']['pops'][:,0:nmeta] = self.data['processed']['pops'][:,0:nmeta] /\
                                             (1 + np.sum(self.data['processed']['pops'][:,0:nmeta],axis=0))

        #setup pec and wavelength stuff start them as lists because
        #we wont know which levels actually have pec values before we loop through them
        #this lists are eventually converted into np arrays
        self.data['processed']['pecs'] = []
        self.data['processed']['pec_levels'] = []
        self.data['processed']['wave_vac'] = []

        pls = []
        for i in range(0,len(self.data['cr_matrix']['A_ji'])):
            for j in range(0,len(levels_to_keep)):
                #if statement here because there will be a number of transitions that have zero
                #for the PEC value and don't want to have to keep track of these.

                if(levels_to_keep[j] > i ):#and self.data['cr_matrix']['A_ji'][j,i] > 1E-31):

                    self.data['processed']['pecs'].append( self.data['cr_matrix']['A_ji'][levels_to_keep[j],i]*\
                                                        self.data['processed']['pops'][j]/ \
                                                        self.data['user']['dens_grid'])
                    #line power, photon energy * PEC, in W cm3
                    pls.append(np.abs(self.data['atomic']['energy'][levels_to_keep[j]] - self.data['atomic']['energy'][i])*                               J_PER_WAVENUMBER*(self.data['cr_matrix']['A_ji'][levels_to_keep[j],i]*                                                        self.data['processed']['pops'][j]/                                                         self.data['user']['dens_grid']))

                    #levels with equal energies (degenerate in the file) have an infinite wavelength
                    with np.errstate(divide='ignore'):
                        self.data['processed']['wave_vac'].append(1.e7/abs(self.data['atomic']['energy'][levels_to_keep[j]]\
                                                                            - self.data['atomic']['energy'][i]))

                    self.data['processed']['pec_levels'].append( np.array([levels_to_keep[j],i]))


        #radiated line power coefficients in W cm3, summed over lines for the driving populations:
        #excitation from the metastables (plt), recombination (prb) and charge exchange (prc)
        #from the parents
        nrecomb = 0
        if(self.data['user']['use_recombination'] or self.data['user']['use_recombination_three_body']):
            nrecomb = len(self.data['atomic']['ion_pot'])
        self.data['processed']['pls'] = np.asarray(pls)
        self.data['processed']['plt'] = np.sum(self.data['processed']['pls'][:,0:nmeta], axis=0)
        self.data['processed']['prb'] = np.sum(self.data['processed']['pls'][:,nmeta:nmeta+nrecomb], axis=0)
        self.data['processed']['prc'] = np.sum(self.data['processed']['pls'][:,nmeta+nrecomb:], axis=0)

        self.data['processed']['pecs'] = np.asarray(self.data['processed']['pecs'])

        self.data['processed']['wave_vac'] = np.asarray(self.data['processed']['wave_vac'])
        self.data['processed']['wave_air'] = vac_to_air(self.data['processed']['wave_vac'])
        self.data['processed']['pec_levels'] = np.asarray(self.data['processed']['pec_levels'])

        if(self.data['user']['temp_dens_pair']):

            self.data['processed']['scd'] = np.zeros((len(self.data['atomic']['metas']),
                                                      len(self.data['atomic']['ion_pot']),
                                                      len(self.data['user']['temp_grid'])))
            self.data['processed']['acd'] = np.zeros((len(self.data['atomic']['metas']),
                                                      len(self.data['atomic']['ion_pot']),
                                                      len(self.data['user']['temp_grid'])))
            self.data['processed']['ccd'] = np.zeros((len(self.data['atomic']['metas']),
                                                      len(self.data['atomic']['ion_pot']),
                                                      len(self.data['user']['temp_grid'])))
            self.data['processed']['xcd'] = np.zeros((len(self.data['atomic']['ion_pot']),
                                                      len(self.data['atomic']['ion_pot']),
                                                      len(self.data['user']['temp_grid'])))
            self.data['processed']['pop_lvl'] = np.zeros((len(self.data['cr_matrix']['cr_red_inv']),
                                                           len(self.data['cr_matrix']['cr_red_inv']),
                                                           len(self.data['atomic']['metas']),
                                                           len(self.data['user']['temp_grid'])))

            #these are how levels get populated
            self.data['processed']['pop_lvl'] = np.einsum('ijk,jmk->ijmk',
                                                          self.data['cr_matrix']['cr_red_inv'],
                                                          self.data['cr_matrix']['beta'][:,:,:])

            self.data['processed']['pops_no_norm'] = np.sum(self.data['processed']['pop_lvl'],axis=1)

            if(self.data['processed']['driving_populations_norm']):
                self.data['processed']['pop_lvl'][:,:,0:nmeta] = self.data['processed']['pop_lvl'][:,:,0:nmeta]/   \
                                                (1+np.sum(np.sum(self.data['processed']['pop_lvl'][:,:,0:nmeta],axis=1),axis=0))

            #the F matrix

            ##############################################################
            #
            # NOT THE SAME AS THE ADAS DEFINITION, ADAS DIVIDES BY n_e
            #
            ###############################################################
            self.data['processed']['F'] = np.sum(self.data['processed']['pop_lvl']\
                                                 [:,:,0:len(self.data['atomic']['metas']),:],axis=1)
            #effective ionization rate
            if(self.data['user']['use_ionization']):
                self.data['processed']['scd'] = np.einsum('ipk,imk->mpk',
                                                          self.data['rates']['ioniz']['ionization'][levels_to_keep,:],
                                                          self.data['processed']['F'])

                if(self.data['processed']['driving_populations_norm']):
                    self.data['processed']['scd'] = self.data['processed']['scd'] + \
                                                  np.einsum('ipk,ik->ipk',
                                                  self.data['rates']['ioniz']['ionization'][self.data['atomic']['metas'],:],
                              1/(1+np.sum(self.data['processed']['pops_no_norm'][:,0:nmeta,:],axis=0)))

                else:
                    self.data['processed']['scd'] = self.data['processed']['scd'] + \
                                             self.data['rates']['ioniz']['ionization'][self.data['atomic']['metas'],:,:]


            if(self.data['user']['use_recombination'] and self.data['user']['use_recombination_three_body']):
                #this is the total recombination with three body and the rates that in included in the adf04 file
                recomb_coeff = np.einsum('ijk,k->ijk',
                                         self.data['rates']['recomb']['recomb_three_body'],
                                         self.data['user']['dens_grid']) + \
                                         self.data['rates']['recomb']['recombination'][:,:,:]
            elif(self.data['user']['use_recombination']):
                recomb_coeff = self.data['rates']['recomb']['recombination'][:,:,:]
            elif(self.data['user']['use_recombination_three_body']):            
                recomb_coeff = np.einsum('ijk,k->ijk',
                                         self.data['rates']['recomb']['recomb_three_body'],
                                         self.data['user']['dens_grid'])
            if(self.data['user']['use_recombination'] or self.data['user']['use_recombination_three_body']):
                #effective recombination rate
                self.data['processed']['acd'] = -np.einsum('njk,jmk->nmk',

                               self.data['cr_matrix']['cr'][np.c_[self.data['atomic']['metas']], levels_to_keep,:],
                               np.einsum('ijk,jmk->imk', self.data['cr_matrix']['cr_red_inv'][0:len(self.data['atomic']['energy']),
                               0:len(self.data['atomic']['energy']),:],
                               recomb_coeff[levels_to_keep,:,:]))

                self.data['processed']['acd'] = self.data['processed']['acd'] + recomb_coeff[self.data['atomic']['metas'],:,:]

            if(self.data['user']['use_cx']):
                #effective thermal charge exchange coefficient
                self.data['processed']['ccd'] = -np.einsum('njk,jmk->nmk',

                               self.data['cr_matrix']['cr'][np.c_[self.data['atomic']['metas']], levels_to_keep,:],
                               np.einsum('ijk,jmk->imk', self.data['cr_matrix']['cr_red_inv'][0:len(self.data['atomic']['energy']),
                               0:len(self.data['atomic']['energy']),:],
                               self.data['rates']['cx']['cx'][levels_to_keep,:,:]))

                self.data['processed']['ccd'] = self.data['processed']['ccd'] + \
                                                self.data['rates']['cx']['cx'][self.data['atomic']['metas'],:,:]
                
            self.data['processed']['qcd'] = np.zeros((len(self.data['atomic']['metas']),
                                                      len(self.data['atomic']['metas']),
                                                      len(self.data['user']['temp_grid'])))

            t = []
            for n in range(0,len(self.data['atomic']['metas'])):
                metas_to_keep = np.setdiff1d( self.data['atomic']['metas'],
                                              self.data['atomic']['metas'][n])
                #calculate the metastable cross coupling coefficent, this the number of atoms that start in one
                #metastable and then end up in a different metastable
                metas_to_keep_ind = np.arange(self.data['atomic']['metas'].shape[0])\
                                         [_isin(self.data['atomic']['metas'], metas_to_keep)]
                if(len(self.data['processed']['qcd']) > 1):

                    self.data['processed']['qcd'][metas_to_keep_ind,n,:] = np.einsum('nk,k->nk',

                                            self.data['cr_matrix']['cr'][self.data['atomic']['metas'][n],metas_to_keep,:] +\
                                            np.einsum('mk,mnk->nk', self.data['cr_matrix']['cr'][self.data['atomic']['metas'][n],
                                            levels_to_keep,:],
                                            self.data['processed']['F'][:,metas_to_keep_ind,:])
                                            ,1/self.data['user']['dens_grid'])

            if(self.data['user']['use_recombination'] or self.data['user']['use_recombination_three_body']):            
                for m in range(0,len(self.data['atomic']['ion_pot'])):

                    metasplus_to_keep = np.setdiff1d( np.linspace(0,len(self.data['atomic']['ion_pot'])-1,
                                                                  len(self.data['atomic']['ion_pot']),dtype='int'),m)
                    metasplus_to_keep_ind = np.arange(self.data['atomic']['ion_pot'].shape[0])\
                                         [_isin(self.data['atomic']['ion_pot_lvl']-1, metasplus_to_keep)]
                    #parent cross coupling coefficient, start in the parent atom, recombine get redistrubted then
                    #ionize back into the parent but into a different parent metastable
                    #Use S*n_e from the CR matrix so XCD has units cm^3/s.
                    if(len(self.data['processed']['xcd']) > 1):
                        self.data['processed']['xcd'][metasplus_to_keep_ind,np.array([m]),:] =-np.einsum('ik,imk->mk',

                                self.data['cr_matrix']['cr'][len(self.data['atomic']['energy'])+m,levels_to_keep,:],np.einsum('ijk,jmk->imk',
                                self.data['cr_matrix']['cr_red_inv'][0:len(self.data['atomic']['energy']),
                                                                     0:len(self.data['atomic']['energy']),:],
                                recomb_coeff[levels_to_keep][:,metasplus_to_keep,:]))

        else:
        
            self.data['processed']['scd'] = np.zeros((len(self.data['atomic']['metas']),
                                                      len(self.data['atomic']['ion_pot']),
                                                      len(self.data['user']['temp_grid']),
                                                      len(self.data['user']['dens_grid'])))
            self.data['processed']['acd'] = np.zeros((len(self.data['atomic']['metas']),
                                                      len(self.data['atomic']['ion_pot']),
                                                      len(self.data['user']['temp_grid']),
                                                      len(self.data['user']['dens_grid'])))
            self.data['processed']['ccd'] = np.zeros((len(self.data['atomic']['metas']),
                                                      len(self.data['atomic']['ion_pot']),
                                                      len(self.data['user']['temp_grid']),
                                                      len(self.data['user']['dens_grid'])))
            
            self.data['processed']['xcd'] = np.zeros((len(self.data['atomic']['ion_pot']),
                                                      len(self.data['atomic']['ion_pot']),
                                                      len(self.data['user']['temp_grid']),
                                                      len(self.data['user']['dens_grid'])))
            self.data['processed']['pop_lvl'] = np.zeros((len(self.data['cr_matrix']['cr_red_inv']),
                                                           len(self.data['cr_matrix']['cr_red_inv']),
                                                           len(self.data['atomic']['metas']),
                                                           len(self.data['user']['temp_grid']),
                                                           len(self.data['user']['dens_grid'])))
            #these are how levels get populated
            self.data['processed']['pop_lvl'] = np.einsum('ijkl,jmkl->ijmkl',
                                                          self.data['cr_matrix']['cr_red_inv'],
                                                          self.data['cr_matrix']['beta'][:,:,:,:])

            self.data['processed']['pops_no_norm'] = np.sum(self.data['processed']['pop_lvl'],axis=1)

            if(self.data['processed']['driving_populations_norm']):
                self.data['processed']['pop_lvl'][:,:,0:nmeta] = self.data['processed']['pop_lvl'][:,:,0:nmeta]/   \
                                                (1+np.sum(np.sum(self.data['processed']['pop_lvl'][:,:,0:nmeta],axis=1),axis=0))


            #the F matrix

            self.data['processed']['F'] = np.sum(self.data['processed']['pop_lvl']\
                                                 [:,:,0:len(self.data['atomic']['metas']),:,:],axis=1)
            #effective ionization rate
            if(self.data['user']['use_ionization']):
                self.data['processed']['scd'] = np.einsum('ipk,imkl->mpkl',
                                                          self.data['rates']['ioniz']['ionization'][levels_to_keep,:,:],
                                                          self.data['processed']['F'])

                if(self.data['processed']['driving_populations_norm']):
                    self.data['processed']['scd'] = self.data['processed']['scd'] + \
                                                  np.einsum('ipk,ikl->ipkl',
                                                  self.data['rates']['ioniz']['ionization'][self.data['atomic']['metas'],:,:],
                              1/(1+np.sum(self.data['processed']['pops_no_norm'][:,0:nmeta,:,:],axis=0)))

                else:
                    self.data['processed']['scd'] = self.data['processed']['scd'] + \
                                             self.data['rates']['ioniz']['ionization'][self.data['atomic']['metas'],:,:,None]

            if(self.data['user']['use_recombination'] and self.data['user']['use_recombination_three_body']):
                #this is the total recombination with three body and the rates that in included in the adf04 file
                recomb_coeff = np.einsum('ijk,l->ijkl',
                                         self.data['rates']['recomb']['recomb_three_body'],
                                         self.data['user']['dens_grid']) + \
                                         self.data['rates']['recomb']['recombination'][:,:,:,None]
            elif(self.data['user']['use_recombination']):
                recomb_coeff = self.data['rates']['recomb']['recombination'][:,:,:,None]
            elif(self.data['user']['use_recombination_three_body']):            
                recomb_coeff = np.einsum('ijk,l->ijkl',
                                         self.data['rates']['recomb']['recomb_three_body'],
                                         self.data['user']['dens_grid'])
            if(self.data['user']['use_recombination'] or self.data['user']['use_recombination_three_body']):
                #effective recombination rate
                self.data['processed']['acd'] = -np.einsum('njkl,jmkl->nmkl',

                               self.data['cr_matrix']['cr'][np.c_[self.data['atomic']['metas']], levels_to_keep,:,:],
                               np.einsum('ijkl,jmkl->imkl', self.data['cr_matrix']['cr_red_inv'][0:len(self.data['atomic']['energy']),
                               0:len(self.data['atomic']['energy']),:,:],
                               recomb_coeff[levels_to_keep,:,:,:]))

                self.data['processed']['acd'] = self.data['processed']['acd'] + recomb_coeff[self.data['atomic']['metas'],:,:,:]

            if(self.data['user']['use_cx']):
                #effective recombination rate
                
                cx_coeff = self.data['rates']['cx']['cx'][:,:,:,None]

                self.data['processed']['ccd'] = -np.einsum('njkl,jmkl->nmkl',

                               self.data['cr_matrix']['cr'][np.c_[self.data['atomic']['metas']], levels_to_keep,:,:],
                               np.einsum('ijkl,jmkl->imkl', self.data['cr_matrix']['cr_red_inv'][0:len(self.data['atomic']['energy']),
                               0:len(self.data['atomic']['energy']),:,:],
                               cx_coeff[levels_to_keep,:,:,:]))

                self.data['processed']['ccd'] = self.data['processed']['ccd'] + cx_coeff[self.data['atomic']['metas'],:,:,:]

            self.data['processed']['qcd'] = np.zeros((len(self.data['atomic']['metas']),
                                                      len(self.data['atomic']['metas']),
                                                      len(self.data['user']['temp_grid']),
                                                      len(self.data['user']['dens_grid'])))
            t = []
            for n in range(0,len(self.data['atomic']['metas'])):
                metas_to_keep = np.setdiff1d( self.data['atomic']['metas'],
                                              self.data['atomic']['metas'][n])
                #calculate the metastable cross coupling coefficent, this the number of atoms that start in one
                #metastable and then end up in a different metastable
                metas_to_keep_ind = np.arange(self.data['atomic']['metas'].shape[0])\
                                         [_isin(self.data['atomic']['metas'], metas_to_keep)]
                self.data['processed']['qcd'][metas_to_keep_ind,n,:,:] = np.einsum('nkl,l->nkl',

                                        self.data['cr_matrix']['cr'][self.data['atomic']['metas'][n],metas_to_keep,:,:] +\
                                        np.einsum('mkl,mnkl->nkl', self.data['cr_matrix']['cr'][self.data['atomic']['metas'][n],
                                        levels_to_keep,:,:],
                                        self.data['processed']['F'][:,metas_to_keep_ind,:,:]
                                        ),1/self.data['user']['dens_grid'])

            if(self.data['user']['use_recombination'] or self.data['user']['use_recombination_three_body']):            
                for m in range(0,len(self.data['atomic']['ion_pot'])):

                    metasplus_to_keep = np.setdiff1d( np.linspace(0,len(self.data['atomic']['ion_pot'])-1,
                                                                  len(self.data['atomic']['ion_pot']),dtype='int'),m)
                    metasplus_to_keep_ind = np.arange(self.data['atomic']['ion_pot'].shape[0])\
                                         [_isin(self.data['atomic']['ion_pot_lvl']-1, metasplus_to_keep)]
                    #parent cross coupling coefficient, start in the parent atom, recombine get redistrubted then
                    #ionize back into the parent but into a different parent metastable
                    #Use S*n_e from the CR matrix so XCD has units cm^3/s.
                    self.data['processed']['xcd'][metasplus_to_keep_ind,np.array([m]),:,:] =-np.einsum('ikl,imkl->mkl',

                            self.data['cr_matrix']['cr'][len(self.data['atomic']['energy'])+m,levels_to_keep,:,:],np.einsum('ijkl,jmkl->imkl',
                            self.data['cr_matrix']['cr_red_inv'][0:len(self.data['atomic']['energy']),
                                                                 0:len(self.data['atomic']['energy']),:,:],
                            recomb_coeff[levels_to_keep][:,metasplus_to_keep,:,:]))

                    
    def solve_time_dependent(self):
        """This function will solve the CR matrix with no assumptions. A matrix expoential is used to solve this problem.
           A source term can be included to mimick erosion of fresh atoms or injection of neutral gas or maybe even LIF

           Uses same CR matrix ".data['cr_matrix']['cr']" used to define the reduced CR matrix that the quasistatic solution solves.
           With charge exchange, the charge exchange takes population from the evolving parents: its rates are moved from
           the charge exchange columns to the parent columns, with the matching loss from the parents, so the solution
           conserves particles. td_n0 and td_source must then be zero for the charge exchange columns.

           Creates a new dictionary under ['processed'] call ['td']
           Stores eigenvalues and eigenvectors that were required for the solution (Theres also physics there see ColRadPy paper).
           Stores the time changing populations.
        
           R. LeVeque, Finite Difference Methods for Ordinary and Par-
           tial Differential Equations: Steady-State and Time-Dependent
           Problems (Classics in Applied Mathematics Classics in Applied
           Mathemat), Society for Industrial and Applied Mathematics,
           Philadelphia, PA, USA, 2007.
        """

        if('processed' not in self.data.keys()):
            self.data['processed'] = {}
        self.data['processed']['td'] = {}

        #Charge exchange takes population from the evolving parents. The quasistatic solution keeps
        #it in separate columns to give the charge exchange driven PECs and CCD; here its rates
        #into the levels are moved to the parent columns, with the matching loss from the parents,
        #so the time dependent solution conserves particles. The charge exchange columns are empty.
        matrix = self.data['cr_matrix']['cr']
        if(self.data['user']['use_cx']):
            nlev = len(self.data['atomic']['energy'])
            nparents = len(self.data['atomic']['ion_pot'])
            parents = nlev + np.arange(nparents)
            cx_cols = nlev + nparents + np.arange(nparents)
            for name in ('td_n0', 'td_source'):
                values = np.asarray(self.data['user'][name])
                if(values.size > 0 and np.any(values[cx_cols] != 0)):
                    raise ValueError(name + ' has non-zero entries ' + str(cx_cols) + ' for the charge ' +
                                     'exchange columns; charge exchange acts on the parents, whose ' +
                                     'entries are ' + str(parents))
            matrix = matrix.copy()
            cx = matrix[:nlev][:, cx_cols]
            matrix[:nlev, parents] += cx
            matrix[parents, parents] -= cx.sum(axis=0)
            matrix[:, cx_cols] = 0

        #check if a source term has been provided, if a source term has been provided solve the problem with s asource
        if(self.data['user']['td_source'].size >0):
            #solve the matrix with a source term
            self.data['processed']['td']['td_pop'],self.data['processed']['td']['eigenvals'],\
            self.data['processed']['td']['eigenvectors'] =\
            solve_matrix_exponential_source(matrix,
                                            self.data['user']['td_n0'],
                                            self.data['user']['td_source'],
                                            self.data['user']['td_t'])

        else:
            #no source term was provided solve just the normal matrix

            self.data['processed']['td']['td_pop'],self.data['processed']['td']['eigenvals'],\
            self.data['processed']['td']['eigenvectors'] = \
            solve_matrix_exponential(matrix,
                                            self.data['user']['td_n0'],
                                            self.data['user']['td_t'])


        levels_to_keep = np.setdiff1d( np.linspace(0,len(self.data['atomic']['energy'])-1,
                                              len(self.data['atomic']['energy']) ,dtype='int64'),
                                       self.data['atomic']['metas'])
        
        self.data['processed']['td']['pecs'] = []#photon emissivity coefficients
        self.data['processed']['td']['pls'] = []

        
        for i in range(0,len(self.data['cr_matrix']['A_ji'])):
            for j in range(0,len(levels_to_keep)):
                #if statement here because there will be a number of transitions that have zero
                #for the PEC value and don't want to have to keep track of these.

                if(levels_to_keep[j] > i ):#and self.data['cr_matrix']['A_ji'][j,i] > 1E-31):
                    #same ways the SS version
                    self.data['processed']['td']['pecs'].append( self.data['cr_matrix']['A_ji'][levels_to_keep[j],i]*\
                                                                 self.data['processed']['td']['td_pop'][levels_to_keep[j]]/\
                                                                 self.data['user']['dens_grid'])

                    #line power, photon energy * PEC, in W cm3
                    self.data['processed']['td']['pls'].append(np.abs(self.data['atomic']['energy'][levels_to_keep[j]] -                                                                       self.data['atomic']['energy'][i])*J_PER_WAVENUMBER*                                           (self.data['cr_matrix']['A_ji'][levels_to_keep[j],i]*self.data['processed']['td']['td_pop'][levels_to_keep[j]]/                                                         self.data['user']['dens_grid']))


        self.data['processed']['td']['pecs'] = np.asarray(self.data['processed']['td']['pecs'])
        self.data['processed']['td']['pls'] = np.asarray(self.data['processed']['td']['pls'])
        self.data['processed']['td']['plt'] =  np.sum(self.data['processed']['td']['pls'],axis=0)#sum up all contributions
        

        #just the time dependent version of the SCD coefficient,calculated the same as SS version
        #just remember not to include the population from the + stage
        self.data['processed']['td']['scd'] =  np.einsum('ipk,itkl->ptkl',self.data['rates']['ioniz']['ionization'],
                            self.data['processed']['td']['td_pop'][0:len(self.data['rates']['ioniz']['ionization']),:,:,:])


    def split_pec_multiplet(self):
        """This function will solve take LS resolved PECs and split them statistically among the j levels
           Note that is only works for dipole transitions. See "split_multiplet.py" for the transitions
           that this will split. Most transitsions of spectroscopic importance should be able to be
           split by this.

           Spin conserving dipole multiplets (``|dL| <= 1``) are split with the Condon and Shortley line
           strengths and lines with a single pair of j levels go to that pair. Other lines, which
           these factors do not describe (spin changing or ``|dL| > 1``), are kept unsplit at the
           wavelength between the statistically weighted centres of their terms, with split levels
           -1 and ['kept_unsplit'] True.

           Wavelengths use the split level energies of shift_j_res_energy_to_nist when available,
           otherwise the energies of the terms.

           Creates the ['split'] sub dictionary to hold the information of the split

        """
        if('processed' not in self.data.keys()):
            self.solve_quasi_static()

        if('split' not in self.data['processed'].keys() or 'term_map' not in self.data['processed']['split']):
            self.split_structure_terms_to_levels()
        split = self.data['processed']['split']

        term_map = split['term_map'].astype(int)
        if('energy' in split):
            level_energy = np.asarray(split['energy'], dtype=float)
        else:
            level_energy = np.asarray(self.data['atomic']['energy'], dtype=float)[term_map]
        #statistically weighted energy of the split levels of each term
        level_weight = 2*np.asarray(split['j'], dtype=float) + 1
        term_energy = np.bincount(term_map, level_weight*level_energy, len(self.data['atomic']['energy']))/\
                      np.maximum(np.bincount(term_map, level_weight, len(self.data['atomic']['energy'])), 1e-300)

        split['j_up'] = []
        split['j_low'] = []
        split['pecs'] = []
        split['wave_vac'] = []
        split['relative_inten'] = []
        split['pec_levels'] = []
        split['unsplit_pec_levels'] = []
        split['unres_pec_map'] = []
        split['kept_unsplit'] = []

        for i in range(0,len(self.data['processed']['pec_levels'])):
            up = self.data['processed']['pec_levels'][i,0]
            low = self.data['processed']['pec_levels'][i,1]
            #splitting the multiplet based on Condon and Shotly
            ju,jl,res = split_multiplet( (self.data['atomic']['S'][low]-1)/2.,
                                         self.data['atomic']['L'][low],
                                         (self.data['atomic']['S'][up]-1)/2.,
                                         self.data['atomic']['L'][up])
            split['j_low'].append(jl)
            split['j_up'].append(ju)
            split['relative_inten'].append(res)

            if(len(ju) == 1):
                #a single pair of levels has the whole line
                weights = np.ones(1)
            elif(self.data['atomic']['S'][up] == self.data['atomic']['S'][low] and
                 np.abs(self.data['atomic']['L'][up] - self.data['atomic']['L'][low]) <= 1 and np.sum(res) > 0):
                weights = res/np.sum(res)
            else:
                weights = None

            if(weights is None):
                #the line is kept unsplit between the centres of its terms
                split['pecs'].append(self.data['processed']['pecs'][i])
                split['pec_levels'].append(np.array([-1, -1]))
                split['unsplit_pec_levels'].append(self.data['processed']['pec_levels'][i])
                split['unres_pec_map'].append(i)
                split['kept_unsplit'].append(True)
                with np.errstate(divide='ignore'):
                    split['wave_vac'].append(1.e7/np.abs(term_energy[up] - term_energy[low]))
                continue

            for j in range(0,len(ju)):
                up_id = np.where((term_map == up) & (split['j'] == ju[j]))[0][0]
                low_id = np.where((term_map == low) & (split['j'] == jl[j]))[0][0]
                #give each split corresponding PEC its relative value add the transition to different maps
                split['pecs'].append(self.data['processed']['pecs'][i]*weights[j])
                split['pec_levels'].append(np.array([up_id, low_id]))
                split['unsplit_pec_levels'].append(self.data['processed']['pec_levels'][i])
                split['unres_pec_map'].append(i)
                split['kept_unsplit'].append(False)
                with np.errstate(divide='ignore'):
                    split['wave_vac'].append(1.e7/np.abs(level_energy[up_id] - level_energy[low_id]))

        split['wave_vac'] = np.asarray(split['wave_vac'])
        split['wave_air'] = vac_to_air(split['wave_vac'])
        split['pecs'] =  np.asarray(split['pecs'])
        split['pec_levels'] =  np.asarray(split['pec_levels'])
        split['unsplit_pec_levels'] =  np.asarray(split['unsplit_pec_levels'])
        split['unsplit_pecs_levels'] = split['unsplit_pec_levels']
        split['unres_pec_map'] =  np.asarray(split['unres_pec_map'])
        split['kept_unsplit'] = np.asarray(split['kept_unsplit'], dtype=bool)

    def auto_check_eissner_config(self):

        if(len(self.data['atomic']['config'][0]) < 3 and  (not any(c.isalpha() for c in self.data['atomic']['config'][0]) )):
            self.convert_eissner_to_ls()
        elif(len(self.data['atomic']['config'][0]) > 3 and (' ' not in self.data['atomic']['config'][0] and '.' not in self.data['atomic']['config'][0])):
            self.convert_eissner_to_ls()
        




        
    def convert_eissner_to_ls(self):

        config_tmp = []
        for i in range(0,len(self.data['atomic']['config'])):
            config_tmp.append(convert_eissner(self.data['atomic']['config'][i]))

        self.data['atomic']['config'] = np.asarray(config_tmp)




    def get_nist_conf_form(self):

        self.data['atomic']['nist_conf_form'] = np.empty_like(self.data['atomic']['config'])
        for i in range(0,len(self.data['atomic']['config'])):

            st = self.data['atomic']['config'][i].lower()
            rem = list(re.finditer(r'\d\w1' , st))
            if rem:
                tmpp = ''
                for ii in range(0,len(rem)):
                    st = st[:rem[ii].span()[1]-1-ii] + st[rem[ii].span()[1]+1-1-ii:]
            
            self.data['atomic']['nist_conf_form'][i] = st
        self.data['atomic']['nist_conf_form'] = self.data['atomic']['nist_conf_form'].astype('object')

        #remove closed subshells because thats what NIST does
        #probably a better way to do this but I cant be bothered
        closed_shells = np.array(['1s2','2s2','2p6','3s2','3p6','4s2','3d10','4p6','5s2','4f14','5d10'])

        #now do the stame thing for the NIST configurations
        shells_arr_tmp = np.empty_like(self.data['atomic']['nist_conf_form'].astype('<U1000'))

        test = np.where(np.char.find(closed_shells,
                              self.data['atomic']['nist_conf_form'][0][0:3])==0)[0] #[0:2]



        if(test.size >0):
            for ij in range(0,test[0]):

                add_shell = np.setdiff1d(range(self.data['atomic']['nist_conf_form'].shape[0]),
                                         np.where(np.char.find(self.data['atomic']['nist_conf_form'].astype('<U'),
                                                               closed_shells[ij][0:2])==0)[0])
                if(add_shell.size>0):
                    for ik in add_shell:

                        if(ij == 0):
                            if(ij == test[0] -1):
                                wat = shells_arr_tmp[ik] +  closed_shells[ij] + '.'
                                shells_arr_tmp[ik] = wat

                            else:
                                wat = shells_arr_tmp[ik] +  closed_shells[ij]
                                shells_arr_tmp[ik] = wat


                        else:
                            if(ij != test[0] -1):
                                wat = shells_arr_tmp[ik] + '.' + closed_shells[ij]
                                shells_arr_tmp[ik] = wat
                            else:
                                wat = shells_arr_tmp[ik] + '.' + closed_shells[ij] + '.'
                                shells_arr_tmp[ik] = wat


            self.data['atomic']['nist_conf_form'] = shells_arr_tmp + self.data['atomic']['nist_conf_form']

            #NIST leaves out the inner closed subshells that are not in its ground configuration
            self.data['atomic']['nist_conf_form'] = np.array(
                [strip_closed_shells(conf, closed_shells, self.data['nist']['levels'][0]['conf'])
                 for conf in self.data['atomic']['nist_conf_form']], dtype=object)

        
    def split_structure_terms_to_levels(self):
        """ split_structure_term_to_levels will take an LS resolved file like is common for low-z
            species and split it to j resolution. This is so the file can be shifted to NIST energy
            values as well as be spectroscopically 'accurate'. This is just using ``j = |l-s| .. l+s`` to
            get the j values. This also puts the configuration string into NIST format.
        """
        
        if('nist' not in self.data):
            self.get_nist_levels_txt()
        self.data.setdefault('processed', {})
        if('split' not in self.data['processed']):
            self.data['processed']['split'] = {}

        self.data['processed']['split']['config'] = []
        self.data['processed']['split']['L'] = []
        self.data['processed']['split']['S'] = []
        self.data['atomic']['nist_conf_form'] = np.empty_like(self.data['atomic']['config'])
        conf = []
        l = []
        s = []
        j = []
        term_map = [] #this will allow for mapping level -> term later
        for i in range(0,len(self.data['atomic']['config'])):
            j_arr = np.arange( np.abs(self.data['atomic']['L'][i] -  (self.data['atomic']['S'][i]-1)/2.),
                       self.data['atomic']['L'][i] + (self.data['atomic']['S'][i]-1)/2.+1,1.)

            conf_arr = np.empty_like(j_arr,dtype=object)
            l_arr = np.empty_like(j_arr,dtype=int)
            s_arr = np.empty_like(j_arr,dtype=int)
            term_map_arr = np.ones_like(j_arr,dtype=int)*i

            st = self.data['atomic']['config'][i].lower()
            rem = list(re.finditer(r'\d\w1' , st))
            if rem:
                tmpp = ''
                for ii in range(0,len(rem)):
                    st = st[:rem[ii].span()[1]-1-ii] + st[rem[ii].span()[1]+1-1-ii:]
                    
                '''
                nstring = ''
                nstring = st[rem[0].span()[0]:rem[0].span()[1]-1]
                for ii in range(1,len(rem)):
                    nstring = nstring + st[rem[ii-1].span()[1]:rem[ii].span()[0]]
                    nstring = nstring + st[rem[ii].span()[0]:rem[ii].span()[1]-1]

                nstring = nstring + st[rem[ii].span()[1]:]
                conf_arr[:] = nstring
                '''
            conf_arr[:] = st
            self.data['atomic']['nist_conf_form'][i] = st
            self.data['atomic']['nist_conf_form'] = self.data['atomic']['nist_conf_form'].astype('object')
            
            l_arr[:] = self.data['atomic']['L'][i]
            s_arr[:] = self.data['atomic']['S'][i]
            
            conf.append(conf_arr)
            l.append(l_arr)
            s.append(s_arr)
            j.append(j_arr)
            term_map.append(term_map_arr)
            
        self.data['processed']['split']['config'] = np.concatenate(conf,axis=0)
        
        #remove closed subshells because thats what NIST does
        #probably a better way to do this but I cant be bothered
        closed_shells = np.array(['1s2','2s2','2p6','3s2','3p6','4s2','3d10','4p6','5s2','4f14','5d10'])

        #because NIST is apparently incapable of creating a standard scheme for the configuration string
        #we are goin got add in all the closed subshells only to remove them later...who came up with this
        #stuff :'(
        shells_arr_tmp = np.empty_like(self.data['processed']['split']['config'].astype('<U1000'))

        test = np.where(np.char.find(closed_shells,
                              self.data['processed']['split']['config'][0][0:3])==0)[0]
        #need the if statement here to easily account for hydrogenic species
        
        if(np.size(test) >0):
            for ij in range(0,test[0]):#len(test)

                add_shell = np.setdiff1d(range(self.data['processed']['split']['config'].shape[0]),
                                        np.where(np.char.find(self.data['processed']['split']['config'].astype('<U'),
                                                              closed_shells[ij][0:2])==0)[0])

                if(add_shell.size>0):
                    for ik in add_shell:

                        if(ij == 0):
                            if(ij == test[0] -1):
                                wat = shells_arr_tmp[ik] +  closed_shells[ij] + '.'
                                shells_arr_tmp[ik] = wat
                                
                            else:
                                wat = shells_arr_tmp[ik] +  closed_shells[ij]
                                shells_arr_tmp[ik] = wat


                        else:
                            if(ij != test[0] -1):
                                wat = shells_arr_tmp[ik] + '.' + closed_shells[ij]
                                shells_arr_tmp[ik] = wat
                            else:
                                wat = shells_arr_tmp[ik] + '.' + closed_shells[ij] + '.'
                                shells_arr_tmp[ik] = wat
                                


            self.data['processed']['split']['config'] =   shells_arr_tmp +self.data['processed']['split']['config']

            #now do the stame thing for the NIST configurations
            shells_arr_tmp = np.empty_like(self.data['atomic']['nist_conf_form'].astype('<U1000'))

            test = np.where(np.char.find(closed_shells,
                                  self.data['atomic']['nist_conf_form'][0][0:3])==0)[0] #[0:2]
            
            for ij in range(0,test[0]):

                add_shell = np.setdiff1d(range(self.data['atomic']['nist_conf_form'].shape[0]),
                                         np.where(np.char.find(self.data['atomic']['nist_conf_form'].astype('<U'),
                                                               closed_shells[ij][0:2])==0)[0])
                if(add_shell.size>0):
                    for ik in add_shell:

                        if(ij == 0):
                            if(ij == test[0] -1):
                                wat = shells_arr_tmp[ik] +  closed_shells[ij] + '.'
                                shells_arr_tmp[ik] = wat
                                
                            else:
                                wat = shells_arr_tmp[ik] +  closed_shells[ij]
                                shells_arr_tmp[ik] = wat

                            
                        else:
                            if(ij != test[0] -1):
                                wat = shells_arr_tmp[ik] + '.' + closed_shells[ij]
                                shells_arr_tmp[ik] = wat
                            else:
                                wat = shells_arr_tmp[ik] + '.' + closed_shells[ij] + '.'
                                shells_arr_tmp[ik] = wat


            self.data['atomic']['nist_conf_form'] = shells_arr_tmp + self.data['atomic']['nist_conf_form']
            
            

            #NIST leaves out the inner closed subshells that are not in its ground configuration
            self.data['atomic']['nist_conf_form'] = np.array(
                [strip_closed_shells(conf, closed_shells, self.data['nist']['levels'][0]['conf'])
                 for conf in self.data['atomic']['nist_conf_form']], dtype=object)
            self.data['processed']['split']['config'] = np.array(
                [strip_closed_shells(conf, closed_shells, self.data['nist']['levels'][0]['conf'])
                 for conf in self.data['processed']['split']['config']], dtype=object)
                            
        self.data['processed']['split']['L'] = np.concatenate(l,axis=0)
        self.data['processed']['split']['S'] = np.concatenate(s,axis=0)
        self.data['processed']['split']['j'] = np.concatenate(j,axis=0)
        self.data['processed']['split']['term_map'] = np.concatenate(term_map,axis=0)




    def format_config_to_nist(self,conf=-1):
        """
            Formats the string for the configuration of the adf04 file to the NIST format
        """

        #allowing for this is be called and operate
        #automatically on the store configuration
        if(conf == -1):
            conf = self.data['atomic']['config']
        
        closed_shells = np.array(['1s2','2s2','2p6','3s2','3p6','4s2','3d10','4p6','5s2',
                                  '4d10','5p6','6s2','4f14','5d10','6p6','7s2','5f14','6d10','7p6'])

        #because NIST is apparently incapable of creating a standard scheme for the configuration string
        #we are goin got add in all the closed subshells only to remove them later...who came up with this
        #stuff :'(
        shells_arr_tmp = np.empty_like(conf.astype('<U1000'))

        test = np.where(np.char.find(closed_shells,
                              conf[0][0:2])==0)[0]
        #need the if statement here to easily account for hydrogenic species
        if(test.size > 0 and test[0] > 0):
            for ij in range(0,test[0]):

                add_shell = np.setdiff1d(range(conf.shape[0]),
                                        np.where(np.char.find(conf.astype('<U'),
                                                              closed_shells[ij][0:2])==0)[0])

                if(add_shell.size>0):
                    for ik in add_shell:
                        if(ij ==0):
                            wat = closed_shells[ij]
                        else:
                            
                            wat = shells_arr_tmp[ik] + '.' + closed_shells[ij]
                        shells_arr_tmp[ik] = wat
            shells_arr_tmp = np.char.add(shells_arr_tmp,  '.')
            self.data['atomic']['nist_conf_form'] = np.char.add(shells_arr_tmp,  conf)
                
            '''            
            shells_arr_tmp = np.empty_like(self.data['atomic']['nist_conf_form'].astype('<U1000'))

            test = np.where(np.char.find(closed_shells,
                                  self.data['atomic']['nist_conf_form'][0][0:2])==0)[0]
            for ij in range(0,test[0]-1):

                add_shell = np.setdiff1d(range(self.data['atomic']['nist_conf_form'].shape[0]),
                                         np.where(np.char.find(self.data['atomic']['nist_conf_form'].astype('<U'),
                                                               closed_shells[ij][0:2])==0)[0])
                if(add_shell.size>0):
                    for ik in add_shell:

                        wat = closed_shells[ij] + '.' + shells_arr_tmp[ik]
                        shells_arr_tmp[ik] = wat
            '''
            #NIST leaves out the inner closed subshells that are not in its ground configuration
            self.data['atomic']['nist_conf_form'] = np.array(
                [strip_closed_shells(conf, closed_shells, self.data['nist']['levels'][0]['conf'])
                 for conf in self.data['atomic']['nist_conf_form']])
        


        
    def _nist_level(self, conf, S, L, j):
        """First NIST level with the configuration, term and J, or None."""
        term = str(S) + L_LETTERS[L]
        j_val = str(Fraction(j))
        for ent in self.data['nist']['levels']:
            if(ent['conf'] == conf and ent['term'] == term and ent['j_val'] == j_val):
                return ent
        return None

    def shift_j_res_energy_to_nist(self,already_j=False,fractional_diff_adf04_nist=0.05):
        """shift_j_res_energy_to_nist maps j resolved structure energy values to the NIST values
           if those values are availble. Saves these in the 'split' sub dictionary. This can be used
           with a j resolved file or with LS file if it is split to j resolution with the
           'split_structure_terms_to_levels' method.

           Levels that are not in NIST keep their energy from the adf04 file, the energy of their
           term for split levels. ['nist_shifted'] marks the levels that were shifted.
        """
        if('nist' not in self.data):
            self.get_nist_levels_txt()

        if( not already_j):#added check for already at j resolution in which case no need to split
            if('split' not in self.data.get('processed', {}) or 'config' not in self.data['processed']['split']):
                self.split_structure_terms_to_levels()

            split = self.data['processed']['split']
            term_energy = self.data['atomic']['energy'][split['term_map'].astype(int)]
            energy = np.array(term_energy, dtype=float)
            shifted = np.zeros(len(energy), dtype=bool)
            for i in range(0,len(split['config'])):
                tmp = self._nist_level(split['config'][i], split['S'][i], split['L'][i], split['j'][i])
                if(tmp is not None):
                    nist_energy = float(tmp['energy'])
                    #the or is to account for the ground state splitting which will definitely lead to bigger than 5% difference
                    #or any term in the file that has a low energy
                    if( (np.abs(nist_energy - term_energy[i]) < fractional_diff_adf04_nist*np.abs(nist_energy)) or
                        term_energy[i] < 10000):
                        energy[i] = nist_energy
                        shifted[i] = True

            split['energy'] = energy
            split['nist_shifted'] = shifted

        else:
            if('nist_conf_form' not in self.data['atomic']):
                self.get_nist_conf_form()
            energy = np.array(self.data['atomic']['energy'], dtype=float)
            shifted = np.zeros(len(energy), dtype=bool)
            for i in range(0,len(self.data['atomic']['nist_conf_form'])):
                tmp = self._nist_level(self.data['atomic']['nist_conf_form'][i], self.data['atomic']['S'][i],
                                       self.data['atomic']['L'][i], self.data['atomic']['w'][i])
                if(tmp is not None):
                    energy[i] = float(tmp['energy'])
                    shifted[i] = True
                    print(self.data['atomic']['nist_conf_form'][i],self.data['atomic']['S'][i],L_LETTERS[self.data['atomic']['L'][i]],Fraction(self.data['atomic']['w'][i]),tmp)

            self.data['atomic']['adf04_energies'] = np.copy(self.data['atomic']['energy'])
            self.data['atomic']['energy'] = energy
            self.data['atomic']['nist_shifted'] = shifted
            print('Energy values that could be shifted to NIST energies have been. Original energies now stored in [atomic][adf04_energies')

    def shift_j_res_pecs(self):
        """Wavelengths of the PECs between levels shifted to NIST energies by shift_j_res_energy_to_nist
           with already_j=True, in ['wave_vac_shift'] and ['wave_air_shift'].
        """
        if('nist_shifted' in self.data['atomic']):
            a = np.where(self.data['atomic']['nist_shifted'])[0]
        else:
            a = np.where( self.data['atomic']['energy'] > -1)[0] #levels that are not in NIST have be given -1 for energies
        in_upper = _isin(self.data['processed']['pec_levels'][:,0],a) #find all the levels that have be shifted upp and lower
        in_lower = _isin(self.data['processed']['pec_levels'][:,1],a)


        #set up 'shift' wave and pecs
        self.data['processed']['pecs_shift'] = self.data['processed']['pecs'][np.logical_and(in_upper,in_lower)]
        self.data['processed']['pec_levels_shift'] = self.data['processed']['pec_levels'][np.logical_and(in_upper,in_lower)]
        self.data['processed']['wave_vac_shift'] = np.zeros(len(self.data['processed']['pec_levels_shift']))

        for i in range(0,len(self.data['processed']['pec_levels_shift'])):
            #levels with equal energies have an infinite wavelength
            with np.errstate(divide='ignore'):
                self.data['processed']['wave_vac_shift'][i] =  1.e7/abs(self.data['atomic']['energy'][self.data['processed']['pec_levels_shift'][i,0]]\
                                                                            - self.data['atomic']['energy'][self.data['processed']['pec_levels_shift'][i,1]])

        #wavelengths below the air cut off stay in vacuum
        self.data['processed']['wave_air_shift'] = vac_to_air(self.data['processed']['wave_vac_shift'])

    def shift_j_res_energy_to_nist2(self,already_j=False):
        """shift_j_res_energy_to_nist maps j resolved structure energy values to the NIST values
           if those values are availble. Saves these in the 'split' sub dictionary. This can be used
           with a j resolved file or with LS file if it is split to j resolution with the
           'split_structure_terms_to_levels' method.

           Levels that are not in NIST keep their energy from the adf04 file, the original
           energies are in ['adf04_energies'] and ['nist_shifted'] marks the shifted levels.
        """
        if('nist' not in self.data):
            self.get_nist_levels_txt()

        if( not already_j):#added check for already at j resolution in which case no need to split
            if('split' not in self.data.get('processed', {}) or 'config' not in self.data['processed']['split']):
                self.split_structure_terms_to_levels()
            confs = self.data['atomic']['config']
        else:
            if('nist_conf_form' not in self.data['atomic']):
                self.get_nist_conf_form()
            confs = self.data['atomic']['nist_conf_form']

        energy = np.array(self.data['atomic']['energy'], dtype=float)
        shifted = np.zeros(len(energy), dtype=bool)
        for i in range(0,len(confs)):
            tmp = self._nist_level(confs[i], self.data['atomic']['S'][i], self.data['atomic']['L'][i],
                                   self.data['atomic']['w'][i])
            if(tmp is not None):
                energy[i] = float(tmp['energy'])
                shifted[i] = True
                if(already_j):
                    print(confs[i],self.data['atomic']['S'][i],L_LETTERS[self.data['atomic']['L'][i]],Fraction(self.data['atomic']['w'][i]),tmp)

        self.data['atomic']['adf04_energies'] = np.copy(self.data['atomic']['energy'])
        self.data['atomic']['energy'] = energy
        self.data['atomic']['nist_shifted'] = shifted
        print('Energy values that could be shifted to NIST energies have been. Original energies now stored in [atomic][adf04_energies')

    def get_nist_levels(self):
        """ get_nist_levels grabs the nist energy levels from the NIST mysql database. The mysql NIST database must
            be installed. There is a plain text file in the works to get around this and simplify for users.
            use get_nist_levels_txt for this
        """
        
        self.data['nist'] = {}

        self.data['nist']['levels'] = get_nist_clean(self.data['atomic']['element'].replace(' ', ''),
                                                     self.data['atomic']['charge_state'] + 1)

    def get_nist_levels_txt(self):
        """ get_nist_levels grabs the nist energy levels from the the stored txt files in 'atomic/nist_energies'.
            This data comes from the NIST database ~09/2016
        """
        
        self.data['nist'] = {}
        self.data['nist']['levels'] = get_nist_txt(nist_energies_dir(),
                                                   self.data['atomic']['element'].replace(' ', ''),
                                                   self.data['atomic']['charge_state'] + 1)

    def solve_cr(self):
        """Solve_cr automates the calls to various function to make the data for
           need to solve the CR equations and get to the quanties that users want.
           This can be thought of as a basic mode for the user doesn't want to mess
           with the default rates that are made.

           Generally what a user just wanting basic output should use.
           Makes ionization if requested
           Makes recombination if requested
           Makes excitation rates
           Populates the CR matrix
           Solve the CR matrix using the quasistatic approximation
        """
        
        #three body recombination is made from the ionization rates by detailed balance, so they
        #are also made when ionization is not in the CR matrix
        if(self.data['user']['use_ionization'] or self.data['user']['use_recombination_three_body']):
            self.make_ioniz_from_reduced_ionizrates()
        if(self.data['user']['suppliment_with_ecip']):
            self.make_ecip()
            self.suppliment_with_ecip()

        if(self.data['user']['use_recombination']):
            self.make_recombination_rates_from_file()
        if(self.data['user']['use_recombination_three_body']):
            self.make_three_body_recombination()
        if((self.data['user']['use_cx']) & (self.data['rates']['cx']['cx_excit'].size > 0)):
            self.make_cx_rates_from_file()
        else:
            self.data['user']['use_cx'] = False
        self.make_electron_excitation_rates()
        self.populate_cr_matrix()
        self.solve_quasi_static()







    def make_nat_broad(self,calc_lorentz = True, dw=0,n=50,gs=5,split_wave=np.array([])):
        """ Calculates the width of natural broadening from the addition of the upper and lower spontaneous emission.
            Also creates wavelength and spectral intensity arrays for each spectral line, of shape
            (#PECs, n). Uses equation 1.27 from Stuart Loch's thesis.

            The profiles are made on offsets from the line centre, ['dwave_arr'] in nm: natural
            widths can be far below the float64 resolution of the wavelength (5e-15 nm for the Be I
            intercombination line at 454.8 nm, where float64 resolves 6e-14 nm), so the wavelengths
            ['wave_arr'] = centre + offset of such lines are all equal. Lines without a finite
            wavelength or width have zero profiles.

        :param calc_lorentz: calculate a Lorentzian for every line of ['processed']['pecs'] or of split_wave
        :type calc_lorentz: bool

        :param dw: half range of the wavelength grids (nm), gs natural widths by default
        :type dw: float

        :param n: number of points in the wavelength grids
        :type n: int

        :param gs: half range of the default wavelength grids in natural widths (FWHM)
        :type gs: float

        :param split_wave: wavelengths of the split PECs to be broadened
        :type split_wave: float array

        """
        
        #check if other broadening processes have already be calculated
        if( 'broadening' not in self.data['processed']):
            self.data['processed']['broadening'] = {}
        self.data['processed']['broadening']['nat'] = {}

        if(split_wave.any()):
            wave = split_wave
            # delta nu_L = A_{n'->n}/2/pi add up both the upper and lower loss rates of the levels
            self.data['processed']['broadening']['nat']['dnu_l'] = \
                                  self.data['cr_matrix']['A_ji_loss'][self.data['processed']['pec_levels'][self.data['processed']['split']['unres_pec_map'],0]]/2/np.pi+\
                                  self.data['cr_matrix']['A_ji_loss'][self.data['processed']['pec_levels'][self.data['processed']['split']['unres_pec_map'],1]]/2/np.pi
            
        else:
            wave = self.data['processed']['wave_air']
        
            # delta nu_L = A_{n'->n}/2/pi add up both the upper and lower loss rates of the levels
            self.data['processed']['broadening']['nat']['dnu_l'] = \
                                  self.data['cr_matrix']['A_ji_loss'][self.data['processed']['pec_levels'][:,0]]/2/np.pi+\
                                  self.data['cr_matrix']['A_ji_loss'][self.data['processed']['pec_levels'][:,1]]/2/np.pi

        if(calc_lorentz):
            nat = self.data['processed']['broadening']['nat']
            wave = np.asarray(wave, dtype=float)
            if(dw==0):
                #set the grid to allow for varrying natural line widths
                with np.errstate(invalid='ignore', over='ignore'):
                    dww = (wave*1.e-9)**2/constants.c*nat['dnu_l']*1.e9*gs
            else:
                dww = np.full(len(wave), float(dw))
            #lines with a finite wavelength and width; the others have zero profiles
            ok = np.isfinite(wave) & np.isfinite(dww) & (dww > 0) & (nat['dnu_l'] > 0)

            #offsets from the line centre (nm), which keep their precision for any width
            nat['dwave_arr'] = np.linspace(-np.where(ok, dww, 0.), np.where(ok, dww, 0.), n, axis=1)
            nat['wave_arr'] = wave[:,None] + nat['dwave_arr']#create 2d wave array

            #frequency offsets from the centre, c/(w + dw) - c/w, without the cancellation
            #of two frequencies near 1e15 Hz
            centre = np.where(ok, wave, 1.)[:,None]
            dnu = -constants.c*nat['dwave_arr']/(centre*(centre + nat['dwave_arr']))*1.e9

            #dnu_l is the full width at half maximum, lorentz takes the half width
            lor_tmp = self.lorentz(np.where(ok, nat['dnu_l'], 1.)/2, dnu, np.zeros(len(wave)))

            #normalize so area under the curve is 1
            nat['theta'] = np.zeros_like(lor_tmp)
            nat['theta'][ok] = lor_tmp[ok]/_trapezoid(lor_tmp[ok], x=nat['dwave_arr'][ok], axis=1)[:,None]


            
    def make_dopp_broad(self, T, m, calc_gauss=True, dw=0., n=50,gs=5,split_wave=np.array([])):
        """ Make the doppler shift for spectral lines ues equation 1.25 and 1.26 from Stuart Lochs thesis

        :param nu0: frequency of spectral line
        :type nu0: float

        :param T:  Temperature (eV)
        :type T: float

        :param m:  Mass of ion (kg)
        :type m: float

        :param calc_gauss: calculated a number of gaussians=['processed']['pecs'] or len(split_wave)
        :type calc_gauss: bool

        :param dw:  wavelength grid spacing if default not chosen
        :type dw:  float

        :param n:  length of wavelength grids
        :type n:  int

        :param split_wave:  PEC/wavelength array to be broadened
        :type split_wave:  int

        """


        if(split_wave.any()):
            wave = split_wave
        else:
            wave = self.data['processed']['wave_air']
        
        #check if other broadening processes have already be calculated
        if( 'broadening' not in self.data['processed']):
            self.data['processed']['broadening'] = {}
        self.data['processed']['broadening']['dopp'] = {}

        #see Loch's thesis equation 1.26
        self.data['processed']['broadening']['dopp']['dnu_g'] = np.sqrt(2*(T*constants.e)/m)*(constants.c/(wave/1.e9))/constants.c

        if(calc_gauss):
            if(dw==0):
                #set a variable delta lambda based on the width of the broadening
                dww = (wave*1.e-9)**2/constants.c* self.data['processed']['broadening']['dopp']['dnu_g']*1.e9*gs

            else:
                dww = dw

            self.data['processed']['broadening']['dopp']['wave_arr'] =np.linspace(wave-dww, wave + dww,n,axis=1)#wavelength array


            self.data['processed']['broadening']['dopp']['theta'] = self.gauss_dopp(self.data['processed']['broadening']['dopp']['wave_arr'],
                                                                                    wave,
                                                                                    self.data['processed']['broadening']['dopp']['dnu_g'])
            
            '''
            #see Loch's thesis equation 1.25
            gau_tmp = np.exp(-1*( (constants.c/(self.data['processed']['broadening']['dopp']['wave_arr']/1.e9) - constants.c/(wave[:,None]/1.e9))/\
                                                             self.data['processed']['broadening']['dopp']['dnu_g'][:,None])**2)/\
                                                   (np.sqrt(np.pi)*self.data['processed']['broadening']['dopp']['dnu_g'][:,None])
            
            #normalize so the area under the curve is 1
            self.data['processed']['broadening']['dopp']['theta'] = np.einsum('ij,i->ij',
                                                                              gau_tmp,
                                1/_trapezoid(gau_tmp,x=self.data['processed']['broadening']['dopp']['wave_arr'],axis=1))

            '''






    def gauss_dopp(self, wave_arr, wave, dnu_g):
        ''' Gives a doppler broadened Gaussian profile

        :param wave_arr: wavelength array for the gaussians
        :param dnu_g: dnu_g parameter for doppler broadening


        '''

        gau_tmp = np.exp(-1*((constants.c/(wave_arr/1.e9)-constants.c/(wave[:,None]/1.e9))/dnu_g[:,None])**2)/\
                                                                                     (np.sqrt(np.pi)*dnu_g[:,None])

        #normalize so the area under the curve is 1
        return np.einsum('ij,i->ij', gau_tmp,1/_trapezoid(gau_tmp,x=wave_arr,axis=1))






        

    def lorentz(self,vl,v,v0):
        '''Lorentzian function used for natural line broadening vl/((v-v0)**2 + vl**2)

        Args:
          :param vl: half width at half maximum
          :type vl: float

          :param v: x array
          :type v: float

          :param v0: center
          :type v0: float

        '''
        return vl[:,None]/((v-v0[:,None])**2 + vl[:,None]**2)













        

        
    def plot_pec_sticks(self,temp=[0],dens=[0],meta=[0],wave_range=None):
        """plot_pec_sticks will plot the the PEC values versus wavelength.
           **WARNING** wavelngth values will be wrong unless energy levels in the file
           have been 'shifted' to NIST values.
           User provides index arrays for the temperature density and metastable
           for the PECs to be plotted.
           
        :param temp: Array of temperature indexes
        :type temp: int array

        :param dens: Array of density indexes
        :type dens: int array

        :param meta: array of metastable indexes
        :type metas: int array

        :param wave_range: wavelength axis (min, max) in nm; by default it spans the lines with
                           PECs of at least 1e-3 of the strongest
        :type wave_range: float tuple
        """

        rc('axes', linewidth=2)
        rc('font', weight='semibold')

        if('processed' not in self.data.keys()):
            self.solve_cr()
        p_t = np.arange(0,len(self.data['user']['temp_grid']))
        p_n = np.arange(0,len(self.data['user']['dens_grid']))
        p_m = np.arange(0,len(self.data['atomic']['metas']))
        if(np.asarray(temp).size>0):
            p_t = p_t[temp]
        if(np.asarray(dens).size>0):
            p_n = p_n[dens]
        if(np.asarray(meta).size>0):
            p_m = p_m[meta]

        for i in p_n:
            for j in p_t:
                for k in p_m:
                    plt.figure()
                    wave = self.data['processed']['wave_air']
                    pecs = self.data['processed']['pecs'][:,k,j,i]
                    #the PECs are plotted in units of 10^scaling ph cm3 s-1, so the largest is from 1 to 10
                    scaling = int(np.floor(np.log10(np.max(pecs)))) if np.max(pecs) > 0 else 0
                    plt.vlines(wave, np.zeros_like(wave), pecs/10.**scaling)

                    plt.xlabel('Wavelength in air (nm)',weight='semibold')
                    plt.ylabel('PEC (10$^{' + str(scaling) + '}$ ph cm$^{3}$ s$^{-1}$)',weight='semibold')
                    plt.title('Temperature ' + str(self.data['user']['temp_grid'][j]) + ' eV,  '+\
                              'Density ' + format(self.data['user']['dens_grid'][i],'.2e') + ' cm$^{-3}$, '+\
                              'Metastable ' + str(self.data['atomic']['metas'][k]),weight='semibold')
                    if(wave_range is not None):
                        plt.xlim(*wave_range)
                    else:
                        #the lines that show on the plot, at least 1e-3 of the strongest
                        shown = np.isfinite(wave) & (pecs > 0) & (pecs >= 1e-3*np.max(pecs))
                        if(shown.any()):
                            low, high = np.min(wave[shown]), np.max(wave[shown])
                            margin = 0.05*(high - low) if high > low else 0.05*high
                            plt.xlim(max(0., low - margin), high + margin)


    def plot_pec_ratio_temp(self,pec1,pec2,dens=np.array([0]),meta = np.array([0])):
        """plot_pec_ratio_temp will plot the ratio of any two user defined PECs versus temperature.
           Density values for the ratio are choosen by the user by specifying the indices from the user
           density array to be plot. Each density chosen will show up as a new line in the figure.
           The user can also choose the metastable by default only the ground is plotted.
           The user specifies the indices of the metastable to be plot. 
           A new figure for each metastable chosen will be made with the number of density lines chosen.

        :param pec1: Index of pec on the top of the ratio
        :type pec1: int

        :param pec2: Index of the pec on the bottom of the ratio
        :type pec2: int

        :param dens: Array of density indexes
        :type dens: int array

        :param meta: array of metastable indexes
        :type metas: int array

        """

        if('processed' not in self.data.keys()):
            self.solve_cr()
        dens = np.array(dens)
        p_n = np.arange(0,len(self.data['user']['dens_grid']),dtype=int)
        p_m = np.arange(0,len(self.data['atomic']['metas']),dtype=int)
        
        if(np.asarray(dens).size>0):
            p_n = p_n[dens]
        if(np.asarray(meta).size>0):
            p_m = p_m[meta]

        for k in p_m:
            plt.figure()                        
            for i in p_n:
                
                plt.plot(self.data['user']['temp_grid'],
                     self.data['processed']['pecs'][pec1,k,:,i]/ \
                         self.data['processed']['pecs'][pec2,k,:,i],
                         label='$n_e$ = ' + format(self.data['user']['dens_grid'][i],'.1e') + ' cm$^{-3}$')

                plt.xlabel('Temperature (eV)',weight='semibold')
                plt.ylabel('Ratio (-)',weight='semibold')
                plt.title('Ratio of PEC '+str(pec1)+', ' + format(self.data['processed']['wave_air'][pec1],'.2f') + ' nm'+\
                          ' to PEC '+str(pec2)+', ' + format(self.data['processed']['wave_air'][pec2],'.2f') + ' nm, '+\
                              'Metastable ' + str(self.data['atomic']['metas'][k]),weight='semibold')
                plt.legend(loc='best')





    def return_level_info(self,level_num,split=False):
        """Retrun all the level identifying information for a given level.
           The default return is the normal level (adf04 file).
           The split level information can also be returned with the flag.

           :param level_num: Level number for info to be returned
           :type level_num: int

           :param split: Flag to return split values
           :type split: bool

        """

        if(split):
            return self.data['processed']['split']['config'][level_num],\
                   self.data['processed']['split']['S'][level_num],\
                   self.data['processed']['split']['L'][level_num],\
                   self.data['processed']['split']['j'][level_num],\
                   self.data['processed']['split']['energy'][level_num]
        else:
                
            return self.data['atomic']['config'][level_num],\
                   self.data['atomic']['S'][level_num],\
                   self.data['atomic']['L'][level_num],\
                   self.data['atomic']['w'][level_num],\
                   self.data['atomic']['energy'][level_num]
           
        

    def plot_pec_ratio_dens(self,pec1,pec2,temp=np.array([0]),meta = np.array([0]),scale='log'):
        """plot_pec_ratio_dens will plot the ratio of any two user defined PECs versus density.
        Temperature values for the ratio are choosen by the user by specifying the indices from the user
        temperature array to be plotted. Each temperature chosen will show up as a new line in the figure.
        The user can also choose the metastable by default only the ground is plotted.
        The user specifies the indices of the metastable to be plot.
        A new figure for each metastable chosen will be made with the number of density lines chosen.
        The density axis default is a log scale but linear can be chosen as well.

        :param pec1: Index of pec on the top of the ratio
        :type pec1: int

        :param pec2: Index of the pec on the bottom of the ratio
        :type pec2: int

        :param temp: Array of temperature indexes
        :type temp: int array

        :param meta: array of metastable indexes
        :type metas: int array

        :param scale: scale for the density axis default is log
        :type scale: str
        """
        
        temp = np.array(temp)
        p_n = np.arange(0,len(self.data['user']['temp_grid']),dtype=int)
        p_m = np.arange(0,len(self.data['atomic']['metas']),dtype=int)
        
        if(np.asarray(temp).size>0):
            p_n = p_n[temp]
        if(np.asarray(meta).size>0):
            p_m = p_m[meta]

        for k in p_m:
            plt.figure()                        
            for i in p_n:
                
                plt.plot(self.data['user']['dens_grid'],
                     self.data['processed']['pecs'][pec1,k,i,:]/ \
                         self.data['processed']['pecs'][pec2,k,i,:],
                         label='$T_e$ = ' + format(self.data['user']['temp_grid'][i],'.1f') + ' eV')

                plt.xlabel('Density (cm$^{-3}$)',weight='semibold')
                plt.ylabel('Ratio (-)',weight='semibold')
                plt.title('Ratio of PEC '+str(pec1)+', ' + format(self.data['processed']['wave_air'][pec1],'.2f') + ' nm'+\
                          ' to PEC '+str(pec2)+', ' + format(self.data['processed']['wave_air'][pec2],'.2f') + ' nm, '+\
                              'Metastable ' + str(self.data['atomic']['metas'][k]),weight='semibold')
                if(scale=='log'):
                    plt.semilogx()
                plt.legend(loc='best')





    def write_pecs_adf15(self,fil_name='', pec_inds = None, num = 8, pecs_split=False):

        """ This function calls the write_adf15 function.

            Wavelengths are written in A, in standard air from 200 nm and in vacuum
            below (see vac_to_air). By default all lines are written except those
            whose wavelength cannot be: infinite or too long for the adf15 format
            (degenerate levels) or not positive (split levels out of order), with a
            warning. PECs need a temperature-density grid, not temp_dens_pair.

        :param fil_name: name of the adf15 file
        :type fil_name: str

        :param pec_inds: indices of the PECs to write, all by default
        :type pec_inds: int arr

        :param num: number of values on each line of the file
        :type num: int

        :param pecs_split: write the PECs split to j levels by split_pec_multiplet
        :type pecs_split: bool

        """

        if(self.data['user']['temp_dens_pair']):
            raise ValueError('adf15 files need PECs on a temperature-density grid, '
                             'they cannot be written with temp_dens_pair=True')

        if(fil_name==''):#defaul file name if the user did not choose one
            fil_name = 'adf15_colradpy_' + os.path.basename(self.data['user']['file_loc'])

        if(pecs_split):#write split PECs if requested
            processed = self.data['processed']['split']
        else:#write un-split PECs
            processed = self.data['processed']
        wave = processed['wave_air']*10#nm to A

        #pec_inds = 0 also selects all, as before
        if(pec_inds is None or (np.isscalar(pec_inds) and pec_inds == 0)):
            #if no PEC indices specified by user just return all that can be written
            pec_inds = np.where((wave > 0) & (wave < MAX_WAVELENGTH))[0]
            if(len(pec_inds) < len(wave)):
                warnings.warn(str(len(wave) - len(pec_inds)) + ' of ' + str(len(wave)) +
                              ' lines are not written to the adf15 file, their wavelengths are '
                              'not positive, infinite or longer than ' + str(MAX_WAVELENGTH) + ' A')

        num_recomb = 0
        if(self.data['user']['use_recombination'] or self.data['user']['use_recombination_three_body']):
            num_recomb = len(self.data['atomic']['ion_pot'])

        write_adf15(fil_name, pec_inds, wave,
                    processed['pecs'], self.data['atomic']['element'],
                    self.data['atomic']['charge_state'], self.data['user']['dens_grid'],
                    self.data['user']['temp_grid'], self.data['atomic']['metas'],
                    self.data['atomic']['ion_pot'],
                    user = self.data['user'], atomic = self.data['atomic'], num = num,
                    num_recomb = num_recomb)



    def dump_hdf5(self,fil_name='colradpy_hdf5_dump.hkl', cut_limit=1e-3, pecs=True, gcrs=False, additional=False):
        """ This function dumps data to a hdf5 file for other codes

        :param fil_name: The file path and name that will be saved
        :type fil_name: str

        :param cut_limit: limits the PEC dumped relative to the largest line in the spectrum
        :type cut_limit: float

        :param pecs: Save the PECs and associated information
        :type pecs: bool

        :param gcrs: Save the GCR coefficients
        :type gcrs: bool

        :param additional: Save data not always used by other codes
        :type additional: bool

        """
        import hdfdict
        import h5py
        #If all the PECs are included this can create hdf5 files that are very large (with many Te, ne)
        #in reality comparison to experiment will only be the strong lines so make a cut

        #find all the pecs above the cut limit, careful the largest pec can change with Te
        inds_above = np.unique(np.where(self.data['processed']['pecs']/\
                               np.max(self.data['processed']['pecs'],axis=0) >cut_limit)[0])
        
        processed = {}#modified processed dictionary to be sent to hdf5

        if(pecs):# dumps the PECs to the hdf file
            processed['pops']                     = self.data['processed']['pops']
            processed['driving_populations_norm'] = self.data['processed']['driving_populations_norm']
            processed['pecs']                     = self.data['processed']['pecs'][inds_above,:,:,:]
            processed['pec_levels']               = self.data['processed']['pec_levels'][inds_above,:]
            processed['wave_air']                 = self.data['processed']['wave_air'][inds_above]
            
        if(gcrs): #dumps the gcrs to the hdf5 file
            processed['scd']          = np.copy(self.data['processed']['scd'])
            processed['acd']          = np.copy(self.data['processed']['acd'])
            processed['qcd']          = np.copy(self.data['processed']['qcd'])
            processed['xcd']          = np.copy(self.data['processed']['xcd'])
            processed['plt']          = np.copy(self.data['processed']['plt'])
            processed['prb']          = np.copy(self.data['processed']['prb'])
            processed['prc']          = np.copy(self.data['processed']['prc'])

        if(additional):# dumps additional data to hdf5 that other codes generally dont care about
            processed['pop_lvl']      = self.data['processed']['pop_lvl']
            processed['pops_no_norm'] = self.data['processed']['pops_no_norm']
            processed['wave_vac']     = self.data['processed']['wave_vac'][inds_above]
            
        fil_tmp = open(fil_name,'w')

        sav_dict = {}
        sav_dict['user']       = self.data['user']
        sav_dict['atomic']     = self.data['atomic']
        sav_dict['input_file'] =  self.data['input_file']
        sav_dict['processed']  =  processed


        sav_dict = myprint(sav_dict)
        hdfdict.dump(sav_dict, h5py.File(fil_name,'a'))




        
        fil_tmp.close()



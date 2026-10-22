"""Reads the atomic data of the Flexible Atomic Code (FAC, M. F. Gu, Can. J. Phys. 86, 675 (2008),
https://github.com/flexible-atomic-code/fac) into the dictionary that read_adf04 makes, so that
ColRadPy can model an ion from them.

FAC writes binary tables, which its PrintTable(binary, ascii, 1) prints as ASCII files with the
energies in eV. This module reads those files without FAC itself (pfac): the levels (a.en), radiative
rates (a.tr), electron impact excitation (a.ce), radiative recombination (a.rr), autoionization and
dielectronic capture (a.ai) and electron impact ionization (a.ci), and the Maxwellian rate
coefficients that pfac.fac.MaxwellRate writes (ce.mr, rr.mr, ci.mr). The files of an ion share a
prefix, e.g. Ar01a.en, Ar01a.tr, ... and Ar01ce.mr for the prefix Ar01.

The levels of the ion are those with nele electrons, sorted by energy; the parents are the levels of
the next ion (nele - 1 electrons). FAC's levels are jj-coupled: their J is exact, but they have no
LS term, so their L and S are -1. FAC's level names are kept in ['atomic']['fac_name'].

The cross-sections are averaged over a thermal electron distribution at the temperatures of the
model (colradpy.convolve_eedf), from each threshold. Between the energies that FAC tabulates, the
collision strengths of excitation and ionization and the bound-free oscillator strength density of
recombination are interpolated log-log. Outside them they follow FAC's own forms:

- excitation: the collision strength of the first energy down to the threshold; above the last energy
  it joins the Bethe-Born limit of FAC, Omega = b ln(E/E0) + c, at the energy FAC computed it for
  (b > 0 for optically allowed transitions, b = 0 for a constant limit), and transitions without a
  Born limit (b < 0, exchange, which falls as E^-2) follow the power law of their last two values,
  at least as steep as E^-1, as FAC does;
- ionization: FAC's fit Omega = p0 ln x + p1 y^2 + p2 y/x + p3 y/x^2 (x = E/I, y = 1 - 1/x), scaled to
  the first and last tabulated values;
- recombination: the bound-free oscillator strength density, finite at threshold, of the first
  energy below it; above the last, FAC's fit (its QK_FIT mode, valid at high energies only) scaled
  to the last value, or else the power law of the last two values.

Dielectronic capture is the capture strength of a.ai times v f(E) at the resonance energy, and adds
to the recombination of the parent into the level. The data of read_adf04 have no autoionization,
so it is folded into the other rates (_fold_autoionization): each flow into an autoionizing level
(capture, excitation, radiative cascade) is multiplied by the fraction of it that stabilizes
radiatively, and the excitation that autoionizes is ionization of the lower level
(excitation-autoionization). The populations of the autoionizing levels, their lines, the
dielectronic recombination and the ionization are then those with autoionization in the model while
collisions from those levels are slow beside their decays (below about 1e20 cm-3 for highly charged
ions). Left out are the capture that autoionizes to another parent (a process of the next ion), the
emission of lines between two autoionizing levels beyond the stabilizing fraction of the lower, and
autoionizing levels without radiative decays in the data. The autoionization rates and the
stabilizing fractions are kept in ['input_file']. The two-photon decays of the 2s level of H-like ions and the 1s2s 1S0 level of He-like ions (Drake, Phys.
Rev. A 34, 2871 (1986), as in FAC) add to the A-value of their decay to the ground level.
"""

import os
import warnings

import numpy as np
from scipy import constants

from colradpy.convolve_eedf import rate_coefficients, resonance_rate_coefficients

#Hartree energy (eV), fine-structure constant and Bohr radius squared (cm2)
HARTREE_EV = constants.physical_constants['Hartree energy in eV'][0]
ALPHA = constants.fine_structure
BOHR_AREA = (constants.physical_constants['Bohr radius'][0]*100)**2
#FAC prints cross-sections in 1e-20 cm2, dielectronic capture strengths in 1e-20 cm2 eV and the rate
#coefficients of MaxwellRate in 1e-10 cm3 s-1
FAC_AREA = 1e-20
FAC_RATE = 1e-10
#qk_mode of FAC tables whose rows carry fit parameters
QK_FIT = 2
#FAC table types (dbase.h) and the processes ColRadPy reads
TABLE_TYPES = {'en': 1, 'tr': 2, 'ce': 3, 'rr': 4, 'ai': 5, 'ci': 6}
PROCESSES = ('en', 'tr', 'ce', 'rr', 'ai', 'ci')
#Rate coefficient of a unit effective collision strength at 1 Ryd as ColRadPy turns upsilons into
#rates (populate_cr_matrix), so that the rates of the CR matrix are those integrated here
UPSILON_RATE = 2.1716E-8
RYDBERG_EV = 13.6058
#K per eV of the temperatures of adf04 files
KELVIN_PER_EV = 11604.5
#eV per cm-1, as in colradpy_class
EV_PER_WAVENUMBER = constants.h*constants.c*100/constants.e


################################################################################
#
# reading the ASCII tables
#
################################################################################

def _lines(path):
    """The lines of a file that are not blank: FAC separates its blocks by blank lines, and their
    sizes are given by their counts."""
    with open(path) as f:
        return [line for line in f.read().splitlines() if line.strip()]


def _value(lines, i, key, kind=float):
    """The value of the line 'key = value' at i, and the next line."""
    if(i >= len(lines)):
        raise ValueError('the file ends where ' + key + ' was expected')
    name, sep, value = lines[i].partition('=')
    if(sep != '=' or not name.split() or name.split()[-1] != key):
        raise ValueError('expected ' + key + ' = ..., found ' + repr(lines[i]))
    return kind(value.strip().split(',')[0]), i + 1


def _grid(lines, i, key):
    """A count 'key = n' followed by n values, one on each line."""
    n, i = _value(lines, i, key, int)
    return np.array([float(lines[k]) for k in range(i, i + n)]), i + n


def _numbers(line, count, kinds):
    """The first count numbers of a line, converted by kinds (one for each)."""
    fields = line.split()
    if(len(fields) < count):
        raise ValueError('expected ' + str(count) + ' numbers in ' + repr(line))
    return [kind(field) for kind, field in zip(kinds, fields[:count])]


def _header(lines, path, table):
    """The header of a FAC ASCII file of the table ('en', 'tr', ...) and the index of its next line."""
    if(not lines or not lines[0].startswith('FAC')):
        raise ValueError(str(path) + ' is not a FAC ASCII table (its first line is not FAC x.y.z)')
    header = {'version': lines[0].split()[1] if len(lines[0].split()) > 1 else ''}
    i = 1
    for key in ('Endian', 'TSess', 'Type', 'Verbose'):
        header[key], i = _value(lines, i, key, int)
    header['symbol'] = lines[i].partition('=')[0].split()[0]
    header['Z'], i = _value(lines, i, 'Z', float)
    header['NBlocks'], i = _value(lines, i, 'NBlocks', int)
    if(header['Type'] != TABLE_TYPES[table]):
        raise ValueError(str(path) + ' holds the FAC table of type ' + str(header['Type']) + ', not the ' +
                         table + ' table (type ' + str(TABLE_TYPES[table]) + ')')
    if(header['Verbose'] != 1):
        raise ValueError(str(path) + ' was printed without the verbose option: ColRadPy reads the energies '
                         'in eV and the cross-sections that PrintTable(binary, ascii, 1) writes')
    return header, i


def read_fac_en(path):
    """Levels of a FAC a.en file (PrintENTable).

    :param path: the a.en file
    :type path: str

    :returns: dict --the header ('symbol', 'Z', ...) and, for each level, 'index' (FAC's ILEV), 'nele',
              'energy' (eV above FAC's ground), 'parity', 'vnl', 'two_j', 'ncomplex', 'sname' (the
              non-relativistic configuration) and 'name' (the jj-coupled level)
    """
    lines = _lines(path)
    header, i = _header(lines, path, 'en')
    if(i < len(lines) and lines[i].startswith('E0')):
        i += 1
    columns = {key: [] for key in ('index', 'nele', 'energy', 'parity', 'vnl', 'two_j', 'ncomplex', 'sname', 'name')}
    while(i < len(lines)):
        nele, i = _value(lines, i, 'NELE', int)
        num, i = _value(lines, i, 'NLEV', int)
        i += 1                                                      #column titles
        for line in lines[i:i + num]:
            fields = line.split()
            index, _, energy, parity, vnl, two_j = _numbers(line, 6, (int, int, float, int, int, int))
            #the bare nucleus has no configuration or name
            names = (fields[6:] + ['', '', ''])[:3]
            for key, value in zip(columns, (index, nele, energy, parity, vnl, two_j) + tuple(names)):
                columns[key].append(value)
        i += num
    out = {key: np.array(value) for key, value in columns.items()}
    out.update(header)
    return out


def read_fac_tr(path):
    """Radiative transition rates of a FAC a.tr file (PrintTRTable).

    :returns: dict --the header and, for each transition, 'upper', 'lower' (FAC level indices),
              'energy' (eV), 'gf', 'rate' (s-1) and 'multipole' (-1 for E1, 1 for M1, ...)
    """
    lines = _lines(path)
    header, i = _header(lines, path, 'tr')
    columns = {key: [] for key in ('upper', 'lower', 'energy', 'gf', 'rate', 'multipole', 'nele')}
    while(i < len(lines)):
        nele, i = _value(lines, i, 'NELE', int)
        num, i = _value(lines, i, 'NTRANS', int)
        multipole, i = _value(lines, i, 'MULTIP', int)
        _, i = _value(lines, i, 'GAUGE', int)
        _, i = _value(lines, i, 'MODE', int)
        for line in lines[i:i + num]:
            fields = line.split()
            upper, _, lower, _, energy = _numbers(line, 5, (int, int, int, int, float))
            #10 columns for UTA transitions, with the width of the array after the energy
            gf, rate = (float(fields[6]), float(fields[7])) if len(fields) >= 10 else (float(fields[5]), float(fields[6]))
            for key, value in zip(columns, (upper, lower, energy, gf, rate, multipole, nele)):
                columns[key].append(value)
        i += num
    out = {key: np.array(value) for key, value in columns.items()}
    out.update(header)
    return out


def _cross_section_blocks(path, table):
    """Transitions of the a.ce, a.rr and a.ci files (PrintCETable, PrintRRTable, PrintCITable): the
    header, and for each transition its line, the line after it (a.ce: the Bethe and Born limits),
    its fit parameters and its rows of the user energy grid."""
    lines = _lines(path)
    header, i = _header(lines, path, table)
    keys = {'ce': ('NELE', 'NTRANS', 'QKMODE', 'NPARAMS', 'MSUB', 'PWTYPE'),
            'rr': ('NELE', 'NTRANS', 'QKMODE', 'MULTIP', 'NPARAMS'),
            'ci': ('NELE', 'NTRANS', 'QKMODE', 'NPARAMS', 'PWTYPE')}[table]
    transitions = []
    while(i < len(lines)):
        block = {}
        for key in keys:
            block[key], i = _value(lines, i, key, int)
        _, i = _grid(lines, i, 'NTEGRID')
        if(table == 'ce'):
            block['TE0'], i = _value(lines, i, 'TE0')
        block['ETYPE'], i = _value(lines, i, 'ETYPE', int)
        _, i = _grid(lines, i, 'NEGRID')
        block['UTYPE'], i = _value(lines, i, 'UTYPE', int)
        usr, i = _grid(lines, i, 'NUSR')
        if(block.get('MSUB')):
            raise ValueError(str(path) + ' holds collision strengths of magnetic sublevels (MSUB = 1), which '
                             'ColRadPy does not read')
        for _ in range(block['NTRANS']):
            transition = {'block': block, 'line': lines[i].split()}
            i += 1
            if(table == 'ce'):
                transition['limits'] = [float(v) for v in lines[i].split()[:3]]
                i += 1
            fit = (table == 'ci' and block['NPARAMS'] > 0) or (table != 'ci' and block['QKMODE'] == QK_FIT)
            if(fit):
                transition['params'] = [float(v) for v in lines[i].split()]
                i += 1
            transition['rows'] = np.array([[float(v) for v in line.split()] for line in lines[i:i + len(usr)]])
            i += len(usr)
            transitions.append(transition)
    return header, transitions


def read_fac_ce(path):
    """Electron impact excitation of a FAC a.ce file (PrintCETable).

    :returns: dict --the header and 'transitions', a dict for each: 'lower', 'upper' (FAC level
              indices), 'lower_2j', 'energy' (transition energy, eV), 'bethe', 'born' (FAC's Bethe
              coefficient b and Born limit c, and the energy of the scattered electron, eV, at which
              FAC computed it), 'scattered' (energies of the scattered electron, eV), 'omega'
              (collision strengths) and 'sigma' (cross-sections, cm2), 'nele'
    """
    header, blocks = _cross_section_blocks(path, 'ce')
    out = dict(header, transitions=[])
    for t in blocks:
        lower, lower_2j, upper, upper_2j, energy = _numbers(' '.join(t['line']), 5, (int, int, int, int, float))
        rows = t['rows']
        #UTYPE = 1: energies of the scattered electron; 0: of the incident one
        scattered = rows[:, 0] - (energy if t['block']['UTYPE'] == 0 else 0.)
        bethe, born, born_energy = t['limits']
        out['transitions'].append({'lower': lower, 'upper': upper, 'lower_2j': lower_2j, 'upper_2j': upper_2j,
                                   'energy': energy, 'bethe': bethe, 'born': (born, born_energy),
                                   'scattered': scattered, 'omega': rows[:, 1], 'sigma': rows[:, 2]*FAC_AREA,
                                   'nele': t['block']['NELE']})
    return out


def read_fac_ci(path):
    """Electron impact ionization of a FAC a.ci file (PrintCITable).

    :returns: dict --the header and 'transitions', a dict for each: 'bound', 'free' (FAC level
              indices), 'bound_2j', 'energy' (ionization energy, eV), 'params' (FAC's fit),
              'ejected' (energies above the threshold, eV), 'omega' (collision strengths in FAC's
              ionization convention) and 'sigma' (cross-sections, cm2), 'nele'
    """
    header, blocks = _cross_section_blocks(path, 'ci')
    out = dict(header, transitions=[])
    for t in blocks:
        bound, bound_2j, free, free_2j, energy = _numbers(' '.join(t['line']), 5, (int, int, int, int, float))
        rows = t['rows']
        out['transitions'].append({'bound': bound, 'free': free, 'bound_2j': bound_2j, 'free_2j': free_2j,
                                   'energy': energy, 'params': t.get('params', []),
                                   'ejected': rows[:, 0] - (energy if t['block']['UTYPE'] == 0 else 0.),
                                   'omega': rows[:, 1], 'sigma': rows[:, 2]*FAC_AREA, 'nele': t['block']['NELE']})
    return out


def read_fac_rr(path):
    """Radiative recombination of a FAC a.rr file (PrintRRTable).

    :returns: dict --the header and 'transitions', a dict for each: 'bound', 'free' (FAC level
              indices), 'bound_2j', 'free_2j', 'energy' (binding energy, eV), 'n' and 'l' of the
              captured electron, 'params' (FAC's fit, the last in eV, when QKMODE is QK_FIT),
              'photoelectron' (energies, eV), 'sigma' (recombination cross-sections, cm2),
              'strength' (bound-free oscillator strength densities), 'nele'
    """
    header, blocks = _cross_section_blocks(path, 'rr')
    out = dict(header, transitions=[])
    for t in blocks:
        bound, bound_2j, free, free_2j, energy, kl = _numbers(' '.join(t['line']), 6, (int, int, int, int, float, int))
        if(free < 0):
            continue
        rows = t['rows']
        out['transitions'].append({'bound': bound, 'free': free, 'bound_2j': bound_2j, 'free_2j': free_2j,
                                   'energy': energy, 'n': kl//1000, 'l': kl % 1000, 'params': t.get('params', []),
                                   'photoelectron': rows[:, 0] - (energy if t['block']['UTYPE'] == 0 else 0.),
                                   'sigma': rows[:, 1]*FAC_AREA, 'strength': rows[:, 3], 'nele': t['block']['NELE']})
    return out


def read_fac_ai(path):
    """Autoionization of a FAC a.ai file (PrintAITable).

    :returns: dict --the header and, for each transition, 'bound', 'free' (FAC level indices),
              'energy' (of the bound level above the free one, eV), 'rate' (autoionization rate, s-1)
              and 'strength' (dielectronic capture strength, cm2 eV), 'nele'
    """
    lines = _lines(path)
    header, i = _header(lines, path, 'ai')
    columns = {key: [] for key in ('bound', 'free', 'energy', 'rate', 'strength', 'nele')}
    while(i < len(lines)):
        nele, i = _value(lines, i, 'NELE', int)
        num, i = _value(lines, i, 'NTRANS', int)
        _, i = _value(lines, i, 'EMIN')
        _, i = _grid(lines, i, 'NEGRID')
        for line in lines[i:i + num]:
            bound, _, free, _, energy, rate, strength = _numbers(line, 7, (int, int, int, int, float, float, float))
            for key, value in zip(columns, (bound, free, energy, rate, strength*FAC_AREA, nele)):
                columns[key].append(value)
        i += num
    out = {key: np.array(value) for key, value in columns.items()}
    out.update(header)
    return out


def read_fac_mr(path):
    """Maxwellian rate coefficients that pfac.fac.MaxwellRate writes from an excitation, ionization or
    recombination table (CEMaxwell, CIMaxwell, RRMaxwell of FAC).

    :returns: list --a dict for each transition: 'initial', 'final' (FAC level indices: lower and
              upper level for excitation, bound level and parent for ionization and recombination),
              'energy' (eV), 'temp' (eV), 'strength' (the effective collision strength of excitation
              and ionization, without the Boltzmann factor) and 'rate' (cm3 s-1)
    """
    lines = _lines(path)
    transitions = []
    i = 0
    while(i < len(lines)):
        if(not lines[i].startswith('#')):
            raise ValueError(str(path) + ': expected the line # initial 2J final 2J energy count of a transition, '
                             'found ' + repr(lines[i]))
        fields = lines[i][1:].split()
        initial, _, final, _, energy, num = _numbers(' '.join(fields), 6, (int, int, int, int, float, int))
        if(len(fields) > 6 and int(fields[6]) != 1):
            raise ValueError(str(path) + ' holds rates of magnetic sublevels, which ColRadPy does not read')
        rows = np.array([[float(v) for v in line.split()] for line in lines[i + 1:i + 1 + num]])
        transitions.append({'initial': initial, 'final': final, 'energy': energy, 'temp': rows[:, 0],
                            'strength': rows[:, 1], 'rate': rows[:, 2]*FAC_RATE})
        i += 1 + num
    return transitions


################################################################################
#
# cross-sections
#
################################################################################

def _momentum_squared(energy):
    """k^2 (atomic units) of electrons with the kinetic energies energy (eV), as FAC relates cross-sections
    to collision strengths: 2E (1 + alpha^2 E/2) with E in Hartree."""
    energy = np.asarray(energy, dtype=float)/HARTREE_EV
    return 2*energy*(1 + 0.5*ALPHA**2*energy)


def _interpolate(x, xp, yp):
    """yp(xp) at x inside xp (increasing): log-log for positive values, else linear in log x."""
    if(len(xp) == 1):
        return np.full(np.shape(x), float(yp[0]))
    if(np.all(yp > 0)):
        return np.exp(np.interp(np.log(x), np.log(xp), np.log(yp)))
    return np.interp(np.log(x), np.log(xp), yp)


class FacCrossSection:
    """Cross-section (cm2) of a FAC transition at the energies (eV) of the incident electron, the free
    electron for recombination, from the values that FAC tabulates and its forms outside them (see the
    module). Create with excitation, ionization or recombination; it is zero below the threshold.

    Attributes: 'threshold' (eV, 0 for recombination), 'initial' and 'final' (FAC level indices).
    """

    def __init__(self, kind, threshold, weight, above, values, initial, final, **extra):
        order = np.argsort(above)
        self.kind, self.threshold, self.weight = kind, float(threshold), float(weight)
        #energies above the threshold (scattered, ejected or photoelectron) and the tabulated values
        keep = above[order] > 0
        self.above, self.values = np.asarray(above, dtype=float)[order][keep], np.asarray(values, dtype=float)[order][keep]
        if(len(self.above) == 0):
            raise ValueError('FAC transition ' + str(initial) + ' - ' + str(final) + ' has no energies above its threshold')
        self.initial, self.final = initial, final
        self.__dict__.update(extra)

    @classmethod
    def excitation(cls, transition, threshold):
        """From a transition of read_fac_ce, with the threshold (eV) of its levels."""
        return cls('ce', threshold, transition['lower_2j'] + 1, transition['scattered'], transition['omega'],
                   transition['lower'], transition['upper'], bethe=transition['bethe'], born=transition['born'])

    @classmethod
    def ionization(cls, transition, threshold):
        """From a transition of read_fac_ci, with the ionization energy (eV) of its levels."""
        return cls('ci', threshold, transition['bound_2j'] + 1, transition['ejected'], transition['omega'],
                   transition['bound'], transition['free'], params=list(transition['params']))

    @classmethod
    def recombination(cls, transition, binding):
        """From a transition of read_fac_rr, with the binding energy (eV) of its levels."""
        return cls('rr', 0., transition['free_2j'] + 1, transition['photoelectron'], transition['strength'],
                   transition['free'], transition['bound'], binding=float(binding), l=transition['l'],
                   bound_weight=transition['bound_2j'] + 1, params=list(transition['params']))

    def _ionization_fit(self, x):
        p = (list(self.params) + [0., 0., 0., 0.])[:4]
        y = 1 - 1/x
        return p[0]*np.log(x) + p[1]*y**2 + p[2]*y/x + p[3]*y/x**2

    def _recombination_fit(self, above):
        p = self.params
        x = 1 + above/p[3]
        return (p[0]*x**(0.5*p[1] - 4.5 - self.l)*((1 + p[2])/(np.sqrt(x) + p[2]))**p[1]*
                (self.binding + above)/p[3])

    def _beyond(self, above, low):
        """Values outside the tabulated energies: below them (low) or above them."""
        edge, value = (self.above[0], self.values[0]) if low else (self.above[-1], self.values[-1])
        if(self.kind == 'ce'):
            if(low):
                return np.full(np.shape(above), value)
            return self._born(above)
        if(self.kind == 'ci'):
            x, x_edge = 1 + above/self.threshold, 1 + edge/self.threshold
            fit, fit_edge = self._ionization_fit(x), self._ionization_fit(x_edge)
            if(len(self.params) and fit_edge > 0):
                return np.clip(value*fit/fit_edge, 0, None)
            return value*(above/edge if low else np.log(x)/np.log(x_edge))
        #FAC's fit is valid above its energies only (FAC manual); the strength is finite at threshold
        if(low):
            return np.full(np.shape(above), value)
        if(len(self.params) == 4 and self.params[3] > 0):
            fit, fit_edge = self._recombination_fit(above), self._recombination_fit(edge)
            if(fit_edge > 0):
                return value*fit/fit_edge
        if(len(self.above) < 2 or self.values[-2] <= 0 or value <= 0):
            return np.full(np.shape(above), value)
        photon = self.binding + self.above[-2:]
        power = min(np.log(value/self.values[-2])/np.log(photon[1]/photon[0]), 0.)
        return value*((self.binding + above)/photon[1])**power

    def _born(self, above):
        """Collision strengths above the last tabulated energy: towards the Bethe-Born limit of FAC."""
        last, value = self.above[-1], self.values[-1]
        bethe, (born, born_above) = self.bethe, self.born
        if(bethe < 0 or self.threshold <= 0):
            #no Born limit (exchange, which falls as E^-2): as FAC's PrepCECrossData, the power law of
            #the last two values, falling at least as E^-1
            power = -2.
            if(len(self.above) > 1 and self.values[-2] > 0 and value > 0):
                power = min(np.log(value/self.values[-2])/
                            np.log((self.threshold + last)/(self.threshold + self.above[-2])), -1.)
            return value*((self.threshold + above)/(self.threshold + last))**power
        x = (self.threshold + above)/self.threshold
        limit = np.clip(bethe*np.log(x) + born, 0, None)
        x_born = (self.threshold + born_above)/self.threshold
        limit_born = bethe*np.log(x_born) + born
        if(born_above <= last or limit_born <= 0):
            return np.where(limit > 0, limit, value)
        #log-log from the last value to the limit at FAC's Born energy, the limit above it
        x_last = (self.threshold + last)/self.threshold
        joined = _interpolate(np.clip(x, x_last, x_born), np.array([x_last, x_born]), np.array([value, limit_born]))
        return np.where(x > x_born, limit, joined)

    def collision_strength(self, energy):
        """Collision strengths (excitation, ionization) or bound-free oscillator strength densities
        (recombination) at the incident (free) electron energies energy (eV)."""
        above = np.asarray(energy, dtype=float) - self.threshold
        out = np.zeros(np.shape(above))
        inside = (above >= self.above[0]) & (above <= self.above[-1])
        low = (above > 0) & (above < self.above[0])
        high = above > self.above[-1]
        out[inside] = _interpolate(above[inside], self.above, self.values)
        if(low.any()):
            out[low] = self._beyond(above[low], True)
        if(high.any()):
            out[high] = self._beyond(above[high], False)
        return out

    def __call__(self, energy):
        energy = np.asarray(energy, dtype=float)
        strength = self.collision_strength(energy)
        with np.errstate(divide='ignore', invalid='ignore'):
            if(self.kind == 'rr'):
                #FAC's PrintRRTable: 2 pi alpha S (alpha E_photon)^2/(2 E (1 + alpha^2 E/2)) a0^2/g_free
                free = energy/HARTREE_EV
                photon = (energy + self.binding)/HARTREE_EV
                sigma = (2*np.pi*ALPHA*strength*(ALPHA*photon)**2/(2*free)/(1 + 0.5*ALPHA**2*free)*
                         BOHR_AREA/self.weight)
            else:
                #excitation pi Omega/(g k^2) a0^2, ionization Omega/(g k^2) a0^2
                sigma = (np.pi if self.kind == 'ce' else 1.)*strength/(self.weight*_momentum_squared(energy))*BOHR_AREA
        return np.where(energy > self.threshold, np.nan_to_num(sigma), 0.)


def fac_cross_sections(fil, process, nele=None):
    """Cross-sections of the transitions of a FAC table, with the thresholds of the transition lines.

    :param fil: prefix of the FAC files (Ar01 for Ar01a.ce)
    :type fil: str

    :param process: 'ce', 'ci' or 'rr'
    :type process: str

    :param nele: number of electrons of the ion (of the initial level of excitation and ionization,
                 of the recombined level), all by default
    :type nele: int

    :returns: list --FacCrossSection of each transition
    """
    reader = {'ce': read_fac_ce, 'ci': read_fac_ci, 'rr': read_fac_rr}[process]
    make = {'ce': FacCrossSection.excitation, 'ci': FacCrossSection.ionization,
            'rr': FacCrossSection.recombination}[process]
    return [make(t, t['energy']) for t in reader(str(fil) + 'a.' + process)['transitions']
            if nele is None or t['nele'] == nele]


################################################################################
#
# the dictionary of read_adf04
#
################################################################################

def two_photon_rate(z, nele):
    """Two-photon decay rate (s-1) of the 2s level of H-like ions (nele = 1) and the 1s2s 1S0 level of
    He-like ions (nele = 2) of nuclear charge z, from G. W. F. Drake, Phys. Rev. A 34, 2871 (1986), as
    in FAC (TwoPhotonRate)."""
    if(nele == 1):
        a2 = (ALPHA*z)**2
        return 8.22943*z**6*(1 + 3.9448*a2 - 2.04*a2**2)/(1 + 4.6019*a2)
    if(nele == 2):
        screened = z - 0.806389
        a2 = (ALPHA*screened)**2
        return 16.458762*(screened**6*(1 + 1.539/(z + 2.5)**2) -
                          screened**6*a2*(0.6571 + 2.04*a2)/(1 + 4.6019*a2))
    raise ValueError('FAC gives two-photon rates for H-like and He-like ions, not ' + str(nele) + ' electrons')


def _kelvin(temps):
    """Temperatures (eV) in K as in adf04 files, the first and last moved by an ulp if needed so that
    dividing by 11604.5 (as ColRadPy does) does not leave them inside the temperatures they came from,
    which would extrapolate the rates."""
    kelvin = np.asarray(temps, dtype=float)*KELVIN_PER_EV
    while(kelvin[0]/KELVIN_PER_EV > temps[0]):
        kelvin[0] = np.nextafter(kelvin[0], 0)
    while(kelvin[-1]/KELVIN_PER_EV < temps[-1]):
        kelvin[-1] = np.nextafter(kelvin[-1], np.inf)
    return kelvin


def _processes(fil, nele, processes, eedf):
    """The processes to read and their files; missing files are skipped with a warning when all are
    requested, and an error when they were named."""
    suffix = {'en': 'a.en', 'tr': 'a.tr', 'ai': 'a.ai'}
    for p in ('ce', 'rr', 'ci'):
        suffix[p] = p + '.mr' if eedf == 'Maxwellian_mr' else 'a.' + p
    requested = list(PROCESSES) if isinstance(processes, str) and processes == 'incl_all' else list(processes)
    unknown = set(requested) - set(PROCESSES)
    if(unknown):
        raise ValueError('FAC processes are ' + str(PROCESSES) + ' or incl_all, not ' + str(sorted(unknown)))
    for p in ('en', 'tr'):
        if(p not in requested):
            raise ValueError('a model needs the levels (en) and radiative rates (tr) of FAC')
    files = {}
    for p in requested:
        path = str(fil) + suffix[p]
        if(os.path.isfile(path)):
            files[p] = path
        elif(p in ('en', 'tr') or not (isinstance(processes, str) and processes == 'incl_all')):
            raise FileNotFoundError('FAC file ' + path + ' (' + p + ') not found')
        elif(not (p == 'ai' and nele == 1)):
            warnings.warn('FAC file ' + path + ' not found, ' + p + ' is left out')
    return files


def _pair_index(pairs, what):
    """Index of each item of a list (levels or pairs of them), which must not repeat."""
    index = {}
    for k, pair in enumerate(pairs):
        if(pair in index):
            raise ValueError('FAC data give the ' + what + ' ' + str(pair) + ' twice')
        index[pair] = k
    return index


def _fold_autoionization(autoionization, a_val, upsilon, recombination, ionization, energy_ev, parent_ev, w,
                         temps):
    """Folds the autoionization of the levels (level, parent, s-1) into the other rates, which it
    changes in place (pairs 1 based: a_val and upsilon (upper, lower), recombination (parent, level),
    ionization (level, parent) as reduced rates). Each flow into an autoionizing level d, by capture,
    excitation or radiative cascade, is multiplied by the fraction b(d) of it that ends in levels that
    do not autoionize, and the rest of the excitation of those levels is their ionization to the
    parents the autoionization reaches. With R(d) its radiative decay rates and A(d) its
    autoionization, from the lowest autoionizing level up,

        b(d) = sum_l R(d->l) b(l) / (R(d) + A(d)),   b = 1 for levels that do not autoionize,
        P(d, p) = (A(d->p) + sum_l R(d->l) P(l, p)) / (R(d) + A(d)),

    and the decays into autoionizing levels become R(d->l) b(l), so that the population of d is that
    with its autoionization, the inflow over R(d) + A(d), while collisions from it are slow beside its
    decays. Returns b of each level.
    """
    num_levels = len(autoionization)
    stabilizing = np.ones(num_levels)
    to_parent = np.zeros(autoionization.shape)
    autoionizing = autoionization.sum(axis=1) > 0
    decays = {}
    for (upper, lower), rate in a_val.items():
        decays.setdefault(upper - 1, []).append((lower - 1, rate))
    for d in np.where(autoionizing)[0]:
        out = decays.get(d, [])
        total = sum(rate for _, rate in out) + autoionization[d].sum()
        stabilizing[d] = sum(rate*stabilizing[l] for l, rate in out)/total
        to_parent[d] = (autoionization[d] + sum(rate*to_parent[l] for l, rate in out))/total
        for l, rate in out:
            if(autoionizing[l]):
                a_val[(d + 1, l + 1)] = rate*stabilizing[l]
    for (upper, lower), values in list(upsilon.items()):
        u, l = upper - 1, lower - 1
        if(autoionizing[u] and not autoionizing[l]):
            #excitation-autoionization: the reduced excitation rate of l times the probability of reaching
            #each parent, as a reduced ionization rate exp(-(E_u - I_p)/kT) times it
            excitation = values*UPSILON_RATE*np.sqrt(RYDBERG_EV/temps)/(2*w[l] + 1)
            for p in np.where(to_parent[u] > 0)[0]:
                pair = (l + 1, p + 1)
                ionization[pair] = ionization.get(pair, 0.) +                     excitation*to_parent[u, p]*np.exp(-(energy_ev[u] - parent_ev[p])/temps)
        upsilon[(upper, lower)] = values*stabilizing[u]*stabilizing[l]
    for (parent, level), values in recombination.items():
        recombination[(parent, level)] = values*stabilizing[level - 1]
    return stabilizing


def read_fac(fil, nele, temp_grid, eedf='Maxwellian', processes='incl_all', relativistic=True,
             fold_autoionization=True):
    """Reads the FAC files of an ion into the dictionary of read_adf04, with the rates of the
    cross-sections on the temperatures temp_grid (see the module).

    :param fil: prefix of the FAC files (Ar01 for Ar01a.en, Ar01a.tr, ...)
    :type fil: str

    :param nele: number of electrons of the ion
    :type nele: int

    :param temp_grid: temperatures of the model (eV), where the cross-sections are averaged
    :type temp_grid: float array

    :param eedf: 'Maxwellian' to average FAC's cross-sections over a thermal (Maxwell-Juttner) electron
                 distribution, or 'Maxwellian_mr' to take the rate coefficients of
                 pfac.fac.MaxwellRate (fil + 'ce.mr', 'rr.mr', 'ci.mr') on their temperatures
    :type eedf: str

    :param processes: 'incl_all' for all the FAC files found, or a list of 'en', 'tr', 'ce', 'rr',
                      'ai', 'ci' (en and tr are needed)
    :type processes: str or list

    :param relativistic: Maxwell-Juttner distribution and relativistic electron speeds
    :type relativistic: bool

    :param fold_autoionization: fold the autoionization into the other rates (see the module); False
                                leaves the autoionizing levels with the rates of the files and without
                                their autoionization, for models that add it themselves
    :type fold_autoionization: bool

    :returns: dict --the 'atomic', 'input_file' and 'rates' dictionaries of read_adf04, and the
              file prefix in ['user']['file_loc'], to give colradpy as its file:
              colradpy(read_fac(fil, nele, temps), metas, temps, dens)
    """
    if(eedf not in ('Maxwellian', 'Maxwellian_mr')):
        raise ValueError("FAC rates are made for eedf = 'Maxwellian' (from the cross-sections) or "
                         "'Maxwellian_mr' (pfac.fac.MaxwellRate files), not " + repr(eedf))
    files = _processes(fil, nele, processes, eedf)

    #levels of the ion and of the next one (the parents), sorted by energy
    en = read_fac_en(files['en'])
    _pair_index(en['index'].tolist(), 'level')
    if(not np.any(en['nele'] == nele)):
        raise ValueError(files['en'] + ' has no levels with ' + str(nele) + ' electrons, only ' +
                         str(sorted(set(en['nele'].tolist()))))
    if(not np.any(en['nele'] == nele - 1)):
        raise ValueError(files['en'] + ' has no levels with ' + str(nele - 1) + ' electrons, the parents '
                         'of the ion: FAC must make the structure of the next ion as well')
    levels = np.where(en['nele'] == nele)[0]
    levels = levels[np.argsort(en['energy'][levels], kind='stable')]
    parents = np.where(en['nele'] == nele - 1)[0]
    parents = parents[np.argsort(en['energy'][parents], kind='stable')]
    ground = en['energy'][levels[0]]
    level_of = {int(en['index'][k]): n for n, k in enumerate(levels)}            #FAC index -> 0 based
    parent_of = {int(en['index'][k]): p for p, k in enumerate(parents)}
    energy_ev = en['energy'][levels] - ground
    parent_ev = en['energy'][parents] - ground
    w = en['two_j'][levels]/2.
    w_ion = en['two_j'][parents]/2.
    z = int(round(en['Z']))

    atomic = {'element': en['symbol'], 'charge_state': z - nele, 'iz0': z, 'iz1': z - nele + 1,
              'ion_pot': parent_ev/EV_PER_WAVENUMBER,
              'ion_term': np.array([str(name) or 'bare' for name in en['name'][parents]]),
              'ion_w': w_ion, 'config': en['sname'][levels], 'fac_name': en['name'][levels],
              'fac_index': en['index'][levels], 'ion_fac_index': en['index'][parents],
              'L': -np.ones(len(levels), dtype=int), 'S': -np.ones(len(levels), dtype=int), 'w': w,
              'energy': energy_ev/EV_PER_WAVENUMBER, 'ion_pot_lvl': np.arange(1, len(parents) + 1)}
    #temperatures of the rates: the model's, or those of the MaxwellRate files
    if(eedf == 'Maxwellian'):
        temps = np.unique(np.asarray(temp_grid, dtype=float))
    else:
        mr = {p: read_fac_mr(files[p]) for p in ('ce', 'rr', 'ci') if p in files}
        grids = [t['temp'] for transitions in mr.values() for t in transitions]
        if(not grids):
            raise ValueError('no MaxwellRate files (ce.mr, rr.mr, ci.mr) of ' + str(fil) + ' were found')
        temps = grids[0]
        if(any(len(g) != len(temps) or not np.allclose(g, temps, rtol=1e-3) for g in grids)):
            raise ValueError('the MaxwellRate files of ' + str(fil) + ' are not on the same temperatures')
    if(np.any(temps <= 0)):
        raise ValueError('FAC rates need positive temperatures')
    sqrt_temp = np.sqrt(temps/RYDBERG_EV)

    #radiative rates, summed over the multipoles of each pair (upper, lower), 1 based
    tr = read_fac_tr(files['tr'])
    a_val = {}
    seen = set()
    for upper, lower, rate, multipole in zip(tr['upper'], tr['lower'], tr['rate'], tr['multipole']):
        if(upper in level_of and lower in level_of):
            i, j = level_of[upper], level_of[lower]
            pair = (max(i, j) + 1, min(i, j) + 1)
            if((pair, multipole) in seen):
                raise ValueError('FAC data give the radiative rate of multipole ' + str(multipole) + ' of ' +
                                 str(pair) + ' twice')
            seen.add((pair, multipole))
            a_val[pair] = a_val.get(pair, 0.) + rate
    two_photon = {}
    if(nele in (1, 2)):
        #2s 2S1/2 -> 1s of H-like ions, 1s2s 1S0 -> 1s2 of He-like ions
        (upper_config, upper_j), lower_config = {1: (('2s1', 0.5), '1s1'), 2: (('1s1.2s1', 0.), '1s2')}[nele]
        upper = np.where((atomic['config'] == upper_config) & (w == upper_j))[0]
        lower = np.where(atomic['config'] == lower_config)[0]
        if(len(upper) and len(lower)):
            pair = (upper[0] + 1, lower[0] + 1)
            two_photon = {'transition': np.array(pair), 'a_val': two_photon_rate(z, nele)}
            a_val[pair] = a_val.get(pair, 0.) + two_photon['a_val']

    #excitation: effective collision strengths (upsilons) of the pairs (upper, lower), 1 based
    upsilon = {}
    if('ce' in files):
        if(eedf == 'Maxwellian'):
            ce = read_fac_ce(files['ce'])['transitions']
            items = [(t['lower'], t['upper'], t) for t in ce]
        else:
            items = [(t['initial'], t['final'], t) for t in mr['ce']]
        _pair_index([(a, b) for a, b, _ in items], 'excitation')
        for a, b, t in items:
            if(a not in level_of or b not in level_of):
                continue
            i, j = sorted((level_of[a], level_of[b]))
            if(eedf == 'Maxwellian'):
                #the effective collision strength from the reduced rate of FAC's lower level, with
                #ColRadPy's constants, so that the CR matrix has the integrated rate
                threshold = max(energy_ev[j] - energy_ev[i], 0.)
                rate = rate_coefficients(FacCrossSection.excitation(t, threshold), threshold, temps,
                                         relativistic, reduced=True)
                values = rate*sqrt_temp/UPSILON_RATE*(t['lower_2j'] + 1)
            else:
                values = t['strength']
            upsilon[(j + 1, i + 1)] = values

    #recombination of the parents into the levels (parent, level), 1 based
    recombination = {}
    if('rr' in files):
        if(eedf == 'Maxwellian'):
            items = [(t['bound'], t['free'], t) for t in read_fac_rr(files['rr'])['transitions']]
        else:
            items = [(t['initial'], t['final'], t) for t in mr['rr']]
        _pair_index([(a, b) for a, b, _ in items], 'radiative recombination')
        for bound, free, t in items:
            if(bound not in level_of or free not in parent_of):
                continue
            level, parent = level_of[bound], parent_of[free]
            if(eedf == 'Maxwellian'):
                section = FacCrossSection.recombination(t, parent_ev[parent] - energy_ev[level])
                values = rate_coefficients(section, 0., temps, relativistic)
            else:
                values = t['rate']
            recombination[(parent + 1, level + 1)] = values

    #autoionization of the levels to the parents (s-1) and dielectronic capture into them
    autoionization = np.zeros((len(levels), len(parents)))
    if('ai' in files):
        ai = read_fac_ai(files['ai'])
        _pair_index(list(zip(ai['bound'].tolist(), ai['free'].tolist())), 'autoionization')
        for bound, free, rate, strength in zip(ai['bound'], ai['free'], ai['rate'], ai['strength']):
            if(bound not in level_of or free not in parent_of):
                continue
            level, parent = level_of[bound], parent_of[free]
            resonance = energy_ev[level] - parent_ev[parent]
            if(resonance <= 0):
                continue
            autoionization[level, parent] += rate
            capture = resonance_rate_coefficients(strength, resonance, temps, relativistic)
            recombination[(parent + 1, level + 1)] = recombination.get((parent + 1, level + 1), 0.) + capture

    #ionization of the levels to the parents (level, parent), 1 based, as reduced rates
    ionization = {}
    if('ci' in files):
        if(eedf == 'Maxwellian'):
            items = [(t['bound'], t['free'], t) for t in read_fac_ci(files['ci'])['transitions']]
        else:
            items = [(t['initial'], t['final'], t) for t in mr['ci']]
        _pair_index([(a, b) for a, b, _ in items], 'ionization')
        for bound, free, t in items:
            if(bound not in level_of or free not in parent_of):
                continue
            level, parent = level_of[bound], parent_of[free]
            threshold = parent_ev[parent] - energy_ev[level]
            if(threshold <= 0):
                continue
            if(eedf == 'Maxwellian'):
                values = rate_coefficients(FacCrossSection.ionization(t, threshold), threshold, temps,
                                           relativistic, reduced=True)
            else:
                #FAC's CIMaxwell: (217.16/pi) sqrt(Hartree/2kT) strength/g e^-I/kT, in 1e-10 cm3 s-1
                values = UPSILON_RATE/np.pi*np.sqrt(HARTREE_EV/(2*t['temp']))*t['strength']/(2*w[level] + 1)
            ionization[(level + 1, parent + 1)] = values

    stabilizing = np.ones(len(levels))
    kept = np.arange(len(levels))
    if(fold_autoionization and autoionization.any()):
        stabilizing = _fold_autoionization(autoionization, a_val, upsilon, recombination, ionization,
                                           energy_ev, parent_ev, w, temps)
        kept = np.where(stabilizing > 0)[0]
        if(len(kept) < len(levels)):
            warnings.warn(str(len(levels) - len(kept)) + ' autoionizing levels of ' + str(fil) + ' have no '
                          'radiative decay in the FAC data and are left out')
            new = {int(old): n for n, old in enumerate(kept)}
            a_val = {(new[u - 1] + 1, new[l - 1] + 1): v for (u, l), v in a_val.items() if u - 1 in new and l - 1 in new}
            upsilon = {(new[u - 1] + 1, new[l - 1] + 1): v for (u, l), v in upsilon.items() if u - 1 in new and l - 1 in new}
            recombination = {(p, new[l - 1] + 1): v for (p, l), v in recombination.items() if l - 1 in new}
            ionization = {(new[l - 1] + 1, p): v for (l, p), v in ionization.items() if l - 1 in new}
            for key in ('config', 'fac_name', 'fac_index', 'L', 'S', 'w', 'energy'):
                atomic[key] = atomic[key][kept]
            if(two_photon):
                two_photon['transition'] = np.array([new[k - 1] + 1 for k in two_photon['transition']])

    pairs = sorted(set(a_val) | set(upsilon))
    num = len(temps)
    rates = {'excit': {'col_transitions': np.array(pairs, dtype=int).reshape(-1, 2),
                       'col_excit': np.array([upsilon.get(p, np.zeros(num)) for p in pairs]).reshape(-1, num)},
             'a_val': np.array([a_val.get(p, 0.) for p in pairs]),
             'inf_engy': np.array([], dtype=float),
             'cx': {'cx_transitions': np.array([]), 'cx_excit': np.array([])}}
    if(two_photon):
        rates['two_photon'] = two_photon
    recomb_pairs = sorted(recombination)
    rates['recomb'] = {'recomb_transitions': np.array(recomb_pairs, dtype=int).reshape(-1, 2),
                       'recomb_excit': np.array([recombination[p] for p in recomb_pairs]).reshape(-1, num)}
    ion_pairs = sorted(ionization)
    rates['ioniz'] = {'ion_transitions': np.array(ion_pairs, dtype=int).reshape(-1, 2),
                      'ion_excit': np.array([ionization[p] for p in ion_pairs]).reshape(-1, num)}
    #the levels ionize to the parents with ionization data (zpla, zpla1 as in read_adf04)
    zpla = -np.ones((len(kept), len(parents)))
    zpla1 = -np.ones((len(kept), len(parents)))
    for level, parent in ion_pairs:
        zpla[level - 1, parent - 1], zpla1[level - 1, parent - 1] = 1., parent
    atomic['zpla'], atomic['zpla1'] = zpla, zpla1

    out = {'atomic': atomic, 'rates': rates, 'user': {'file_loc': str(fil)},
           'input_file': {'temp_grid': _kelvin(temps), 'fac': {key: os.path.abspath(path) for key, path in files.items()},
                          'eedf': eedf, 'relativistic': relativistic,
                          'autoionization': autoionization[kept], 'stabilizing': stabilizing[kept]}}
    return out

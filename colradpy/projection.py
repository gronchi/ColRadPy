"""Projection of the high n-shells onto the levels of an adf04 file.

The levels of an adf04 file usually cover the lowest n-shells of the ion. Recombination into the
higher n-shells, and the redistribution and ionization through them, are missing from a
collisional-radiative model built on those levels alone, and they dominate recombination at most
temperatures. A bundle-nS calculation (ADAS204) describes all n-shells, grouped into n-shell
populations for each parent and spin system, and condenses them onto its lowest n-shells: the
condensed (projected) couplings between those shells, their ionization to the parents, the
recombination into them and the coupling between parents through the high n-shells (adf17
files). It also gives the direct part of each coupling, which an adf04 file already has, so that
projected minus direct leaves the indirect couplings through the high n-shells. An adf18
a17_p208 file assigns each adf04 level to an n-shell, spin system and parent, with a weight
normalised over the levels of the n-shell, over which the indirect couplings are spread:

- recombination from the parent into level i of shell n: w_i r_n;
- ionization of level i in shell n to the parent: the ionization of shell n (the shell population
  is the sum of its level populations). Ionization through the n-shells of one parent to another
  parent, the autoionization of core-excited Rydberg states, is excitation-autoionization, which
  the ionization rates of adf04 files (from adf07 data) already include, so it is left out;
- transfer from level j in shell m to level i in shell n (n != m): w_i times the transfer from
  shell m to shell n; transfers within a shell are not described by n-shell populations;
- coupling between parents through the high n-shells, added to the parent cross coupling.

The losses of the levels follow from these, so the collisional-radiative matrix conserves
particles. The bundle-nS couplings are in the convention of the adf17 files: losses positive on
the diagonal and gains negative off it, in s-1; recombination in cm3 s-1.
"""

import warnings

import numpy as np
from scipy import constants
from scipy.interpolate import RegularGridInterpolator


#temperature of 1 eV in K
KELVIN_PER_EV = 11604.5
#energy of 1 cm-1 in eV, as in colradpy_class
EV_PER_WAVENUMBER = constants.h*constants.c*100/constants.e
#exponents of the power laws that the condensed couplings follow below the first temperature of their
#adf17 grid, through their two lowest temperatures, as the rates of adf04 files below theirs
#(colradpy_class): recombination, and the parent coupling that starts with it, grows at most as
#T^-1.5 towards lower temperatures and does not fall; the reduced ionization and transfers (times
#exp(dE/kT), dE their threshold) do not grow
RECOMBINATION_EXPONENTS = (-1.5, 0.)
REDUCED_EXPONENTS = (0., None)
#effective zero of the log interpolation of the indirect couplings, which are not negative
EFFECTIVE_ZERO = 1e-300
#negative indirect couplings up to this fraction of the projected and direct values they are the
#difference of are round-off (the adf17 files have 11 digits) and are set to zero; larger ones show
#condensed data that are not reliable, as at the lowest temperatures of many adf17 files
ROUND_OFF = 1e-3


def _grid_first(array, grid_axes):
    """A copy of the array with its temperature and density axes moved to the front."""
    return np.moveaxis(np.array(array, dtype=float), grid_axes, (0, 1))


def _complete(system, switches, spin_weight):
    """Projected and direct couplings of a spin system (temperature and density axes first), with
    the direct parts that the adf17 file left out of the projected ones added back (switches IECION,
    IETREC, IEDREC, IERREC, IEXREC set to 1) and the recombination multiplied by the spin system
    weight when the file did not (IERSYS = 1)."""
    data = {'crmat': _grid_first(system['crmat'], (2, 3)), 'ion': _grid_first(system['ion'], (2, 3)),
            'rhs': _grid_first(system['rhs'], (1, 2)),
            'dir_crmat': _grid_first(system['dir']['crmat'], (2, 3)),
            'dir_ion': _grid_first(system['dir']['ion'], (2, 3))}
    for key in ('recomb_three_body', 'recomb_diel', 'recomb_rad', 'recomb_cx'):
        data[key] = _grid_first(system['dir'][key], (1, 2))
    num_parents = data['ion'].shape[-1]
    if(system['parent_qc'] is None):
        data['parent_qc'] = np.zeros(data['rhs'].shape[:2] + (num_parents,))
    else:
        data['parent_qc'] = _grid_first(system['parent_qc'], (1, 2))
    if(switches.get('IECION', 0) == 1):
        data['ion'] = data['ion'] + data['dir_ion']
        shells = np.arange(data['crmat'].shape[-1])
        data['crmat'][..., shells, shells] += data['dir_ion'].sum(axis=-1)
    for switch, key in (('IETREC', 'recomb_three_body'), ('IEDREC', 'recomb_diel'),
                        ('IERREC', 'recomb_rad'), ('IEXREC', 'recomb_cx')):
        if(switches.get(switch, 0) == 1):
            data['rhs'] = data['rhs'] + data[key]
    if(switches.get('IERSYS', 0) == 1):
        for key in ('rhs', 'recomb_three_body', 'recomb_diel', 'recomb_rad', 'recomb_cx', 'parent_qc'):
            data[key] = data[key]*spin_weight
    return data


def _unreliable(value, scale):
    """Points of the temperature and density grid (the first two axes) where an indirect coupling is
    negative beyond the round-off of the values it is the difference of (scale)."""
    beyond = value < -ROUND_OFF*scale
    return beyond.reshape(beyond.shape[:2] + (-1,)).any(axis=-1)


def _indirect(data, kept):
    """Indirect couplings between the n-shells kept (a boolean mask), through the higher n-shells
    of the bundle-nS model and the shells above the kept ones, which are condensed onto them.

    With P the projected couplings (loss positive), r the recombination, I the ionization and Q the
    parent coupling, shells u above the kept ones l are eliminated from P x = r (x the shell
    populations per unit parent population and electron density):
        P' = P_ll - P_lu P_uu^-1 P_ul      r' = r_l - P_lu P_uu^-1 r_u
        I' = I_l - (P_uu^-1 P_ul)^T I_u    Q' = Q + I_u^T P_uu^-1 r_u
    and the direct transfers from kept to eliminated shells become indirect. The direct couplings
    between kept shells, their direct ionization and their direct radiative, dielectronic,
    three-body and charge exchange recombination are then subtracted. Negative results are set to
    zero; 'unreliable' marks the grid points where one is negative beyond round-off (ROUND_OFF).
    """
    P, I, r, Q = data['crmat'], data['ion'], data['rhs'], data['parent_qc']
    D, DI = data['dir_crmat'], data['dir_ion']
    upper = ~kept
    l, u = np.where(kept)[0], np.where(upper)[0]
    P_ll = P[..., l[:, None], l]
    D_ll = D[..., l[:, None], l].copy()
    r_l, I_l = r[..., l], I[..., l, :]
    Q_upper = np.zeros_like(Q)
    if(len(u)):
        P_uu = P[..., u[:, None], u]
        P_ul, P_lu = P[..., u[:, None], l], P[..., l[:, None], u]
        X = np.linalg.solve(P_uu, P_ul)                       #P_uu^-1 P_ul
        y = np.linalg.solve(P_uu, r[..., u, None])[..., 0]    #P_uu^-1 r_u
        P_ll = P_ll - P_lu @ X
        r_l = r_l - np.einsum('...ab,...b->...a', P_lu, y)
        I_l = I_l - np.swapaxes(X, -1, -2) @ I[..., u, :]
        Q_upper = np.einsum('...ap,...a->...p', I[..., u, :], y)
        #direct transfers to the eliminated shells (negative) no longer count as direct losses
        D_ll[..., np.arange(len(l)), np.arange(len(l))] += D[..., u[:, None], l].sum(axis=-2)
    K = P_ll - D_ll
    K[..., np.arange(len(l)), np.arange(len(l))] -= DI[..., l, :].sum(axis=-1)
    direct_recombination = sum(data[key][..., l] for key in
                               ('recomb_rad', 'recomb_diel', 'recomb_three_body', 'recomb_cx'))
    off_diagonal = ~np.eye(len(l), dtype=bool)
    gains = -K
    gains[..., ~off_diagonal] = 0.
    ionization = I_l - DI[..., l, :]
    recombination = r_l - direct_recombination
    parent_coupling = Q + Q_upper
    unreliable = {'gains': _unreliable(gains[..., off_diagonal],
                                       np.abs(P_ll[..., off_diagonal]) + np.abs(D_ll[..., off_diagonal])),
                  'ionization': _unreliable(ionization, np.abs(I_l) + np.abs(DI[..., l, :])),
                  'recombination': _unreliable(recombination, np.abs(r_l) + np.abs(direct_recombination)),
                  'parent_coupling': _unreliable(parent_coupling, np.abs(Q) + np.abs(Q_upper))}
    return {'gains': np.clip(gains, 0., None),                          #[to, from], s-1
            'loss': K[..., np.arange(len(l)), np.arange(len(l))],      #s-1, for checks
            'ionization': np.clip(ionization, 0., None),                #[shell, parent], s-1
            'recombination': np.clip(recombination, 0., None),          #cm3 s-1
            'parent_coupling': np.clip(parent_coupling, 0., None),      #[to parent], cm3 s-1
            'unreliable': unreliable}                                   #(temperature, density)


def indirect_couplings(adf17, adf18):
    """The indirect couplings through the high n-shells for each parent and spin system of an
    adf18 a17_p208 file, from the condensed matrices of its adf17 file, on the temperatures and
    densities of the adf17 file.

    :param adf17: adf17 data, read_adf17(...)['input_file']
    :type adf17: dict

    :param adf18: adf18 a17_p208 data, read_adf18(...)
    :type adf18: dict

    :returns: list -- for each parent and spin system with levels in the adf18 file: 'parent'
              (index from 0), 'spin' (2S+1), 'shells' (the n-shells kept, those of its levels and
              the lower ones of the bundle), and on (temperature, density, ...): 'gains' (shell,
              shell) [to, from] and 'ionization' (shell, parent) in s-1, 'recombination' (shell)
              and 'parent_coupling' (parent) in cm3 s-1, and 'loss' (shell) in s-1, the diagonal
              of the indirect couplings, whose difference with the gains and ionization out of a
              shell shows how well the condensed data conserve particles; 'unreliable', for each
              of 'gains', 'ionization', 'recombination' and 'parent_coupling', the (temperature,
              density) points where it is negative beyond round-off (ROUND_OFF), where the
              condensed data are not reliable.
    """
    switches = adf17['switches']
    if(switches.get('IEFPRE', 0) == 1 or switches.get('IEFPRS', 0) == 1):
        raise NotImplementedError('the fractional parentage corrections of the lowest shell (IEFPRE or IEFPRS = 1, '
                                  'set by the cbnm93 adf17 files) are not implemented; the cbnm96 files do not use them')
    if(len(adf17['parents']) != adf18['num_parents']):
        raise ValueError('the adf17 file has ' + str(len(adf17['parents'])) + ' parents and the adf18 file ' +
                         str(adf18['num_parents']))
    groups = []
    for p, (parent, mapped) in enumerate(zip(adf17['parents'], adf18['parents'])):
        if(parent['term'].replace(' ', '').upper() != mapped['term'].replace(' ', '').upper()):
            warnings.warn('parent ' + str(p + 1) + ' is (' + parent['term'] + ') in the adf17 file and (' +
                          mapped['term'] + ') in the adf18 file')
        for system in parent['spin_systems']:
            spin = int(round(system['spin']))
            records = [r for r in adf18['records'] if r['parent'] == p + 1 and r['spin'] == spin]
            if(not records):
                continue
            highest = max(r['shell'] for r in records)
            kept = np.asarray(system['n']) <= highest
            if(not kept.any()):
                continue
            spin_weight = spin/(2.*parent['spin'])
            indirect = _indirect(_complete(system, switches, spin_weight), kept)
            indirect.update({'parent': p, 'spin': spin, 'shells': np.asarray(system['n'])[kept]})
            groups.append(indirect)
    return groups


def interpolate(values, temps, dens, user_temps, user_dens, pairs=False):
    """Values (temperature, density, ...) at the temperatures temps (K) and densities dens (cm-3)
    of an adf17 file, interpolated on the user temperatures (eV) and densities (cm-3): linearly in
    the logs of temperature, density and value, which are not negative (zeros stay effectively
    zero), and held at the values of the closest grid edge outside the grid. With pairs, the user
    temperatures and densities are pairs, else a grid.

    :returns: array (user temperature, user density, ...) or (pair, ...)
    """
    values = np.asarray(values, dtype=float)
    log_values = np.log(np.maximum(values, EFFECTIVE_ZERO))
    axes = (np.log(np.asarray(temps)/KELVIN_PER_EV), np.log(np.asarray(dens)))
    interpolator = RegularGridInterpolator(axes, log_values, method='linear')
    user_t = np.clip(np.log(np.asarray(user_temps, dtype=float)), axes[0][0], axes[0][-1])
    user_n = np.clip(np.log(np.asarray(user_dens, dtype=float)), axes[1][0], axes[1][-1])
    if(pairs):
        points = np.column_stack([user_t, user_n])
    else:
        points = np.stack(np.meshgrid(user_t, user_n, indexing='ij'), axis=-1)
    result = np.exp(interpolator(points))
    result[result <= EFFECTIVE_ZERO*1e10] = 0.
    return result


def _factor_below(user_temps, first, second, slope, energy, limits):
    """Factors that turn couplings held at their value at the first temperature first (eV) of their
    adf17 grid into their values at the user temperatures below it: (T/first)^a exp(-energy (1/T -
    1/first)), with energy the threshold (eV) of the coupling and a the slope d ln/d ln T of the
    reduced coupling, the coupling times exp(energy/T), between first and the second temperature
    second, clipped to limits. slope is the slope of the coupling itself on the user grid (user
    temperatures first); the factors are 1 at and above first."""
    t = np.asarray(user_temps, dtype=float).reshape((-1,) + (1,)*(np.ndim(slope) - 1))
    reduced = np.clip(np.nan_to_num(slope, nan=0.) + energy*(1/second - 1/first)/np.log(second/first), *limits)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        factor = (t/first)**reduced*np.exp(-energy*(1/t - 1/first))
    return np.where(t < first, factor, 1.)


#the couplings of a parent and spin system (indirect_couplings), with their names in the warnings
COUPLINGS = {'gains': 'transfers', 'ionization': 'ionization', 'recombination': 'recombination',
             'parent_coupling': 'parent coupling'}


def _on_user_grid(values, unreliable, temps, dens, user_temps, user_dens, pairs):
    """A coupling given on the adf17 grid (temperatures temps in K, densities dens in cm-3),
    interpolated on the user temperatures (eV) and densities without the temperatures where it is
    not reliable (unreliable, temperature x density), and held at the first temperature kept below
    it.

    :returns: the values on the user grid; (first, second, slope) for _factor_below, the two lowest
              temperatures kept (eV) and the slope of the log of the coupling between them on the
              user grid, or None when no user temperature is below the first; and the temperatures
              left out (eV) that the user temperatures need, those between their neighbours.
    """
    user_t = np.asarray(user_temps, dtype=float)
    t_eV = np.asarray(temps)/KELVIN_PER_EV
    kept = ~unreliable.any(axis=1)
    neighbours = np.concatenate([[-np.inf], t_eV, [np.inf]])
    needed = [t_eV[k] for k in np.where(~kept)[0] if np.any((user_t > neighbours[k]) & (user_t < neighbours[k + 2]))]
    if(kept.sum() < 2):
        grid = (len(user_t),) if pairs else (len(user_t), len(user_dens))
        return np.zeros(grid + values.shape[2:]), None, needed
    values, kept_temps = values[kept], np.asarray(temps)[kept]
    on_grid = interpolate(values, kept_temps, dens, user_t, user_dens, pairs)
    first, second = kept_temps[:2]/KELVIN_PER_EV
    if(not np.any(user_t < first)):
        return on_grid, None, needed
    #the values at the second temperature for each density (or pair), against those held at the first
    at_second = interpolate(values, kept_temps, dens, np.full(len(user_t) if pairs else 1, second), user_dens, pairs)
    with np.errstate(divide='ignore', invalid='ignore'):
        slope = np.log(at_second/on_grid)/np.log(second/first)
    return on_grid, (first, second, slope), needed


def _thresholds(records, shells, parent_energy, level_energies):
    """Thresholds (eV) of the paths through the high n-shells out of the levels of a parent and spin
    system (their adf18 records, its kept n-shells): their ionization energies to the parent, of
    energy parent_energy (cm-1), and the energies to excite them to the shell above those kept,
    whose binding energy is that of the highest kept shell scaled as 1/n^2.

    :returns: dicts {level index: ionization energy} and {level index: excitation energy}
    """
    ionization = {r['level'] - 1: max(parent_energy - level_energies[r['level'] - 1], 0.)*EV_PER_WAVENUMBER
                  for r in records}
    top = [r for r in records if r['shell'] == max(shells)]
    weights = sum(r['weight'] for r in top)
    upper = sum(r['weight']*ionization[r['level'] - 1] for r in top)/weights if weights > 0 else 0.
    upper *= (max(shells)/(max(shells) + 1.))**2
    return ionization, {i: max(energy - upper, 0.) for i, energy in ionization.items()}


def project(adf17, adf18, level_energies, ion_pot, user_temps, user_dens, pairs=False):
    """The indirect couplings through the high n-shells spread over the levels of an adf04 file, on
    the user temperatures (eV) and densities (cm-3). The temperatures where the condensed data of a
    coupling are not reliable (indirect_couplings) are left out of its interpolation, with a warning
    when the user temperatures need them. Below the first temperature of the adf17 grid, the
    couplings follow power laws through their two lowest temperatures times their Boltzmann factors
    exp(-dE/kT), each with the threshold of its path through the high n-shells: the ionization of
    a level its ionization energy; the transfers out of a level the energy to reach the shell above
    those kept (from the binding energy of the highest kept shell, scaled as 1/n^2); the parent
    coupling the energy between the two parents; recombination none. Above the grid they keep the
    values at its edge.

    :param adf17: adf17 data, read_adf17(...)['input_file']
    :type adf17: dict

    :param adf18: adf18 a17_p208 data, read_adf18(...)
    :type adf18: dict

    :param level_energies: energies of the levels of the adf04 file (cm-1)
    :type level_energies: float array

    :param ion_pot: ionization potentials of the parents of the adf04 file (cm-1, its first line)
    :type ion_pot: float array

    :param user_temps: temperatures (eV)
    :type user_temps: float array

    :param user_dens: densities (cm-3)
    :type user_dens: float array

    :param pairs: the temperatures and densities are pairs instead of a grid
    :type pairs: bool

    :returns: dict -- with the user temperature and density axes last (one pair axis with pairs):
              'couplings' (level, level) [to, from], transfer rates between levels through the
              high n-shells in s-1, zero on the diagonal; 'ionization' (level, parent) and
              'recombination' (level, parent) rate coefficients in cm3 s-1; 'parent_coupling'
              (parent, parent) [to, from] in cm3 s-1, to add to the parent cross coupling (XCD);
              'groups', the indirect couplings of each parent and spin system on the adf17 grid
              (indirect_couplings).
    """
    level_energies, ion_pot = np.asarray(level_energies, dtype=float), np.asarray(ion_pot, dtype=float)
    num_levels, num_parents = len(level_energies), len(ion_pot)
    if(adf18['num_parents'] != num_parents):
        raise ValueError('the adf18 file has ' + str(adf18['num_parents']) + ' parents and the adf04 file ' +
                         str(num_parents))
    if(adf18['num_levels'] and adf18['num_levels'] != num_levels):
        warnings.warn('the adf18 file maps ' + str(adf18['num_levels']) + ' levels and the adf04 file has ' +
                      str(num_levels))
    temps, dens = adf17['temp_grid'], adf17['dens_grid']
    t_eV = np.asarray(temps)/KELVIN_PER_EV
    outside = ((np.min(user_temps) < t_eV[0]) or (np.max(user_temps) > t_eV[-1]) or
               (np.min(user_dens) < dens[0]) or (np.max(user_dens) > dens[-1]))
    if(outside):
        warnings.warn('the user temperatures or densities go beyond those of the adf17 file (' +
                      format(t_eV[0], '.3g') + '-' + format(t_eV[-1], '.3g') + ' eV, ' + format(dens[0], '.3g') +
                      '-' + format(dens[-1], '.3g') + ' cm-3); below its temperatures the projection extrapolates '
                      'the couplings with power laws and Boltzmann factors, elsewhere it takes the values at its edges')
    user_t = np.asarray(user_temps, dtype=float)
    grid = (len(user_temps),) if pairs else (len(user_temps), len(user_dens))
    density = np.asarray(user_dens, dtype=float) if pairs else \
              np.broadcast_to(np.asarray(user_dens, dtype=float), grid)
    couplings = np.zeros(grid + (num_levels, num_levels))
    ionization = np.zeros(grid + (num_levels, num_parents))
    recombination = np.zeros(grid + (num_levels, num_parents))
    parent_coupling = np.zeros(grid + (num_parents, num_parents))
    groups = indirect_couplings(adf17, adf18)
    for group in groups:
        p = group['parent']
        on_grid, below, left_out = {}, {}, {}
        for key in COUPLINGS:
            on_grid[key], below[key], needed = _on_user_grid(group[key], group['unreliable'][key], temps, dens,
                                                             user_t, user_dens, pairs)
            if(needed):
                left_out[key] = needed
        if(left_out):
            warnings.warn('the condensed data of parent ' + str(p + 1) + ' (' + adf17['parents'][p]['term'] +
                          ') and spin system ' + str(group['spin']) + ' of the adf17 file give negative indirect ' +
                          ', '.join(COUPLINGS[key] for key in left_out) + ' beyond round-off at ' +
                          ', '.join(format(t, '.3g') for t in sorted(set(sum(left_out.values(), [])))) + ' eV; the '
                          'projection leaves those temperatures out and uses the neighbouring ones')

        def factor(key, index, energy):
            """Factor of on_grid[key][index] below its first temperature (1 if no user temperature is)."""
            if(below[key] is None):
                return 1.
            first, second, slope = below[key]
            limits = RECOMBINATION_EXPONENTS if key in ('recombination', 'parent_coupling') else REDUCED_EXPONENTS
            return _factor_below(user_t, first, second, slope[index], energy, limits)

        shell_index = {n: k for k, n in enumerate(group['shells'])}
        records = [r for r in adf18['records'] if r['parent'] == p + 1 and r['spin'] == group['spin'] and
                   r['shell'] in shell_index and 1 <= r['level'] <= num_levels]
        ionization_energy, excitation_energy = _thresholds(records, group['shells'], ion_pot[p], level_energies)
        for record in records:
            i, a = record['level'] - 1, shell_index[record['shell']]
            recombination[..., i, p] += record['weight']*on_grid['recombination'][..., a]* \
                                        factor('recombination', (..., a), 0.)
            #ionization to the parent of the group only: through the n-shells of one parent to
            #another (autoionization of core-excited Rydberg states) is excitation-autoionization,
            #which the ionization rates of adf04 files (adf07) already have
            ionization[..., i, p] += on_grid['ionization'][..., a, p]/density* \
                                     factor('ionization', (..., a, p), ionization_energy[i])
            for other in records:
                b, j = shell_index[other['shell']], other['level'] - 1
                if(b != a):
                    #every path out of level j through the high n-shells starts with its excitation to them
                    couplings[..., i, j] += record['weight']*on_grid['gains'][..., a, b]* \
                                            factor('gains', (..., a, b), excitation_energy[j])
        coupling = on_grid['parent_coupling'][..., :num_parents]* \
                   factor('parent_coupling', (..., slice(0, num_parents)),
                          np.maximum(ion_pot - ion_pot[p], 0.)*EV_PER_WAVENUMBER)
        coupling[..., p] = 0.
        parent_coupling[..., :, p] += coupling
    levels = np.arange(num_levels)
    couplings[..., levels, levels] = 0.
    last = lambda a: np.moveaxis(a, tuple(range(len(grid))), tuple(range(-len(grid), 0)))
    return {'couplings': last(couplings), 'ionization': last(ionization),
            'recombination': last(recombination), 'parent_coupling': last(parent_coupling), 'groups': groups}

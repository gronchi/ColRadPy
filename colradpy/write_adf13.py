import numpy as np
import warnings

from colradpy.write_adf15 import (VALUES_PER_LINE, _filmem, _ion, _wavelength_field, _write_block_table,
                                  _write_information, _write_values)


#Smallest S/XB written, as the smallest PEC of adf15 files: values are e9.2 fields, and ADAS writes
#S/XB that is not defined (no excitation emission) as 1.00E-74
MIN_SXB = 1.e-74
#Largest S/XB an e9.2 field holds, with a two-digit exponent
MAX_SXB = 9.99e99
#The ADAS reader of adf13 files (xxdata_13) keeps three characters of ISEL
MAX_BLOCKS = 999
#CODE of the block headers, the code that made the coefficients (ADAS writes ADAS208)
CODE = 'ColRadPy'


def write_adf13(fil_name, sxb_inds, wave, sxb, element_sym, charge_state, dens, temp, metas,
                user='', atomic='', num=8, total=False, sxb_levels=None):
    """Write S/XB coefficients, the ionizations per photon of lines, to an adf13 file (ADAS
    appxa-13).

    Units: wave in [A], S/XB dimensionless, dens in [cm-3] and temp in [eV].

    One block is written for each selected line and metastable, ordered by metastable.
    The block headers give the source file (FILMEM), CODE = ColRadPy, the metastable
    (INDM = its index starting at 1, or T for populations normalized to the whole ion with
    one metastable) and ISEL, the block index from 1. S/XB below MIN_SXB, including zeros,
    is written as MIN_SXB, and above MAX_SXB as MAX_SXB, with a warning.

    :param fil_name: name of the adf13 file
    :type fil_name: str

    :param sxb_inds: indices of the lines to write
    :type sxb_inds: int arr

    :param wave: wavelengths in A
    :type wave: float arr

    :param sxb: S/XB with shape (lines, metastables, temperatures, densities)
    :type sxb: float arr

    :param element_sym: element symbol
    :type element_sym: str

    :param charge_state: charge of the ion
    :type charge_state: int

    :param dens: electron densities in cm-3
    :type dens: float arr

    :param temp: electron temperatures in eV
    :type temp: float arr

    :param metas: metastable level indices
    :type metas: int arr

    :param user: the 'user' dictionary of colradpy, written in the comments
    :type user: dict

    :param atomic: the 'atomic' dictionary of colradpy, for the nuclear charge and the levels in the comments
    :type atomic: dict

    :param num: number of values on each line of the file
    :type num: int

    :param total: the S/XB of the populations normalized to the whole ion, with one metastable
                  (INDM = T)
    :type total: bool

    :param sxb_levels: upper and lower level of each line of sxb (all lines, as sxb), to list the
                       blocks with their levels in the comments
    :type sxb_levels: int arr

    """

    sxb_inds = np.atleast_1d(sxb_inds)
    filmem = _filmem(user)
    if(num != VALUES_PER_LINE):
        warnings.warn('ADAS reads ' + str(VALUES_PER_LINE) + ' values on each line of adf13 files (8d9.2), '
                      'its readers will not read the ' + str(num) + ' written on each line')
    sxb = np.array(sxb, dtype=float)[sxb_inds]#only the lines requested, a copy
    wave = np.asarray(wave, dtype=float)[sxb_inds]
    if(sxb_levels is not None):
        sxb_levels = np.asarray(sxb_levels)[sxb_inds]

    if(sxb.ndim != 4):
        raise ValueError('adf13 files need S/XB on a temperature-density grid, '
                         'shape (lines, metastables, temperatures, densities)')
    num_metas = np.shape(sxb)[1]
    if(num_metas != len(metas)):
        raise ValueError('sxb has ' + str(num_metas) + ' metastable columns, not the ' + str(len(metas)) +
                         ' metastables')
    if(total and num_metas != 1):
        raise ValueError('INDM = T (total) is for populations normalized to the whole ion with one metastable, '
                         'not ' + str(num_metas))
    num_blocks = len(wave)*num_metas
    if(num_blocks > MAX_BLOCKS):
        warnings.warn(str(num_blocks) + ' blocks are written, but the ADAS reader of adf13 files keeps three '
                      'digits of ISEL; select at most ' + str(MAX_BLOCKS) + ' lines and metastables')
    wave_fields = [_wavelength_field(w) for w in wave]
    if(not np.isfinite(sxb).all()):
        raise ValueError('S/XB that is not finite cannot be written to an adf13 file')
    if((sxb < 0).any()):
        warnings.warn('negative S/XB is written as ' + str(MIN_SXB) + ' in the adf13 file')
    if((sxb > MAX_SXB).any()):
        warnings.warn('S/XB above ' + str(MAX_SXB) + ', lines with almost no excitation emission, is written as ' +
                      str(MAX_SXB) + ' in the adf13 file')
    sxb = np.clip(sxb, MIN_SXB, MAX_SXB)

    with open(fil_name, 'w') as f:
        #first line: the number of blocks and, in 35 characters between slashes, the ion
        f.write('{:5d}'.format(num_blocks) + '    /' +
                '{:<35s}'.format(_ion(element_sym, charge_state) + ' IONISATIONS PER PHOTON') + '/\n')

        isel = 0
        blocks = []#ISEL, wavelength, type, INDM and line of each block, for the comments
        for j in range(num_metas):
            indm = 'T' if total else str(j + 1)
            for i in range(len(wave)):
                isel = isel + 1
                blocks.append((isel, wave_fields[i], 'EXCIT', indm, i))
                f.write(wave_fields[i] + ' A' + '{:4d}{:4d}'.format(len(dens), len(temp)) +
                        ' /FILMEM = ' + '{:<8s}'.format(filmem) + '/CODE = ' + '{:<8s}'.format(CODE) +
                        ' /INDM =' + '{:>2s}'.format(indm) + '/ISEL =' + '{:5d}'.format(isel) + '\n')
                _write_values(f, dens, num)
                _write_values(f, temp, num)
                #all the temperatures for each density
                for d in range(len(dens)):
                    _write_values(f, sxb[i, j, :, d], num)

        _write_information(f, 'IONISATIONS PER PHOTON:', charge_state, user, atomic)
        f.write('C\n')
        f.write('C  S/XB = (SCD of the metastable to all parents) / (excitation PEC of the line from it)\n')
        if(sxb_levels is not None):
            _write_block_table(f, blocks, sxb_levels)

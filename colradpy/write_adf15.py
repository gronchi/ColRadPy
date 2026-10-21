import numpy as np
import itertools
import re
import warnings


#Smallest PEC written. Values are e9.2 fields, which cannot hold three-digit
#exponents, and ADAS files write negligible PECs as 1.00E-74.
MIN_PEC = 1.e-74
#Wavelengths in A must fit the f9 field of the block header.
MAX_WAVELENGTH = 1.e8
#ADAS reads the values as 8d9.2, eight on each line
VALUES_PER_LINE = 8
#FILMEM of the block headers: ADAS keeps 8 characters, and its reader (xxdata_15) takes blocks
#with a blank FILMEM for superstage blocks and stops
FILMEM_LENGTH = 8


# group subset of an array
# this is so there are 8 entries per line
# in the adf15 file
# see stackexchange
def grouper(n, iterable):
    it = iter(iterable)
    while True:
       chunk = tuple(itertools.islice(it, n))
       if not chunk:
           return
       yield chunk


def _wavelength_field(wave):
    """Wavelength in A as an f9 field, with as many decimals as fit."""
    if(not np.isfinite(wave) or not 0 < wave < MAX_WAVELENGTH):
        raise ValueError('wavelength ' + str(wave) + ' A cannot be written to an adf15 file')
    #a leading space and up to 8 characters, or 9 when rounding adds a digit
    decimals = max(0, 7 - len(str(int(wave))))
    return '{:9.{}f}'.format(wave, decimals)


def _write_values(f, values, num):
    """Write values as e9.2 fields, 'num' entries per line."""
    for chunk in grouper(num, values):
        f.write(''.join(map('{:9.2E}'.format, chunk)) + '\n')


def _filmem(user):
    """The FILMEM of the block headers: the name of the adf04 file, without its extension."""
    name = ''
    if(type(user) == dict and user.get('file_loc')):
        name = re.split(r'[\\/]', str(user['file_loc']))[-1].rsplit('.', 1)[0]
        name = re.sub(r'\s', '', name)
    return (name or 'colradpy')[:FILMEM_LENGTH]


def _comment(text):
    """One comment line, with any line breaks in the text removed."""
    return 'C  ' + ' '.join(str(text).split()) + '\n'


def write_adf15(fil_name,pec_inds, wave, pecs, element_sym,
                charge_state, dens, temp, metas, ion_pot, user='', atomic='', num=8,
                num_recomb=None):
    """Write all or a subset of PECs to an adf15 formatted file.

    Standard adf15 units: wave in [A], PEC in [ph cm3 s-1], dens in [cm-3] and
    temp in [eV].

    One block is written for each selected line and driving population, ordered by
    driver. The driver columns of pecs are the metastables (TYPE = EXCIT, INDM = the
    metastable index starting at 1), then the recombining parents (TYPE = RECOM,
    INDM = +parent index) and then the charge exchange parents (TYPE = CHEXC).
    ISEL numbers the blocks from 1. PECs below MIN_PEC, including zeros, are written
    as MIN_PEC.

    :param fil_name: name of the adf15 file
    :type fil_name: str

    :param pec_inds: indices of the lines to write
    :type pec_inds: int arr

    :param wave: wavelengths in A
    :type wave: float arr

    :param pecs: PECs with shape (lines, drivers, temperatures, densities)
    :type pecs: float arr

    :param element_sym: element symbol
    :type element_sym: str

    :param charge_state: charge of the ion
    :type charge_state: int

    :param dens: electron densities in cm-3
    :type dens: float arr

    :param temp: electron temperatures in eV
    :type temp: float arr

    :param metas: metastable level indices
    :type metas: float_arr

    :param ion_pot: ionization potentials, one for each recombining parent
    :type ion_pot: float arr

    :param user: the 'user' dictionary of colradpy, written in the comments
    :type user: dict

    :param atomic: the 'atomic' dictionary of colradpy, for the nuclear charge and the levels in the comments
    :type atomic: dict

    :param num: number of values on each line of the file
    :type num: int

    :param num_recomb: number of recombination driver columns in pecs, len(ion_pot) by default
    :type num_recomb: int

    """

    pec_inds = np.atleast_1d(pec_inds)
    filmem = _filmem(user)
    if(num != VALUES_PER_LINE):
        warnings.warn('ADAS reads ' + str(VALUES_PER_LINE) + ' values on each line of adf15 files (8d9.2), '
                      'its readers will not read the ' + str(num) + ' written on each line')
    pecs = np.array(pecs, dtype=float)[pec_inds]#only take PECs requested, a copy
    wave = np.asarray(wave, dtype=float)[pec_inds]#only take wavelengths requested

    if(pecs.ndim != 4):
        raise ValueError('adf15 files need PECs on a temperature-density grid, '
                         'shape (lines, drivers, temperatures, densities)')
    if(num_recomb is None):
        num_recomb = len(ion_pot)
    num_drivers = np.shape(pecs)[1]
    if(num_drivers < len(metas) + num_recomb):
        raise ValueError('pecs has ' + str(num_drivers) + ' driver columns, fewer than the ' +
                         str(len(metas)) + ' metastables and ' + str(num_recomb) + ' recombining parents')
    wave_fields = [_wavelength_field(w) for w in wave]
    if(not np.isfinite(pecs).all()):
        raise ValueError('PECs that are not finite cannot be written to an adf15 file')
    if((pecs < 0).any()):
        warnings.warn('negative PECs are written as ' + str(MIN_PEC) + ' in the adf15 file')
    pecs[pecs < MIN_PEC] = MIN_PEC

    with open(fil_name,'w') as f:

        #first line in the adf15 file format, the number of blocks
        #the ion as ADAS writes it, a2,'+',i2 (C + 2, AR+16): the ADAS reader needs the + (or
        #a :) and stops otherwise
        f.write('{:5d}'.format(len(wave)*num_drivers) + '    /' +
                '{:<2s}+{:2d}'.format(str(element_sym).strip(), int(charge_state)) +
                ' PHOTON EMISSIVITY COEFFICIENTS/\n')

        isel = 0
        for j in range(0,num_drivers):#loop over driving populations
            #letting the user know if the PEC is excitation, recombination or charge exchange
            if(j <len(metas)):
                typ = 'EXCIT'
                indm = str(j + 1)
            elif(j < len(metas) + num_recomb):
                typ = 'RECOM'
                indm = '+' + str(j - len(metas) + 1)
            else:
                typ = 'CHEXC'
                indm = '+' + str(j - len(metas) - num_recomb + 1)

            for i in range(0, len(wave)):#loop over PECs
                isel = isel + 1
                #header information for the PEC, important things here are the wavelength
                # number of temps and dens and metastable level
                f.write(wave_fields[i] + ' A' + '{:4d}{:4d}'.format(len(dens), len(temp)) +
                        ' /FILMEM = ' + '{:<8s}'.format(filmem) + '/TYPE = ' + '{:<8s}'.format(typ) +
                        ' /INDM =' + '{:>2s}'.format(indm) + '/ISEL =' + '{:5d}'.format(isel) + '\n')

                #write the density grid to the file 'num' entries per line
                _write_values(f, dens, num)
                #write the temperature grid to the file 'num' entries per line
                _write_values(f, temp, num)
                #write the PEC grid to the file 'num' entries per line
                #all the temperatures for each density
                for d in range(0,len(dens)):
                    _write_values(f, pecs[i,j,:,d], num)

        #write out calculation details
        f.write('C-----------------------------------------------------------------------\n')
        f.write('C\n')
        f.write('C  PHOTON EMISSIVITY COEFFICIENTS:\n')
        f.write('C\n')
        f.write('C\n')
        f.write('C  INFORMATION\n')
        f.write('C  -----------\n')
        f.write('C\n')
        f.write('C   ***MADE WITH ColRadPy***\n')
        if(type(atomic) == dict and 'iz0' in atomic):
            f.write('C  NUCLEAR CHARGE =  ' + str(atomic['iz0']) + '\n')
        f.write('C  ION CHARGE +1  =  ' + str(charge_state + 1) + '\n')
        f.write('C\n')
        if(type(user) == dict and 'file_loc' in user):
            f.write(_comment('SPECIFIC ION FILE  : ' + str(user['file_loc'])))
        f.write('C\n')
        f.write('C\n')
        if(type(user) == dict):
            for i in user.keys():
                f.write(_comment(i + ' ' + str(user[i])))

        if(type(atomic) == dict):
            f.write('C  ' + 'lvl' +'  '+ 'config' +
                        '  ' + 'S'+
                        '  ' + 'L'+
                        '  ' + 'w'+
                        '  ' + 'eng\n')

            f.write('C-----------------------------------------------------------------------\n')
            f.write('C\n')
            for ii in range(0,len(atomic['config'])):

                f.write(_comment(str(ii) +'  '+ str(atomic['config'][ii])+\
                        '  ' + str(atomic['S'][ii]) + \
                        '  ' + str(atomic['L'][ii]) + \
                        '  ' + str(atomic['w'][ii]) + \
                        '  ' + str(atomic['energy'][ii])))

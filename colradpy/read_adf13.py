import numpy as np
import re


#a number written by ADAS (8e9.2): 1.00D+11, 1.00E+11, or 1.41+100 when the exponent has three digits
NUMBER = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+)(?:[eEdD][-+]?\d+|[-+]\d+)?')
#the line of a block: WLNG (f9, Angstrom) 'A', NDENS, NTE (2i4), then '/' delimited KEY = value
BLOCK = re.compile(r'^\s*(\d+\.?\d*|\.\d+)\s*A?\s+(\d+)\s+(\d+)\s*/(.*)$')
#the element and charge at the start of the text of the first line: BE+0, H + 0, HE+ 1
ION = re.compile(r'^\s*([A-Za-z]{1,2})\s*\+\s*(\d+)')


def _float(text):
    """A Fortran number: D exponents, and three-digit exponents written without the letter."""
    text = text.replace('D', 'e').replace('d', 'e')
    if('e' not in text and 'E' not in text):
        text = re.sub(r'(?<=[\d.])([-+])', r'e\1', text)
    return float(text)


def read_adf13(fil):
    """Reads an adf13 file of ionisations per photon (S/XB, ADAS appxa-13).

    :param fil: path of the adf13 file
    :type fil: str

    :returns: dict -- adf13['input_file'] holds 'num_trans', 'element', 'charge_state' and, for
              each block (transition), 'wave_air' the wavelength (A), 'orig_file' the source
              file FILMEM, 'meta_indx' the metastable INDM (a string: an index, or 'T' for the
              total over metastables of unresolved files) and 'isel', with 'dens_grid' (cm-3),
              'temp_grid' (eV), 'num_dens', 'num_temp' and 'sxb' (blocks, temperatures, densities).
              Blocks can have their own grids (appxa-13): then 'dens_grid', 'temp_grid',
              'num_dens' and 'num_temp' are lists with one entry for each block and 'sxb' a list
              of (temperatures, densities) arrays.
    """
    with open(fil) as f:
        lines = [line.rstrip('\n') for line in f]

    adf13 = {'input_file': {}}
    data = adf13['input_file']
    first = lines[0].split('/') if lines else ['']
    try:
        data['num_trans'] = int(first[0])
    except ValueError:
        raise ValueError(str(fil) + ' is not an adf13 file: its first line does not start with the '
                         'number of transitions') from None
    ion = ION.match(first[1]) if len(first) > 1 else None
    data['element'] = ion.group(1) if ion else ''
    data['charge_state'] = int(ion.group(2)) if ion else None

    blocks = []
    k = 1
    for block in range(data['num_trans']):
        header = BLOCK.match(lines[k]) if k < len(lines) else None
        if(header is None):
            raise ValueError(str(fil) + ': expected the line of transition ' + str(block + 1) +
                             ' (wavelength A, number of densities and temperatures, /FILMEM = ...), found ' +
                             (repr(lines[k][:80]) if k < len(lines) else 'the end of the file'))
        num_dens, num_temp = int(header.group(2)), int(header.group(3))
        keys = {key.upper(): value.strip() for key, value in re.findall(r'(\w+)\s*=\s*([^/]*)', header.group(4))}
        k += 1
        size = num_dens + num_temp + num_dens*num_temp
        values = []
        while(len(values) < size and k < len(lines) and not BLOCK.match(lines[k]) and
              not lines[k].startswith('C')):
            values += [_float(x) for x in NUMBER.findall(lines[k])]
            k += 1
        if(len(values) != size):
            raise ValueError(str(fil) + ': transition ' + str(block + 1) + ' has ' + str(len(values)) +
                             ' numbers, not ' + str(size) + ' (' + str(num_dens) + ' densities, ' +
                             str(num_temp) + ' temperatures and their S/XB)')
        blocks.append({'wave_air': float(header.group(1)), 'orig_file': keys.get('FILMEM', ''),
                       'meta_indx': keys.get('INDM', ''), 'isel': keys.get('ISEL', ''),
                       'dens_grid': np.array(values[:num_dens]),
                       'temp_grid': np.array(values[num_dens:num_dens + num_temp]),
                       #SXB(IN,IT) is written for each density: (temperatures, densities)
                       'sxb': np.array(values[num_dens + num_temp:]).reshape(num_dens, num_temp).T})

    for key in ('wave_air', 'orig_file', 'meta_indx', 'isel'):
        data[key] = np.array([b[key] for b in blocks])
    same_grids = all(np.array_equal(b['dens_grid'], blocks[0]['dens_grid']) and
                     np.array_equal(b['temp_grid'], blocks[0]['temp_grid']) for b in blocks)
    if(same_grids and blocks):
        data['dens_grid'] = blocks[0]['dens_grid']
        data['temp_grid'] = blocks[0]['temp_grid']
        data['num_dens'] = len(data['dens_grid'])
        data['num_temp'] = len(data['temp_grid'])
        data['sxb'] = np.array([b['sxb'] for b in blocks])
    else:
        data['dens_grid'] = [b['dens_grid'] for b in blocks]
        data['temp_grid'] = [b['temp_grid'] for b in blocks]
        data['num_dens'] = [len(b['dens_grid']) for b in blocks]
        data['num_temp'] = [len(b['temp_grid']) for b in blocks]
        data['sxb'] = [b['sxb'] for b in blocks]
    return adf13

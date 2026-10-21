import os
import numpy as np
import re


#ADF11 data classes, file names start with them (e.g. scd96_c.dat, acd96r_c.dat)
ADF11_CLASSES = ('acd', 'scd', 'ccd', 'prb', 'prc', 'qcd', 'xcd', 'plt', 'pls', 'met', 'ecd', 'ycd', 'zcd')
#a fixed point number such as -12.34567 or, without the leading zero, -.69877 (ADAS resolved
#files); numbers can follow each other without spaces
NUMBER = r'[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?'
#the labelled indices of a block line, '/ IPRT= 2  / IGRD= 1  /--------/ Z1= 1   / DATE= 16/11/93'
LABEL = re.compile(r'(IPRT|IGRD|JPRT|JGRD|Z1)\s*=\s*(\d+)')
#the line wavelength (A) of pls blocks, '/ IGRD= 1  / IPRT= 0  / 1638.59/ Z1= 1'
WAVELENGTH = re.compile(r'/\s*(\d+\.\d*)\s*/')


def _block_index(cls, block, metas, pairs):
    """Stage, array shape and (row, column) of a block of the class cls. The stage is Z1 - 1 (Z1
    is the charge of the recombining or ionised ion). Missing labels (unresolved files) are 1.
    acd, scd, ccd: (metastable IGRD of the stage, parent IPRT); qcd: IGRD to JGRD of the stage;
    xcd: parent IPRT to JPRT; plt, pls: metastable IGRD of the stage (IPRT = 0); prb, prc: parent
    IPRT (IGRD = 0, 96r files) or, when the file gives grounds too (93r), (IGRD, IPRT)."""
    stage = block['z1'] - 1
    get = lambda key: block.get(key, 1)
    if(cls in ('acd', 'scd', 'ccd') or (cls in ('prb', 'prc') and pairs)):
        return stage, (metas[stage], metas[stage+1]), (get('igrd') - 1, get('iprt') - 1)
    if(cls == 'qcd'):
        return stage, (metas[stage], metas[stage]), (get('igrd') - 1, get('jgrd') - 1)
    if(cls == 'xcd'):
        return stage, (metas[stage+1], metas[stage+1]), (get('iprt') - 1, get('jprt') - 1)
    if(cls in ('plt', 'pls')):
        return stage, (metas[stage], 1), (get('igrd') - 1, 0)
    if(cls in ('prb', 'prc')):
        return stage, (metas[stage+1], 1), (get('iprt') - 1, 0)
    return None


def adf11_class(fil):
    """Class (e.g. 'scd') and resolution of an adf11 file from its name, e.g. acd96r_c.dat
       is a resolved acd file. Only the file name is used, not the directories.

    :param fil: The file path
    :type fil: string

    :returns: (str or None, bool) --class and whether the file is metastable resolved

    """
    prefix = re.split('_', os.path.split(str(fil))[-1])[0].lower()
    match = re.match(r'(' + '|'.join(ADF11_CLASSES) + r')(\d*)(r?)', prefix)
    if(match):
        return match.group(1), match.group(3) == 'r'
    #names that do not follow the convention: the first class in the name
    for cls in ADF11_CLASSES:
        if(cls in os.path.split(str(fil))[-1].lower()):
            return cls, False
    return None, False


def read_adf11(fil):
    """ Reads data formatted in adf11 format (ADAS appxa-11). This format is generally used for
    GCR coefficients. The data is stored in arrays of log_10 (the ecd, ycd and zcd classes give
    their values as they are).
    Creates a dictionary to hold the data

    :param fil: The file path to the input file
    :type fil: string

    :returns: dict -- adf11['input_file'] holds 'class', 'resolved', 'nuc_charge', 'num_dens',
              'num_temp', 'charge_min', 'charge_max', 'metas' (metastables of each stage),
              'dens_grid', 'temp_grid' and, for each stage i (charge i), str(i): an array
              (rows, columns, temperatures, densities) placed by the labels of the blocks
              (-inf where the file gives no block): acd, scd and ccd (metastable of stage i,
              parent of stage i+1); qcd (metastable to metastable of stage i); xcd (parent to
              parent of stage i+1); plt and pls (metastable of stage i, 1); prb and prc (parent
              of stage i+1, 1), or (metastable, parent) when the file gives both. 'blocks' lists
              every block as given: its labels 'z1', 'iprt', 'igrd', 'jprt', 'jgrd' when present,
              'wavelength' (A, pls), 'date' and 'data' (temperatures, densities); the classes
              ecd, ycd and zcd are only given there.
    """
    cls, resolved = adf11_class(fil)

    with open(fil) as f:
        lines = [line.rstrip('\n') for line in f]
    adf11 = {}
    adf11['input_file'] = {}
    data = adf11['input_file']
    data['class'] = cls

    #reading the first line: IZMAX, IDMAXD, ITMAXD, IZ1MIN, IZ1MAX
    tmp = re.findall(r'(\d+)', lines[0]) if lines else []
    if(len(tmp) < 5):
        raise ValueError(str(fil) + ' is not an adf11 file: its first line does not give the nuclear charge, '
                         'the numbers of densities and temperatures and the lowest and highest charges')
    data['nuc_charge'] = int(tmp[0])
    data['num_dens'] = int(tmp[1])
    data['num_temp'] = int(tmp[2])
    data['charge_min'] = int(tmp[3])
    data['charge_max'] = int(tmp[4])

    #after a line of dashes, resolved files give the number of metastables of each stage
    #(integers, then another line of dashes) before the grids
    k = 2
    if(k < len(lines) and lines[k].strip() and '.' not in lines[k]):
        metas = []
        while(len(metas) < data['nuc_charge'] + 1 and k < len(lines) and '.' not in lines[k] and
              set(lines[k].strip()) != {'-'}):
            metas += [int(x) for x in re.findall(r'\d+', lines[k])]
            k += 1
        data['metas'] = np.array(metas)
        data['resolved'] = True
        with_metas = True
        if(k < len(lines) and set(lines[k].strip()) == {'-'}):
            k += 1
    else:
        data['metas'] = np.ones(data['charge_max']+1, dtype='int')
        data['resolved'] = resolved
        with_metas = False

    #the density and temperature grids
    for key, size in (('dens_grid', data['num_dens']), ('temp_grid', data['num_temp'])):
        values = []
        while(len(values) < size and k < len(lines)):
            values += [float(x) for x in re.findall(NUMBER, lines[k])]
            k += 1
        if(len(values) != size):
            raise ValueError(str(fil) + ': the ' + key + ' has ' + str(len(values)) + ' values, not ' + str(size))
        data[key] = np.array(values)
    num_values = data['num_temp']*data['num_dens']

    #the blocks, each a line with its labels and the values of each temperature for each
    #density, up to the comments (C-----, or a line of dashes only in some files)
    blocks = []
    for line in lines[k:]:
        if(len(line.strip()) == 0 or line.lstrip().startswith('C') or set(line.strip()) == {'-'}):
            break
        if('----------' in line):
            labels = {key.lower(): int(value) for key, value in LABEL.findall(line)}
            if('z1' not in labels):
                raise ValueError(str(fil) + ': the block line ' + repr(line.strip()) + ' does not give Z1')
            wavelength = WAVELENGTH.search(line)
            if(wavelength):
                labels['wavelength'] = float(wavelength.group(1))
            date = re.search(r'DATE\s*=\s*(\S+)', line)
            labels['date'] = date.group(1) if date else ''
            labels['data'] = []
            blocks.append(labels)
        elif(blocks):
            blocks[-1]['data'] += [float(x) for x in re.findall(NUMBER, line)]
        else:
            raise ValueError(str(fil) + ': expected a block line after the grids, found ' + repr(line.strip()[:60]))
    for block in blocks:
        if(len(block['data']) != num_values):
            raise ValueError(str(fil) + ': the block of Z1 = ' + str(block['z1']) + ' has ' + str(len(block['data'])) +
                             ' values, not ' + str(num_values) + ' (temperatures x densities)')
        block['data'] = np.array(block['data']).reshape(data['num_temp'], data['num_dens'])
    data['blocks'] = blocks

    #the GCR stages; blocks that the file does not give (e.g. the diagonal of QCD and XCD) are
    #zero coefficients, log10 = -inf. Files without metastables have one block for each stage,
    #whatever their labels say (IPRT= 1 / IGRD= 0 in plt12_h, IGRD= 1 / IPRT= 0 in plt89).
    metas = data['metas']
    labels = lambda block: block if with_metas else {'z1': block['z1']}
    pairs = any(labels(block).get('igrd', 1) > 0 for block in blocks)
    if(_block_index(cls, {'z1': 1}, metas, pairs) is None):
        return adf11
    for i in range(0, len(metas)-1):
        shape = _block_index(cls, {'z1': i + 1}, metas, pairs)[1]
        data[str(i)] = np.full(shape + (data['num_temp'], data['num_dens']), -np.inf)
    for block in blocks:
        stage, shape, (row, column) = _block_index(cls, labels(block), metas, pairs)
        if(not (0 <= stage < len(metas)-1 and 0 <= row < shape[0] and 0 <= column < shape[1])):
            raise ValueError(str(fil) + ': the block ' + str({key: value for key, value in block.items()
                                                               if key in ('z1', 'iprt', 'igrd', 'jprt', 'jgrd')}) +
                             ' is outside the metastables of the file ' + str(list(metas)))
        data[str(stage)][row, column] = block['data']
    return adf11

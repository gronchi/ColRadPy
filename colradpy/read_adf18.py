import re

import numpy as np


#key = value pairs of the &SEQINF namelist that opens the file
NAMELIST_ITEM = re.compile(r"(\w+)\s*=\s*('[^']*'|[^,\s]+)")


def _fields(line):
    """The values between the colons of a table line such as 'SPINSYS     :2          :4          :'."""
    return [field.strip() for field in line.split(':')[1:]]


def read_adf18_a17_p208(fil):
    """Reads an adf18 expansion file of the a17_p208 kind (ADAS appxa-18), which maps the n-shells of
    the bundle-nS condensed matrices of an adf17 file onto the levels of an adf04 file, for the
    projection of the high n-shells (see colradpy.projection).

    :param fil: path of the adf18 file
    :type fil: str

    :returns: dict -- 'sequence', 'adf04_file' (DSNREF) and 'adf17_file' (DSNCPM) as written,
              'num_parents', 'num_shells' and 'num_levels' of the namelist; 'parents', a list with,
              for each parent, its 'term', 'spin' (2S+1) and 'spin_systems', each with its 'spin'
              of the recombined system, 'lowest_n', 'imax', 'phase_space' (LOWESTP) and
              'parentage' (FRPARNT, or PRTWHT in older files, the fractional parentage of the lowest
              level for ionisation; 1 when not given);
              'records', a list of the level records with 'level' (index in the adf04 file, from
              1), 'metastable' (marked with *), 'code', 'szd', 'spin', 'shell' (active n-shell),
              'parent' (index from 1, 0 for none) and 'weight' (the normalised weight of the level
              in its n-shell).
    """
    with open(fil) as f:
        lines = [line.rstrip('\n') for line in f]

    #the namelist &SEQINF ... &END
    start = next((k for k, line in enumerate(lines) if '&SEQINF' in line.upper()), None)
    end = next((k for k, line in enumerate(lines) if '&END' in line.upper()), None)
    if(start is None or end is None):
        raise ValueError(str(fil) + ' is not an adf18 a17_p208 file: it has no &SEQINF ... &END namelist')
    items = {key.upper(): value.strip("'") for key, value in NAMELIST_ITEM.findall(' '.join(lines[start:end + 1]))}
    adf18 = {'sequence': items.get('SEQ', '').strip(), 'adf04_file': items.get('DSNREF', ''),
             'adf17_file': items.get('DSNCPM', ''), 'num_parents': int(items.get('NPARNT', 0)),
             'num_shells': int(items.get('NSHEL', 0)), 'num_levels': int(items.get('NLEV', 0))}

    #the parents and, for each, its spin systems
    k = end + 1
    table = {}
    while(k < len(lines) and not re.match(r'\s*PARENT\s*=', lines[k]) and 'RESOLVED' not in lines[k]):
        if(':' in lines[k]):
            table[lines[k].split(':')[0].strip().upper()] = _fields(lines[k])
        k += 1
    terms = table.get('PARENT', [])
    spins = table.get('SPINPRT', [])
    adf18['parents'] = [{'term': terms[p].strip('()') if p < len(terms) else '',
                         'spin': int(spins[p]) if p < len(spins) and spins[p] else 0,
                         'spin_systems': []} for p in range(adf18['num_parents'])]
    #older files write the fractional parentage as PRTWHT
    keys = {'SPINSYS': 'spin', 'LOWESTN': 'lowest_n', 'IMAX': 'imax', 'LOWESTP': 'phase_space',
            'FRPARNT': 'parentage', 'PRTWHT': 'parentage'}
    while(k < len(lines) and 'RESOLVED' not in lines[k]):
        parent = re.match(r'\s*PARENT\s*=\s*(\d+)', lines[k])
        if(parent):
            index = int(parent.group(1)) - 1
            values = {}
            k += 1
            while(k < len(lines) and not re.match(r'\s*PARENT\s*=', lines[k]) and 'RESOLVED' not in lines[k]):
                name = lines[k].split(':')[0].strip().upper()
                if(name in keys and ':' in lines[k]):
                    values[keys[name]] = [v for v in _fields(lines[k]) if v]
                k += 1
            systems = []
            for s in range(len(values.get('spin', []))):
                systems.append({'spin': int(values['spin'][s]),
                                'lowest_n': int(values['lowest_n'][s]),
                                'imax': int(values['imax'][s]),
                                'phase_space': float(values['phase_space'][s]) if 'phase_space' in values else 1.,
                                'parentage': float(values['parentage'][s]) if 'parentage' in values else 1.})
            if(not 0 <= index < len(adf18['parents'])):
                raise ValueError(str(fil) + ': parent ' + str(index + 1) + ' is not among the ' +
                                 str(adf18['num_parents']) + ' of the namelist')
            adf18['parents'][index]['spin_systems'] = systems
        else:
            k += 1

    #the level records, after the lines of headings, up to the comments
    adf18['records'] = []
    for line in lines[k:]:
        if(line.startswith('C')):
            break
        head, colon, tail = line.partition(':')
        tokens = head.split()
        if(not colon or len(tokens) < 6 or not re.fullmatch(r'\d+\*?', tokens[0])):
            continue
        weights = [float(v) if v else 0. for v in _fields(':' + tail)]
        shell = int(tokens[-2])
        adf18['records'].append({'level': int(tokens[0].rstrip('*')), 'metastable': tokens[0].endswith('*'),
                                 'code': ' '.join(tokens[1:-4]), 'szd': int(tokens[-4]),
                                 'spin': int(tokens[-3]), 'shell': shell, 'parent': int(tokens[-1]),
                                 'weight': weights[shell - 1] if 0 < shell <= len(weights) else 0.})
    if(not adf18['records']):
        raise ValueError(str(fil) + ': no level records (INDX CODE SZD SP SH PT and the n-shell weights)')
    return adf18

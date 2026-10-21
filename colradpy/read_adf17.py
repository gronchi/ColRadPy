import numpy as np
import re


#Fixed columns of the data lines (ADAS appxa-17, written by ADAS208): fields of 17 characters
FIELD = 17
#projected (or direct) collisional-radiative matrix, PCRTMP(I,J) J=1,5
MATRIX_COLUMNS = [19 + FIELD*j for j in range(5)]
#projected (or direct) ionisation to the parents, PIOTMP(I,J) J=1,4; also the indirect parent
#coefficients PQPTMP(J) J=1,4 that end each temperature
ION_COLUMNS = [106 + FIELD*j for j in range(4)]
#projected right hand side PRHTMP(I) or, on the direct lines, the direct three-body,
#dielectronic, radiative and charge exchange recombination DTREC, DDREC, DRREC, DXREC
RHS_COLUMNS = [176 + FIELD*j for j in range(4)]

HEADER = re.compile(r'SEQUENCE\s*=\s*(\S+)\s+NUCCHG\s*=\s*(\d+)\s+NPARNT\s*=\s*(\d+)\s+'
                    r'MAXD\s*=\s*(\d+)\s+MAXT\s*=\s*(\d+)')
PARENT = re.compile(r'IPRT\s*=\s*(\d+)\s+TRMPRT\s*=\s*\(([^)]*)\)\s+SPNPRT\s*=\s*([\d.]+)')
SPIN_SYSTEM = re.compile(r'SPNSYS\s*=\s*([\d.]+)\s+NSHEL\s*=\s*(\d+)')
SHELL = re.compile(r'N=\s*(\d+)')
NUMBER = re.compile(r'[-+]?\d*\.?\d+(?:[DdEe][-+]?\d+)?')


def _float(text):
    """A Fortran number: D exponents, and three-digit exponents written without the D
    (9.0835111029+102)."""
    text = text.strip().replace('D', 'e').replace('d', 'e')
    match = re.fullmatch(r'([-+]?\d*\.\d*)([-+]\d+)', text)
    if(match):
        text = match.group(1) + 'e' + match.group(2)
    return float(text)


def _value(line, start):
    """The number in the field of the line starting at the column start, 0 if it is blank."""
    text = line[start:start + FIELD].strip()
    return _float(text) if text else 0.


def _numbers(text):
    return [_float(x) for x in NUMBER.findall(text)]


def read_adf17(fil):
    """Reads an adf17 file of condensed collisional-radiative projection matrices (ADAS appxa-17,
    written by ADAS208), with every parent and spin system.

    :param fil: path of the adf17 file
    :type fil: str

    :returns: dict -- adf17['input_file'] holds 'sequence', 'nuc_chg', 'num_parent',
              'num_dens', 'num_temp', the switches (IEDMAT, ...) in 'switches', 'dens_grid'
              (cm-3), 'temp_grid' (K) and 'parents', a list with, for each parent, its 'index',
              'term', 'spin' and 'spin_systems'. Each spin system has its 'spin', the n-shells
              'n' and, with temperatures and densities as the last axes:
              'crmat' (shells, shells, ...) the projected collisional-radiative matrix PCRTMP,
              'ion' (shells, parents, ...) the projected ionisation PIOTMP,
              'rhs' (shells, ...) the projected right hand side PRHTMP,
              'dir' with the direct 'crmat' DCRTMP, 'ion' DIOMAT, 'recomb_three_body' DTREC,
              'recomb_diel' DDREC, 'recomb_rad' DRREC and 'recomb_cx' DXREC,
              'parent_qc' (parents, ...) the indirect parent coefficients PQPTMP (None if the
              file does not give them), and 'prb' (...) the value at the start of the same line when
              it is labelled (PRB) as in the cbnm96 files, which the specification does not
              describe (recombination-bremsstrahlung power by its label; None otherwise).
    """
    with open(fil) as f:
        lines = [line.rstrip('\n') for line in f]

    header = HEADER.search(lines[0]) if lines else None
    if(header is None):
        raise ValueError(str(fil) + ' is not an adf17 file of condensed projection matrices (ADAS208): '
                         'its first line does not give SEQUENCE, NUCCHG, NPARNT, MAXD and MAXT')
    adf17 = {'input_file': {}}
    data = adf17['input_file']
    data['sequence'] = header.group(1)
    data['nuc_chg'] = int(header.group(2))
    data['num_parent'] = int(header.group(3))
    data['num_dens'] = int(header.group(4))
    data['num_temp'] = int(header.group(5))
    num_dens, num_temp, num_parent = data['num_dens'], data['num_temp'], data['num_parent']

    #switches, then the density and temperature grids, which can continue on further lines
    k = 1
    data['switches'] = {}
    while(k < len(lines) and 'NE(CM-3)' not in lines[k]):
        data['switches'].update({key: int(value) for key, value in re.findall(r'(\w+)\s*=\s*(-?\d+)', lines[k])})
        k += 1
    grids = {}
    for key, label, size in (('dens_grid', 'NE(CM-3)', num_dens), ('temp_grid', 'TE(K)', num_temp)):
        if(k >= len(lines) or label not in lines[k]):
            raise ValueError(str(fil) + ': the ' + label + ' grid is missing')
        values = _numbers(lines[k].split('=', 1)[1])
        k += 1
        while(len(values) < size and k < len(lines)):
            values += _numbers(lines[k])
            k += 1
        if(len(values) != size):
            raise ValueError(str(fil) + ': the ' + label + ' grid has ' + str(len(values)) + ' values, not ' + str(size))
        grids[key] = np.array(values)
    data.update(grids)

    #the data lines, without blank lines and lines of dashes, up to the comments (C-----)
    body = []
    for line in lines[k:]:
        if(line.startswith('C')):
            break
        if(line.strip() and set(line.strip()) != {'-'}):
            body.append(line)
    i = 0
    data['parents'] = []
    while(i < len(body)):
        parent = PARENT.search(body[i])
        if(parent is None):
            raise ValueError(str(fil) + ': expected a parent (IPRT = ...) at ' + repr(body[i].strip()[:60]))
        entry = {'index': int(parent.group(1)), 'term': parent.group(2).strip(),
                 'spin': float(parent.group(3)), 'spin_systems': []}
        data['parents'].append(entry)
        i += 1
        while(i < len(body) and SPIN_SYSTEM.search(body[i])):
            spin, num_shells = SPIN_SYSTEM.search(body[i]).groups()
            num_shells = int(num_shells)
            if(num_shells > len(MATRIX_COLUMNS)):
                raise ValueError(str(fil) + ': ' + str(num_shells) + ' n-shells, at most ' + str(len(MATRIX_COLUMNS)))
            i += 1
            grid = (num_temp, num_dens)
            system = {'spin': float(spin), 'n': None,
                      'crmat': np.zeros((num_shells, num_shells) + grid),
                      'ion': np.zeros((num_shells, num_parent) + grid),
                      'rhs': np.zeros((num_shells,) + grid),
                      'dir': {'crmat': np.zeros((num_shells, num_shells) + grid),
                              'ion': np.zeros((num_shells, num_parent) + grid),
                              'recomb_three_body': np.zeros((num_shells,) + grid),
                              'recomb_diel': np.zeros((num_shells,) + grid),
                              'recomb_rad': np.zeros((num_shells,) + grid),
                              'recomb_cx': np.zeros((num_shells,) + grid)},
                      'parent_qc': np.zeros((num_parent,) + grid),
                      'prb': np.zeros(grid)}
            with_prb = False
            #the line of indirect parent coefficients PQPTMP after each temperature is not in
            #every file (cbnm93#he)
            with_qc = None
            shells = []
            #densities outermost, then temperatures: projected rows, direct rows, PQPTMP
            for d in range(num_dens):
                for t in range(num_temp):
                    if(with_qc is None):
                        following = body[i + 2*num_shells] if i + 2*num_shells < len(body) else ''
                        with_qc = bool(following.strip()) and not SHELL.search(following) and                                   not SPIN_SYSTEM.search(following) and not PARENT.search(following)
                    rows = body[i:i + 2*num_shells + int(with_qc)]
                    if(len(rows) < 2*num_shells + int(with_qc) or
                       not all(SHELL.search(r) for r in rows[:2*num_shells]) or '(DIR)' not in rows[num_shells]):
                        raise ValueError(str(fil) + ': unexpected lines for density ' + str(d + 1) +
                                         ', temperature ' + str(t + 1) + ' of spin system ' + spin +
                                         ' of parent ' + str(entry['index']))
                    for s in range(num_shells):
                        projected, direct = rows[s], rows[num_shells + s]
                        if(d == 0 and t == 0):
                            shells.append(int(SHELL.search(projected).group(1)))
                        for j in range(num_shells):
                            system['crmat'][s, j, t, d] = _value(projected, MATRIX_COLUMNS[j])
                            system['dir']['crmat'][s, j, t, d] = _value(direct, MATRIX_COLUMNS[j])
                        for p in range(num_parent):
                            system['ion'][s, p, t, d] = _value(projected, ION_COLUMNS[p])
                            system['dir']['ion'][s, p, t, d] = _value(direct, ION_COLUMNS[p])
                        system['rhs'][s, t, d] = _value(projected, RHS_COLUMNS[0])
                        for c, name in enumerate(('recomb_three_body', 'recomb_diel', 'recomb_rad', 'recomb_cx')):
                            system['dir'][name][s, t, d] = _value(direct, RHS_COLUMNS[c])
                    if(with_qc):
                        if(SHELL.search(rows[-1])):
                            raise ValueError(str(fil) + ': the line of parent coefficients is missing for density ' +
                                             str(d + 1) + ', temperature ' + str(t + 1))
                        for p in range(num_parent):
                            system['parent_qc'][p, t, d] = _value(rows[-1], ION_COLUMNS[p])
                        if('(PRB)' in rows[-1]):
                            with_prb = True
                            system['prb'][t, d] = _value(rows[-1], MATRIX_COLUMNS[0])
                    i += 2*num_shells + int(with_qc)
            if(not with_qc):
                system['parent_qc'] = None
            if(not with_prb):
                system['prb'] = None
            system['n'] = np.array(shells)
            entry['spin_systems'].append(system)
        if(not entry['spin_systems']):
            raise ValueError(str(fil) + ': parent ' + str(entry['index']) + ' has no spin system')
    if(len(data['parents']) != num_parent):
        raise ValueError(str(fil) + ': ' + str(len(data['parents'])) + ' parents, NPARNT is ' + str(num_parent))
    return adf17

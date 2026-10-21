################################################################################
# file name         : read_adf04_py3.py
# author            : Curt Johnson
# description       : This code reads adf04 formatted files
# version           : 0.9
# python version    : 3.6.5 |Anaconda, INC.| ipython 6.4.0
# dependencies      : numpy, re, sys
# license           : to be freed after code is stable and papers are published
#
# todo          check with all the adf04 files in adas
#
#
################################################################################

import numpy as np
import re
import sys
import copy
import warnings


#a number: 1.23+08, -1.23-04 (adf04 values omit the e), 1.23E+08, 1.23D+08, 1.000 or .0
NUMBER = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD]?[-+]\d+)?'
#ionisation potential and parent term of the first line: 196622.4(1S); a missing or blank term is 1S
IONIZATION_POTENTIAL = re.compile(r'(' + NUMBER + r')\s*(?:\(([^)]*)\)?)?')
#(IS)IL(XJ) of a level line: (2)1( 2.5), ( 2)1(  1.5), (3)10(10.0)
LEVEL_QUANTUM_NUMBERS = re.compile(r'\(\s*(\d+)\s*\)\s*(\d+)\s*\(\s*(\d*\.?\d*)\s*\)')
#{parent}fractional parentage, or {X} when ionisation from the level is excluded
PARENTAGE = re.compile(r'\{\s*(\d+|X)\s*[})]\s*(' + NUMBER + r')?')
#the level indices after the code of a transition line: '  2   1' (excitation), '  1  +1' (R, S, H);
#some R lines leave out the parent
TRANSITION_INDICES = re.compile(r'\s*(\d+)(?:\s*([-+]?)\s*(\d+)(?![\d.]))?')
#values of the transition lines are fields of 8 characters (16e8.2)
FIELD = 8
FIELD_VALUE = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+)(?:[eEdD]?[-+]\d+)?')
#codes of the transition lines that ColRadPy does not use
UNUSED_CODES = {'P': 'positive ion impact excitation', 'I': 'ionisation from the stage below',
                'L': 'unresolved dielectronic satellite lines'}

def adf04_float(text):
    """Number written in the adf04 form 1.23+08 or 1.23-08 (exponent without E), or a plain
       number such as 1.000 or 1.23E+08, with an optional sign."""
    text = str(text).strip()
    if(not re.search('[eEdD]', text)):
        text = re.sub(r'(?<=[\d.])([+-])(?=\d)', r'E\1', text)
    return float(text.replace('D', 'E').replace('d', 'e'))

def config_format(config_arr):
    return '.'.join(config_arr)

def value_fields(line, start, num_temp):
    """The values of a transition line after its indices, which end at the column start: the
    AVAL/AAUG/EBEAM/WVLN slot, a value for each temperature and possibly the Bethe limit, None
    when blank. They are fields of 8 characters (16e8.2), which can hold negative numbers without a
    space between them; the specification puts the first at column 9, older files shift them, so
    they are placed by the end of the first number, which ends its field. Values that are not in
    such fields are read as numbers separated by spaces, with a blank slot when there is one value
    for each temperature."""
    first = FIELD_VALUE.search(line, start)
    if(first is None):
        return []
    begin = first.end() - FIELD
    while(begin - FIELD >= start):
        begin -= FIELD
    if(begin >= start):
        try:
            fields = []
            for i in range(begin, len(line), FIELD):
                text = line[i:i+FIELD].strip()
                fields.append(adf04_float(text) if text else None)
            return fields
        except ValueError:
            pass
    fields = [adf04_float(text) for text in line[start:].split()]
    return [None] + fields if len(fields) == num_temp else fields

def read_adf04(fil):
    file_path = fil

    file_name = file_path
    adf04 = {}
    adf04['atomic'] = {}
    adf04['input_file'] = {}
    adf04['rates'] = {}
    f = open(file_name)
    ############################################################################
    #
    # reading the first line
    #
    ############################################################################
    try:
        first_line = f.readline().rstrip()#needed for some adf04 that don't follow the correct format
        adf04['atomic']['element'] = re.split('[+]',first_line)[0]
        first_linep = np.asarray(re.split('[ ]',re.split('[+]',first_line)[1]))
        first_linep_inds = np.where(first_linep !='')[0]
        adf04['atomic']['charge_state'] = int(first_linep[first_linep_inds[0]])
        adf04['atomic']['iz0'] = int(first_linep[first_linep_inds[1]])
        adf04['atomic']['iz1'] = int(first_linep[first_linep_inds[2]])
        #the ionisation potentials (cm-1) with the parent terms, written with or without spaces
        potentials = ' '.join(first_linep[first_linep_inds[3]:])
        ion_pot = []
        ion_term = []
        for match in IONIZATION_POTENTIAL.finditer(potentials):
            ion_pot.append(float(match.group(1)))
            #a parent without a term, or with a blank one, is 1S (ADF04 specification)
            ion_term.append((match.group(2) or '').strip() or '1S')
        if(not ion_pot or IONIZATION_POTENTIAL.sub('', potentials).strip()):
            raise ValueError('expected ionisation potentials with their parent terms, found ' + repr(potentials))
        adf04['atomic']['ion_pot'] = np.asarray(ion_pot)
        adf04['atomic']['ion_term'] = np.asarray(ion_term)

    except Exception as err:
        f.close()
        raise ValueError("**Failed first line of the adf04**\n\n"+first_line+"\n\nfile is formatted incorrectly failed line.\nCHECK WHITE SPACE.\nIt should be formatted as below (Neutral Tungsten example). \nIf more than one ionization metastable,\nadd more ionization potentials and terms after the first\nW+ 0        74         1       63427.7(6D3)" + "\n\nError: " + str(err)) from err

    ############################################################################
    #
    # done reading the first line, reading the level information
    #
    ############################################################################
    try:
        config = []
        L = []
        S = []
        w = []
        energy = []
        zpla = []
        zpla1 = []
        level_num = 0
        tmp_line =   first_line
        zpla = []
        zpla1 = []
        num_parents = len(adf04['atomic']['ion_pot'])
        while True:
            level_num = level_num + 1
            tmp_line = f.readline()
            if(not tmp_line):
                raise ValueError('the file ends before the line -1 that closes the levels')
            tmp_line = tmp_line.rstrip()
            if(re.match(r'\s*-1(\s|$)', tmp_line)):
                break
            #index, configuration (free text that can hold * or a term in parentheses), (IS)IL(XJ),
            #energy (cm-1) and the {parent}fractional parentage of each parent
            index = re.match(r'\s*\d+', tmp_line)
            quantum_numbers = list(LEVEL_QUANTUM_NUMBERS.finditer(tmp_line))
            if(index is None or not quantum_numbers):
                raise ValueError('expected the level index, configuration and (IS)IL(XJ) in ' + repr(tmp_line))
            quantum_numbers = quantum_numbers[-1]
            level_energy = re.compile(r'\s*(' + NUMBER + ')').match(tmp_line, quantum_numbers.end())
            if(level_energy is None):
                raise ValueError('expected the energy after (IS)IL(XJ) in ' + repr(tmp_line))
            config.append(config_format(tmp_line[index.end():quantum_numbers.start()].split()))
            S.append(int(quantum_numbers.group(1)))
            L.append(int(quantum_numbers.group(2)))
            w.append(float(quantum_numbers.group(3)))
            energy.append(adf04_float(level_energy.group(1)))
            zz = np.ones(len(adf04['atomic']['ion_pot']))*-1
            zz1 = np.ones(len(adf04['atomic']['ion_pot']))*-1

            #{parent}fractional parentage: the column of a parent is its index - 1
            rest = tmp_line[level_energy.end():]
            if(PARENTAGE.sub('', rest).strip()):
                raise ValueError('expected {parent}fractional parentage or {X} after the energy in ' + repr(tmp_line))
            for parentage in PARENTAGE.finditer(rest):
                if(parentage.group(1) == 'X'):
                    continue
                if(parentage.group(2) is None):
                    raise ValueError('parent ' + parentage.group(1) + ' has no fractional parentage in ' + repr(tmp_line))
                zpla1_line = int(parentage.group(1))
                zpla_line = adf04_float(parentage.group(2))
                if(not 1 <= zpla1_line <= num_parents):
                    raise ValueError('level ' + str(level_num) + ' ionizes to parent ' + str(zpla1_line) +
                                     ' but the first line gives ' + str(num_parents) +
                                     ' ionization potentials')
                zz[zpla1_line-1] = zpla1_line
                zz1[zpla1_line-1] = zpla_line

            zpla.append(zz1)
            zpla1.append(zz)

        adf04['atomic']['config'] = np.asarray(config)
        adf04['atomic']['L'] = np.asarray(L)
        adf04['atomic']['S'] = np.asarray(S)
        adf04['atomic']['w'] = np.asarray(w)
        adf04['atomic']['energy'] = np.asarray(energy)
        adf04['atomic']['zpla'] = np.asarray(zpla)
        adf04['atomic']['zpla1'] = np.asarray(zpla1)
        adf04['atomic']['ion_pot_lvl'] = np.arange(1, num_parents+1)
    except Exception as err:
        f.close()
        raise ValueError("\n**FAILED LEVEL " +str(level_num)+"**\n\n"+"\n\nOne or more configuration/energy lines in the adf04 file are formatted incorrectly.\nIt should be formatted as below (C+3 example). \nIf more than one ionization metastable,\nadd more apla,zpla after the first ie. {2}1.00\n      2 1S2 2P1           (2)1( 2.5)    64555.4   {1}1.000" + "\n\nError: " + str(err)) from err
    ############################################################################
    #
    # end reading level info start reading temperature grid
    #
    ############################################################################
    try:
        temp_line = f.readline().rstrip()#remove trailing because people cant follow directions
        temp_line_arr = np.asarray(re.split('[ ]',temp_line))
        temp_line_inds = np.where( temp_line_arr!='')[0]
        #ZEFF, ITYP and the temperatures (K)
        ityp = int(adf04_float(temp_line_arr[temp_line_inds[1]]))
        adf04_temp_grid = []
        for i in range(2,len(temp_line_inds)):
            adf04_temp_grid.append(adf04_float(temp_line_arr[temp_line_inds[i]]))
        adf04['input_file']['temp_grid'] = {}
        adf04['input_file']['temp_grid'] = np.asarray(adf04_temp_grid)
    except Exception as err:
        f.close()
        raise ValueError("\n**FAILED TEMPERATURE GRID**\n\n" +temp_line+"\n\nadf04 temperature grid formatted incorrectly.\nShould be formatted as below (C+3 example).\n  4.0    3       8.00+03 1.60+04 3.20+04 8.00+04 1.60+05 3.20+05 8.00+05 1.60+06 3.20+06 8.00+06 1.60+07" + "\n\nError: " + str(err)) from err
    if(ityp != 3):
        f.close()
        raise ValueError('adf04 ITYP = ' + str(ityp) + ' is not supported: ColRadPy reads effective collision '
                         'strengths (upsilons) against electron temperature, ITYP = 3. ITYP = 1 gives '
                         'collision strengths against threshold parameter and 4 non-Maxwellian rates.')
    ################################################################################
    #
    # end reading temp grid start reading in the atomic data from the adf04 file
    #
    ################################################################################
    #declare the colisional stuff
    col_excit = []
    col_transitions = []
    #declare the recombination stuff
    recomb_excit = []
    recomb_transitions = []
    #declare the ion stuff
    ion_excit = []
    ion_transitions = []
    #declare the cx stuff
    cx_excit = []
    cx_transitions = []
    #declare the a value stuff
    a_val = []
    inf_engy= []
    has_inf_engy = False
    num_temp = len(adf04['input_file']['temp_grid'])
    unused = {}
    #transition lines (ADAS appxa-04): the code in column 1 (' ', '1', '2' or '3' excitation, 'R'
    #recombination, 'S' ionisation, 'H' charge exchange), the level indices in columns 2-4 and 5-8
    #('+' and the parent for R, S and H lines), then fields of 8 characters (16e8.2) with the A-value
    #(the AAUG/EBEAM slot, or blank, for R, S and H lines), a value for each temperature and the
    #Bethe limit of an excitation
    for tmp_line in f:
        tmp_line = tmp_line.rstrip()
        if(re.match(r'\s*-1(\s|$)', tmp_line)):
            break
        if(not tmp_line.strip()):
            continue
        code = tmp_line[0].upper()
        if(code not in ' 123RSH'):
            unused[code] = unused.get(code, 0) + 1
            continue
        try:
            indices = TRANSITION_INDICES.match(tmp_line, 1)
            if(indices is None):
                raise ValueError('expected the level indices after the code')
            fields = value_fields(tmp_line, indices.end(), num_temp)
            values = np.array([v if v is not None else 0. for v in fields[1:1+num_temp]] +
                              [0.]*max(0, num_temp - len(fields[1:1+num_temp])))
            if(code in ' 123'):
                if(indices.group(3) is None or indices.group(2)):
                    raise ValueError('expected the upper and lower level of the excitation')
                col_transitions.append(np.array([int(indices.group(1)), int(indices.group(3))]))
                a_val.append(fields[0] if fields and fields[0] is not None else 0.)
                col_excit.append(values)
                #the Bethe limit (infinite energy point), 0 when it is not given
                bethe = fields[1+num_temp] if len(fields) > 1+num_temp else None
                has_inf_engy = has_inf_engy or bethe is not None
                inf_engy.append(bethe if bethe is not None else 0.)
            else:
                #the parent follows a '+'; R lines of some files leave it out when there is one parent
                if(indices.group(3) is None and num_parents == 1):
                    parent = 1
                elif(indices.group(3) is None or indices.group(2) == '-'):
                    raise ValueError('expected + and the parent after the level')
                else:
                    parent = int(indices.group(3))
                level = int(indices.group(1))
                if(code == 'R'):
                    recomb_transitions.append(np.array([parent, level]))
                    recomb_excit.append(values)
                elif(code == 'S'):
                    ion_transitions.append(np.array([level, parent]))
                    ion_excit.append(values)
                else:
                    cx_transitions.append(np.array([level, parent]))
                    cx_excit.append(values)
        except ValueError as err:
            f.close()
            raise ValueError('adf04 transition line formatted incorrectly: ' + repr(tmp_line) + '\n' + str(err) +
                             '\nIt should be formatted as below (C+3 example)\n'
                             '   2   1 1.04+08 1.03+00 1.03+00 1.03+00 1.04+00 1.05+00 1.08+00 1.18+00') from err
    if(unused):
        warnings.warn('ColRadPy does not use the adf04 lines ' +
                      ', '.join(str(n) + ' with code ' + c + ' (' + UNUSED_CODES.get(c, 'unknown code') + ')'
                                for c, n in sorted(unused.items())))

    f.close()
    if(not has_inf_engy):
        inf_engy = []

    #recombination, ionization and charge exchange connect to the parents of the first line
    for code, transitions, column in (('R', recomb_transitions, 0), ('S', ion_transitions, 1),
                                      ('H', cx_transitions, 1)):
        parents = np.asarray(transitions, dtype=int).reshape(-1, 2)[:,column]
        if(((parents < 1) | (parents > num_parents)).any()):
            raise ValueError(code + ' lines give parents ' + str(sorted(set(parents.tolist()))) +
                             ' but the first line gives ' + str(num_parents) + ' ionization potentials')

    #check if energies indexes wrong
    #Can these atomic physics people ever get a file format correct.........
    sort_energy = np.argsort(adf04['atomic']['energy'])#the correct sorted indexes
    #indexes that need to swapped around
    indd = np.where( np.linspace(0,len(adf04['atomic']['energy'])-1,len(adf04['atomic']['energy']),dtype=int) -\
                       sort_energy !=0)[0]
    #indexes as listed in the adf04 file
    inds = np.linspace(1,len(adf04['atomic']['energy']),len(adf04['atomic']['energy']),dtype=int)
    if(indd.size > 0):
        print('Energy levels in adf04 are not in order')
        print(inds[indd],sort_energy[indd])
        #create tmp arrays to hold the corrected swapped levels
        col_transitions_tmp = np.asarray(col_transitions)
        col_transitions = np.asarray(col_transitions)

        ion_transitions_tmp = np.asarray(ion_transitions)
        ion_transitions = np.asarray(ion_transitions)
        
        recomb_transitions_tmp = np.asarray(recomb_transitions)
        recomb_transitions = np.asarray(recomb_transitions)

        cx_transitions_tmp = np.asarray(cx_transitions)
        cx_transitions = np.asarray(cx_transitions)

        
        for ii in range(0,len(indd)):

            swap_indexes = np.where(col_transitions == sort_energy[indd[ii]]+1)
            col_transitions_tmp[swap_indexes] = inds[indd[ii]]#swapping around the indexes


            if(ion_transitions.size >0):            
                if(np.where(ion_transitions[:,0] == sort_energy[indd[ii]]+1)[0].size >0):
                    #swapping ionization
                    ion_transitions_tmp[np.where(ion_transitions[:,0] == sort_energy[indd[ii]]+1)[0],0] =  inds[indd[ii]]
                    
            if(recomb_transitions.size >0):
                if(np.where(recomb_transitions[:,1] == sort_energy[indd[ii]]+1)[0].size >0):
                    #swapping recombination
                    recomb_transitions_tmp[np.where(recomb_transitions[:,1] == sort_energy[indd[ii]]+1),1] =  inds[indd[ii]]
                
            if(cx_transitions.size >0):
                if(np.where(cx_transitions[:,0] == sort_energy[indd[ii]]+1)[0].size >0):
                    #swapping charge exchange
                    cx_transitions_tmp[np.where(cx_transitions[:,0] == sort_energy[indd[ii]]+1)[0],0] =  inds[indd[ii]]


            #col_transitions_tmp[np.where(col_transitions == sort_energy[indd[ii]]+1)] = inds[indd[ii]]
        col_transitions = col_transitions_tmp
        recomb_transitions = recomb_transitions_tmp
        ion_transitions = ion_transitions_tmp
        cx_transitions = cx_transitions_tmp

        #swapping around the level labeling information
        adf04['atomic']['config'] =       adf04['atomic']['config'][sort_energy]
        adf04['atomic']['L'] =            adf04['atomic']['L'][sort_energy]
        adf04['atomic']['S'] =            adf04['atomic']['S'][sort_energy]
        adf04['atomic']['w'] =            adf04['atomic']['w'][sort_energy]
        adf04['atomic']['energy'] =       adf04['atomic']['energy'][sort_energy]
        adf04['atomic']['zpla'] =         adf04['atomic']['zpla'][sort_energy]
        adf04['atomic']['zpla1'] =        adf04['atomic']['zpla1'][sort_energy]

    #there are some instances where the level transistions are backwards (lower level first),
    #the upper level is always written first here
    col_transitions = np.asarray(col_transitions)
    if(col_transitions.size > 0):
        ttt = np.where(col_transitions[:,0] < col_transitions[:,1])[0]
        if(ttt.size>0):
            print(str(ttt.size) + ' transitions in the adf04 file give the lower level first, they were swapped')
            ttmp = col_transitions[ttt,0]
            col_transitions[ttt,0] = col_transitions[ttt,1]
            col_transitions[ttt,1] = ttmp



    adf04['rates']['excit'] = {}
    adf04['rates']['excit']['col_transitions'] = np.asarray(col_transitions)
    adf04['rates']['excit']['col_excit'] = np.asarray(col_excit)
    adf04['rates']['a_val'] = np.asarray(a_val)
    adf04['rates']['inf_engy'] = np.asarray(inf_engy)
    adf04['rates']['recomb'] = {}
    adf04['rates']['recomb']['recomb_transitions'] = np.asarray(recomb_transitions)
    adf04['rates']['recomb']['recomb_excit'] = np.asarray(recomb_excit)
    adf04['rates']['ioniz'] = {}
    adf04['rates']['ioniz']['ion_transitions'] = np.asarray(ion_transitions)
    adf04['rates']['ioniz']['ion_excit'] = np.asarray(ion_excit)
    adf04['rates']['cx'] = {}
    adf04['rates']['cx']['cx_transitions'] = np.asarray(cx_transitions)
    adf04['rates']['cx']['cx_excit'] = np.asarray(cx_excit)

    

    adf04['input_file']['rates'] = {}
    adf04['input_file']['atomic'] = {}
    adf04['input_file']['rates'] = copy.deepcopy(adf04['rates'])
    adf04['input_file']['atomic'] = copy.deepcopy(adf04['atomic'])
    return adf04

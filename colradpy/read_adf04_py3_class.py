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


def first_substring(strings, substring):
    return next(i for i, string in enumerate(strings) if substring in string[0])

def adf04_float(text):
    """Number written in the adf04 form 1.23+08 or 1.23-08 (exponent without E), or a plain
       number such as 1.000 or 1.23E+08, with an optional sign."""
    text = str(text).strip()
    if(not re.search('[eEdD]', text)):
        text = re.sub(r'(?<=[\d.])([+-])(?=\d)', r'E\1', text)
    return float(text.replace('D', 'E').replace('d', 'e'))

def config_format(config_arr):
    conf = config_arr[0]
    if(len(config_arr)>1):
        for i in range(1,len(config_arr)):
            conf = conf+'.' + config_arr[i]
    return conf

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
        adf04['atomic']['ion_pot'] = np.zeros(len(first_linep_inds)-3)

        ion_term = []
        for i in range(0,len(adf04['atomic']['ion_pot'])):
            ion_pot_term = re.split('[()]',first_linep[first_linep_inds[i+3]])
            adf04['atomic']['ion_pot'][i] =float(ion_pot_term[0])
            #a parent without a term is 1S (ADF04 specification)
            ion_term.append(ion_pot_term[1] if len(ion_pot_term) > 1 else '1S')

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
        while('-1' not in tmp_line or '-1\n' not in tmp_line):
            level_num = level_num + 1
            tmp_line = f.readline().rstrip()#remove trailing
            #remove the white space that occurs when L=10 in the adf04 file
            tmp_line =     tmp_line[0:tmp_line.rfind('(')-3] + \
               tmp_line[tmp_line.rfind('(')-3:tmp_line.rfind('(')].replace(' ','')+\
               tmp_line[tmp_line.rfind('('):]

            tmp = np.asarray(re.split('[  ]',tmp_line))

            if('-1' in tmp or '-1\n' in tmp):
                break
            else:
                tmp_inds = np.where(tmp !='')[0]
                #finding end of config
                config_stop = first_substring(tmp[tmp_inds],'(')

                config.append(config_format(tmp[tmp_inds[1:config_stop]]))

                #there is offset that can occur due to L >=10
                if(len(tmp[tmp_inds[config_stop]])==3):
                    offset_l = 1
                    offset_w = 2
                else:
                    offset_l = 0
                    offset_w = 0

                S.append(int(re.split('[)]',re.split('[(]',tmp[tmp_inds[config_stop]])[1])[0]))

                L.append(int(re.split('[)]',re.split('[(]',tmp[tmp_inds[config_stop+offset_l]])[1-offset_l])[1-offset_l]))



                

                if( tmp[tmp_inds[config_stop]].count(')') > 1):
                    w.append(float(re.split('[()]',tmp[tmp_inds[config_stop]])[3]))
                    offset = -1
                else:
                    w.append(float(re.split('[)]',re.split('[(]',tmp[tmp_inds[config_stop+1]])[0])[0]))
                    offset=0
                    
                energy.append(float(tmp[tmp_inds[config_stop+2+offset]]))
                zz = np.ones(len(adf04['atomic']['ion_pot']))*-1
                zz1 = np.ones(len(adf04['atomic']['ion_pot']))*-1
                

                #{parent}fractional parentage: the column of a parent is its index - 1
                for i in range(config_stop+3+offset,len(tmp_inds)):
                    if ('X' not in tmp[tmp_inds[i]]):
                        zpla1_line = int(re.split('[}]',tmp[tmp_inds[i]])[0][1])
                        zpla_line = float(re.split('[}]',tmp[tmp_inds[i]])[1])
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
    while('-1' not in tmp or '-1\n' not in tmp):
        level_num = level_num + 1
        tmp_line = f.readline().rstrip()#remove trailing whitespace from some adf04 files :'(
        tmp = np.asarray(re.split('[  ]',tmp_line))
        if('-1' in tmp or '-1\n' in tmp):
            break
        if(tmp[0] ==''):
            tmp_inds = np.where(tmp!='')[0]
            # Transition numbers
            col_transitions.append(np.array([int(tmp[tmp_inds[0]]),int(tmp[tmp_inds[1]])]))
            # A-values
            a_val.append(adf04_float(tmp[tmp_inds[2]]))
            #collisional values, and the infinite energy point that may follow them, which can
            #be joined to the last value when it is negative (1.23-03-4.56+01)
            num_temp = len(adf04['input_file']['temp_grid'])
            col_excit_row = np.zeros(num_temp)
            inf_engy_line = 0.
            for i in range(0,len(tmp_inds) - 3):
                token = tmp[tmp_inds[i+3]]
                if(len(token) < 9):
                    if(i>num_temp-1):
                        inf_engy_line = adf04_float(token)
                        has_inf_engy = True
                    else:
                        col_excit_row[i] = adf04_float(token)
                else:
                    col_excit_row[i] = adf04_float(token[0:7])
                    inf_engy_line = adf04_float(token[7:])
                    has_inf_engy = True
            #one infinite energy point for every transition, 0 when it is not given
            inf_engy.append(inf_engy_line)
            col_excit.append(col_excit_row)
            #infinite energy

    ######################################################################
    #
    # end excitation and a values from the adf04, start looking for recomb
    #
    ######################################################################
        elif(tmp[0] == 'R' or tmp[0] == 'r'):
            tmp_inds = np.where(tmp!='')[0]
            recomb_transitions.append(np.array([int(tmp[tmp_inds[2]]),int(tmp[tmp_inds[1]])]))

            recomb_excit_row = np.zeros(len(adf04['input_file']['temp_grid']))
            for i in range(0,len(adf04['input_file']['temp_grid'])):
                recomb_excit_row[i] = adf04_float(tmp[tmp_inds[i+3]])
            recomb_excit.append(recomb_excit_row)

    ######################################################################
    #
    # end recombination and values from the adf04, start looking for ionization
    #
    ######################################################################
        elif(tmp[0] == 'S' or tmp[0] =='s'):
            tmp_inds = np.where(tmp!='')[0]
            ion_transitions.append(np.array([int(tmp[tmp_inds[1]]),int(tmp[tmp_inds[2]])]))
            ion_excit_row = np.zeros(len(adf04['input_file']['temp_grid']))
            for i in range(0,len(adf04['input_file']['temp_grid'])):
                ion_excit_row[i] = adf04_float(tmp[tmp_inds[i+3]])
            ion_excit.append(ion_excit_row)

    ######################################################################
    #
    # end ionization and values from the adf04, start looking for cx
    #
    ######################################################################

        elif(tmp[0] == 'H' or tmp[0] =='h'):
            tmp_inds = np.where(tmp!='')[0]
            cx_transitions.append(np.array([int(tmp[tmp_inds[1]]),int(tmp[tmp_inds[2]])]))
            cx_excit_row = np.zeros(len(adf04['input_file']['temp_grid']))
            for i in range(0,len(adf04['input_file']['temp_grid'])):
                cx_excit_row[i] = adf04_float(tmp[tmp_inds[i+3]])
            cx_excit.append(cx_excit_row)


    f.close()
    if(not has_inf_engy):
        inf_engy = []

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

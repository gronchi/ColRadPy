import os
import numpy as np
import re


#ADF11 data classes, file names start with them (e.g. scd96_c.dat, acd96r_c.dat)
ADF11_CLASSES = ('acd', 'scd', 'ccd', 'prb', 'prc', 'qcd', 'xcd', 'plt', 'pls', 'met')
#a fixed point number such as -12.34567 or, without the leading zero, -.69877 (ADAS resolved
#files); numbers can follow each other without spaces
NUMBER = r'[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?'


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
    """ Reads data formatted in adf11 format. This format is generally used for GCR coefficients.
    The data is stored in arrays of log_10.
    Creates a dictionary to hold the data

    :param fil: The file path to the input file
    :type fil: string
    """
    cls, resolved = adf11_class(fil)

    with open(fil) as f:
        adf11 = {}
        adf11['input_file'] = {}
        adf11['input_file']['class'] = cls
        adf11['input_file']['resolved'] = resolved

        #reading the first line
        tmp = re.findall(r'(\d+)',f.readline())
        adf11['input_file']['nuc_charge'] = int(tmp[0])
        adf11['input_file']['num_dens'] = int(tmp[1])
        adf11['input_file']['num_temp'] = int(tmp[2])
        adf11['input_file']['charge_min'] = int(tmp[3])
        adf11['input_file']['charge_max'] = int(tmp[4])

        f.readline() #reading '-------------'
        if(resolved):

            adf11['input_file']['metas'] = np.array(list(   #metastables
                           map(int,re.findall(r'(\d+)',f.readline()))))
            f.readline() #reading '---------------'
        else:
            adf11['input_file']['metas'] = np.ones(adf11['input_file']['charge_max']+1,dtype='int')

        #read in the density grid
        adf11['input_file']['dens_grid'] = np.array([])
        while adf11['input_file']['dens_grid'].size <adf11['input_file']['num_dens']:
            adf11['input_file']['dens_grid'] = np.append(adf11['input_file']['dens_grid'],
                                                         np.array(list(map(float,re.findall(NUMBER,f.readline())))))
        #read in the temperature grid
        adf11['input_file']['temp_grid'] = np.array([])
        while adf11['input_file']['temp_grid'].size <adf11['input_file']['num_temp']:
            adf11['input_file']['temp_grid'] = np.append(adf11['input_file']['temp_grid'],
                                                         np.array(list(map(float,re.findall(NUMBER,f.readline())))))

        #setting up the GCR stages
        if(adf11['input_file']['nuc_charge'] == len(adf11['input_file']['metas'])-1):
            num_stages = adf11['input_file']['nuc_charge']
        else:
            num_stages = len(adf11['input_file']['metas'])
        for i in range(0,num_stages):
            adf11['input_file'][str(i)] = {}
            if(cls in ('scd', 'acd', 'ccd')):
                adf11['input_file'][str(i)] = np.zeros((adf11['input_file']['metas'][i],
                                                         adf11['input_file']['metas'][i+1],
                                                 len(adf11['input_file']['temp_grid']),
                                                 len(adf11['input_file']['dens_grid'])))
            if(cls == 'qcd'):
                adf11['input_file'][str(i)] = np.zeros((adf11['input_file']['metas'][i],
                                                         adf11['input_file']['metas'][i],
                                                 len(adf11['input_file']['temp_grid']),
                                                 len(adf11['input_file']['dens_grid'])))
            if(cls == 'xcd'):
                adf11['input_file'][str(i)] = np.zeros((adf11['input_file']['metas'][i+1],
                                                         adf11['input_file']['metas'][i+1],
                                                 len(adf11['input_file']['temp_grid']),
                                                 len(adf11['input_file']['dens_grid'])))
            if(cls in ('plt', 'prb', 'prc')):
                adf11['input_file'][str(i)] = np.zeros((adf11['input_file']['metas'][i],
                                                         adf11['input_file']['metas'][i],
                                                 len(adf11['input_file']['temp_grid']),
                                                 len(adf11['input_file']['dens_grid'])))

        #Reading the GCR value portion, which ends at the comments (C-----, or a line of
        #dashes only in some files)
        gcr_line = f.readline()
        ii = 0

        while len(gcr_line.strip()) != 0 and not gcr_line.lstrip().startswith('C') and \
              set(gcr_line.strip()) != {'-'}:
            #look for the stage identifying line
            if('----------' in gcr_line):
                dens_count = 0
                temp_count = 0
                stage_id = np.array(list(map(int,re.findall(r'(\d+ )',gcr_line))))
                if(len(stage_id) == 1):#unresolved files only give Z1
                    stage_id = np.array([1,1,stage_id[0]])
                elif(cls in ('scd', 'acd', 'ccd')):#IPRT, IGRD to the recombined ion first
                    tmp = stage_id[0]
                    stage_id[0] = stage_id[1]
                    stage_id[1] = tmp

                ii = ii+1
            else:
                gcr_vals = np.array(list(map(float,re.findall(NUMBER,gcr_line))))
                adf11['input_file'][str(stage_id[2]-1)][stage_id[0]-1,stage_id[1]-1,
                                               temp_count,dens_count:dens_count+len(gcr_vals)] = gcr_vals
                dens_count = dens_count + len(gcr_vals)
                if(dens_count == len(adf11['input_file']['dens_grid'])):
                    temp_count = temp_count + 1
                    dens_count = 0
            gcr_line = f.readline()
    return adf11

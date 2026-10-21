import contextlib
import io
import warnings
import numpy as np
from colradpy.colradpy_class import *
from colradpy.colradpy_class import parent_levels
from colradpy.read_adf04_py3_class import read_adf04
from colradpy.solve_matrix_exponential import *
from scipy.interpolate import RectBivariateSpline
from colradpy.read_adf11 import *

def interp_rates_adf11(logged_temp,logged_dens,temp,dens,logged_gcr): # y are logged_temp and logged_dens args alongside temp and dens? seems redundant?
    # Dane did some optimization, Curt seemed to cobble this together (Dane optimized just this block of code as a self-contained unit)
    # parallelization should be implemented if more performance is needed
    source_logged_temp = np.asarray(logged_temp)
    source_logged_dens = np.asarray(logged_dens)
    target_logged_temp = np.log10(np.asarray(temp))
    target_logged_dens = np.log10(np.asarray(dens))

    gcr_arr = np.zeros((np.shape(logged_gcr)[0],
                        np.shape(logged_gcr)[1],
                        len(target_logged_temp),
                        len(target_logged_dens)))

    # Fitpack needs spline order < number of grid points.
    kx = min(3, len(source_logged_temp) - 1)
    ky = min(3, len(source_logged_dens) - 1)
    
    for i in range(0,np.shape(logged_gcr)[0]):
        for j in range(0,np.shape(logged_gcr)[1]):
            if(np.all(np.isneginf(logged_gcr[i,j]))):
                gcr_arr[i,j] = -np.inf #a block that the file does not give: zero coefficients
                continue
            interp_gcr = RectBivariateSpline(source_logged_temp,
                source_logged_dens,
                logged_gcr[i,j,:,:],
                kx=kx,
                ky=ky
            )
            gcr_arr[i,j] = interp_gcr(target_logged_temp, target_logged_dens) # array size matching works when Dane tests it
    return 10**gcr_arr


class ionization_balance():
    """The ionization balance class takes adf04 file inputs runs CR calcualations
       'colradpy_class.py' to get GCR coefficients. The GCR coefficients are then
       assemble into the ionization balance matrix. This matrix is solved in the 
       same way as the time dependent CR problem with 'solve_matrix_exponential.py'.
       The matrix can be solved time dependently with and without a source.
       The time-independent solution can also be solved. See dictionary structure documentation.

    :param fils: Array of the input adf04 file
    :type fils: string array

    :param metas: One array for the first stage (later stages inferred from the
        preceding SCD parent count), or one array per adf04 file, which can be followed by
        one for the final stage (its number of metastables must be the parents of the last file)
    :type metas: array or list of arrays

    :param temp_grid: Array of the temperatures in (eV)
    :type temp_grid: float array

    :param dens_grid: Array of the densities in (cm-3)
    :type dens_grid: float array

    :param htemp_grid: Temperature grid of thermal CX hydrogen (eV)
    :type htemp_grid: float array

    :param hdens_grid: Density grid of the thermal CX hydrogen densities in (cm-3)
    :type hdens_grid: float array

    :param soln_times: Times to calculate the solution for the time dependent solutions (s)
    :type soln_times: float array

    :param use_ionization: Choose if ionization will be use. This will proced the SCD rates.
        This should probably always be true if running an ionization balance but the option is there.
        Can be array of bools or just a single bool,
        if just a single bool is supplied then all
        charge states will have the same value.
    :type use_ionization: bool array or bool

    :param suppliment_with_ecip: Suppliments ionization rates with ECIP when ionization rates are not 
        present in the adf04 file.
        Can be array of bools or just a single bool,
        if just a single bool is supplied then all
        charge states will have the same value.
    :type suppliment_with_ecip: bool array or bool

    :param use_recombination_three_body: Adds in three body recombination, there must be ionization present for this to work
        Can be array of bools or just a single bool,
        if just a single bool is supplied then all
        charge states will have the same value.
    :type use_recombination_three_body: bool array or bool

    :param use_recombination: Use recombination in the ionization balance. This will produce the ACD and possible XCD rates
        Can be array of bools or just a single bool,
        if just a single bool is supplied then all
        charge states will have the same value.
    :type use_recombination: bool array or bool


    :param use_cx: Use thermal charge exchange  in the ionization balance. 
        This will produce the CCD rates the input can be array of bools or just a single bool,
        if just a single bool is supplied then all charge states will have the same value.
    :type use_cx: bool array or bool


    :param keep_species_data: When True this will keep all of the data associated with the CR solutions for each charge state.
        The can potentially take up lots of memory, the default is False
    :type keep_species_data: bool


    :param init_abund: The initial fractional abundances at t=0
    :type init_abund: float array

    :param source: Source rate into any charge state
    :type source: float array

    :param scale_file_ioniz: Scale ionization in the file
    :type scale_file_ioniz: bool

    :param rate_interp_ion: Interpolation of the adf04 ionization rates, see colradpy; one string
        for every stage or one for each adf04 file. Also rate_interp_recomb, rate_interp_col and
        rate_interp_cx for recombination, excitation and charge exchange
    :type rate_interp_ion: str or list of str

    :param ne_tau: n_e*tau values for time dependent calc to be evaluated at can not be defined at the same time as td_t (cm-3 s)
    :type ne_tau: float array

    :param adf11_files: the files provided are adf11 formatted files
    :type adf11_files: bool

    """
 
    def __init__(self,fils,metas,temp_grid, dens_grid, htemp_grid=np.array([]), hdens_grid=np.array([]),
                 soln_times=np.array([]), use_ionization=True,
                 suppliment_with_ecip=True, use_recombination_three_body=True,use_recombination=True,
                 use_cx=False,
                 keep_charge_state_data=False,init_abund = np.array([]), source= np.array([]),
                 temp_dens_pair=False,scale_file_ioniz=False, ne_tau=np.array([-1]),adf11_files=False,
                 hdf5_files=False, rate_interp_ion='slinear', rate_interp_recomb='log_slinear',
                 rate_interp_col='log_slinear', rate_interp_cx='log_quadratic'):


        #this is basic input data that is the same whether adf04 or adf11 inputs are used
        self.data = {}        
        self.data['cr_data'] = {}
        self.data['cr_data']['gcrs'] = {}
        self.data['user'] = {}
        self.data['user']['temp_grid'] = np.asarray(temp_grid) #eV
        self.data['user']['dens_grid'] = np.asarray(dens_grid)#cm-3
        self.data['user']['fils'] = np.asarray(fils)
        self.data['user']['init_abund'] = np.asarray(init_abund)
        self.data['user']['soln_times'] = np.asarray(soln_times)
        self.data['user']['source'] = np.asarray(source)
        self.data['user']['htemp_grid'] = np.asarray(htemp_grid) #eV
        self.data['user']['hdens_grid'] = np.asarray(hdens_grid) #cm-3
        self.data['user']['ne_tau'] = np.atleast_1d(ne_tau)
        if(self.data['user']['ne_tau'][0] != -1):
            print('ne_tau was not equal to -1 so overwiting solution times with ne_tau calculated values')
            self.data['user']['soln_times'] = self.ne_tau_times(self.data['user']['ne_tau'])





        



        
########################################################################################################################
        if(adf11_files):#adf11 specific formatting
            self.data['input_file'] = {}
            self.data['user']['temp_dens_pair'] = False
            if(type(use_cx) != bool):
                self.data['user']['use_cx'] = np.asarray(use_cx, dtype=bool)
            for i in range(0,len(fils)):
                adf11 = read_adf11(fils[i])
                #the class comes from the file name, not from its directories
                cls = adf11['input_file']['class']

                if(type(use_cx) == bool):
                    self.data['user']['use_cx'] = np.ones(len(adf11['input_file']['metas'])-1,dtype=bool)
                    self.data['user']['use_cx'][:] = use_cx

                if(cls not in ('scd', 'acd', 'qcd', 'xcd', 'ccd')):
                    continue
                self.data['input_file'][cls] = adf11['input_file']
                for j in range(0,len(adf11['input_file']['metas'])-1):
                    if( str(j) not in self.data['cr_data']['gcrs']):
                        self.data['cr_data']['gcrs'][str(j)] = {}
                    self.data['cr_data']['gcrs'][str(j)][cls] = interp_rates_adf11(adf11['input_file']['temp_grid'],
                                                                                   adf11['input_file']['dens_grid'],
                                                                                   temp_grid,dens_grid,
                                                                                   adf11['input_file'][str(j)])

            #coefficients that were not given are made after all the files are read, so the
            #order of the files does not matter
            for j in range(0,len(self.data['cr_data']['gcrs'])):
                gcrs = self.data['cr_data']['gcrs'][str(j)]
                for cls in ('scd', 'acd'):
                    if(cls not in gcrs):
                        raise ValueError('ADF11 ionization balances need scd and acd files, there is no ' +
                                         cls + ' for stage ' + str(j))
                num_met, num_ion = np.shape(gcrs['scd'])[0:2]
                if('qcd' not in gcrs):
                    gcrs['qcd'] = np.zeros((num_met, num_met, len(temp_grid), len(dens_grid)))
                if('xcd' not in gcrs):
                    gcrs['xcd'] = np.zeros((num_ion, num_ion, len(temp_grid), len(dens_grid)))
                if(self.data['user']['use_cx'][j] and 'ccd' not in gcrs):
                    raise ValueError('use_cx needs ccd files, there is no ccd for stage ' + str(j))


        elif(hdf5_files):

            import h5py
            import hdfdict

            if(type(use_cx) == bool):
                self.data['user']['use_cx'] = np.ones_like(fils,dtype=bool)
                self.data['user']['use_cx'][:] = use_cx

            
            self.data['user']['temp_dens_pair'] = False#probably should remove at some point
            self.data['cr_data']['stage_data'] = {}
            for i,j in enumerate(fils):
                c_hdf5 = hdfdict.load(j,lazy=False)
                self.data['cr_data']['gcrs'][str(i)] = {}
                self.data['cr_data']['gcrs'][str(i)]['scd'] = np.copy(c_hdf5['processed']['scd'])
                self.data['cr_data']['gcrs'][str(i)]['acd'] = np.copy(c_hdf5['processed']['acd'])
                self.data['cr_data']['gcrs'][str(i)]['qcd'] = np.copy(c_hdf5['processed']['qcd'])
                self.data['cr_data']['gcrs'][str(i)]['xcd'] = np.copy(c_hdf5['processed']['xcd'])
                self.data['cr_data']['stage_data'][str(i)] = c_hdf5
                
        else:#adf04 specific formating this is the default
            
            # give the option to the user to choose different
            # ionization and recombination settings for each charge state
            # this also just allows the same thing to be specified for all charge states




            self.data['user']['temp_dens_pair'] = temp_dens_pair
            metas_input = [metas] if np.isscalar(metas) else list(metas)
            if not metas_input:
                raise ValueError('metas must select at least one level in the first stage')
            explicit_metas = not np.isscalar(metas_input[0])
            #one level array per adf04 file, which can be followed by one for the final stage that
            #has no adf04 file; its number of metastables must be the parents of the last file
            final_metas = None
            if explicit_metas and len(metas_input) == len(fils) + 1:
                final_metas = np.atleast_1d(np.asarray(metas_input[-1], dtype=int))
                metas_input = metas_input[:-1]
            if explicit_metas and len(metas_input) != len(fils):
                raise ValueError('metas must have one level array per adf04 file, which can be '
                                 'followed by one for the final stage')
            self.data['user']['metas'] = []







            
            if(type(use_ionization) == bool):
                self.data['user']['use_ionization'] = np.ones_like(fils,dtype=bool)
                self.data['user']['use_ionization'][:] = use_ionization
            else:
                self.data['user']['use_ionization'] = use_ionization
            if(type(suppliment_with_ecip) == bool):
                self.data['user']['suppliment_with_ecip'] = np.ones_like(fils,dtype=bool)
                self.data['user']['suppliment_with_ecip'][:] = suppliment_with_ecip
            else:
                self.data['user']['suppliment_with_ecip'] = suppliment_with_ecip
            if(type(use_recombination_three_body) == bool):
                self.data['user']['use_recombination_three_body'] = np.ones_like(fils,dtype=bool)
                self.data['user']['use_recombination_three_body'][:] = use_recombination_three_body
            else:
                self.data['user']['use_recombination_three_body'] = use_recombination_three_body
            if(type(use_recombination) == bool):
                self.data['user']['use_recombination'] = np.ones_like(fils,dtype=bool)
                self.data['user']['use_recombination'][:] = use_recombination
            else:
                self.data['user']['use_recombination'] = use_recombination
            if(type(use_cx) == bool):
                self.data['user']['use_cx'] = np.ones_like(fils,dtype=bool)
                self.data['user']['use_cx'][:] = use_cx
            else:
                self.data['user']['use_cx'] = use_cx

            if(type(use_ionization) == bool):
                self.data['user']['use_ionization'] = np.ones_like(fils,dtype=bool)
                self.data['user']['use_ionization'][:] = use_ionization
            else:
                self.data['user']['use_ionization'] = use_ionization
            if(type(scale_file_ioniz) == bool):
                self.data['user']['scale_file_ioniz'] = np.ones_like(fils,dtype=bool)
                self.data['user']['scale_file_ioniz'][:] = scale_file_ioniz
            else:
                self.data['user']['scale_file_ioniz'] = scale_file_ioniz
            #interpolations of the file rates, one for every stage or one for each adf04 file
            for name, value in (('rate_interp_ion', rate_interp_ion), ('rate_interp_recomb', rate_interp_recomb),
                                ('rate_interp_col', rate_interp_col), ('rate_interp_cx', rate_interp_cx)):
                values = [value]*len(fils) if isinstance(value, str) else list(value)
                if(len(values) != len(fils)):
                    raise ValueError(name + ' must be one string or one for each adf04 file')
                self.data['user'][name] = values

            self.data['user']['keep_charge_state_data'] = keep_charge_state_data

            #default is to not keep all of the data from the CR calculations
            #this can take up a lot of memory and the general use case for this class only
            #calls for the use of GCR coefficients so its kind of a waste to keep all the data
            if(keep_charge_state_data):
                self.data['cr_data']['stage_data'] = {}
            #cycle over all of the ion stages that the user chose and run the CR calculation
            #to get the GCR coefficients for that stage
            for i in range(0,len(fils)):
                #default is for user to just give metastables of the ground state and then
                #choose metastables just based off of ionization potentials in adf04 file
                #the user can also give a list of metastable states for every ionstage
                #and this will override the metastables in the adf04 files
                if explicit_metas:
                    meta = np.atleast_1d(np.asarray(metas_input[i], dtype=int))
                elif i == 0:
                    meta = np.asarray(metas_input, dtype=int)
                if(i > 0):
                    #the parents of the stage before are levels of this stage, found by their terms
                    #and energies; ionization to parent p populates metastable p of this stage
                    with contextlib.redirect_stdout(io.StringIO()):
                        levels = read_adf04(str(fils[i]))['atomic']
                    try:
                        parents = parent_levels(tmp.data['atomic']['ion_pot'], tmp.data['atomic']['ion_term'],
                                                levels['energy'], levels['S'], levels['L'], levels['w'])
                    except ValueError as err:
                        if not explicit_metas:
                            raise ValueError('the metastables of stage ' + str(i) + ' cannot be inferred: ' +
                                             str(err) + '; give one level array per stage') from err
                        parents = None
                    if not explicit_metas:
                        meta = parents
                    elif(parents is not None and len(meta) == len(parents) and not np.array_equal(meta, parents)):
                        warnings.warn('the metastables ' + str(meta) + ' of stage ' + str(i) + ' are not the ' +
                                      'parents ' + str(parents) + ' of stage ' + str(i - 1) + ' (by term and ' +
                                      'energy), which ionization from stage ' + str(i - 1) + ' populates')
                #the metastables of a stage take the places of the parents of the stage before
                if(i > 0 and len(meta) != self.data['cr_data']['gcrs'][str(i - 1)]['scd'].shape[1]):
                    raise ValueError('stage ' + str(i) + ' has ' + str(len(meta)) + ' metastables but stage ' +
                                     str(i - 1) + ' ionizes to ' +
                                     str(self.data['cr_data']['gcrs'][str(i - 1)]['scd'].shape[1]) + ' parents')
                self.data['user']['metas'].append(meta)
                #setup the CR
                print(i)
                tmp = colradpy(fil=str(fils[i]),metas=meta,temp_grid=temp_grid,electron_den=dens_grid,
                               htemp_grid = htemp_grid, hdens_grid = hdens_grid,
                               use_ionization = self.data['user']['use_ionization'][i],
                               suppliment_with_ecip = self.data['user']['suppliment_with_ecip'][i],
                               use_recombination_three_body = self.data['user']['use_recombination_three_body'][i],
                               use_recombination = self.data['user']['use_recombination'][i],
                               use_cx = self.data['user']['use_cx'][i],
                               temp_dens_pair = self.data['user']['temp_dens_pair'],
                               scale_file_ioniz = self.data['user']['scale_file_ioniz'][i],
                               rate_interp_ion = self.data['user']['rate_interp_ion'][i],
                               rate_interp_recomb = self.data['user']['rate_interp_recomb'][i],
                               rate_interp_col = self.data['user']['rate_interp_col'][i],
                               rate_interp_cx = self.data['user']['rate_interp_cx'][i])

                tmp.solve_cr() #Solving the CR set of equations
                #keep all the CR data if requested
                if(keep_charge_state_data):
                    self.data['cr_data']['stage_data'][str(i)] = tmp.data
                #keeping the GCR data, ionization can be run from this data only
                self.data['cr_data']['gcrs'][str(i)] = {}
                self.data['cr_data']['gcrs'][str(i)]['qcd'] = tmp.data['processed']['qcd']
                self.data['cr_data']['gcrs'][str(i)]['scd'] = tmp.data['processed']['scd']
                self.data['cr_data']['gcrs'][str(i)]['acd'] = tmp.data['processed']['acd']
                self.data['cr_data']['gcrs'][str(i)]['xcd'] = tmp.data['processed']['xcd']
                if(self.data['user']['use_cx'][i]):
                    self.data['cr_data']['gcrs'][str(i)]['ccd'] = tmp.data['processed']['ccd']

            if final_metas is not None:
                last = len(fils) - 1
                num_parents = self.data['cr_data']['gcrs'][str(last)]['scd'].shape[1]
                if(len(final_metas) != num_parents):
                    raise ValueError('the final stage has ' + str(len(final_metas)) + ' metastables but stage ' +
                                     str(last) + ' ionizes to ' + str(num_parents) + ' parents')
                self.data['user']['metas'].append(final_metas)

            if self.data['user']['init_abund'].size == 0:
                gcrs = self.data['cr_data']['gcrs']
                num_states = gcrs['0']['qcd'].shape[0] + sum(
                    gcrs[str(i)]['scd'].shape[1] for i in range(len(fils)))
                self.data['user']['init_abund'] = np.zeros(num_states)
                self.data['user']['init_abund'][0] = 1.

    def ne_tau_times(self, ne_tau):
        """Solution times ne_tau/n_e (s) for the density grid, as solve_matrix_exponential takes
           them: one set of times for a single density, otherwise times for each density with
           shape (densities, len(ne_tau)).

        :param ne_tau: n_e*tau values (cm-3 s)
        :type ne_tau: float array

        :returns: array --solution times

        """
        ne_tau = np.atleast_1d(np.asarray(ne_tau, dtype=float))
        dens = self.data['user']['dens_grid']
        if(len(dens) == 1):
            return ne_tau/dens[0]
        return np.einsum('i,j->ji', ne_tau, 1/dens)

    def default_abundances(self):
        """Initial abundances: init_abund, or all in the first state of the ion matrix."""
        if(self.data['user']['init_abund'].size > 0):
            return self.data['user']['init_abund']
        n0 = np.zeros(len(self.data['ion_matrix']))
        n0[0] = 1.
        return n0

    def populate_ion_matrix(self):
        """This will populate the ionization matrix from the various GCR coefficients
           for all of the ionization stages

           This matrix in mostly zeros because we don't have ways to connect charge states
           that are more than one change away. For example if only ground states are included
           this matrix becomes diagonal.

           QCDs bring the atom between metastables states in a charge state
           SCDS bring the atom to the next charge state
           ACDS bring the atom to the previous charge state
           XCDS bring the atom to next charge state between the metastables of that level,
           through the previous charge state

           The columns of this matrix sum to zero
        """
        #finding the total number of states to be tracked (metastables)
        m = np.shape(self.data['cr_data']['gcrs']['0']['qcd'])[0]#num metastables in ground
        m = m + np.shape(self.data['cr_data']['gcrs']['0']['scd'])[1]#num metastables in plus

        for i in range(1,len(self.data['cr_data']['gcrs'])):
            m = m + np.shape(self.data['cr_data']['gcrs'][str(i)]['scd'])[1]#num of metastales in the plus loop
        #create an empty matrix to hold the GCR rates



        if(self.data['user']['temp_dens_pair']):
            self.data['ion_matrix'] = np.zeros((m, m,len(self.data['user']['temp_grid'])))


            m = 0
            for i in range(0,len(self.data['cr_data']['gcrs'])):
                num_met = np.shape(self.data['cr_data']['gcrs'][str(i)]['qcd'])[0]
                num_ion = np.shape(self.data['cr_data']['gcrs'][str(i)]['scd'])[1]
                diag_met = np.diag_indices(num_met)
                diag_ion = np.diag_indices(num_ion)

                #populate QCDs in ion balance
                self.data['ion_matrix'][m+diag_met[0],m+diag_met[1]] = self.data['ion_matrix'][m+diag_met[0],m+diag_met[1]] \
                                       -np.sum(self.data['cr_data']['gcrs'][str(i)]['qcd'],axis=1)

                self.data['ion_matrix'][m:m+num_met,m:m+num_met,:]=self.data['ion_matrix'][m:m+num_met,m:m+num_met,:]\
                                        +self.data['cr_data']['gcrs'][str(i)]['qcd'].transpose(1,0,2)

                #populate SCDs in ion balance
                self.data['ion_matrix'][m+diag_met[0],m+diag_met[1]] = self.data['ion_matrix'][m+diag_met[0],m+diag_met[1]]\
                                          -np.sum(self.data['cr_data']['gcrs'][str(i)]['scd'],axis=1)

                self.data['ion_matrix'][m+num_met:m+num_met+num_ion,m:m+num_met,:] = \
                                    self.data['ion_matrix'][m+num_met:m+num_met+num_ion,m:m+num_met,:] \
                                                       +self.data['cr_data']['gcrs'][str(i)]['scd'].transpose(1,0,2)

                #populate ACDs in ion balance
                self.data['ion_matrix'][m+num_met+diag_ion[0], m+num_met+diag_ion[1] ] = \
                                            self.data['ion_matrix'][m+num_met+diag_ion[0], m+num_met+diag_ion[1] ]-\
                                                   np.sum(self.data['cr_data']['gcrs'][str(i)]['acd'],axis=0)

                self.data['ion_matrix'][m:m+num_met,m+num_met:m+num_met+num_ion,:] = \
                                self.data['ion_matrix'][m:m+num_met,m+num_met:m+num_met+num_ion,:] \
                                                 +  self.data['cr_data']['gcrs'][str(i)]['acd']

                #populate CCDs in ion balance



                
                if(self.data['user']['use_cx'][i]):
                    self.data['ion_matrix'][m+num_met+diag_ion[0], m+num_met+diag_ion[1] ] = \
                                                self.data['ion_matrix'][m+num_met+diag_ion[0], m+num_met+diag_ion[1] ]-\
                                                       np.sum(np.einsum('ijn,n->ijn',self.data['cr_data']['gcrs'][str(i)]['ccd'],
                                                                        self.data['user']['hdens_grid']/self.data['user']['dens_grid']),axis=0)


                    self.data['ion_matrix'][m:m+num_met,m+num_met:m+num_met+num_ion,:] = \
                                    self.data['ion_matrix'][m:m+num_met,m+num_met:m+num_met+num_ion,:] \
                                                     +  np.einsum('ijn,n->ijn',self.data['cr_data']['gcrs'][str(i)]['ccd'],
                                                                  self.data['user']['hdens_grid']/self.data['user']['dens_grid'])                                                                  

                #populate XCD in ion balance
                self.data['ion_matrix'][m+num_met+diag_ion[0], m+num_met+diag_ion[1] ] = \
                                            self.data['ion_matrix'][m+num_met+diag_ion[0], m+num_met+diag_ion[1] ]-\
                                            np.sum(self.data['cr_data']['gcrs'][str(i)]['xcd'],axis=1)

                self.data['ion_matrix'][m+num_met:m+num_met+num_ion,m+num_met:m+num_met+num_ion,:]=\
                            self.data['ion_matrix'][m+num_met:m+num_met+num_ion,m+num_met:m+num_met+num_ion,:] + \
                            self.data['cr_data']['gcrs'][str(i)]['xcd'].transpose(1,0,2)

                m = m + num_met

            
        else:
            self.data['ion_matrix'] = np.zeros((m, m,len(self.data['user']['temp_grid']),
                                                     len(self.data['user']['dens_grid'])))
                                                     
            m = 0
            for i in range(0,len(self.data['cr_data']['gcrs'])):
                num_met = np.shape(self.data['cr_data']['gcrs'][str(i)]['qcd'])[0]
                num_ion = np.shape(self.data['cr_data']['gcrs'][str(i)]['scd'])[1]
                diag_met = np.diag_indices(num_met)
                diag_ion = np.diag_indices(num_ion)

                #populate QCDs in ion balance
                self.data['ion_matrix'][m+diag_met[0],m+diag_met[1]] = self.data['ion_matrix'][m+diag_met[0],m+diag_met[1]] \
                                       -np.sum(self.data['cr_data']['gcrs'][str(i)]['qcd'],axis=1)

                self.data['ion_matrix'][m:m+num_met,m:m+num_met,:,:]=self.data['ion_matrix'][m:m+num_met,m:m+num_met,:,:]\
                                        +self.data['cr_data']['gcrs'][str(i)]['qcd'].transpose(1,0,2,3)

                #populate SCDs in ion balance
                self.data['ion_matrix'][m+diag_met[0],m+diag_met[1]] = self.data['ion_matrix'][m+diag_met[0],m+diag_met[1]]\
                                          -np.sum(self.data['cr_data']['gcrs'][str(i)]['scd'],axis=1)

                self.data['ion_matrix'][m+num_met:m+num_met+num_ion,m:m+num_met,:,:] = \
                                    self.data['ion_matrix'][m+num_met:m+num_met+num_ion,m:m+num_met,:,:] \
                                                       +self.data['cr_data']['gcrs'][str(i)]['scd'].transpose(1,0,2,3)

                #populate ACDs in ion balance
                self.data['ion_matrix'][m+num_met+diag_ion[0], m+num_met+diag_ion[1] ] = \
                                            self.data['ion_matrix'][m+num_met+diag_ion[0], m+num_met+diag_ion[1] ]-\
                                                   np.sum(self.data['cr_data']['gcrs'][str(i)]['acd'],axis=0)

                self.data['ion_matrix'][m:m+num_met,m+num_met:m+num_met+num_ion,:,:] = \
                                self.data['ion_matrix'][m:m+num_met,m+num_met:m+num_met+num_ion,:,:] \
                                                 +  self.data['cr_data']['gcrs'][str(i)]['acd']
                #populate CCDs in ion balance
                if(self.data['user']['use_cx'][i]):                
                    self.data['ion_matrix'][m+num_met+diag_ion[0], m+num_met+diag_ion[1] ] = \
                                            self.data['ion_matrix'][m+num_met+diag_ion[0], m+num_met+diag_ion[1] ]-\
                                                   np.sum(np.einsum('ijkn,n->ijkn',self.data['cr_data']['gcrs'][str(i)]['ccd'],
                                                                    self.data['user']['hdens_grid']/self.data['user']['dens_grid']),axis=0)

                    self.data['ion_matrix'][m:m+num_met,m+num_met:m+num_met+num_ion,:,:] = \
                                    self.data['ion_matrix'][m:m+num_met,m+num_met:m+num_met+num_ion,:,:] \
                                                     +  np.einsum('ijkn,n->ijkn',self.data['cr_data']['gcrs'][str(i)]['ccd'],
                                                                  self.data['user']['hdens_grid']/self.data['user']['dens_grid'])

                #populate XCD in ion balance
                self.data['ion_matrix'][m+num_met+diag_ion[0], m+num_met+diag_ion[1] ] = \
                                            self.data['ion_matrix'][m+num_met+diag_ion[0], m+num_met+diag_ion[1] ]-\
                                            np.sum(self.data['cr_data']['gcrs'][str(i)]['xcd'],axis=1)

                self.data['ion_matrix'][m+num_met:m+num_met+num_ion,m+num_met:m+num_met+num_ion,:,:]=\
                            self.data['ion_matrix'][m+num_met:m+num_met+num_ion,m+num_met:m+num_met+num_ion,:,:] + \
                            self.data['cr_data']['gcrs'][str(i)]['xcd'].transpose(1,0,2,3)

                m = m + num_met


    def solve_no_source(self,n0=np.array([]),td_t=np.array([])):
        """Solves the ionization balance matrix given an initial populations and times

        :param n0: The initial populations of levels at t=0
        :type n0: float array

        :param td_t: Solution times
        :type td_t: float array


        populates the pops, eigen_val and eigen_vec

        """

        if(n0.size < 1):
            n0 = self.default_abundances()
        if(td_t.size < 1):
            td_t = self.data['user']['soln_times']

        if('processed' not in self.data.keys()):
            self.data['processed'] = {}
            self.data['processed']['pops_td'] = 0
            self.data['processed']['eigen_val'] = 0
            self.data['processed']['eigen_vec'] = 0            
            #solve the ionization balance set of equation with no source term
        if(self.data['user']['temp_dens_pair']):
            self.data['processed']['pops_td'],\
                self.data['processed']['eigen_val'],\
                self.data['processed']['eigen_vec'] = solve_matrix_exponential(
                                               np.einsum('ijk,k->ijk',self.data['ion_matrix'],
                                               self.data['user']['dens_grid']),n0,td_t)

        else:
            self.data['processed']['pops_td'],\
                self.data['processed']['eigen_val'],\
                self.data['processed']['eigen_vec'] = solve_matrix_exponential(
                                               np.einsum('ijkl,l->ijkl',self.data['ion_matrix'],
                                               self.data['user']['dens_grid']),n0,td_t)
    def solve_source(self,n0=np.array([]), s0=np.array([]), td_t=np.array([]),ne_tau = np.array([-1])):
        """Solves the ionization balance matrix given an
             initial populations and times when a source term is present

        :param n0: The initial populations of levels at t=0
        :type n0: float array

        :param td_t: Solution times
        :type td_t: float array

        :param ne_tau: n_e*tau values (cm-3 s); when given, the solution times are ne_tau/n_e
        :type ne_tau: float array


        populates the pops, eigen_val and eigen_vec

        """

        if(n0.size < 1):
            n0 = self.default_abundances()
        ne_tau = np.atleast_1d(ne_tau)
        if(ne_tau[0] != -1):
            print('ne_tau was not equal to -1 so overwiting solution times with ne_tau calculated values')
            td_t = self.ne_tau_times(ne_tau)
            self.data['user']['soln_times'] = td_t
        elif(td_t.size < 1):
            td_t = self.data['user']['soln_times']

        if(s0.size < 1):
            if(self.data['user']['source'].size < 1):
                s0 = np.zeros(len(self.data['ion_matrix']))
            else:
                s0 = self.data['user']['source']


                
        if('processed' not in self.data.keys()):
            self.data['processed'] = {}




        if(self.data['user']['temp_dens_pair']):
            self.data['processed']['pops_td_source'],\
                self.data['processed']['eigen_val'],\
                self.data['processed']['eigen_vec'] = solve_matrix_exponential_source(
                                               np.einsum('ijk,k->ijk',self.data['ion_matrix'],
                                               self.data['user']['dens_grid']),n0,s0,td_t)
        else:
            self.data['processed']['pops_td_source'],\
                self.data['processed']['eigen_val'],\
                self.data['processed']['eigen_vec'] = solve_matrix_exponential_source(
                                               np.einsum('ijkl,l->ijkl',self.data['ion_matrix'],
                                               self.data['user']['dens_grid']),n0,s0,td_t)

        self.data['processed']['pops_source'] = self.data['processed']['pops_td_source']

        '''
        self.data['processed']['pops_td'],\
        self.data['processed']['eigen_val'],\
        self.data['processed']['eigen_vec'] = solve_matrix_exponential_source(
                                                 np.einsum('ijkl,l->ijkl',self.data['ion_matrix'],
                                                 self.data['user']['dens_grid']),n0,s0,td_t)
        '''

    def solve_time_independent(self):
        """Solves the ionization balance matrix for the steady-state (time-independent) solution.

           The steady state is the eigenvector of the ionization balance matrix whose eigenvalue
           is zero. solve_matrix_exponential_steady_state takes the eigenvector of the eigenvalue
           closest to zero and checks that it is finite, non-negative, normalized and balanced;
           otherwise it solves the balance equations with the normalization directly.

           Populates ['processed']['pops_ss'], ['eigen_val'] and ['eigen_vec'].
        """
        if('processed' not in self.data.keys()):
            self.data['processed'] = {}
            self.data['processed']['pops_ss'] = 0
            
        #for the steady state the initial conditions don't matter so just make it simple        
        n0 = np.zeros((len(self.data['ion_matrix'])))
        n0[0] = 1.
        if(self.data['user']['temp_dens_pair']):
            self.data['processed']['pops_ss'],\
                self.data['processed']['eigen_val'],\
                self.data['processed']['eigen_vec'] = solve_matrix_exponential_steady_state(
                                               np.einsum('ijk,k->ijk',self.data['ion_matrix'],
                                               self.data['user']['dens_grid']))
        else:
            self.data['processed']['pops_ss'],\
                self.data['processed']['eigen_val'],\
                self.data['processed']['eigen_vec'] = solve_matrix_exponential_steady_state(
                                               np.einsum('ijkl,l->ijkl',self.data['ion_matrix'],
                                               self.data['user']['dens_grid']))

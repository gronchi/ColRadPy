import sys
sys.path.append('../')
from colradpy import colradpy
import matplotlib.pyplot as plt
import numpy as np
from os.path import exists
from os import mkdir
from pathlib import Path


# Variables
# Set up output folders
EXAMPLES_PATH: Path = Path(__file__).parent
EXAMPLES_INPUT_PATH: Path = EXAMPLES_PATH / "input"
EXAMPLES_OUTPUT_PATH: Path = EXAMPLES_PATH / "output"
OUTPUT_PATH: Path = EXAMPLES_OUTPUT_PATH / Path(__file__).name.split('.')[0]
# Making output directories
paths = [EXAMPLES_OUTPUT_PATH, OUTPUT_PATH]
for p in paths:
    if not exists(p):
        mkdir(p)

fil = str(EXAMPLES_INPUT_PATH / 'cpb03_ls#be0.dat') #adf04 file
temperature_arr = np.linspace(1,100,100) #eV
metastable_levels = np.array([0])   #metastable level, just ground chosen here
density_arr =     np.array([1.e13,4.e14]) # cm-3
be = colradpy(fil,metastable_levels,temperature_arr,density_arr,use_recombination=True,
              use_recombination_three_body = True,use_ionization=True,suppliment_with_ecip=True)
be.solve_cr()

#pop_lvl[i,j,m] is the part of the population of excited level i that the metastable m
#populates through level j; i and j index the excited (non-metastable) levels, whose level
#numbers are in excited_levels
excited_levels = be.data['processed']['excited_levels']
level = 1 #the level whose populating mechanisms are plotted, 2s2p 3P
i = np.where(excited_levels == level)[0][0]

#plotting the populating levels
fig, ax1 = plt.subplots(1,1,figsize=(16/3.,9/3.),dpi=300)
fig.subplots_adjust(bottom=0.15,top=0.93,left=0.125,right=0.965)

for t in (0, 10, -1):
    ax1.plot(excited_levels,
             be.data['processed']['pop_lvl'][i,:,0,t,0]/\
             np.sum(be.data['processed']['pop_lvl'][i,:,0,t,0]),
             label='Te= '+str(be.data['user']['temp_grid'][t])+ ' eV')


ax1.set_title('Populating mechanisms of level ' + str(level))
ax1.legend(fontsize='x-small',loc='best')
ax1.set_xlabel('Level number (#)')
ax1.set_ylabel('Populating fraction (-)')

fig.savefig(OUTPUT_PATH / "populating_levels.pdf", format='pdf')

#plotting the populating fraction from the ground versus temperature
fig, ax1 = plt.subplots(1,1,figsize=(16/3.,9/3.),dpi=300)
fig.subplots_adjust(bottom=0.15,top=0.93,left=0.125,right=0.965)
ax1.plot(be.data['user']['temp_grid'],
          be.data['processed']['pop_lvl'][i,i,0,:,0]/\
          np.sum(be.data['processed']['pop_lvl'][i,:,0,:,0],axis=0))
ax1.set_title('Populating fraction from ground to level ' + str(level))
ax1.set_xlabel('Temperature (eV)')
ax1.set_ylabel('Populating fraction from ground (-)')

fig.savefig(OUTPUT_PATH / "fractional_excited_states_vs_te.pdf", format='pdf')


API reference
=============

The ``colradpy`` package exports the main classes, so

.. code-block:: python

   from colradpy import colradpy, ionization_balance

works without naming the modules. The pages below document each module.

Main classes
------------

The classes that solve the collisional-radiative and balance problems.

.. toctree::
   :maxdepth: 1

   colradpy_class
   ionization_balance_class
   energy_balance

Atomic data files
-----------------

Readers and writers of ADAS, NIST and Flexible Atomic Code (FAC) data.

.. toctree::
   :maxdepth: 1

   read_adf04_py3_class
   read_adf11
   read_adf13
   read_adf17
   read_adf18
   read_fac
   write_adf15
   nist_read_txt
   nist_read
   convert_eissner
   adf04_file_paths

Rate coefficients
-----------------

Extrapolation of excitation rates, ECIP ionization rates, the projection of the high n-shells and the
Maxwellian averages of cross-sections.

.. toctree::
   :maxdepth: 1

   burgess_tully_rates
   convolve_eedf
   ecip_rates
   projection
   r8necip
   r8yip

Populations and solutions
-------------------------

LTE populations, time-dependent solutions and line shapes.

.. toctree::
   :maxdepth: 1

   saha_boltzmann_lte_pops
   solve_matrix_exponential
   split_multiplet
   colradpy_utility

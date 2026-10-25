Escape factors
==============

The optional ``opacity`` argument models isolated one-photon lines in a uniform
single zone with complete frequency redistribution. Absorber densities are
absolute densities in cm\ :sup:`-3`, fixed across the electron grid and in time.
They are not reconstructed from the solved populations. Full spatial radiation
transport, continuum absorption and population-dependent opacity are outside
this model.

For example, using the bundled Be I input from the repository root:

.. code-block:: python

   import numpy as np
   from colradpy import colradpy

   model = colradpy(
       "examples/input/cpb03_ls#be0.dat", metas=np.array([0]),
       temp_grid=np.array([5., 20.]), electron_den=np.array([1e12, 1e13]),
       opacity=dict(length=5., densities={0: 1e12}, temperature=1., mass=9.012),
   )
   model.solve_cr()
   net = model.data["processed"]["pecs"]
   chord = model.data["processed"]["pecs_los"]
   chord_sxb = model.get_sxb(emission="los")
   model.write_pecs_adf15("be_los.adf15", emission="los")
   model.write_sxb_adf13("be_los.adf13", emission="los")

``length`` is the radius R or slab half thickness in cm; the central chord has
length 2R. ``temperature`` is the absorber temperature in eV, independent of
electron temperature, and ``mass`` is in amu. ``densities`` uses zero-based
indices in the model's sorted level list. All physical inputs are scalar;
densities and length must be finite and nonnegative, temperature and mass
finite and positive. Geometry defaults to ``'cylinder'``; ``'sphere'`` and
``'slab'`` are also available. Profiles are ``'doppler'`` (default) or ``'voigt'``
with natural widths calculated from the thin radiative rates.

The population factor defaults to ``position='average'``, which equates net
radiative losses to boundary-integrated photon escape for the assumed uniform
populations. ``position='centre'`` selects the ADAS214 local centre closure.
Neither option solves for a spatial population distribution. Central-chord
brightness can exceed the thin value without violating photon conservation.

``pecs`` and the power coefficients use A times the population escape factor;
``pecs_los`` uses the thin A times the central-chord line escape factor with the
same solved populations. For a uniform isotropic emitter, chord photon radiance
is the corresponding thin-volume emissivity convention times 2R/(4 pi), using
``pecs_los``. The coefficient itself contains no chord-length or solid-angle
factor. With the centre closure, ``pecs`` describes a local net-loss estimate.

S/XB and ADF13/ADF15 exports accept ``emission='net'`` (default) or ``'los'``.
File comments record the choice and opacity inputs. Existing plotting methods
use net PECs; use ``pecs_los`` explicitly for chord diagnostics.
Time-dependent solutions include ``processed['td']['pecs_los']`` with fixed
escape factors and an instantaneous radiation closure. They do not evolve
the radiation field or absorber densities.

Alternatively, supply both direct-factor mappings with the same keys:

.. code-block:: python

   opacity = dict(theta_p={(2, 0): 0.2}, theta_l={(2, 0): 0.3})

Factors must be finite scalars in [0, 1], with valid ``upper > lower`` indices.
Unlisted transitions remain thin. Direct factors cannot be mixed with physical
opacity inputs. The caller is responsible for supplying factors from a
consistent geometry and radiation closure.

Precomputed high-shell projection and statistical PEC multiplet splitting
reject opacity: both need a consistent treatment of trapping in the population
solution. Term- and n-resolved atomic models also require justification of
their assumed sublevel population distribution before quantitative use.

``dump_hdf5`` retains lines significant in either net or LOS emission and writes
``processed/pecs_los``, ``rates/escape`` and ``cr_matrix/A_ji_thin``. Its
``user/opacity`` mappings are numeric indexed tables: rows are (level, density)
or (upper, lower, factor). The live input mappings retain their integer and
tuple keys.

Population-dependent opacity would require absolute densities reconstructed
from all retained and excited populations and their normalization. Stage density
alone is not ground-state density. Such a nonlinear solution would make its
PECs and GCR coefficients conditional on the converged population state.
A future supplied total radiation field must use thin spontaneous rates plus
explicit absorption and stimulated emission; also reducing A for the same
internal photons would count reabsorption twice.

The closure and chord convention follow
`ADAS214 <https://www.adas.ac.uk/man/chap2-14.pdf>`_; the volume-average option
is a separate uniform-source closure. Numerical tests establish selected
invariants and transport benchmarks, not validation against measured plasmas.

.. automodule:: colradpy.escape_factors
   :members: thermal_speed, line_profile, optical_depth, population_escape_factor, line_escape_factor

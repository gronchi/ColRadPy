<p align="center">
  <img src="logo.jpg" alt="ColRadPy" width="200">
</p>

# ColRadPy

ColRadPy is a collisional-radiative (CR) solver written in Python. It reads atomic data
in the ADAS adf04 format, builds the CR matrix and solves it for level populations,
photon emissivity coefficients (PECs) and generalized collisional-radiative (GCR)
coefficients. It also solves ionization balances and energy balances.

Documentation: <https://colradpy.readthedocs.io/en/latest/>

## Features

- **Atomic rates:** electron-impact excitation from effective collision strengths, with
  Burgess–Tully extrapolation outside the file's temperatures. Ionization comes from
  the adf04 file or from ECIP. The solver also includes radiative recombination,
  three-body recombination by detailed balance, and thermal charge exchange with
  hydrogen.
- **Quasistatic solutions:**
  - metastable-resolved populations;
  - PECs;
  - the GCR coefficients SCD, ACD, QCD, XCD and CCD;
  - radiated power coefficients PLT, PRB and PRC.
- **Time-dependent solutions:** the CR equations, with or without a source, solved by
  matrix exponentials.
- **Ionization balance:** built from adf04 files (through ColRadPy) or from adf11
  files, solved time dependently or at steady state. The energy balance follows the
  charge-state temperatures as they are heated by the main ions.
- **Spectra:**
  - LS-resolved PECs split into j-resolved lines;
  - level energies shifted to NIST values;
  - Doppler and natural broadening;
  - output written as adf15 files.
- **LTE references:** Boltzmann and Saha populations.

## Installation

ColRadPy needs Python 3 with NumPy (below 2.5), SciPy and matplotlib. Clone the
repository and install it in editable mode:

```bash
git clone https://github.com/johnson-c/ColRadPy.git
cd ColRadPy
pip install -e .
```

An editable install follows changes to the clone; `pip install .` makes a regular
install. Both include the NIST level energies of `atomic/nist_energies`, which the
NIST energy shifts and line splitting use.

Optional packages:

- `h5py` and `hdfdict`, for HDF5 output (`dump_hdf5`) and HDF5 input to the
  ionization balance;
- `pymysql`, for the `colradpy.nist_read` module, which queries a local copy of the
  NIST database.

## Quick start

```python
import numpy as np
from colradpy import colradpy

be = colradpy('examples/input/cpb03_ls#be0.dat',   # adf04 file
              metas=np.array([0]),                  # metastable levels: the ground level
              temp_grid=np.linspace(1, 100, 100),   # electron temperatures (eV)
              electron_den=np.array([1e13, 4e14]))  # electron densities (cm-3)
be.solve_cr()  # quasistatic solution of the CR matrix

pecs = be.data['processed']['pecs']           # PECs (ph cm3 s-1): lines, drivers, temperatures, densities
wave = be.data['processed']['wave_air']       # wavelengths in air (nm)
levels = be.data['processed']['pec_levels']   # upper and lower level of each line
scd = be.data['processed']['scd']             # effective ionization coefficients (cm3 s-1)
```

All results are stored in the `data` dictionary, which is described in the
[documentation](https://colradpy.readthedocs.io/en/latest/). The ionization balance
is the `ionization_balance` class:

```python
from colradpy import ionization_balance
```

## Examples

The [`examples`](examples) folder has scripts for:

| Script | Shows |
|---|---|
| `run_colradpy_example.py` | a CR calculation, step by step |
| `gcr_example.py` | GCR coefficients |
| `populating_lvls_example.py` | which levels populate a level |
| `split_multiplet_example.py` | splitting PECs into j-resolved lines |
| `time_dependent_cr_example.py` | time-dependent populations |
| `time_dependent_line_ratios.py` | time-dependent line ratios |
| `ion_bal.py` | ionization balance from adf04 files |
| `ss_ion_bal_adf11.py`, `td_ion_bal_adf11.py` | steady-state and time-dependent ionization balance from adf11 files |
| `time_dependent_temperature_evolution.py` | energy balance |

Run the scripts from the `examples` folder. Most of them use the beryllium adf04
files in `examples/input`. The ionization balance examples download their data
from OpenADAS. For the energy balance example, set the path to local adf11 files in
the script.

## Documentation

The documentation is built with Sphinx and hosted on Read the Docs. It includes a
tutorial, the structure of the data dictionaries, the API reference and a theory
manual. To build it locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs/source docs/build/html
```

`make html` in `docs` does the same. The LaTeX theory manual is in
`docs/source/theory`.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Citing ColRadPy

If you use ColRadPy in your work, please cite:

> C. A. Johnson, S. D. Loch and D. A. Ennis, "ColRadPy: A Python collisional
> radiative solver", *Nuclear Materials and Energy* (2019),
> [doi:10.1016/j.nme.2019.01.013](https://doi.org/10.1016/j.nme.2019.01.013).

## Community

- Discord: <https://discord.gg/xMN8nJq>
- YouTube tutorials:
  <https://www.youtube.com/channel/UCjF3AQGUeS0Y4yLcgIO6muw/videos>

## License

ColRadPy is released under the GNU General Public License v3 (see
[`license.txt`](license.txt)). The author would be glad to hear from you if you want
to collaborate.

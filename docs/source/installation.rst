.. _installation:

Installation
============

ColRadPy needs Python 3 with NumPy (below 2.5), SciPy and matplotlib. Clone the
repository and install it in editable mode::

    git clone https://github.com/johnson-c/ColRadPy.git
    cd ColRadPy
    pip install -e .

The classes can then be imported from any directory:

.. code-block:: python

   from colradpy import colradpy, ionization_balance

Install from the clone rather than copying the package elsewhere: the NIST level
energies (``atomic/nist_energies``) are read from the repository.

Optional packages
-----------------

- ``h5py`` and ``hdfdict``, for HDF5 output (``dump_hdf5``) and HDF5 input to the
  ionization balance.
- ``pymysql``, for the ``colradpy.nist_read`` module, which queries a local copy of the
  NIST database.

Building this documentation
---------------------------

From the repository root::

    pip install -r docs/requirements.txt
    sphinx-build -b html docs/source docs/build/html

``make html`` in ``docs`` does the same. Read the Docs builds the published pages
with ``.readthedocs.yaml``, and a GitHub Actions workflow checks that every push and
pull request builds without warnings.

Running the tests
-----------------

From the repository root::

    python -m unittest discover -s tests -v

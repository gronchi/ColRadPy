"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks quasistatic radiated line power coefficients: pls = photon energy * PEC
in W cm^3 (the unit of ADAS adf11 PLT/PRB files), and its sums over the
excitation (plt), recombination (prb) and charge exchange (prc) drivers.
"""

import contextlib
import io
from pathlib import Path
import unittest

import numpy as np
from scipy import constants

from colradpy import colradpy


INPUT = Path(__file__).resolve().parents[1] / "examples" / "input"
# Photon energy per cm^-1 of transition energy, in J.
J_PER_WAVENUMBER = constants.h * constants.c * 100


def photon_energies(model):
    energy = model.data["atomic"]["energy"]
    levels = model.data["processed"]["pec_levels"]
    return np.abs(energy[levels[:, 0]] - energy[levels[:, 1]]) * J_PER_WAVENUMBER


def driven_model(paired):
    """Ground level, one excited level (A = 1 s^-1 to ground, 1e4 cm^-1 above it), one
    parent feeding it by recombination at 2 s^-1 and by charge exchange at 3 s^-1."""
    density = 1e13
    shape = (1,) if paired else (1, 1)
    matrix = np.zeros((4, 4) + shape)
    matrix[0, 1] = 1.0
    matrix[1, 1] = -1.0
    matrix[1, 2] = 2.0
    matrix[2, 2] = -2.0
    matrix[1, 3] = 3.0
    matrix[3, 3] = -3.0
    recombination = np.zeros((2, 1, 1))
    recombination[1, 0] = 2.0 / density
    model = colradpy.__new__(colradpy)
    model.data = {
        "user": {
            "temp_grid": np.array([10.0]), "dens_grid": np.array([density]),
            "temp_dens_pair": paired, "default_pop_norm": True,
            "use_ionization": False, "use_recombination": True,
            "use_recombination_three_body": False, "use_cx": True,
        },
        "atomic": {
            "energy": np.array([0.0, 1e4]), "metas": np.array([0]),
            "ion_pot": np.array([1e5]), "ion_pot_lvl": np.array([1]),
        },
        "rates": {
            "ioniz": {"ionization": np.zeros((2, 1, 1))},
            "recomb": {"recombination": recombination,
                       "recomb_three_body": np.zeros((2, 1, 1))},
            "cx": {"cx_transitions": np.array([[2, 1]]), "cx": np.zeros((2, 1, 1))},
        },
        "cr_matrix": {"cr": matrix, "A_ji": np.array([[0.0, 0.0], [1.0, 0.0]])},
    }
    model.solve_quasi_static()
    return model


class RadiatedPowerTests(unittest.TestCase):
    def test_line_power_is_photon_energy_times_pec(self):
        for paired in (False, True):
            with self.subTest(paired=paired):
                model = colradpy(str(INPUT / "cpb03_ls#be0.dat"), metas=np.array([0]),
                                 temp_grid=np.array([2., 10.]), electron_den=np.array([1e12, 1e14]),
                                 temp_dens_pair=paired)
                # Degenerate levels in bundled data produce a pre-existing
                # zero-energy wavelength warning, outside this check's scope.
                with contextlib.redirect_stdout(io.StringIO()), np.errstate(divide="ignore"):
                    model.solve_cr()
                processed = model.data["processed"]
                photon = photon_energies(model).reshape((-1, 1) + (1,) * (processed["pecs"].ndim - 2))
                np.testing.assert_allclose(processed["pls"], photon * processed["pecs"], rtol=1e-12)
                np.testing.assert_allclose(processed["plt"], processed["pls"][:, :1].sum(axis=0))
                np.testing.assert_allclose(processed["prb"], processed["pls"][:, 1:].sum(axis=0))
                self.assertEqual(processed["prc"].shape[0], 0)
                # Be I radiates of order 1e-25 W cm^3 at 10 eV (1e-18 erg cm^3 s^-1).
                self.assertTrue((1e-28 < processed["plt"].max() < 1e-24))

    def test_drivers_are_separated(self):
        for paired in (False, True):
            with self.subTest(paired=paired):
                processed = driven_model(paired).data["processed"]
                photon = 1e4 * J_PER_WAVENUMBER
                # The excited level holds 2 (recombination) and 3 (charge exchange)
                # per parent, so the line power is photon * A * N / n_e.
                np.testing.assert_allclose(processed["pls"][0, 1:].ravel(),
                                           [photon * 2 / 1e13, photon * 3 / 1e13])
                np.testing.assert_allclose(processed["prb"].ravel(), photon * 2 / 1e13)
                np.testing.assert_allclose(processed["prc"].ravel(), photon * 3 / 1e13)
                np.testing.assert_allclose(processed["plt"].ravel(), 0.0)


if __name__ == "__main__":
    unittest.main()

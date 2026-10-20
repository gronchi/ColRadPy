"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks the normalization of quasistatic populations. Excitation populations,
relative to a metastable of the ion, are normalized to the ion population by
default with one metastable; recombination and charge exchange populations are
relative to the parent populations and must not be.
"""

import contextlib
import io
from pathlib import Path
import unittest

import numpy as np

from colradpy import colradpy


INPUT = Path(__file__).resolve().parents[1] / "examples" / "input"


def one_level_model(paired, default_pop_norm=True):
    """Ground level, one excited level and one parent: the excited level decays at
    A = 1 s^-1 and is fed from the parent at n_e alpha = 2 s^-1, without excitation."""
    density = 1e13
    shape = (1,) if paired else (1, 1)
    matrix = np.zeros((3, 3) + shape)
    matrix[0, 1] = 1.0
    matrix[1, 1] = -1.0
    matrix[1, 2] = 2.0
    matrix[2, 2] = -2.0
    recombination = np.zeros((2, 1, 1))
    recombination[1, 0] = 2.0 / density
    model = colradpy.__new__(colradpy)
    model.data = {
        "user": {
            "temp_grid": np.array([10.0]), "dens_grid": np.array([density]),
            "temp_dens_pair": paired, "default_pop_norm": default_pop_norm,
            "use_ionization": False, "use_recombination": True,
            "use_recombination_three_body": False, "use_cx": False,
        },
        "atomic": {
            "energy": np.array([0.0, 1e4]), "metas": np.array([0]),
            "ion_pot": np.array([1e5]), "ion_pot_lvl": np.array([1]),
        },
        "rates": {
            "ioniz": {"ionization": np.zeros((2, 1, 1))},
            "recomb": {"recombination": recombination,
                       "recomb_three_body": np.zeros((2, 1, 1))},
        },
        "cr_matrix": {"cr": matrix, "A_ji": np.array([[0.0, 0.0], [1.0, 0.0]])},
    }
    model.solve_quasi_static()
    return model


def solve(name, metas, default_pop_norm, temperatures=(0.5, 2.0), densities=(1e12, 1e17)):
    model = colradpy(str(INPUT / name), metas=np.array(metas), temp_grid=np.array(temperatures),
                     electron_den=np.array(densities), default_pop_norm=default_pop_norm)
    # Degenerate levels in bundled data produce a pre-existing zero-energy
    # wavelength warning, outside this check's scope.
    with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
        model.solve_cr()
    return model


class PopulationNormalizationTests(unittest.TestCase):
    def test_recombination_population_relative_to_parent(self):
        # Quasistatic balance -N_exc + 2 N_parent = 0 gives N_exc / N_parent = 2.
        for paired in (False, True):
            with self.subTest(paired=paired):
                processed = one_level_model(paired).data["processed"]
                self.assertTrue(processed["driving_populations_norm"])
                np.testing.assert_allclose(processed["pops"][0, 1], 2.0)
                np.testing.assert_allclose(processed["pop_lvl"][0, 0, 1], 2.0)
                # The recombination PEC is A N_exc / (n_e N_parent).
                np.testing.assert_allclose(processed["pecs"][0, 1], 2.0 / 1e13)

    def test_only_excitation_columns_normalized(self):
        # Low temperature and high density, where recombination populations are large.
        for name in ("cpb03_ls#be0.dat", "cpb03_ls#be1.dat", "be2_adf04"):
            with self.subTest(file=name):
                conditions = dict(temperatures=(0.2, 1.0), densities=(1e12, 1e19))
                norm = solve(name, [0], True, **conditions).data["processed"]
                raw = solve(name, [0], False, **conditions).data["processed"]
                self.assertTrue(norm["driving_populations_norm"])
                self.assertFalse(raw["driving_populations_norm"])
                self.assertGreater(raw["pops"][:, 1:].sum(axis=0).max(), 1.0)
                for key in ("pops", "pecs"):
                    np.testing.assert_array_equal(norm[key][:, 1:], raw[key][:, 1:])
                np.testing.assert_array_equal(norm["pop_lvl"][:, :, 1:], raw["pop_lvl"][:, :, 1:])
                scale = 1 + raw["pops"][:, 0].sum(axis=0)
                np.testing.assert_allclose(norm["pops"][:, 0], raw["pops"][:, 0] / scale)
                np.testing.assert_allclose(norm["pecs"][:, 0], raw["pecs"][:, 0] / scale)
                np.testing.assert_allclose(norm["acd"], raw["acd"])

    def test_ionization_normalized_by_metastable_columns(self):
        # With metastables [0, 2] the driver columns are 0 and 1, not the levels.
        model = solve("cpb03_ls#be1.dat", [0, 2], False)
        processed = model.data["processed"]
        self.assertTrue(processed["driving_populations_norm"])
        ionization = model.data["rates"]["ioniz"]["ionization"]
        excited = processed["excited_levels"]
        metastable_fraction = 1 / (1 + processed["pops_no_norm"][:, 0:2].sum(axis=0))
        expected = np.einsum("ipk,imkl->mpkl", ionization[excited], processed["F"]) \
            + np.einsum("ipk,ikl->ipkl", ionization[[0, 2]], metastable_fraction)
        np.testing.assert_allclose(processed["scd"], expected)


if __name__ == "__main__":
    unittest.main()

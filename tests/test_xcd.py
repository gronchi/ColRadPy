"""Check parent coupling against elimination of the full rate matrix."""

from pathlib import Path
import unittest

import numpy as np

from colradpy import colradpy
from colradpy.ionization_balance_class import ionization_balance


def make_model(paired, metas, parents=3, radiative=True, three_body=True,
               use_ionization=True):
    """Four bound levels with distinct rates into each parent and bound level."""
    temperatures = np.array([2.0, 8.0])
    densities = np.array([1e10, 3e11]) if paired else np.array([1e10, 3e11, 8e12])
    shape = (2,) if paired else (2, 3)
    level, parent, temperature = np.indices((4, parents, 2))
    ionization = (1 + level + 2 * parent + temperature) * 1e-11
    recombination = (1 + 3 * level + parent + temperature) * 1e-12
    recomb_three_body = (1 + level + 3 * parent + temperature) * 1e-25
    a_values = np.tril(np.arange(1, 17).reshape(4, 4), -1).astype(float)
    matrix = np.zeros((4 + parents, 4 + parents) + shape)
    for index in np.ndindex(shape):
        t = index[0]
        density = densities[t if paired else index[1]]
        rates = np.zeros((4 + parents, 4 + parents))
        rates[:4, :4] = a_values.T
        for upper in range(1, 4):
            for lower in range(upper):
                rates[upper, lower] = density * (upper + lower + 1) * 1e-12
        if use_ionization:
            rates[4:, :4] = density * ionization[:, :, t].T
        if radiative:
            rates[:4, 4:] += density * recombination[:, :, t]
        if three_body:
            rates[:4, 4:] += density**2 * recomb_three_body[:, :, t]
        rates[np.diag_indices(4 + parents)] = -rates.sum(axis=0)
        matrix[(slice(None), slice(None)) + index] = rates

    model = colradpy.__new__(colradpy)
    model.data = {
        "user": {
            "temp_grid": temperatures, "dens_grid": densities,
            "temp_dens_pair": paired, "default_pop_norm": True,
            "use_ionization": use_ionization, "use_recombination": radiative,
            "use_recombination_three_body": three_body, "use_cx": False,
        },
        "atomic": {
            "energy": np.arange(4) * 1e4, "metas": np.array(metas),
            "ion_pot": np.arange(1, parents + 1) * 1e5,
            "ion_pot_lvl": np.arange(1, parents + 1),
        },
        "rates": {
            "ioniz": {"ionization": ionization},
            "recomb": {
                "recombination": recombination,
                "recomb_three_body": recomb_three_body,
            },
        },
        "cr_matrix": {"cr": matrix, "A_ji": a_values},
    }
    model.solve_quasi_static()
    return model


def reduced_coefficients(model):
    """Eliminate excited levels using a linear solve, then divide rates by n_e."""
    bound = len(model.data["atomic"]["energy"])
    metas = model.data["atomic"]["metas"]
    matrix = model.data["cr_matrix"]["cr"]
    excited = np.setdiff1d(np.arange(bound), metas)
    retained = np.r_[metas, np.arange(bound, matrix.shape[0])]
    coefficients = np.empty((len(retained), len(retained)) + matrix.shape[2:])
    for index in np.ndindex(matrix.shape[2:]):
        rates = matrix[(slice(None), slice(None)) + index]
        reduced = rates[np.ix_(retained, retained)] - (
            rates[np.ix_(retained, excited)] @ np.linalg.solve(
                rates[np.ix_(excited, excited)], rates[np.ix_(excited, retained)]
            )
        )
        density_index = index[0] if model.data["user"]["temp_dens_pair"] else index[1]
        coefficients[(slice(None), slice(None)) + index] = (
            reduced / model.data["user"]["dens_grid"][density_index]
        )
    return coefficients


class XcdTests(unittest.TestCase):
    def assert_xcd_matches_reduction(self, model):
        metastables = len(model.data["atomic"]["metas"])
        parent_block = reduced_coefficients(model)[metastables:, metastables:]
        # XCD stores [source parent, destination parent]; rate matrices use
        # [destination, source]. Its diagonal excludes returns to the same parent.
        expected = parent_block.swapaxes(0, 1).copy()
        expected[np.diag_indices(len(expected))] = 0.0
        np.testing.assert_allclose(
            model.data["processed"]["xcd"], expected, rtol=1e-12, atol=0.0
        )

    def test_xcd_matches_full_matrix_reduction(self):
        for paired in (False, True):
            for metas in ([0, 1], [0, 2]):
                for parents in (1, 2, 3):
                    for radiative, three_body in ((True, False), (False, True), (True, True)):
                        with self.subTest(paired=paired, metas=metas, parents=parents,
                                          radiative=radiative, three_body=three_body):
                            model = make_model(paired, metas, parents, radiative, three_body)
                            self.assert_xcd_matches_reduction(model)

    def test_ionization_balance_matches_full_matrix_reduction(self):
        for paired in (False, True):
            with self.subTest(paired=paired):
                model = make_model(paired, [0, 2])
                balance = ionization_balance.__new__(ionization_balance)
                balance.data = {
                    "user": dict(model.data["user"], use_cx=[False]),
                    "cr_data": {"gcrs": {"0": model.data["processed"]}},
                }
                balance.populate_ion_matrix()
                np.testing.assert_allclose(
                    balance.data["ion_matrix"], reduced_coefficients(model),
                    rtol=1e-12, atol=1e-26,
                )
                matrix = balance.data["ion_matrix"]
                np.testing.assert_array_less(
                    np.abs(matrix.sum(axis=0)), 1e-14 * np.abs(matrix).sum(axis=0)
                )

    def test_disabled_ionization_has_no_parent_cross_coupling(self):
        for paired in (False, True):
            for cached_rates in (False, True):
                with self.subTest(paired=paired, cached_rates=cached_rates):
                    model = make_model(paired, [0, 2], use_ionization=False)
                    if not cached_rates:
                        del model.data["rates"]["ioniz"]["ionization"]
                        model.solve_quasi_static()
                    self.assertEqual(np.count_nonzero(model.data["processed"]["xcd"]), 0)
                    self.assert_xcd_matches_reduction(model)

    def test_bundled_be1_with_contiguous_and_noncontiguous_metastables(self):
        path = Path(__file__).resolve().parents[1] / "examples/input/cpb03_ls#be1.dat"
        for paired in (False, True):
            for metas in ([0, 1], [0, 2]):
                with self.subTest(paired=paired, metas=metas):
                    model = colradpy(
                        str(path), np.array(metas), np.array([5.0, 20.0]),
                        np.array([1e12, 1e14]), temp_dens_pair=paired,
                    )
                    model.solve_cr()
                    self.assert_xcd_matches_reduction(model)
                    self.assertTrue(np.any(model.data["processed"]["xcd"] > 0))

    def test_projected_parent_coupling_direction(self):
        # The projection gives the recombination from a parent through the high n-shells that
        # ionizes to another parent as [to, from]; it was added to XCD, which is [from, to], without
        # transposing, so the O0 coupling from O+ 2D down to 4S went from 4S up to 2D. A coupling
        # from parent 1 to parent 0 must change XCD[1, 0] only.
        path = Path(__file__).resolve().parents[1] / "examples/input/cpb03_ls#be1.dat"
        temps, dens = np.array([5.0, 20.0]), np.array([1e12, 1e14])
        model = colradpy(str(path), np.array([0]), temps, dens)
        model.solve_cr()
        plain = model.data["processed"]["xcd"].copy()
        levels, parents = len(model.data["atomic"]["energy"]), len(model.data["atomic"]["ion_pot"])
        self.assertEqual(parents, 2)
        grid = (len(temps), len(dens))
        coupling = np.zeros((parents, parents) + grid)
        coupling[0, 1] = 3e-9                                 # [to, from]: from parent 1 to parent 0
        model.data["user"]["projection"] = ("adf17", "adf18")
        model.data["rates"]["projection"] = {"couplings": np.zeros((levels, levels) + grid),
                                             "ionization": np.zeros((levels, parents) + grid),
                                             "recombination": np.zeros((levels, parents) + grid),
                                             "parent_coupling": coupling}
        model.populate_cr_matrix()
        model.solve_quasi_static()
        change = model.data["processed"]["xcd"] - plain
        np.testing.assert_allclose(change[1, 0], 3e-9)
        np.testing.assert_allclose(change[0, 1], 0, atol=1e-25)


if __name__ == "__main__":
    unittest.main()

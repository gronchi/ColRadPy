"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks the time grids of the ionization balance: n_e*tau values become solution
times n_e*tau/n_e for each density, and the time dependent solutions agree with
the matrix exponential of the ionization balance matrix.
"""

import contextlib
import io
from pathlib import Path
import unittest

import numpy as np
from scipy.linalg import expm

from colradpy import colradpy
from colradpy.ionization_balance_class import ionization_balance


INPUT = Path(__file__).resolve().parents[1] / "examples" / "input"
NE_TAU = np.array([1e10, 1e11, 1e12])  # cm-3 s


def be0_balance(densities, ne_tau=np.array([-1])):
    """One Be0 stage and its Be+ parent at 10 eV."""
    with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
        balance = ionization_balance([str(INPUT / "cpb03_ls#be0.dat")], [np.array([0])],
                                     np.array([10.]), np.asarray(densities), ne_tau=ne_tau)
        balance.populate_ion_matrix()
    return balance


def expected_populations(balance, times):
    """expm(A n_e t) n0 for every density, times with shape (densities, times)."""
    matrix = balance.data["ion_matrix"]
    dens = balance.data["user"]["dens_grid"]
    n0 = balance.default_abundances()
    out = np.zeros((len(n0), times.shape[1], 1, len(dens)))
    for d in range(len(dens)):
        for a, t in enumerate(times[d]):
            out[:, a, 0, d] = expm(matrix[:, :, 0, d] * dens[d] * t) @ n0
    return out


class NeTauTests(unittest.TestCase):
    def test_times_for_each_density(self):
        balance = be0_balance([1e12, 1e13], NE_TAU)
        times = balance.data["user"]["soln_times"]
        self.assertEqual(times.shape, (2, 3))
        np.testing.assert_allclose(times, NE_TAU[None, :] / np.array([1e12, 1e13])[:, None])
        # One density gives one set of times.
        single = be0_balance([1e13], NE_TAU)
        np.testing.assert_allclose(single.data["user"]["soln_times"], NE_TAU / 1e13)

    def test_solution_at_ne_tau(self):
        # Unequal numbers of ages and densities failed; equal numbers paired the
        # ages with the wrong densities.
        for densities in ([1e12, 1e13], [1e12, 1e13, 1e14], [1e13]):
            with self.subTest(densities=densities):
                balance = be0_balance(densities, NE_TAU)
                with np.errstate(all="ignore"):
                    balance.solve_no_source()
                pops = balance.data["processed"]["pops_td"]
                times = np.atleast_2d(balance.ne_tau_times(NE_TAU))
                if len(densities) == 1:
                    times = times.reshape(1, -1)
                self.assertEqual(pops.shape, (2, 3, 1, len(densities)))
                np.testing.assert_allclose(pops, expected_populations(balance, times),
                                           rtol=1e-8, atol=1e-12)

    def test_solve_source_uses_its_ne_tau(self):
        # The ne_tau argument was ignored unless ne_tau was given to the constructor.
        balance = be0_balance([1e12, 1e13])
        with np.errstate(all="ignore"):
            balance.solve_source(ne_tau=NE_TAU)
        np.testing.assert_allclose(balance.data["user"]["soln_times"], balance.ne_tau_times(NE_TAU))
        pops = balance.data["processed"]["pops_td_source"]
        times = balance.ne_tau_times(NE_TAU)
        # Without a source the solution is the one without source.
        np.testing.assert_allclose(pops, expected_populations(balance, times), rtol=1e-8, atol=1e-12)

    def test_defaults_without_adf04_inputs(self):
        # ADF11 and HDF5 balances have no metastable lists or default abundances;
        # solve_source read keys only the adf04 branch set.
        balance = be0_balance([1e12, 1e13])
        for key in ("metas", "init_abund"):
            del balance.data["user"][key]
        balance.data["user"]["init_abund"] = np.array([])
        np.testing.assert_array_equal(balance.default_abundances(), [1., 0.])
        with np.errstate(all="ignore"):
            balance.solve_source(td_t=np.array([1e-6, 1e-4]))
            balance.solve_no_source(td_t=np.array([1e-6, 1e-4]))
        np.testing.assert_allclose(balance.data["processed"]["pops_td_source"],
                                   balance.data["processed"]["pops_td"], rtol=1e-10, atol=1e-14)


class MetastableInputTests(unittest.TestCase):
    def make_balance(self, metas):
        files = [str(INPUT / "cpb03_ls#be0.dat"), str(INPUT / "cpb03_ls#be1.dat")]
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
            balance = ionization_balance(files, metas, np.array([10.]), np.array([1e13]))
            balance.populate_ion_matrix()
        return balance

    def test_flat_selection_infers_later_stage_and_sizes_abundances(self):
        for metas in (np.array([0]), [0], np.array([0, 1])):
            with self.subTest(metas=metas):
                balance = self.make_balance(metas)
                selections = balance.data["user"]["metas"]
                np.testing.assert_array_equal(selections[0], metas)
                np.testing.assert_array_equal(selections[1],
                    np.arange(balance.data["cr_data"]["gcrs"]["0"]["scd"].shape[1]))
                self.assertEqual(len(balance.default_abundances()),
                                 balance.data["ion_matrix"].shape[0])
                np.testing.assert_array_equal(balance.default_abundances()[1:], 0)

    def test_unequal_per_stage_selections(self):
        balance = self.make_balance([np.array([0, 1]), np.array([0])])
        selections = balance.data["user"]["metas"]
        self.assertIsInstance(selections, list)
        np.testing.assert_array_equal(selections[0], [0, 1])
        np.testing.assert_array_equal(selections[1], [0])
        self.assertEqual(len(balance.default_abundances()),
                         balance.data["ion_matrix"].shape[0])

    def test_invalid_selections(self):
        # Be I ionizes to one Be+ parent, so Be+ can only have one metastable.
        with self.assertRaisesRegex(ValueError, "stage 1 has 2 metastables but stage 0 ionizes to 1"):
            self.make_balance([np.array([0]), np.array([0, 2])])
        with self.assertRaisesRegex(ValueError, "one level array per adf04 file"):
            self.make_balance([np.array([0])])
        # The ground level must be a metastable (upstream issue #37).
        with self.assertRaisesRegex(ValueError, "ground level 0"):
            self.make_balance([np.array([1]), np.array([0])])


class ColradpyMetastableTests(unittest.TestCase):
    def test_ground_level_required(self):
        path = str(INPUT / "cpb03_ls#be0.dat")
        for metas in ([1], [2, 3]):
            with self.subTest(metas=metas), self.assertRaisesRegex(ValueError, "ground level 0"):
                colradpy(path, metas=np.array(metas), temp_grid=np.array([10.]), electron_den=np.array([1e13]))
        for metas in ([0, 0], [0, 99], [0, -1]):
            with self.subTest(metas=metas), self.assertRaisesRegex(ValueError, "different levels"):
                colradpy(path, metas=np.array(metas), temp_grid=np.array([10.]), electron_den=np.array([1e13]))
        # The ground with other metastables in any order, or none for reading only.
        for metas in ([2, 0], [0]):
            colradpy(path, metas=np.array(metas), temp_grid=np.array([10.]), electron_den=np.array([1e13]))
        colradpy(path, temp_grid=np.array([10.]), electron_den=np.array([1e13]))


if __name__ == "__main__":
    unittest.main()

"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks the time grids of the ionization balance: n_e*tau values become solution
times n_e*tau/n_e for each density, and the time dependent solutions agree with
the matrix exponential of the ionization balance matrix.
"""

import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
from scipy.linalg import expm

from colradpy import colradpy
from colradpy.colradpy_class import parent_levels
from colradpy.ionization_balance_class import ionization_balance, metastable_fractions


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

    def test_selection_for_the_final_stage(self):
        # One array per charge state, the final stage included (examples/ion_bal.py), was
        # rejected; the final stage has no adf04 file and takes the parents of the last one.
        per_file = self.make_balance([np.array([0]), np.array([0])])
        per_stage = self.make_balance([np.array([0]), np.array([0]), np.array([0, 1])])
        np.testing.assert_array_equal(per_stage.data["ion_matrix"], per_file.data["ion_matrix"])
        np.testing.assert_array_equal(per_stage.data["user"]["metas"][2], [0, 1])
        # Be II ionizes to two Be2+ parents
        with self.assertRaisesRegex(ValueError, "final stage has 1 metastables but stage 1 ionizes to 2"):
            self.make_balance([np.array([0]), np.array([0]), np.array([0])])

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


class ParentLevelTests(unittest.TestCase):
    """The metastables of a later stage are the parents on the first line of the stage before,
    which need not be the lowest levels of the next ion."""

    def test_neon_parents(self):
        # Ne0 ionizes to Ne+ 2p5 2P and 2p4 3s 4P (219178.6 cm-1 above it); level 1 of the Ne+
        # file is 2s2p6 2S, which decays in a nanosecond (OpenADAS cop98#10_ls#ne0/ne1).
        energy, S, L, w = [0., 216787.5, 219179.3, 224031.0], [2, 2, 4, 2], [1, 0, 1, 1], [2.5, 0.5, 5.5, 2.5]
        np.testing.assert_array_equal(
            parent_levels([173929.7, 393108.3], ["2P", "4P"], energy, S, L, w), [0, 2])
        # IC terms also match J
        np.testing.assert_array_equal(
            parent_levels([100., 2100.], ["2P1.5", "2P0.5"], [0., 2000., 2100.], [2, 2, 2], [1, 1, 1],
                          [1.5, 1.5, 0.5]), [0, 2])
        with self.assertRaisesRegex(ValueError, "no level of the next ion has the term 3D"):
            parent_levels([0., 5000.], ["2P", "3D"], energy, S, L, w)
        with self.assertRaisesRegex(ValueError, "matches no level"):
            parent_levels([0., 100000.], ["2P", "4P"], energy, S, L, w)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # Be II whose second parent is the Be2+ 1s2s 1S (level 2 of be2_adf04) instead of 3S
        lines = (INPUT / "cpb03_ls#be1.dat").read_text().splitlines()
        lines[0] = lines[0].replace("1103374.0(3S)", "1128050.0(1S)")
        self.be1 = os.path.join(tmp.name, "be1_1s_parent.dat")
        with open(self.be1, "w") as f:
            f.write("\n".join(lines) + "\n")
        self.files = [str(INPUT / "cpb03_ls#be0.dat"), self.be1, str(INPUT / "be2_adf04")]

    def balance(self, metas):
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
            return ionization_balance(self.files, metas, np.array([10.]), np.array([1e13]))

    def test_inferred_metastables_are_the_parents(self):
        # The inferred metastables were the lowest levels, [0, 1] (the 1s2s 3S) here.
        balance = self.balance(np.array([0]))
        np.testing.assert_array_equal(balance.data["user"]["metas"][2], [0, 2])

    def test_explicit_metastables_that_are_not_the_parents(self):
        with self.assertWarnsRegex(UserWarning, r"metastables \[0 1\] of stage 2 are not the parents \[0 2\]"):
            self.balance([np.array([0]), np.array([0]), np.array([0, 1])])
        with warnings.catch_warnings():
            warnings.filterwarnings("error", "the metastables", UserWarning)
            self.balance([np.array([0]), np.array([0]), np.array([0, 2])])


class PackageNameTests(unittest.TestCase):
    def test_ionization_balance_is_the_class(self):
        # An older copy of the class lived in colradpy/ionization_balance.py; importing it as
        # colradpy.ionization_balance replaced the package attribute, so that
        # 'from colradpy import ionization_balance' then gave the module instead of the class.
        import importlib.util
        import colradpy as package
        from colradpy import ionization_balance_class
        self.assertIsNone(importlib.util.find_spec("colradpy.ionization_balance"))
        self.assertIs(package.ionization_balance, ionization_balance_class.ionization_balance)


class InterpolationOptionTests(unittest.TestCase):
    def test_rate_interpolations_reach_the_stages(self):
        # The ionization balance had no rate_interp_* options, so its stages always used the
        # default interpolations of colradpy.
        files = [str(INPUT / "cpb03_ls#be0.dat"), str(INPUT / "cpb03_ls#be1.dat")]
        temps, dens = np.array([3., 7., 30.]), np.array([1e13])
        options = dict(rate_interp_ion="log_slinear", rate_interp_recomb="slinear",
                       rate_interp_col="slinear")

        def balance(**kwargs):
            with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return ionization_balance(files, [np.array([0]), np.array([0])], temps, dens, **kwargs)

        chosen, default = balance(**options), balance()
        direct = colradpy(files[0], metas=np.array([0]), temp_grid=temps, electron_den=dens, **options)
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            direct.solve_cr()
        for c in ("scd", "acd", "qcd"):
            np.testing.assert_array_equal(chosen.data["cr_data"]["gcrs"]["0"][c], direct.data["processed"][c])
        self.assertFalse(np.allclose(chosen.data["cr_data"]["gcrs"]["0"]["scd"],
                                     default.data["cr_data"]["gcrs"]["0"]["scd"], rtol=1e-3, atol=0))
        # one interpolation for each file
        per_file = balance(rate_interp_col=["slinear", "log_slinear"])
        np.testing.assert_array_equal(per_file.data["cr_data"]["gcrs"]["1"]["scd"],
                                      default.data["cr_data"]["gcrs"]["1"]["scd"])
        with self.assertRaisesRegex(ValueError, "rate_interp_col must be one string or one for each"):
            balance(rate_interp_col=["slinear"])


class UseMetTests(unittest.TestCase):
    # use_met collapses every stage to one population as ADAS404 makes unresolved ADF11 files.
    BE = [str(INPUT / f) for f in ("cpb03_ls#be0.dat", "cpb03_ls#be1.dat", "be2_adf04", "be3_adf04")]
    TEMPS, DENS = np.array([2., 10., 50.]), np.array([1e12, 1e14])

    def balance(self, files, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            balance = ionization_balance(files, np.array([0, 1]), self.TEMPS, self.DENS, **kwargs)
            balance.populate_ion_matrix()
            balance.solve_time_independent()
        return balance

    def test_metastable_fractions(self):
        rng = np.random.default_rng(1)
        # two metastables: N1/N0 = Q(0->1) / (Q(1->0) + S1)
        qcd = rng.uniform(1e-9, 1e-7, (2, 2, 3, 2))
        scd = rng.uniform(1e-9, 1e-7, (2, 2, 3, 2))
        f = metastable_fractions(qcd, scd)
        np.testing.assert_allclose(f[1] / f[0], qcd[0, 1] / (qcd[1, 0] + scd[1].sum(axis=0)), rtol=1e-12)
        np.testing.assert_allclose(f.sum(axis=0), 1, rtol=1e-12)
        # four metastables: the fractions are the steady state of the gains and losses
        qcd = rng.uniform(1e-9, 1e-7, (4, 4, 3))
        scd = rng.uniform(1e-9, 1e-7, (4, 3, 3))
        f = metastable_fractions(qcd, scd)
        off = qcd * (1 - np.eye(4))[:, :, None]
        rate = np.einsum("sr...,s...->r...", off, f) - (off.sum(axis=1) + scd.sum(axis=1)) * f
        np.testing.assert_allclose(rate[1:], 0, atol=1e-12 * np.abs(off).max())
        self.assertTrue((f >= 0).all())
        np.testing.assert_array_equal(metastable_fractions(qcd[:1, :1], scd[:1]), np.ones((1, 3)))

    def test_collapse_of_the_be_stages(self):
        resolved, collapsed = self.balance(self.BE), self.balance(self.BE, use_met=True)
        r, c = resolved.data["cr_data"]["gcrs"], collapsed.data["cr_data"]["gcrs"]
        # the default is the metastable resolved balance
        self.assertNotIn("met", r["0"])
        self.assertEqual(r["0"]["qcd"].shape[:2], (2, 2))
        np.testing.assert_array_equal(collapsed.data["cr_data"]["gcrs_resolved"]["2"]["acd"], r["2"]["acd"])
        # one population per stage: Be0 (2 metastables), Be+, Be2+ (2), Be3+ and Be4+
        self.assertEqual(collapsed.data["ion_matrix"].shape[:2], (5, 5))
        np.testing.assert_allclose(collapsed.data["ion_matrix"].sum(axis=0), 0,
                                   atol=1e-12 * np.abs(collapsed.data["ion_matrix"]).max())
        np.testing.assert_allclose(collapsed.data["processed"]["pops_ss"].sum(axis=0), 1, rtol=1e-10)
        for i in range(4):
            f = metastable_fractions(r[str(i)]["qcd"], r[str(i)]["scd"])
            f_plus = (metastable_fractions(r[str(i + 1)]["qcd"], r[str(i + 1)]["scd"]) if i < 3
                      else np.ones((1,) + f.shape[1:]))
            np.testing.assert_allclose(c[str(i)]["met"], f)
            np.testing.assert_allclose(c[str(i)]["scd"][0, 0], (f * r[str(i)]["scd"].sum(axis=1)).sum(axis=0))
            np.testing.assert_allclose(c[str(i)]["plt"][0], (f * r[str(i)]["plt"]).sum(axis=0))
            # ACD weighted by the fractions of the recombining stage (Be2+ for Be+)
            np.testing.assert_allclose(c[str(i)]["acd"][0, 0], (f_plus * r[str(i)]["acd"].sum(axis=0)).sum(axis=0))
            np.testing.assert_array_equal(c[str(i)]["qcd"], 0)
            np.testing.assert_array_equal(c[str(i)]["xcd"], 0)
        # Be0 2s2p 3P is populated, Be2+ 1s2s 3S almost not at these temperatures
        self.assertGreater(c["0"]["met"][1].min(), 0.02)
        self.assertLess(c["2"]["met"][1].max(), 0.05)

    def test_final_stage_and_inputs(self):
        # Be2+ (two metastables) ends a Be0-Be+ balance without its adf04 file
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"), \
                self.assertWarnsRegex(UserWarning, "final stage has 2 metastables"):
            ionization_balance(self.BE[:2], np.array([0, 1]), self.TEMPS, self.DENS, use_met=True)
        with self.assertRaisesRegex(ValueError, "use_met collapses the coefficients made from adf04"):
            ionization_balance([str(INPUT / "Open-ADAS" / "adas#26" / "scd89_fe.dat")], np.array([0]),
                               self.TEMPS, self.DENS, adf11_files=True, use_met=True)


if __name__ == "__main__":
    unittest.main()

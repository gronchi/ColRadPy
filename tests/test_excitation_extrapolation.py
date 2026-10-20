"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks electron collision strengths and reduced ionization rates outside the
temperatures calculated in the adf04 file: Burgess-Tully extrapolation of all
four transition types above them, extrapolation towards the finite T = 0 limit
below them, and no negative rates.
"""

import contextlib
import io
from pathlib import Path
import unittest

import numpy as np

from colradpy import colradpy
from colradpy.burgess_tully_rates import burgess_tully_rates, type2_xconvert


INPUT = Path(__file__).resolve().parents[1] / "examples" / "input"
FILES = ("cpb03_ls#be0.dat", "cpb03_ls#be1.dat", "be2_adf04", "be3_adf04")


def file_temperatures(name):
    model = colradpy(str(INPUT / name), metas=np.array([0]), temp_grid=np.array([1.]),
                     electron_den=np.array([1e13]))
    return model.data["input_file"]["temp_grid"] / 11604.5


def solve(name, temperatures, rates_only=False):
    model = colradpy(str(INPUT / name), metas=np.array([0]), temp_grid=np.asarray(temperatures),
                     electron_den=np.array([1e13]))
    # Degenerate levels in bundled data produce a pre-existing zero-energy
    # wavelength warning; extreme temperatures overflow Boltzmann factors.
    with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
        if rates_only:
            model.make_ioniz_from_reduced_ionizrates()
            model.make_electron_excitation_rates()
        else:
            model.solve_cr()
    return model


class ExcitationExtrapolationTests(unittest.TestCase):
    def test_no_negative_collision_strengths(self):
        for name in FILES:
            with self.subTest(file=name):
                calculated = file_temperatures(name)
                temps = np.concatenate([[1e-3, 1e-2], calculated[:1] * (1 - 1e-9), calculated,
                                        calculated[-1:] * (1 + 1e-9), [1e3, 1e4]])
                model = solve(name, temps, rates_only=True)
                upsilon = model.data["rates"]["excit"]["col_excit_interp"]
                tabulated = model.data["rates"]["excit"]["col_excit"]
                self.assertTrue((upsilon > 0).all())
                # Tabulated values are kept, and the extrapolations join them continuously.
                np.testing.assert_allclose(upsilon[:, 3:3 + len(calculated)], tabulated, rtol=1e-10)
                np.testing.assert_allclose(upsilon[:, 2], tabulated[:, 0], rtol=1e-6)
                np.testing.assert_allclose(upsilon[:, 3 + len(calculated)], tabulated[:, -1], rtol=1e-6)

    def test_low_temperature_burgess_tully(self):
        calculated = file_temperatures("cpb03_ls#be0.dat")
        low = np.array([1e-3, 1e-2, 0.05])
        model = solve("cpb03_ls#be0.dat", np.concatenate([low, calculated]), rates_only=True)
        rates = model.data["rates"]
        bt = rates["burg_tully"]
        upsilon = rates["excit"]["col_excit_interp"]
        tabulated = rates["excit"]["col_excit"]
        np.testing.assert_array_equal(bt["extrap_temp_inds_low"], [0, 1, 2])
        for j in range(4):
            constant = bt["ind_arrs"][j][bt["low_constant_inds"][j]]
            np.testing.assert_array_equal(upsilon[constant, :3],
                                          np.repeat(tabulated[constant, :1], 3, axis=1))
        # Type 2 collision strengths are y and linear in x below the calculated temperatures.
        inds = np.setdiff1d(np.arange(len(bt["ind_arrs"][1])), bt["low_constant_inds"][1])
        self.assertGreater(len(inds), 10)
        transitions = rates["excit"]["col_transitions"][bt["ind_arrs"][1][inds]] - 1
        energy = model.data["atomic"]["energy"]
        energies = np.array([energy[transitions[:, 0]], energy[transitions[:, 1]]])
        x = type2_xconvert(np.concatenate([low, calculated[:2]]) * 11604.5, energies)
        y = upsilon[bt["ind_arrs"][1][inds]][:, [0, 1, 2, 3, 4]]
        slope = (y[:, 4] - y[:, 3]) / (x[:, 4] - x[:, 3])
        np.testing.assert_allclose(y[:, :3], y[:, [3]] + slope[:, None] * (x[:, :3] - x[:, [3]]),
                                   rtol=1e-8)

    def test_type_4_above_calculated_temperatures(self):
        model = solve("be2_adf04", [10., 1e3, 1e4], rates_only=True)
        bt = model.data["rates"]["burg_tully"]
        self.assertEqual(len(bt["ind_arrs"][3]), 2)
        upsilon = model.data["rates"]["excit"]["col_excit_interp"]
        np.testing.assert_array_equal(upsilon[np.ix_(bt["ind_arrs"][3], bt["extrap_temp_inds_hi"])],
                                      bt["excit_extrap_lin"][3])
        self.assertTrue((upsilon[bt["ind_arrs"][3]] > 0).all())

    def test_linear_extrapolations_keep_one_entry_per_type(self):
        # Infinite energy points for types 1 and 2 only, so just types 3 and 4 need
        # the linear extrapolation from the last two calculated points.
        model = solve("be2_adf04", [10., 1e4], rates_only=True)
        rates, atomic = model.data["rates"], model.data["atomic"]
        types = model.data["rates"]["burg_tully"]["ind_arrs"]
        inf_engy = np.zeros(len(rates["excit"]["col_transitions"]))
        inf_engy[np.concatenate(types[:2])] = 1.0
        # A zero infinite energy point (log of 0) marks the transitions needing it.
        with np.errstate(divide="ignore", invalid="ignore"):
            bt = burgess_tully_rates(np.array([10., 1e4]), model.data["input_file"]["temp_grid"],
                                     rates["excit"]["col_transitions"], rates["excit"]["col_excit"],
                                     atomic["energy"], atomic["w"], rates["a_val"], atomic["S"],
                                     atomic["L"], inf_engy)["burg_tully"]
        self.assertEqual([len(z) for z in bt["zero_inds"]], [0, 0, 88, 2])
        self.assertEqual(len(bt["excit_extrap_lin"]), 4)
        self.assertEqual([np.shape(e) for e in bt["excit_extrap_lin"]], [(0,), (0,), (88, 1), (2, 1)])


class IonizationExtrapolationTests(unittest.TestCase):
    def test_power_law_below_calculated_temperatures(self):
        calculated = file_temperatures("cpb03_ls#be0.dat")
        low = np.array([1e-3, 1e-2, 0.05])
        model = solve("cpb03_ls#be0.dat", np.concatenate([low, calculated]), rates_only=True)
        ioniz = model.data["rates"]["ioniz"]
        reduced = ioniz["ion_excit"]
        levels, parents = (ioniz["ion_transitions"] - 1).T
        grid = ioniz["ionization_reduced"][levels, parents]
        np.testing.assert_allclose(grid[:, 3:], reduced, rtol=1e-12)
        power = np.log(reduced[:, 1] / reduced[:, 0]) / np.log(calculated[1] / calculated[0])
        # The ground state rises with temperature; level 2 falls and is held constant.
        self.assertGreater(power[0], 0)
        self.assertLess(power[1], 0)
        np.testing.assert_allclose(grid[0, :3], reduced[0, 0] * (low / calculated[0])**power[0])
        np.testing.assert_allclose(grid[1, :3], reduced[1, 0])

    def test_low_temperature_solution(self):
        model = solve("cpb03_ls#be0.dat", [0.01, 0.03, 0.1])
        processed = model.data["processed"]
        for key in ("pops", "pecs", "scd", "acd"):
            self.assertTrue(np.isfinite(processed[key]).all(), key)
        self.assertTrue((processed["pops"] >= 0).all())
        self.assertTrue((processed["pecs"] >= 0).all())


if __name__ == "__main__":
    unittest.main()

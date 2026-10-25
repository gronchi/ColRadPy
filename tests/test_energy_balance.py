"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks the plasma parameters the energy balance derives from its inputs, which
are given in cm^-3 while the collision formulas work in SI units.
"""

import contextlib
import io
import runpy
import unittest
import warnings

import numpy as np
from scipy import constants

from colradpy.energy_balance import coulomb_logarithm, energy_balance
from colradpy.solve_matrix_exponential import solve_matrix_exponential


class StubIonizationBalance:
    """Neutral, singly and doubly charged states with fixed rates (s^-1)."""

    def __init__(self, electron_dens, times):
        self.rates = np.array([[-2.0, 1.0, 0.0], [2.0, -2.0, 1.0], [0.0, 1.0, -1.0]])
        self.data = {"user": {
            "init_abund": np.array([1.0, 0.0, 0.0]),
            "soln_times": times,
            "dens_grid": np.array([electron_dens]),
        }}

    def populate_ion_matrix(self):
        # ionization_balance stores rate coefficients; solve multiplies by n_e.
        density = self.data["user"]["dens_grid"][0]
        self.data["ion_matrix"] = (self.rates / density)[:, :, None, None]

    def solve_no_source(self, n0, td_t):
        pops, eigenvalues, eigenvectors = solve_matrix_exponential(
            self.rates[:, :, None, None], n0, td_t)
        self.data["processed"] = {
            "pops_td": pops, "eigen_val": eigenvalues, "eigen_vec": eigenvectors,
        }


def make_balance(electron_temp, electron_dens):
    times = np.array([1e-3, 1e-2, 1e-1])
    balance = energy_balance.__new__(energy_balance)
    balance.ion_balance = StubIonizationBalance(electron_dens, times)
    balance.data = {"user": {
        "mass": 12.0, "polarizability": 12.0,
        "electron_temp": electron_temp, "electron_dens": electron_dens,
        "ion_temp": electron_temp, "ion_dens": electron_dens, "ion_mass": 2.0,
        "ion_charge_number": 1, "init_temp": np.array([1.0, 1.0, 1.0]),
    }}
    return balance


class EnergyBalanceTests(unittest.TestCase):
    def test_debye_length_uses_density_in_si_units(self):
        electron_temp, electron_dens = 10.0, 1e13  # eV, cm^-3
        balance = make_balance(electron_temp, electron_dens)
        balance.solve()
        # lambda_D = sqrt(epsilon_0 T_e / (n_e e)) with T_e in eV and n_e in m^-3.
        expected = np.sqrt(constants.epsilon_0 * electron_temp
                           / (electron_dens * 1e6 * constants.e))
        np.testing.assert_allclose(balance.debye_length, expected, rtol=1e-12)
        np.testing.assert_allclose(balance.debye_length, 7.43394e-6, rtol=1e-5)
        self.assertTrue(np.isfinite(balance.data["processed"]["temperatures"]).all())

    def test_coulomb_logarithm_for_carbon_on_deuterium(self):
        # C2+ on D+ at 10 eV with n_e = 1e13 cm^-3; a Debye length computed from
        # the density in cm^-3 gave ln(Lambda) = 17.07 instead.
        balance = make_balance(10.0, 1e13)
        balance.solve()
        log = coulomb_logarithm(
            masses=np.array([12.0, 2.0]) * constants.value("atomic mass constant"),
            charges=np.array([2, 1]),
            temperatures=np.array([10.0, 10.0]),
            debye_length=balance.debye_length,
        )
        np.testing.assert_allclose(log, 10.15863, rtol=1e-5)

    def test_failed_integration(self):
        # A failed integration returned energies for fewer times than the abundances,
        # which were broadcast against them without a warning.
        balance = make_balance(10.0, 1e13)
        rate = balance._evolve_energy
        # the energies diverge at 5 ms, between the first and second solution times
        balance._evolve_energy = lambda t, e, f: rate(t, e, f) + 1 / (5e-3 - t)**2
        with self.assertWarnsRegex(RuntimeWarning, "failed after 1 of 3 solution times"):
            balance.solve()
        processed = balance.data["processed"]
        self.assertFalse(processed["success"])
        np.testing.assert_array_equal(processed["time"], [1e-3])
        for key in ("abundances", "energies", "temperatures"):
            self.assertEqual(processed[key].shape, (3, 1))

        # a successful integration keeps every time and gives no warning
        balance = make_balance(10.0, 1e13)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            balance.solve()
        self.assertTrue(balance.data["processed"]["success"])
        self.assertEqual(balance.data["processed"]["temperatures"].shape, (3, 3))

    def test_charge_states_without_population(self):
        # The temperatures were energies / (3/2 abundances), 0/0 with a RuntimeWarning for the
        # empty charge states, e.g. every ion at t = 0.
        balance = make_balance(10.0, 1e13)
        balance.ion_balance.data["user"]["soln_times"] = np.array([0.0, 1e-3, 1e-1])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            balance.solve()
        processed = balance.data["processed"]
        empty = processed["abundances"] <= 0
        self.assertTrue(empty[:, 0].any())
        self.assertTrue(np.isnan(processed["temperatures"][empty]).all())
        resolved = processed["abundances"] >= processed["min_abundance"]
        self.assertTrue(np.isfinite(processed["temperatures"][resolved]).all())
        np.testing.assert_allclose(processed["temperatures"][0, 0], 1.0)  # init_temp

    def test_temperatures_below_the_tolerance(self):
        # The temperature of a barely populated charge state is its energy, integration noise
        # near the absolute tolerance, over 3/2 its abundance: C at 1e-17 gave -1800 eV.
        balance = make_balance(10.0, 1e13)
        balance.ion_balance.data["user"]["soln_times"] = np.array([1e-7, 1e-3, 1e-1])
        balance.solve()
        processed = balance.data["processed"]
        # 1e-12 of 3/2 T_max = 15 eV, the default atol, times 1e4, over 15 eV
        np.testing.assert_allclose(processed["min_abundance"], 1e-8)
        abundances = processed["abundances"]
        self.assertLess(abundances[2, 0], 1e-8)    # about 1e-14 after 1e-7 s
        self.assertGreater(abundances[1, 0], 1e-8)
        self.assertTrue(np.isnan(processed["temperatures"][2, 0]))
        self.assertTrue(np.isfinite(processed["temperatures"][abundances >= 1e-8]).all())

    def test_default_tolerances_converge(self):
        # solve_ivp's defaults (rtol 1e-3, atol 1e-6 eV) left the temperatures of carbon in a
        # 100 eV plasma up to 10% off for abundances above 1e-6.
        processed = []
        for options in ({}, {"rtol": 1e-12, "atol": 1e-18}):
            balance = make_balance(10.0, 1e13)
            balance.solve(**options)
            processed.append(balance.data["processed"])
        default, converged = processed
        resolved = default["abundances"] >= default["min_abundance"]
        np.testing.assert_allclose(default["temperatures"][resolved],
                                   converged["temperatures"][resolved], rtol=1e-6)

    def test_module_example(self):
        # The example run as __main__ called plasma_energy_transfer_frequency without
        # the Debye length and raised TypeError.
        output = io.StringIO()
        with contextlib.redirect_stdout(output), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # runpy: module already imported
            runpy.run_module("colradpy.energy_balance", run_name="__main__")
        rates = [float(line.split(": ")[1].split()[0]) for line in output.getvalue().splitlines()]
        self.assertEqual(len(rates), 5)
        self.assertTrue(all(np.isfinite(rate) and rate > 0 for rate in rates))


if __name__ == "__main__":
    unittest.main()

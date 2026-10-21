"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks the ECIP ionization rate coefficients of r8ecip against the same
integrand summed with the 5-point Gauss-Laguerre rule of numpy.
"""

import contextlib
import io
from pathlib import Path
import unittest

import numpy as np
from numpy.polynomial.laguerre import laggauss

import colradpy
from colradpy.ecip_rates import ecip_rates
from colradpy.r8necip import r8necip
from colradpy.r8yip import R8YIP, r8ecip


INPUT = Path(__file__).resolve().parents[1] / "examples" / "input"


def reference_reduced(charge, ion_pot, energy, zeta, temperature):
    """exp(dE/kT) S from the ECIP integrand, temperature in K, energies in cm-1."""
    nodes, weights = laggauss(5)
    z = charge + 1.0
    xi = (ion_pot - energy) * 9.11269e-6  # Rydberg
    ate = 157890.0 / temperature
    en = z / np.sqrt(xi)
    y = xi * ate
    total = 0.0
    for node, weight in zip(nodes, weights):
        b = node / y
        b1 = b + 1.0
        c = np.sqrt(b1)
        r = (1.25 * en**2 + 0.25) / z
        d = (z / en) * (r + 2.0 * en**2 * c / ((b + 2.0) * z**2)) / (c + np.sqrt(b))
        c1 = 1.0 / (b + 2.0)
        c2 = c1 * (b - b1 * c1 * np.log(b1)) + 0.65343 * (1.0 - 1.0 / b1**3) * R8YIP(0.0, d) / en
        total += weight * c2
    return 8.68811e-8 * np.sqrt(ate) * zeta * total / xi


class EcipTests(unittest.TestCase):
    def test_all_quadrature_points(self):
        # Leaving out the fifth point changes the rates by 2e-5 to 4e-4 here.
        temperatures = np.geomspace(1e3, 1e8, 9)
        for charge in (0, 3, 10, 25):
            for ion_pot in (5e4, 3e5, 2e6):
                with self.subTest(charge=charge, ion_pot=ion_pot):
                    reduced = r8ecip(charge, ion_pot, np.array([0.0]), np.array([1.0]),
                                     temperatures, reduced=True)[0]
                    expected = [reference_reduced(charge, ion_pot, 0.0, 1.0, t) for t in temperatures]
                    np.testing.assert_allclose(reduced, expected, rtol=1e-5)

    def test_rates_and_reduced_rates(self):
        temperatures = np.geomspace(1e2, 1e6, 13)
        energy = np.array([0.0, 2e4])
        rates = r8ecip(1, 1.5e5, energy, np.array([2.0, 1.0]), temperatures)
        reduced = r8ecip(1, 1.5e5, energy, np.array([2.0, 1.0]), temperatures, reduced=True)
        exponent = np.outer(1.5e5 - energy, 9.11269e-6 * 157890.0 / temperatures)
        self.assertTrue((exponent > 180).any() and (exponent <= 180).any())
        kept = exponent <= 180
        np.testing.assert_allclose(rates[kept], (reduced * np.exp(-exponent))[kept], rtol=1e-12)
        np.testing.assert_array_equal(rates[~kept], 0.0)
        self.assertTrue((reduced > 0).all())

    def test_r8necip_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            r8necip(0, 1.5e5, np.array([0.0]), np.array([1.0]), np.array([1e4]))
        # The package still provides the ECIP functions through it.
        self.assertIs(colradpy.r8ecip, r8ecip)
        self.assertIs(colradpy.r8necip, r8necip)


class EcipLevelSelectionTests(unittest.TestCase):
    """A level is ionized by ECIP to each parent it is marked {p} for in the level list and lies
    below, also when it is above the first ionization potential (OpenADAS mom97_ls#c0 levels
    56-60 are between the C+ 2P and 4P limits and are marked {2})."""

    def test_ecip_rates_select_levels_below_each_parent(self):
        energy = np.array([0., 100000., 150000.])
        ion_pot = np.array([120000., 300000.])
        zpla = np.array([[1., -1.], [1., 1.], [1., 1.]])
        zpla1 = np.array([[1., -1.], [1., 2.], [1., 2.]])
        rates = ecip_rates(energy, ion_pot, zpla, zpla1, 1, np.array([5., 20.]), reduced=True)
        self.assertTrue((rates[:2, 0] > 0).all())
        self.assertTrue((rates[1:, 1] > 0).all())
        np.testing.assert_array_equal(rates[2, 0], 0)  # above the first ionization potential
        np.testing.assert_array_equal(rates[0, 1], 0)  # not marked for the second parent
        np.testing.assert_allclose(rates[2, 1], r8ecip(1, 300000., energy[2:], np.array([1.]),
                                                       np.array([5., 20.]) * 11604.5, True)[0], rtol=1e-12)

    def test_levels_above_the_first_ionization_potential(self):
        # Be II levels above 90000 cm-1 are put above a lowered first ionization potential and
        # marked for the second parent; they had no ECIP rates.
        model = colradpy.colradpy(str(INPUT / "cpb03_ls#be1.dat"), metas=np.array([0]),
                                  temp_grid=np.array([5., 20.]), electron_den=np.array([1e13]))
        atomic = model.data["atomic"]
        atomic["ion_pot"] = np.array([90000., atomic["ion_pot"][1]])
        above = atomic["energy"] > 90000.
        atomic["zpla"][above, 1] = 1.
        atomic["zpla1"][above, 1] = 2.
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
            model.make_ecip()
        ecip = model.data["rates"]["ioniz"]["ecip"]
        self.assertGreater(above.sum(), 3)
        self.assertTrue((ecip[above, 1] > 0).all())
        np.testing.assert_array_equal(ecip[above, 0], 0)


if __name__ == "__main__":
    unittest.main()

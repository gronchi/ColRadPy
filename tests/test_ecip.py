"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks the ECIP ionization rate coefficients of r8ecip against the same
integrand summed with the 5-point Gauss-Laguerre rule of numpy.
"""

import unittest

import numpy as np
from numpy.polynomial.laguerre import laggauss

import colradpy
from colradpy.r8necip import r8necip
from colradpy.r8yip import R8YIP, r8ecip


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


if __name__ == "__main__":
    unittest.main()

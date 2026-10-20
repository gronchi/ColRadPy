"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks the vacuum-to-air wavelength conversion: the revised Edlen formula in
standard air, vacuum wavelengths kept below 200 nm, and the model outputs.
"""

from pathlib import Path
import unittest

import numpy as np

from colradpy import colradpy
from colradpy.colradpy_class import (
    MIN_AIR_WAVELENGTH,
    air_refractive_index,
    convert_to_air,
    vac_to_air,
)


INPUT = Path(__file__).resolve().parents[1] / "examples" / "input"


def edlen_1966(wave_vac):
    """Refractive index of dry standard air (15 C, 101325 Pa), Edlen (1966)."""
    s2 = (1e3 / wave_vac)**2
    return 1 + 8342.54e-8 + 2406147e-8 / (130 - s2) + 15998e-8 / (38.9 - s2)


class AirWavelengthTests(unittest.TestCase):
    def test_standard_air_matches_edlen_1966(self):
        wave = np.geomspace(200., 2000., 50)
        np.testing.assert_allclose(air_refractive_index(wave) - 1,
                                   edlen_1966(wave) - 1, rtol=1e-5)

    def test_sodium_d2_matches_nist_air_wavelength(self):
        # NIST ASD Na I D2: 589.1582641 nm in vacuum, 588.9950954 nm in air.
        self.assertAlmostEqual(float(vac_to_air(589.1582641)), 588.9950954, delta=2e-4)

    def test_air_conditions(self):
        wave = np.array([400., 600., 800.])
        standard = air_refractive_index(wave) - 1
        # Less dense air (lower pressure, warmer) or water vapor lowers the index.
        self.assertTrue((air_refractive_index(wave, pressure_pa=9e4) - 1 < standard).all())
        self.assertTrue((air_refractive_index(wave, temperature_c=25.) - 1 < standard).all())
        self.assertTrue((air_refractive_index(wave, water_vapor_pressure_pa=1e3) - 1
                         < standard).all())
        # The index of a vacuum is one.
        np.testing.assert_allclose(air_refractive_index(wave, pressure_pa=0.), 1.)

    def test_ultraviolet_stays_in_vacuum(self):
        # Wavelengths around both poles of the formula (87.7 and 160.3 nm).
        ultraviolet = np.array([10., 87., 87.7058, 88., 160.2, 160.33384, 160.4, 199.99])
        with np.errstate(all="raise"):
            np.testing.assert_array_equal(vac_to_air(ultraviolet), ultraviolet)
        self.assertEqual(MIN_AIR_WAVELENGTH, 200.)
        self.assertLess(float(vac_to_air(200.)), 200.)
        # Air wavelengths keep the order of the vacuum wavelengths above the cutoff.
        wave = np.linspace(200., 2000., 1000)
        self.assertTrue((np.diff(vac_to_air(wave)) > 0).all())

    def test_scalar_and_array_inputs(self):
        self.assertEqual(np.shape(vac_to_air(500.)), ())
        wave = np.array([[150., 500.], [1000., 2500.]])
        air = vac_to_air(wave)
        self.assertEqual(air.shape, wave.shape)
        np.testing.assert_allclose(air[1], wave[1] / air_refractive_index(wave[1]))

    def test_convert_to_air_returns_refractive_index(self):
        wave = np.array([300., 600.])
        np.testing.assert_allclose(convert_to_air(wave), air_refractive_index(wave))

    def test_model_wavelengths(self):
        for name in ("cpb03_ls#be0.dat", "be2_adf04"):
            with self.subTest(file=name):
                model = colradpy(str(INPUT / name), metas=np.array([0]),
                                 temp_grid=np.array([10.]), electron_den=np.array([1e13]))
                # Degenerate levels in bundled data produce a pre-existing
                # zero-energy wavelength warning, outside this check's scope.
                with np.errstate(divide="ignore"):
                    model.solve_cr()
                vac = model.data["processed"]["wave_vac"]
                air = model.data["processed"]["wave_air"]
                ultraviolet = vac < 200.
                np.testing.assert_array_equal(air[ultraviolet], vac[ultraviolet])
                finite = ~ultraviolet & np.isfinite(vac)
                self.assertTrue(finite.any())
                ratio = vac[finite] / air[finite]
                self.assertTrue(((ratio > 1.00026) & (ratio < 1.00033)).all())


if __name__ == "__main__":
    unittest.main()

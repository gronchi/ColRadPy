"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks the natural (Lorentzian) and Doppler (Gaussian) line profiles. With
gs=5 and n=41 the wavelength grid steps by a quarter of the width that sets
it, so grid points fall at known fractions of each profile's width.
"""

from fractions import Fraction
from pathlib import Path
import unittest
import warnings

import numpy as np
from scipy import constants

from colradpy import colradpy


INPUT = Path(__file__).resolve().parents[1] / "examples" / "input"
CENTER, HALF_WIDTH, WIDTH = 20, 22, 24  # grid indices for n=41, gs=5


def one_line_model(a_value, wave):
    """A single line whose upper level decays at a_value (s^-1) to a stable lower level."""
    model = colradpy.__new__(colradpy)
    model.data = {
        "processed": {"wave_air": np.array([wave]), "pec_levels": np.array([[1, 0]])},
        "cr_matrix": {"A_ji_loss": np.array([0.0, a_value])},
    }
    return model


def be0_model():
    model = colradpy(str(INPUT / "cpb03_ls#be0.dat"), metas=np.array([0]),
                     temp_grid=np.array([10.]), electron_den=np.array([1e13]))
    model.solve_cr()
    return model


class BroadeningTests(unittest.TestCase):
    def check_area(self, profile, x="wave_arr"):
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        areas = trapezoid(profile["theta"], x=profile[x], axis=1)
        np.testing.assert_allclose(areas, 1.0, rtol=1e-10)

    def test_natural_width_is_lifetime_width(self):
        model = one_line_model(1e8, 500.)
        model.make_nat_broad(n=41, gs=5)
        nat = model.data["processed"]["broadening"]["nat"]
        # The full width at half maximum is A / (2 pi) in frequency.
        fwhm = 1e8 / (2 * np.pi)
        np.testing.assert_allclose(nat["dnu_l"], fwhm)
        nu = constants.c / (nat["wave_arr"][0] * 1e-9)
        np.testing.assert_allclose(abs(nu[HALF_WIDTH] - nu[CENTER]), fwhm / 2, rtol=1e-6)
        theta = nat["theta"][0]
        self.assertAlmostEqual(theta[HALF_WIDTH] / theta[CENTER], 0.5, places=6)
        self.assertAlmostEqual(theta[WIDTH] / theta[CENTER], 0.2, places=6)
        # The profile is normalized on the offsets from the centre; on the wavelengths the
        # area also carries their float64 rounding (5.7e-14 nm at 500 nm).
        self.check_area(nat, x="dwave_arr")
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        np.testing.assert_allclose(trapezoid(theta, x=nat["wave_arr"][0]), 1.0, rtol=1e-8)

    def test_width_below_the_wavelength_resolution(self):
        # The Be I 2s2p 3P - 2s2 1S line at 454.8 nm is 5e-15 nm wide, while float64 resolves
        # 6e-14 nm there: the wavelength grid collapsed to one value and the profile was NaN.
        model = one_line_model(0.046, 454.832)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            model.make_nat_broad(n=41, gs=5)
        nat = model.data["processed"]["broadening"]["nat"]
        fwhm = (454.832e-9)**2 / constants.c * nat["dnu_l"][0] * 1e9  # nm
        self.assertLess(fwhm, np.spacing(454.832))
        np.testing.assert_allclose(nat["dwave_arr"][0, [HALF_WIDTH, WIDTH]], [fwhm / 2, fwhm], rtol=1e-12)
        theta = nat["theta"][0]
        self.assertAlmostEqual(theta[HALF_WIDTH] / theta[CENTER], 0.5, places=9)
        self.assertAlmostEqual(theta[WIDTH] / theta[CENTER], 0.2, places=9)
        self.check_area(nat, x="dwave_arr")

    def test_profile_in_exact_arithmetic(self):
        # Frequencies near 1e15 Hz differ by 0.1 Hz in float64, so c/w - c/w0 lost the shape of
        # narrow profiles (6.7e-4 for the 717 Hz wide Be I line at 175.8 nm).
        a_value, wave = 4500., 175.8
        model = one_line_model(a_value, wave)
        model.make_nat_broad(n=41, gs=5)
        nat = model.data["processed"]["broadening"]["nat"]
        c, w0, half = Fraction(constants.c), Fraction(wave) / 10**9, Fraction(nat["dnu_l"][0]) / 2
        exact = []
        for dw in nat["dwave_arr"][0]:
            dnu = c / (w0 + Fraction(dw) / 10**9) - c / w0
            exact.append(float(half / (dnu**2 + half**2)))
        exact = np.array(exact)
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        exact /= trapezoid(exact, x=nat["dwave_arr"][0])
        np.testing.assert_allclose(nat["theta"][0], exact, rtol=1e-12)

    def test_bundled_model_profiles(self):
        model = be0_model()
        processed = model.data["processed"]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            model.make_nat_broad(n=41, gs=5)
        nat = processed["broadening"]["nat"]
        loss = model.data["cr_matrix"]["A_ji_loss"]
        upper, lower = processed["pec_levels"].T
        np.testing.assert_allclose(nat["dnu_l"], (loss[upper] + loss[lower]) / (2 * np.pi))
        # Every line with a wavelength has a profile, including those narrower than the
        # float64 resolution of their wavelength in nm (the 454.8 nm line, 5e-15 nm wide).
        lines = np.isfinite(processed["wave_air"])
        np.testing.assert_array_equal(np.where(~lines)[0], [405])  # degenerate levels
        np.testing.assert_array_equal(nat["theta"][~lines], 0)
        # The grid is symmetric in wavelength and the profile in frequency, which
        # shifts the ratio by up to the relative width (4e-4 for far-infrared lines).
        np.testing.assert_allclose(nat["theta"][lines, HALF_WIDTH] / nat["theta"][lines, CENTER],
                                   0.5, rtol=1e-2)
        self.check_area({key: nat[key][lines] for key in ("theta", "dwave_arr")}, x="dwave_arr")
        np.testing.assert_allclose(nat["wave_arr"][lines],
                                   processed["wave_air"][lines, None] + nat["dwave_arr"][lines])

        temperature, mass = 10.0, 9.012182 * constants.atomic_mass
        with np.errstate(all="ignore"):  # the line between degenerate levels
            model.make_dopp_broad(temperature, mass, n=41, gs=5)
        dopp = processed["broadening"]["dopp"]
        lines = np.isfinite(processed["wave_air"])
        # 1/e half width sqrt(2 k T / m) nu0 / c in frequency.
        nu0 = constants.c / (processed["wave_air"][lines] * 1e-9)
        np.testing.assert_allclose(
            dopp["dnu_g"][lines],
            np.sqrt(2 * temperature * constants.e / mass) * nu0 / constants.c)
        np.testing.assert_allclose(dopp["theta"][lines, WIDTH] / dopp["theta"][lines, CENTER],
                                   np.exp(-1), rtol=1e-3)
        self.check_area({key: dopp[key][lines] for key in ("theta", "wave_arr")})


if __name__ == "__main__":
    unittest.main()

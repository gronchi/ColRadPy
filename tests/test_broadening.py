"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks the natural (Lorentzian) and Doppler (Gaussian) line profiles. With
gs=5 and n=41 the wavelength grid steps by a quarter of the width that sets
it, so grid points fall at known fractions of each profile's width.
"""

from pathlib import Path
import unittest

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
    # Degenerate levels in bundled data produce a pre-existing
    # zero-energy wavelength warning, outside this check's scope.
    with np.errstate(divide="ignore"):
        model.solve_cr()
    return model


class BroadeningTests(unittest.TestCase):
    def check_area(self, profile):
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        areas = trapezoid(profile["theta"], x=profile["wave_arr"], axis=1)
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
        self.check_area(nat)

    def test_bundled_model_profiles(self):
        model = be0_model()
        processed = model.data["processed"]
        with np.errstate(all="ignore"):
            model.make_nat_broad(n=41, gs=5)
        nat = processed["broadening"]["nat"]
        loss = model.data["cr_matrix"]["A_ji_loss"]
        upper, lower = processed["pec_levels"].T
        np.testing.assert_allclose(nat["dnu_l"], (loss[upper] + loss[lower]) / (2 * np.pi))
        # Widths below ~1e-9 of the wavelength cannot be resolved on a grid in nm.
        relative_width = processed["wave_air"] * 1e-9 / constants.c * nat["dnu_l"]
        lines = np.isfinite(processed["wave_air"]) & (relative_width > 1e-9)
        self.assertGreater(lines.sum(), 300)
        # The grid is symmetric in wavelength and the profile in frequency, which
        # shifts the ratio by up to the relative width (4e-4 for far-infrared lines).
        np.testing.assert_allclose(nat["theta"][lines, HALF_WIDTH] / nat["theta"][lines, CENTER],
                                   0.5, rtol=1e-2)
        self.check_area({key: nat[key][lines] for key in ("theta", "wave_arr")})

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

"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks scale_file_ioniz, which multiplies the ionization rates of the adf04 file by the
{parent} values of the level list, for every interpolation of the rates and in the
ionization balance.
"""

import contextlib
import io
from pathlib import Path
import unittest

import numpy as np

from colradpy import colradpy, ionization_balance


INPUT = Path(__file__).resolve().parents[1] / "examples" / "input"
BE0 = str(INPUT / "cpb03_ls#be0.dat")  # S lines for levels 1 and 2, which have {1}2.000
TEMPS = np.array([2., 10., 50.])
DENS = np.array([1e12, 1e14])


def model(scale, rate_interp_ion="slinear", **kwargs):
    m = colradpy(BE0, metas=np.array([0]), temp_grid=TEMPS, electron_den=DENS,
                 rate_interp_ion=rate_interp_ion, scale_file_ioniz=scale, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        m.make_ioniz_from_reduced_ionizrates()
    return m


class ScaleFileIonizationTests(unittest.TestCase):
    def test_every_interpolation(self):
        # The scaling was only applied when the ionization rates were not interpolated in log.
        for kind in ("slinear", "log_slinear", "log_quadratic", "loglog_slinear"):
            with self.subTest(rate_interp_ion=kind):
                plain = model(False, kind)
                scaled = model(True, kind)
                trans = plain.data["rates"]["ioniz"]["ion_transitions"]
                zpla = plain.data["atomic"]["zpla"][trans[:, 0] - 1, trans[:, 1] - 1]
                np.testing.assert_array_equal(zpla, 2)  # the S lines are of levels 1 and 2
                for key in ("ionization", "ionization_reduced"):
                    a = plain.data["rates"]["ioniz"][key][trans[:, 0] - 1, trans[:, 1] - 1]
                    b = scaled.data["rates"]["ioniz"][key][trans[:, 0] - 1, trans[:, 1] - 1]
                    self.assertTrue((a > 0).all())
                    np.testing.assert_allclose(b, a * zpla[:, None], rtol=1e-12)

    def test_levels_without_a_parent_value(self):
        # A level with an S line but no {1} value in the level list keeps its rate.
        plain = model(False)
        scaled = model(True)
        trans = plain.data["rates"]["ioniz"]["ion_transitions"]
        level = trans[0, 0] - 1
        for m in (plain, scaled):
            m.data["atomic"]["zpla"][level, 0] = -1
            with contextlib.redirect_stdout(io.StringIO()):
                m.make_ioniz_from_reduced_ionizrates()
        np.testing.assert_allclose(scaled.data["rates"]["ioniz"]["ionization"][level, 0],
                                   plain.data["rates"]["ioniz"]["ionization"][level, 0], rtol=1e-12)

    def test_ionization_balance(self):
        # The ionization balance stored scale_file_ioniz but did not pass it to colradpy.
        def scd(scale):
            with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
                ion = ionization_balance([BE0], [np.array([0])], TEMPS, DENS,
                                         use_recombination=False, use_recombination_three_body=False,
                                         suppliment_with_ecip=False, scale_file_ioniz=scale)
            return ion.data["cr_data"]["gcrs"]["0"]["scd"]

        direct = model(True, "loglog_slinear", use_recombination=False, use_recombination_three_body=False,
                       suppliment_with_ecip=False)   # the default of both
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
            direct.solve_cr()
        self.assertFalse(np.allclose(scd(True), scd(False)))
        np.testing.assert_allclose(scd(True), direct.data["processed"]["scd"], rtol=1e-12)


if __name__ == "__main__":
    unittest.main()

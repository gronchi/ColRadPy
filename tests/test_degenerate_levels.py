"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks that lines between levels with equal energies get an infinite wavelength
without a divide-by-zero warning.
"""

import contextlib
import io
from pathlib import Path
import unittest
import warnings

import numpy as np

from colradpy import colradpy


INPUT = Path(__file__).resolve().parents[1] / "examples" / "input"


def model():
    # Be I gives the 2s5g 3G and 1G terms (levels 27 and 28) the same energy.
    return colradpy(str(INPUT / "cpb03_ls#be0.dat"), metas=np.array([0]),
                    temp_grid=np.array([10.]), electron_den=np.array([1e13]))


@contextlib.contextmanager
def divide_by_zero_is_an_error():
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.filterwarnings("error", "divide by zero", RuntimeWarning)
        yield


class DegenerateLevelTests(unittest.TestCase):
    def test_quasistatic_wavelengths(self):
        m = model()
        with divide_by_zero_is_an_error():
            m.solve_cr()
        p = m.data["processed"]
        energy = m.data["atomic"]["energy"]
        diff = np.abs(energy[p["pec_levels"][:, 0]] - energy[p["pec_levels"][:, 1]])
        degenerate = diff == 0
        np.testing.assert_array_equal(p["pec_levels"][degenerate], [[28, 27]])
        np.testing.assert_array_equal(p["wave_vac"][degenerate], np.inf)
        np.testing.assert_array_equal(p["wave_air"][degenerate], np.inf)
        np.testing.assert_allclose(p["wave_vac"][~degenerate], 1e7 / diff[~degenerate], rtol=1e-15)

    def test_shifted_wavelengths(self):
        m = model()
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
            m.solve_cr()
            m.shift_j_res_energy_to_nist(already_j=True)
        shifted = np.where(m.data["atomic"]["nist_shifted"])[0]
        low, up = shifted[0], shifted[1]
        m.data["atomic"]["energy"][up] = m.data["atomic"]["energy"][low]  # equal NIST energies
        with divide_by_zero_is_an_error():
            m.shift_j_res_pecs()
        p = m.data["processed"]
        line = (p["pec_levels_shift"][:, 0] == up) & (p["pec_levels_shift"][:, 1] == low)
        self.assertEqual(line.sum(), 1)
        self.assertEqual(p["wave_vac_shift"][line][0], np.inf)
        self.assertTrue(np.isfinite(p["wave_vac_shift"][~line]).all())


if __name__ == "__main__":
    unittest.main()

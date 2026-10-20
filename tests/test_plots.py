"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks the units and scaling of the PEC stick plot.
"""

import contextlib
import io
from pathlib import Path
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from colradpy import colradpy


INPUT = Path(__file__).resolve().parents[1] / "examples" / "input"


class PecPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_pec_sticks_units_and_scaling(self):
        # The label gave PECs in cm^-3 and a factor 1E<n> while the sticks were multiplied
        # by 10^|n|, so the label read as PEC x 10^n instead of PEC / 10^n.
        m = colradpy(str(INPUT / "cpb03_ls#be0.dat"), metas=np.array([0]),
                     temp_grid=np.array([10.]), electron_den=np.array([1e13]))
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
            m.solve_cr()
            m.plot_pec_sticks(temp=[0], dens=[0], meta=[0])
        ax = plt.gca()
        pecs = m.data["processed"]["pecs"][:, 0, 0, 0]
        scaling = int(np.floor(np.log10(pecs.max())))
        self.assertEqual(ax.get_ylabel(), "PEC (10$^{" + str(scaling) + "}$ ph cm$^{3}$ s$^{-1}$)")
        # lines between degenerate levels have an infinite wavelength and no stick
        finite = np.isfinite(m.data["processed"]["wave_air"])
        segments = ax.collections[0].get_segments()
        heights = np.array([s[1, 1] for s, f in zip(segments, finite) if f])
        np.testing.assert_allclose(heights * 10.**scaling, pecs[finite], rtol=1e-12)
        self.assertTrue(1 <= heights.max() < 10)


if __name__ == "__main__":
    unittest.main()

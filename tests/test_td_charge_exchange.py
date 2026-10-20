"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks that charge exchange in the time-dependent CR solution takes population from the
evolving parent, so that the solution conserves particles.
"""

import contextlib
import io
import os
import sys
import tempfile
import unittest

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(__file__))
from test_cr_options import INPUT, cx_line  # noqa: E402

from colradpy import colradpy  # noqa: E402


TEMPS = np.array([5., 20.])
DENS = np.array([1e12, 1e14])
TIMES = np.array([0., 1e-8, 1e-6, 1e-4, 1e-2])


class TimeDependentChargeExchangeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # Be I with charge exchange from the parent into levels 2 and 3
        text = (INPUT / "cpb03_ls#be0.dat").read_text().splitlines()
        k = next(i for i, l in enumerate(text) if l.startswith("S  2  +1"))
        text[k + 1:k + 1] = [cx_line(2, 1, 11), cx_line(3, 1, 11)]
        self.path = os.path.join(tmp.name, "be0_cx.dat")
        with open(self.path, "w") as f:
            f.write("\n".join(text) + "\n")

    def model(self, use_cx, td_n0):
        m = colradpy(self.path, metas=np.array([0]), temp_grid=TEMPS, electron_den=DENS,
                     htemp_grid=TEMPS, hdens_grid=DENS / 10, use_cx=use_cx,
                     td_t=TIMES, td_n0=td_n0)
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
            m.solve_cr()
            m.solve_time_dependent()
        return m

    def test_parent_loses_what_charge_exchange_gives(self):
        # The charge exchange column never lost population, so it was a source that never
        # emptied: the levels it fed ionized into the parent, which grew without bound.
        nlev = len(colradpy(self.path, metas=np.array([0]), temp_grid=TEMPS,
                            electron_den=DENS).data["atomic"]["energy"])
        n0 = np.zeros(nlev + 2)
        n0[0] = 0.3     # ground level
        n0[nlev] = 0.7  # parent
        m = self.model(True, n0)
        pop = m.data["processed"]["td"]["td_pop"]
        self.assertEqual(pop.shape, (nlev + 2, len(TIMES), len(TEMPS), len(DENS)))
        np.testing.assert_allclose(pop.sum(axis=0), 1., rtol=1e-8)
        np.testing.assert_array_equal(pop[nlev + 1], 0)

        # the same solution from the matrix without charge exchange, with the charge exchange
        # rates times the hydrogen density moved onto the parent
        base = self.model(False, n0[:nlev + 1])
        cx = m.data["rates"]["cx"]["cx"][:, 0, :, None] * (DENS / 10)  # (levels, temps, dens)
        for t in range(len(TEMPS)):
            for d in range(len(DENS)):
                a = base.data["cr_matrix"]["cr"][:, :, t, d].copy()
                a[:nlev, nlev] += cx[:, t, d]
                a[nlev, nlev] -= cx[:, t, d].sum()
                expected = np.array([expm(a * time) @ n0[:nlev + 1] for time in TIMES]).T
                np.testing.assert_allclose(pop[:nlev + 1, :, t, d], expected, rtol=1e-6, atol=1e-12)

        # charge exchange changes the solution
        self.assertFalse(np.allclose(pop[:nlev + 1], base.data["processed"]["td"]["td_pop"]))

    def test_charge_exchange_columns_of_the_initial_populations(self):
        nlev = len(colradpy(self.path, metas=np.array([0]), temp_grid=TEMPS,
                            electron_den=DENS).data["atomic"]["energy"])
        n0 = np.zeros(nlev + 2)
        n0[nlev] = 1.
        n0[nlev + 1] = 1.  # the parent given again as the charge exchange column
        with self.assertRaisesRegex(ValueError, "td_n0 has non-zero entries"):
            self.model(True, n0)


if __name__ == "__main__":
    unittest.main()

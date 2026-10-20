"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks cr_loss, the loss terms of the levels in the CR matrix, against the radiative,
collisional and ionization rates of each level.
"""

import contextlib
import io
from pathlib import Path
import unittest

import numpy as np

from colradpy import colradpy


INPUT = Path(__file__).resolve().parents[1] / "examples" / "input"
TEMPS = np.array([5., 20., 80.])
DENS = np.array([1e11, 1e13, 1e15])


def matrix(temp_dens_pair, use_ionization):
    m = colradpy(str(INPUT / "cpb03_ls#be0.dat"), metas=np.array([0]), temp_grid=TEMPS,
                 electron_den=DENS, temp_dens_pair=temp_dens_pair, use_ionization=use_ionization,
                 suppliment_with_ecip=use_ionization)
    with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
        if use_ionization:
            m.make_ioniz_from_reduced_ionizrates()
            m.make_ecip()
            m.suppliment_with_ecip()
        m.make_recombination_rates_from_file()
        m.make_three_body_recombination()
        m.make_electron_excitation_rates()
        m.populate_cr_matrix()
    return m


class CrLossTests(unittest.TestCase):
    def test_level_losses(self):
        # cr_loss counted ionization twice, and collisions twice (grid) or not at all
        # (temperature-density pairs) without ionization; its pair shape had no CX rows.
        for pair in (False, True):
            for ion in (False, True):
                with self.subTest(temp_dens_pair=pair, use_ionization=ion):
                    m = matrix(pair, ion)
                    cr = m.data["cr_matrix"]
                    nlev = len(m.data["atomic"]["energy"])
                    radiative = cr["A_ji"].sum(axis=1)                       # (levels,)
                    collisional = (cr["q_ji"] + cr["q_ij"]).sum(axis=1)      # (levels, temperatures)
                    rate = collisional
                    if ion:
                        rate = rate + m.data["rates"]["ioniz"]["ionization"].sum(axis=1)
                    if pair:  # one density for each temperature
                        expected = -(radiative[:, None] + rate * DENS)
                    else:
                        expected = -(radiative[:, None, None] + rate[:, :, None] * DENS)

                    loss = cr["cr_loss"]
                    self.assertEqual(loss.shape, cr["cr"].shape)
                    levels = np.arange(nlev)
                    np.testing.assert_allclose(loss[levels, levels], expected, rtol=1e-12)
                    off = loss.copy()
                    off[levels, levels] = 0
                    np.testing.assert_array_equal(off, 0)


if __name__ == "__main__":
    unittest.main()

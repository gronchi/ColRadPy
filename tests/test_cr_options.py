"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks combinations of the rate options of colradpy: charge exchange without
recombination, charge exchange from one of several parents, and three-body
recombination without ionization in the CR matrix.
"""

import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from colradpy import colradpy


INPUT = Path(__file__).resolve().parents[1] / "examples" / "input"
TEMPS = np.array([5., 20.])
DENS = np.array([1e12, 1e14])


def cx_line(level, parent, n):
    """A charge exchange (H) line with n rate coefficients rising with temperature."""
    values = " ".join(f"{1e-9 * (1 + 0.1 * k):.2e}".replace("e-0", "-") for k in range(n))
    return f"H{level:3d}  +{parent}         {values}"


def model(path, **kwargs):
    options = dict(metas=np.array([0]), temp_grid=TEMPS, electron_den=DENS,
                   htemp_grid=TEMPS, hdens_grid=DENS / 10)
    options.update(kwargs)
    m = colradpy(str(path), **options)
    with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
        m.solve_cr()
    return m


class RateOptionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def with_lines(self, name, after, lines):
        """Copy of a bundled file with lines inserted after the line starting with `after`."""
        text = (INPUT / name).read_text().splitlines()
        k = next(i for i, l in enumerate(text) if l.startswith(after))
        text[k + 1:k + 1] = lines
        path = os.path.join(self.tmp.name, name)
        with open(path, "w") as f:
            f.write("\n".join(text) + "\n")
        return path

    def test_charge_exchange_without_recombination(self):
        # Raised UnboundLocalError: the charge exchange columns were placed after the
        # recombination columns, which were only defined with recombination.
        n = 11
        path = self.with_lines("cpb03_ls#be0.dat", "S  2  +1", [cx_line(2, 1, n), cx_line(3, 1, n)])
        cx_only = model(path, use_cx=True, use_recombination=False, use_recombination_three_body=False)
        both = model(path, use_cx=True)
        pops_cx = cx_only.data["processed"]["pops"]
        pops_both = both.data["processed"]["pops"]
        self.assertEqual(pops_cx.shape[1], 2)  # ground metastable and one charge exchange parent
        self.assertEqual(pops_both.shape[1], 3)
        # Recombination does not change the excited levels' matrix, so the charge exchange
        # driven populations are the same with and without it.
        np.testing.assert_allclose(pops_cx[:, 1], pops_both[:, 2], rtol=1e-10)
        cr = cx_only.data["cr_matrix"]["cr"]
        nlev = len(cx_only.data["atomic"]["energy"])
        np.testing.assert_allclose(cr[1, nlev + 1], cx_only.data["rates"]["cx"]["cx"][1, 0][:, None] * DENS / 10)
        self.assertTrue((cx_only.data["processed"]["pecs"][:, 1] >= 0).all())

    def test_charge_exchange_from_second_parent(self):
        # The charge exchange rates had one column per parent found in the file but were
        # filled by parent index, so charge exchange only from parent 2 failed.
        n = 12
        path = self.with_lines("cpb03_ls#be1.dat", "S  1  +2", [cx_line(2, 2, n)])
        m = model(path, use_cx=True)
        cx = m.data["rates"]["cx"]["cx"]
        self.assertEqual(cx.shape[1], 2)
        np.testing.assert_array_equal(cx[:, 0], 0)
        self.assertTrue((cx[1, 1] > 0).all())
        # metastable, two recombining parents and two charge exchange parents
        self.assertEqual(m.data["processed"]["pops"].shape[1], 5)
        np.testing.assert_array_equal(m.data["processed"]["pops"][:, 3], 0)
        self.assertTrue((m.data["processed"]["pops"][:, 4] > 0).any())

    def test_parents_beyond_the_first_line(self):
        n = 11
        path = self.with_lines("cpb03_ls#be0.dat", "S  2  +1", [cx_line(2, 2, n)])
        with self.assertRaisesRegex(ValueError, "H lines give parents"):
            model(path, use_cx=True)

    def test_three_body_without_ionization(self):
        # Raised KeyError: 'ionization', which detailed balance needs even when
        # ionization is left out of the CR matrix.
        path = INPUT / "cpb03_ls#be0.dat"
        for ecip in (False, True):
            with self.subTest(suppliment_with_ecip=ecip):
                no_ion = model(path, use_ionization=False, suppliment_with_ecip=ecip)
                with_ion = model(path, suppliment_with_ecip=ecip)
                np.testing.assert_allclose(no_ion.data["rates"]["recomb"]["recomb_three_body"],
                                           with_ion.data["rates"]["recomb"]["recomb_three_body"])
                cr = no_ion.data["cr_matrix"]["cr"]
                nlev = len(no_ion.data["atomic"]["energy"])
                np.testing.assert_array_equal(cr[nlev:, :nlev], 0)  # no ionization
                self.assertTrue((cr[:nlev, nlev] > 0).any())          # recombination
                np.testing.assert_array_equal(no_ion.data["processed"]["xcd"], 0)


if __name__ == "__main__":
    unittest.main()

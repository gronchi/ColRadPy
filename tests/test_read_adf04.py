"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks reading adf04 files (ADAS appxa-04) with edge cases made by editing
copies of the bundled files.
"""

import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from colradpy import colradpy
from colradpy.read_adf04_py3_class import adf04_float, read_adf04


INPUT = Path(__file__).resolve().parents[1] / "examples" / "input"


def read(path):
    with contextlib.redirect_stdout(io.StringIO()):
        return read_adf04(str(path))


class Adf04Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def edited(self, name, *replacements):
        """Copy of a bundled file with each (old, new) replaced once."""
        text = (INPUT / name).read_text()
        for old, new in replacements:
            self.assertEqual(text.count(old), 1, old)
            text = text.replace(old, new)
        path = os.path.join(self.tmp.name, name)
        with open(path, "w") as f:
            f.write(text)
        return path

    def test_numbers(self):
        for text, value in (("1.23+08", 1.23e8), ("1.23-08", 1.23e-8), ("-1.23-01", -0.123),
                            ("-1.23+01", -12.3), ("1.000", 1.0), ("1.23E+08", 1.23e8), ("5", 5.0)):
            with self.subTest(text=text):
                self.assertEqual(adf04_float(text), value)

    def test_a_value_without_exponent(self):
        # It was skipped, shifting every following A-value.
        original = read(INPUT / "cpb03_ls#be0.dat")["rates"]
        edited = read(self.edited("cpb03_ls#be0.dat", ("   3   1 5.51+08", "   3   1 1.000  ")))["rates"]
        self.assertEqual(len(edited["a_val"]), len(original["a_val"]))
        np.testing.assert_array_equal(edited["a_val"][1], 1.0)
        np.testing.assert_array_equal(np.delete(edited["a_val"], 1), np.delete(original["a_val"], 1))

    def test_infinite_energy_point_of_one_transition(self):
        # Infinite energy points were only appended when present, so one point
        # was taken as that of the first transition.
        line = "   2   1 4.60-02 1.40+00 2.14+00 3.13+00 3.56+00 3.56+00 2.88+00 2.13+00 1.44+00 7.14-01 3.86-01 2.01-01"
        rates = read(self.edited("cpb03_ls#be0.dat", ("   4   1 1.00-30 2.55-02",
                                                      "   4   1 1.00-30 2.55-02"),
                                 (line, line)))["rates"]
        self.assertEqual(len(rates["inf_engy"]), 0)  # none given
        text = (INPUT / "cpb03_ls#be0.dat").read_text().splitlines()
        k = next(i for i, l in enumerate(text) if l.startswith("   4   1 1.00-30"))
        path = self.edited("cpb03_ls#be0.dat", (text[k] + "\n", text[k] + " 2.50+00\n"))
        rates = read(path)["rates"]
        self.assertEqual(len(rates["inf_engy"]), len(rates["a_val"]))
        index = np.where((rates["excit"]["col_transitions"] == [4, 1]).all(axis=1))[0][0]
        self.assertEqual(rates["inf_engy"][index], 2.5)
        self.assertEqual(np.count_nonzero(rates["inf_engy"]), 1)

    def test_temperature_with_negative_exponent(self):
        # Raised NameError (temp_grid_inds), which the reader turned into an exit.
        adf04 = read(self.edited("cpb03_ls#be0.dat", ("  1.0    3       1.00+03", "  1.0    3       5.00-01")))
        self.assertEqual(adf04["input_file"]["temp_grid"][0], 0.5)

    def test_reversed_transition(self):
        # Lower level first was only corrected when level energies were reordered.
        rates = read(self.edited("cpb03_ls#be0.dat", ("   3   1 5.51+08", "   1   3 5.51+08")))["rates"]
        transitions = rates["excit"]["col_transitions"]
        self.assertTrue((transitions[:, 0] > transitions[:, 1]).all())
        self.assertEqual(rates["a_val"][1], 5.51e8)
        np.testing.assert_array_equal(transitions[1], [3, 1])

    def test_j_without_space(self):
        # (1)4(4.0) without a space before J (upstream PR #31).
        atomic = read(self.edited("cpb03_ls#be0.dat", ("(1)4( 4.0)    71874.2", "(1)4(4.0)     71874.2")))["atomic"]
        self.assertEqual(atomic["w"][-1], 4.0)
        self.assertEqual(atomic["energy"][-1], 71874.2)

    def test_parents_by_index(self):
        # Parent columns followed the order in which parents first appear.
        original = read(INPUT / "cpb03_ls#be1.dat")["atomic"]
        atomic = read(self.edited("cpb03_ls#be1.dat", ("{1}1.000 {2}1.500", "{2}1.500 {1}1.000")))["atomic"]
        np.testing.assert_array_equal(atomic["zpla"], original["zpla"])
        np.testing.assert_array_equal(atomic["zpla1"], original["zpla1"])
        np.testing.assert_array_equal(atomic["ion_pot_lvl"], [1, 2])

    def test_unused_ionization_potential(self):
        # A first line with more ionization potentials than the levels use gave a
        # shorter ion_pot_lvl, which broke the parent cross coupling (upstream issue #42).
        path = self.edited("cpb03_ls#be0.dat", ("75190.0(2S)", "75190.0(2S)  250000.0(2P)"))
        atomic = read(path)["atomic"]
        np.testing.assert_array_equal(atomic["ion_pot_lvl"], [1, 2])
        self.assertTrue((atomic["zpla1"][:, 1] == -1).all())
        model = colradpy(path, metas=np.array([0]), temp_grid=np.array([10.]), electron_den=np.array([1e13]))
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
            model.solve_cr()
        self.assertEqual(model.data["processed"]["xcd"].shape[:2], (2, 2))
        self.assertTrue(np.isfinite(model.data["processed"]["pecs"]).all())

    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "parent 2"):
            read(self.edited("cpb03_ls#be0.dat", ("(1)4( 4.0)    71874.2 {1}1.000", "(1)4( 4.0)    71874.2 {2}1.000")))
        with self.assertRaisesRegex(ValueError, "ITYP = 4"):
            read(self.edited("cpb03_ls#be0.dat", ("  1.0    3       1.00+03", "  1.0    4       1.00+03")))
        # A malformed file raises instead of exiting Python.
        with self.assertRaises(ValueError):
            read(self.edited("cpb03_ls#be0.dat", ("75190.0(2S)", "")))


if __name__ == "__main__":
    unittest.main()

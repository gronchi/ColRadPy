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

    LINE = "   2   1 4.60-02 1.40+00 2.14+00 3.13+00 3.56+00 3.56+00 2.88+00 2.13+00 1.44+00 7.14-01 3.86-01 2.01-01"
    UPSILONS = [1.40, 2.14, 3.13, 3.56, 3.56, 2.88, 2.13, 1.44, 0.714, 0.386, 0.201]

    def test_values_without_spaces(self):
        # Values are fields of 8 characters, so a negative one follows the previous value without
        # a space, as in clike_bmf#o2 and slike_al97#fe10 of ADAS. The reader split lines on
        # spaces and read 1.44+00-7.14-01 as an upsilon and the Bethe limit.
        line = self.LINE.replace(" 7.14-01", "-7.14-01").replace(" 2.01-01", "-2.01-01")
        rates = read(self.edited("cpb03_ls#be0.dat", (self.LINE, line)))["rates"]
        upsilons = self.UPSILONS[:8] + [-0.714, 0.386, -0.201]
        np.testing.assert_array_equal(rates["excit"]["col_excit"][0], upsilons)
        self.assertEqual(len(rates["inf_engy"]), 0)

    def test_text_after_the_last_value(self):
        # copss#b_ss#o3l of ADAS writes a 6 just after the last upsilon of every line: 2.01-016
        # was read as one upsilon, 2.01e-16. The field after the last temperature is the Bethe limit.
        rates = read(self.edited("cpb03_ls#be0.dat", (self.LINE, self.LINE + "6")))["rates"]
        np.testing.assert_array_equal(rates["excit"]["col_excit"][0], self.UPSILONS)
        self.assertEqual(rates["inf_engy"][0], 6.)

    def test_shifted_fields(self):
        # Older files shift the fields (copssh#18 of ADAS writes the indices as 2i6)
        original = read(INPUT / "cpb03_ls#be0.dat")["rates"]
        rates = read(self.edited("cpb03_ls#be0.dat", (self.LINE, "  " + self.LINE.replace("   1 ", "     1 "))))["rates"]
        np.testing.assert_array_equal(rates["excit"]["col_excit"], original["excit"]["col_excit"])
        np.testing.assert_array_equal(rates["a_val"], original["a_val"])

    def test_configuration_with_term_or_star(self):
        # The configuration is free text before (IS)IL(XJ). ADAS files write terms and * in it
        # (helike_adw01#ar16: '1s2      (1S) (1)0( 0.0)', belike_dhb94#o: '2S2P(3P0)        *(3)1( 4.0)'),
        # which the reader failed on.
        original = read(INPUT / "cpb03_ls#be0.dat")["atomic"]
        atomic = read(self.edited("cpb03_ls#be0.dat",
                                  ("    1 2S2               (1)0( 0.0)", "    1 2S2      (1S)     (1)0( 0.0)"),
                                  ("    2 2S1 2P1           (3)1( 4.0)", "    2 2S1 2P1 (3P)     *(3)1( 4.0)")))["atomic"]
        self.assertEqual(list(atomic["config"][:3]), ["2S2.(1S)", "2S1.2P1.(3P).*", "2S1.2P1"])
        for key in ("S", "L", "w", "energy", "zpla", "zpla1"):
            np.testing.assert_array_equal(atomic[key], original[key], key)

    def test_blank_parent_term(self):
        # olike_lgy11#s8 of ADAS writes 3138303.(  ). A missing term is 1S (appxa-04); the
        # reader failed.
        atomic = read(self.edited("cpb03_ls#be0.dat", ("75190.0(2S)", "75190.0(  )")))["atomic"]
        self.assertEqual(list(atomic["ion_term"]), ["1S"])
        np.testing.assert_array_equal(atomic["ion_pot"], [75190.])

    def test_recombination_slot_and_parent(self):
        # The AAUG field before the recombination coefficients was read as the first coefficient,
        # and R lines without the parent (helike_fpk94#al of ADAS: 'R  2     0.00-00 4.96-13')
        # failed.
        original = read(INPUT / "cpb03_ls#be0.dat")["rates"]["recomb"]
        rates = read(self.edited("cpb03_ls#be0.dat", ("R  1  +1         1.29-14", "R  1  +1 2.00+10 1.29-14"),
                                 ("R  2  +1         7.54-13", "R  2     0.00-00 7.54-13")))["rates"]["recomb"]
        np.testing.assert_array_equal(rates["recomb_excit"], original["recomb_excit"])
        np.testing.assert_array_equal(rates["recomb_transitions"], original["recomb_transitions"])

    def test_transition_codes(self):
        # Codes 1, 2 and 3 mark dipole, non-dipole and spin change excitations (appxa-04), which
        # were skipped like the lines ColRadPy does not use, such as positive ion impact excitation
        # (P), which were skipped without a warning.
        original = read(INPUT / "cpb03_ls#be0.dat")["rates"]
        path = self.edited("cpb03_ls#be0.dat", ("   3   1 5.51+08", "1  3   1 5.51+08"),
                           (self.LINE, self.LINE + "\nP  2   1         1.00-01 2.00-01"))
        with self.assertWarnsRegex(UserWarning, "1 with code P"):
            rates = read(path)
        for key in ("a_val", "inf_engy"):
            np.testing.assert_array_equal(rates["rates"][key], original[key])
        np.testing.assert_array_equal(rates["rates"]["excit"]["col_excit"], original["excit"]["col_excit"])

    def test_bare_nucleus(self):
        # ADAS gives files of one level without a configuration for bare nuclei (copmm#6/ls#c6),
        # which the reader failed on.
        path = os.path.join(self.tmp.name, "bare.dat")
        with open(path, "w") as f:
            f.write("Be+ 4         4         5            0.0\n"
                    "    1                   (1)0( 0.0)              0.0\n"
                    "   -1\n"
                    "  5.0    3       1.00+04 1.00+05\n"
                    "  -1\n"
                    "  -1  -1\n")
        adf04 = read(path)
        self.assertEqual(list(adf04["atomic"]["config"]), [""])
        np.testing.assert_array_equal(adf04["atomic"]["energy"], [0.])
        self.assertEqual(len(adf04["rates"]["excit"]["col_transitions"]), 0)

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

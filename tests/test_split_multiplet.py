"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks splitting LS resolved levels and PECs to j levels shifted to NIST
energies: configurations in the NIST form, levels missing from NIST, lines the
multiplet factors cannot split, and hydrogen-like ions.
"""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from colradpy import colradpy
from colradpy.colradpy_class import strip_closed_shells


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "examples" / "input"
CLOSED = ["1s2", "2s2", "2p6", "3s2", "3p6"]


def split_model(name, already_j=False):
    model = colradpy(str(INPUT / name), metas=np.array([0]), temp_grid=np.array([10.]),
                     electron_den=np.array([1e13]))
    # Degenerate levels in bundled data produce a pre-existing zero-energy
    # wavelength warning, outside this check's scope.
    with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
        model.solve_cr()
        # Without get_nist_levels_txt or split_structure_terms_to_levels first.
        model.shift_j_res_energy_to_nist(already_j=already_j)
        if not already_j:
            model.split_pec_multiplet()
    return model


class ClosedShellTests(unittest.TestCase):
    def test_strip_closed_shells(self):
        self.assertEqual(strip_closed_shells("1s2.2s2.2p6.3s.3p", CLOSED, "3s2"), "3s.3p")
        self.assertEqual(strip_closed_shells("1s2.2s2.2p6.3s", CLOSED, "2p6.3s"), "2p6.3s")
        self.assertEqual(strip_closed_shells("1s2.2s.2p", CLOSED, "1s2.2s2"), "1s2.2s.2p")
        self.assertEqual(strip_closed_shells("2s2", CLOSED, "1s"), "2s2")
        self.assertEqual(strip_closed_shells("1s", CLOSED, "1s"), "1s")

    def test_hydrogen_like_ion_finishes(self):
        # The closed shell removal looped forever when NIST's ground configuration
        # (1s for Be3+) has no closed subshell.
        script = (
            "import io, contextlib, warnings, numpy as np\n"
            "warnings.simplefilter('ignore')\n"
            "from colradpy import colradpy\n"
            f"m = colradpy(r'{INPUT / 'be3_adf04'}', metas=np.array([0]), temp_grid=np.array([10.]),"
            " electron_den=np.array([1e13]))\n"
            "with contextlib.redirect_stdout(io.StringIO()), np.errstate(all='ignore'):\n"
            "    m.split_structure_terms_to_levels()\n"
            "print(m.data['processed']['split']['config'][0])\n"
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                                timeout=120, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "1s")


class NistShiftTests(unittest.TestCase):
    def test_levels_missing_from_nist_keep_their_term_energy(self):
        model = split_model("cpb03_ls#be0.dat")
        split = model.data["processed"]["split"]
        atomic = model.data["atomic"]
        terms = split["term_map"].astype(int)
        shifted = split["nist_shifted"]
        self.assertEqual((~shifted).sum(), 4)
        np.testing.assert_array_equal(split["energy"][~shifted], atomic["energy"][terms[~shifted]])
        self.assertTrue((split["energy"] >= 0).all())
        for i in np.where(shifted)[0]:
            level = model._nist_level(split["config"][i], split["S"][i], split["L"][i], split["j"][i])
            self.assertEqual(split["energy"][i], float(level["energy"]))

    def test_j_resolved_shift_keeps_unmatched_energies(self):
        model = split_model("cpb03_ls#be0.dat", already_j=True)
        atomic = model.data["atomic"]
        shifted = atomic["nist_shifted"]
        self.assertTrue(shifted.any() and not shifted.all())
        np.testing.assert_array_equal(atomic["energy"][~shifted], atomic["adf04_energies"][~shifted])
        model.shift_j_res_pecs()
        self.assertTrue((model.data["processed"]["wave_vac_shift"] > 0).all())


class SplitPecTests(unittest.TestCase):
    def test_split_lines(self):
        for name in ("cpb03_ls#be0.dat", "cpb03_ls#be1.dat", "be2_adf04", "be3_adf04"):
            with self.subTest(file=name):
                model = split_model(name)
                processed = model.data["processed"]
                split = processed["split"]
                atomic = model.data["atomic"]
                # Every line is kept: the split PECs of a line add up to its PEC.
                total = np.zeros_like(processed["pecs"])
                np.add.at(total, split["unres_pec_map"], split["pecs"])
                np.testing.assert_allclose(total, processed["pecs"], rtol=1e-12, atol=0)
                self.assertFalse((split["wave_vac"] < 0).any())
                np.testing.assert_allclose(split["wave_air"][split["wave_vac"] < 200],
                                           split["wave_vac"][split["wave_vac"] < 200])

                upper, lower = split["unsplit_pec_levels"].T
                spin_change = atomic["S"][upper] != atomic["S"][lower]
                large_dl = np.abs(atomic["L"][upper] - atomic["L"][lower]) > 1
                kept = split["kept_unsplit"]
                self.assertTrue(kept.any())
                self.assertTrue((spin_change | large_dl)[kept].all())
                np.testing.assert_array_equal(split["pec_levels"][kept], -1)
                # Split entries are j levels of the line's own terms.
                terms = split["term_map"].astype(int)
                np.testing.assert_array_equal(terms[split["pec_levels"][~kept, 0]], upper[~kept])
                np.testing.assert_array_equal(terms[split["pec_levels"][~kept, 1]], lower[~kept])
                energy = split["energy"][split["pec_levels"][~kept]]
                with np.errstate(divide="ignore"):
                    np.testing.assert_allclose(split["wave_vac"][~kept],
                                               1e7 / np.abs(energy[:, 0] - energy[:, 1]))

    def test_unsplit_line_between_term_centres(self):
        model = split_model("cpb03_ls#be0.dat")
        split = model.data["processed"]["split"]
        weight = 2 * split["j"] + 1
        terms = split["term_map"].astype(int)
        k = np.where(split["kept_unsplit"])[0][0]
        upper, lower = split["unsplit_pec_levels"][k]
        centre = [np.sum((weight * split["energy"])[terms == t]) / np.sum(weight[terms == t])
                  for t in (upper, lower)]
        np.testing.assert_allclose(split["wave_vac"][k], 1e7 / abs(centre[0] - centre[1]))


if __name__ == "__main__":
    unittest.main()

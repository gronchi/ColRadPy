"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks reading the bundled NIST level files, whose text columns np.genfromtxt
returns as bytes on NumPy < 2 and as str on NumPy >= 2.
"""

from pathlib import Path
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from colradpy import colradpy
import colradpy.nist_read_txt as nist_read_txt


ROOT = Path(__file__).resolve().parents[1]
NIST = str(ROOT / "atomic" / "nist_energies") + "/"
GENFROMTXT = np.genfromtxt


def genfromtxt_bytes(*args, **kwargs):
    """np.genfromtxt returning bytes for text, as NumPy < 2 did by default."""
    with warnings.catch_warnings():
        # NumPy >= 2 deprecates this (VisibleDeprecationWarning).
        warnings.simplefilter("ignore")
        return GENFROMTXT(*args, encoding="bytes", **kwargs)


class NistReadTxtTests(unittest.TestCase):
    def check_levels(self, read):
        neutral = read(NIST, "Be", 1)
        self.assertEqual(neutral[0], {"conf": "1s2.2s2", "term": "1S", "j_val": "0",
                                      "energy": 0.0})
        self.assertEqual((neutral[2]["conf"], neutral[2]["term"], neutral[2]["j_val"]),
                         ("1s2.2s.2p", "3P", "1"))
        self.assertAlmostEqual(neutral[2]["energy"], 21978.925)
        # Fractional J values are text in the file.
        ion = read(NIST, "Be", 2)
        self.assertEqual((ion[2]["term"], ion[2]["j_val"]), ("2P", "3/2"))
        for level in neutral + ion:
            for key in ("conf", "term", "j_val"):
                self.assertIs(type(level[key]), str)

    def test_text_columns_are_str(self):
        self.check_levels(nist_read_txt.get_nist_txt)

    def test_bytes_from_older_numpy(self):
        self.assertIsInstance(genfromtxt_bytes(NIST + "be_1", dtype=None, delimiter=",")[0][0],
                              bytes)
        with patch.object(nist_read_txt.np, "genfromtxt", genfromtxt_bytes):
            self.check_levels(nist_read_txt.get_nist_txt)

    def test_model_reads_bundled_levels(self):
        model = colradpy(str(ROOT / "examples" / "input" / "cpb03_ls#be0.dat"),
                         metas=np.array([0]), temp_grid=np.array([10.]),
                         electron_den=np.array([1e13]))
        model.get_nist_levels_txt()
        self.assertEqual(model.data["nist"]["levels"][0]["conf"], "1s2.2s2")


if __name__ == "__main__":
    unittest.main()

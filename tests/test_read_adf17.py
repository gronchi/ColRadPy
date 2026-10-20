"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks that the adf17 reader is a function and that importing it reads no file.
"""

import importlib
import os
import sys
import tempfile
import unittest


class ReadAdf17Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_import_reads_no_file(self):
        # Importing the module opened '../cbnm93#h_h0ls.dat' and raised FileNotFoundError.
        cwd = os.getcwd()
        os.chdir(self.tmp.name)  # a working directory without that file
        try:
            sys.modules.pop("colradpy.read_adf17", None)
            module = importlib.import_module("colradpy.read_adf17")
        finally:
            os.chdir(cwd)
        self.assertTrue(callable(module.read_adf17))

    def test_errors(self):
        from colradpy.read_adf17 import read_adf17
        with self.assertRaises(FileNotFoundError):
            read_adf17(os.path.join(self.tmp.name, "missing.dat"))
        path = os.path.join(self.tmp.name, "bad.dat")
        with open(path, "w") as f:
            f.write("SEQ = H  NUCL.CHG. = 1\n")  # fewer than the five fields of the first line
        with self.assertRaisesRegex(ValueError, "first line"):
            read_adf17(path)


if __name__ == "__main__":
    unittest.main()

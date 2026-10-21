"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks adf04_file_paths: fix_whitespace on Python 3, and importing the module without the
download of an adf04 file from OpenADAS.
"""

import contextlib
import io
import os
import runpy
import tempfile
import unittest

from colradpy import adf04_file_paths
from colradpy.adf04_file_paths import fix_whitespace


class FixWhitespaceTests(unittest.TestCase):
    def fix(self, contents):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "be0_adf04")
            with open(path, "wb") as f:
                f.write(contents)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                fix_whitespace(path)
            with open(path, "rb") as f:
                return f.read(), out.getvalue()

    def test_trailing_whitespace_and_line_endings(self):
        # The function used the "rU" mode (removed in Python 3.11), os without importing it, and
        # wrote a str to a file opened "wb".
        contents, report = self.fix(b"Be+ 0  4  1 \r\n   1 1s2 2s2   (1)0( 0.0) \t\r\n  -1\rlast  \n")
        self.assertEqual(contents, b"Be+ 0  4  1\n   1 1s2 2s2   (1)0( 0.0)\n  -1\nlast\n")
        self.assertIn("be0_adf04", report)
        self.assertIn("Fixed trailing whitespace on 3 lines", report)
        self.assertIn("Fixed line endings to Unix", report)

    def test_clean_file_unchanged(self):
        contents, report = self.fix(b"Be+ 0  4  1\n  -1\n")
        self.assertEqual(contents, b"Be+ 0  4  1\n  -1\n")
        self.assertEqual(report, "")

    def test_bytes_of_any_encoding(self):
        contents, _ = self.fix("Ångström  \r\n".encode("latin-1"))
        self.assertEqual(contents, "Ångström\n".encode("latin-1"))


class ImportTests(unittest.TestCase):
    def test_import_does_not_download(self):
        # Importing the module downloaded h1_adf04 from OpenADAS into the current folder.
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                runpy.run_path(adf04_file_paths.__file__)       #as an import, not as __main__
                self.assertEqual(os.listdir(tmp), [])
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()

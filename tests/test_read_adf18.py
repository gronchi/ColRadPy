"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks the reader of adf18 a17_p208 expansion files (ADAS appxa-18), which map the n-shells of
condensed bundle-nS matrices onto the levels of an adf04 file, with a file written here in the
layout of the exp96 files.
"""

import os
import tempfile
import unittest

from colradpy.read_adf18 import read_adf18


EXP = """ &SEQINF SEQ='F',
         DSNREF='"ADASCENT"/adf04/test/test_ls#ne1.dat',
         DSNCPM='"ADASCENT"/adf17/test/test_ne1ls.dat',
         NPARNT=2,
         NSHEL=3,
         NLEV=  5,
 &END

       PARENT      :(3P)       :(1D)       :
       SPINPRT     :3          :1          :
       NSPNSYS     :2          :1          :

    PARENT =  1
    -----------
       SPINSYS     :2          :4          :
       LOWESTN     :2          :3          :
       IMAX        :2          :1          :
       LOWESTP     :1.0000     :1.0000     :
       {key}     :0.6000     :0.3750     :

    PARENT =  2
    -----------
       SPINSYS     :2          :
       LOWESTN     :2          :
       IMAX        :1          :
       LOWESTP     :1.0000     :
       {key}     :0.2000     :

 RESOLVED NL  <-- BUNDLE-N                N:1     N:2     N:3     N:4     N:5

INDX CODE                 SZD SP SH PT   N-SHELL NORMALISED WEIGHTS
---- ----                 --- -- -- --   --------------------------
  1* 2P5        (2)1( 2.5)   0  2  2  1  :       : 1.0000:       :       :
  1* 2P5        (2)1( 2.5)   0  2  2  2  :       : 1.0000:       :       :
  2 2S 2P6     (2)0( 0.5)   0  2  2  0  :       :       :       :       :
  3 2P4 3S     (4)1( 5.5)   0  4  3  1  :       :       : 1.0000:       :
  4 2P4 3S     (2)1( 2.5)   0  2  3  1  :       :       : 0.3000:       :
  5 2P4 3P     (2)2( 4.5)   0  2  3  1  :       :       : 0.7000:       :

C-----------------------------------------------------------------
C  written for the ColRadPy tests
"""


class ReadAdf18Tests(unittest.TestCase):
    def read(self, key="FRPARNT"):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "exp96#f_ne1ls.dat")
            with open(path, "w") as f:
                f.write(EXP.replace("{key}", f"{key:<7s}"))
            return read_adf18(path)

    def test_namelist_and_parents(self):
        data = self.read()
        self.assertEqual((data["sequence"], data["num_parents"], data["num_shells"], data["num_levels"]), ("F", 2, 3, 5))
        self.assertEqual(data["adf04_file"], '"ADASCENT"/adf04/test/test_ls#ne1.dat')
        self.assertEqual(data["adf17_file"], '"ADASCENT"/adf17/test/test_ne1ls.dat')
        self.assertEqual([p["term"] for p in data["parents"]], ["3P", "1D"])
        self.assertEqual([p["spin"] for p in data["parents"]], [3, 1])
        first = data["parents"][0]["spin_systems"]
        self.assertEqual([(s["spin"], s["lowest_n"], s["imax"]) for s in first], [(2, 2, 2), (4, 3, 1)])
        self.assertEqual([s["parentage"] for s in first], [0.6, 0.375])

    def test_records(self):
        records = self.read()["records"]
        self.assertEqual([r["level"] for r in records], [1, 1, 2, 3, 4, 5])
        self.assertEqual([r["parent"] for r in records], [1, 2, 0, 1, 1, 1])
        self.assertTrue(records[0]["metastable"])
        self.assertFalse(records[3]["metastable"])
        self.assertEqual((records[4]["spin"], records[4]["shell"], records[4]["weight"]), (2, 3, 0.3))
        self.assertEqual(records[2]["weight"], 0.)
        self.assertEqual(records[3]["code"], "2P4 3S (4)1( 5.5)")

    def test_older_parentage_label(self):
        # older files (exp93#li, exp96#li_c3ls) write the fractional parentage as PRTWHT
        self.assertEqual(self.read("PRTWHT")["parents"][1]["spin_systems"][0]["parentage"], 0.2)

    def test_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.dat")
            with open(path, "w") as f:
                f.write("not an expansion file\n")
            with self.assertRaisesRegex(ValueError, "not an adf18 file of the a17_p208 kind"):
                read_adf18(path)
            with open(path, "w") as f:   #p310_a17 and p311_a17 files have a namelist without DSNCPM
                f.write(" &SEQINF SEQ='H',REFMEM=' ',NPARNT=1,NSHEL=4,NLEV=4,&END\n")
            with self.assertRaisesRegex(ValueError, "other kinds .* are not read"):
                read_adf18(path)
            with open(path, "w") as f:
                f.write(EXP.replace("{key}", "FRPARNT").split(" RESOLVED")[0])
            with self.assertRaisesRegex(ValueError, "no level records"):
                read_adf18(path)


if __name__ == "__main__":
    unittest.main()

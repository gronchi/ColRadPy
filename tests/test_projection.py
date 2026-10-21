"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks the projection of the high n-shells (colradpy.projection): the condensation of n-shells
onto the lower ones, and the indirect couplings added to the
collisional-radiative model of the bundled Be I file, with adf17 and adf18 files written here.
"""

import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

from colradpy import colradpy, ionization_balance
from colradpy.projection import _indirect, indirect_couplings
from colradpy.read_adf17 import read_adf17
from colradpy.read_adf18 import read_adf18


INPUT = Path(__file__).resolve().parents[1] / "examples" / "input"
BE0 = str(INPUT / "cpb03_ls#be0.dat")
TEMPS_K = np.array([1.e4, 1.e5, 1.e6])
DENS = np.array([1.e4, 1.e8, 1.e12])
SHELLS = [2, 3, 4, 5]
#levels of the Be I file by spin system and n-shell (2p2 levels 6 and 8 belong to no shell)
LEVELS = {1: {2: [1, 3], 3: [5, 9, 11], 4: [13, 15, 18, 19], 5: [21, 23, 26, 27, 29]},
          3: {2: [2], 3: [4, 7, 10], 4: [12, 14, 16, 17], 5: [20, 22, 24, 25, 28]}}


def field(value):
    return f"{value:.10E}".replace("E", "D").rjust(17)


def write_adf17(path, systems, switches="IEDMAT = 0   IECION = 0   IETREC = 0   IEDREC = 0   IERREC = 0   IEXREC = 0   IERSYS = 0"):
    """An adf17 file of one parent (2S) with the spin systems {spin: arrays} on TEMPS_K x DENS, in the
    fixed columns of ADAS208 (cbnm96 layout)."""
    def grid(label, values):
        return " " + label + "".join(f"{v:.2E}".replace("E", "D").rjust(10) for v in values)

    lines = [f" SEQUENCE = BE   NUCCHG = 4   NPARNT = 1   MAXD = {len(DENS)}   MAXT ={len(TEMPS_K):2d}", " ",
             " " + switches, " ", grid("NE(CM-3) =", DENS), grid("TE(K)    =", TEMPS_K), " ",
             " IPRT =  1     TRMPRT = (2S)     SPNPRT =   2.", " " + "-" * 44]
    for spin, s in systems.items():
        n = len(s["n"])
        lines += [" " * 50 + f"SPNSYS = {spin:2d}.     NSHEL = {n:2d}", " " * 50 + "-" * 27]
        for d in range(len(DENS)):
            for t in range(len(TEMPS_K)):
                for kind in ("proj", "dir"):
                    for i in range(n):
                        prefix = (f" ID={d + 1:2d} IT={t + 1:2d} " if kind == "proj" and i == 0 else
                                  "       (DIR) " if i == 0 else " " * 13) + f"N={s['n'][i]:2d}  "
                        line = [" "] * 260
                        line[0:len(prefix)] = prefix
                        matrix = s["crmat"] if kind == "proj" else s["dir_crmat"]
                        ion = s["ion"] if kind == "proj" else s["dir_ion"]
                        cells = {19 + 17 * j: matrix[i, j, t, d] for j in range(n)}
                        cells[106] = ion[i, t, d]
                        tail = ([s["rhs"][i, t, d]] if kind == "proj" else
                                [s[k][i, t, d] for k in ("three", "diel", "rad", "cx")])
                        cells.update({176 + 17 * c: v for c, v in enumerate(tail)})
                        for column, value in cells.items():
                            line[column:column + 17] = field(value)
                        lines.append("".join(line).rstrip())
                qc = [" "] * 130
                qc[106:123] = field(0.)
                lines.append("".join(qc).rstrip())
                lines.append(" ")
    lines += ["C" + "-" * 79, "C  written for the ColRadPy tests", "C" + "-" * 79]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def write_adf18(path, systems_shells=SHELLS):
    """An adf18 a17_p208 file mapping the Be I levels onto the n-shells of parent 1 (2S)."""
    lines = [" &SEQINF SEQ='BE',", "         DSNREF='\"ADASCENT\"/adf04/test/cpb03_ls#be0.dat',",
             "         DSNCPM='\"ADASCENT\"/adf17/test/test.dat',", "         NPARNT=1,", "         NSHEL=4,",
             "         NLEV= 29,", " &END", " ",
             "       PARENT      :(2S)       :", "       SPINPRT     :2          :", "       NSPNSYS     :2          :", " ",
             "    PARENT =  1", "    -----------", "       SPINSYS     :1          :3          :",
             "       LOWESTN     :2          :2          :", "       IMAX        :4          :4          :",
             "       LOWESTP     :1.0000     :1.0000     :", "       FRPARNT     :1.0000     :1.0000     :", " ",
             " RESOLVED NL  <-- BUNDLE-N                N:1     N:2     N:3     N:4     N:5",
             "INDX CODE                 SZD SP SH PT   N-SHELL NORMALISED WEIGHTS",
             "---- ----                 --- -- -- --   --------------------------"]
    with contextlib.redirect_stdout(io.StringIO()):
        atomic = colradpy(BE0, np.array([0]), np.array([10.]), np.array([1e10])).data["atomic"]
    for level in range(1, 30):
        spin = int(atomic["S"][level - 1])
        shell = next((n for n, levels in LEVELS.get(spin, {}).items() if level in levels and n in systems_shells), 0)
        weight = 0.
        if shell:
            g = lambda k: 2 * atomic["L"][k - 1] + 1
            weight = g(level) / sum(g(k) for k in LEVELS[spin][shell])
        cells = ["       "] * 5
        if shell:
            cells[shell - 1] = f" {weight:.4f}"
        lines.append(f"{level:3d}  {'LEVEL':<24s}  0 {spin:2d} {shell:2d} {1 if shell else 0:2d}  :" + ":".join(cells) + ":")
    lines += ["C" + "-" * 60, "C  written for the ColRadPy tests"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def bundle(seed, shells=SHELLS, indirect_gain=0., indirect_ion=0., indirect_recomb=0.):
    """Condensed matrices of a spin system with chosen indirect parts: direct transfers between the
    n-shells (loss positive on the diagonal), direct ionization and recombination, and indirect
    transfers, ionization and recombination scaled by the given factors."""
    rng = np.random.default_rng(seed)
    n = len(shells)
    shape = (len(TEMPS_K), len(DENS))
    scale = np.sqrt(TEMPS_K / 1e5)[:, None] * (1 + DENS[None, :] / 1e10)
    direct_gain = rng.uniform(1e2, 1e5, (n, n))[..., None, None] * scale
    indirect = rng.uniform(1e0, 1e3, (n, n))[..., None, None] * scale * indirect_gain
    for g in (direct_gain, indirect):
        g[np.arange(n), np.arange(n)] = 0.
    dir_ion = rng.uniform(1e-2, 1e2, n)[:, None, None] * scale
    ind_ion = rng.uniform(1e-3, 1e1, n)[:, None, None] * scale * indirect_ion
    direct = -direct_gain.copy()
    direct[np.arange(n), np.arange(n)] = direct_gain.sum(axis=0)
    projected = direct - indirect
    projected[np.arange(n), np.arange(n)] += indirect.sum(axis=0) + dir_ion + ind_ion
    rad = rng.uniform(1e-14, 1e-13, n)[:, None, None] * np.ones(shape)
    diel = rng.uniform(1e-15, 1e-14, n)[:, None, None] * np.ones(shape)
    three = rng.uniform(1e-30, 1e-29, n)[:, None, None] * DENS[None, None, :] * np.ones(shape)
    ind_recomb = rng.uniform(1e-14, 1e-13, n)[:, None, None] * np.ones(shape) * indirect_recomb
    return {"n": shells, "crmat": projected, "dir_crmat": direct, "ion": dir_ion + ind_ion, "dir_ion": dir_ion,
            "rhs": rad + diel + three + ind_recomb, "rad": rad, "diel": diel, "three": three,
            "cx": np.zeros((n,) + shape), "indirect": {"gains": indirect, "ionization": ind_ion, "recombination": ind_recomb}}


def solve(projection=None, temps=None, dens=None, **kwargs):
    temps = TEMPS_K / 11604.5 if temps is None else temps
    dens = DENS if dens is None else dens
    with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = colradpy(BE0, np.array([0]), temps, dens, projection=projection, **kwargs)
        model.solve_cr()
    return model


class CondensationTests(unittest.TestCase):
    def test_upper_shells_condensed(self):
        # Eliminating the shells above the kept ones gives the populations, ionization and parent
        # coupling of the full bundle-nS system for any recombination source.
        s = bundle(1, shells=[2, 3, 4, 5, 6], indirect_gain=1., indirect_ion=1., indirect_recomb=1.)
        data = {"crmat": np.moveaxis(s["crmat"], (2, 3), (0, 1)), "ion": np.moveaxis(s["ion"], (1, 2), (0, 1))[..., None],
                "rhs": np.moveaxis(s["rhs"], (1, 2), (0, 1)), "dir_crmat": np.moveaxis(s["dir_crmat"], (2, 3), (0, 1)),
                "dir_ion": np.moveaxis(s["dir_ion"], (1, 2), (0, 1))[..., None], "parent_qc": np.zeros((3, 3, 1))}
        for key in ("rad", "diel", "three", "cx"):
            data["recomb_" + {"rad": "rad", "diel": "diel", "three": "three_body", "cx": "cx"}[key]] = \
                np.moveaxis(s[key], (1, 2), (0, 1))
        kept = np.array([True, True, True, False, False])
        out = _indirect(data, kept)
        for t in range(3):
            for d in range(3):
                P, r, I = data["crmat"][t, d], data["rhs"][t, d], data["ion"][t, d, :, 0]
                x = np.linalg.solve(P, r)                    #full system
                #direct couplings between the kept shells, without the direct transfers to the others
                D = data["dir_crmat"][t, d].copy()
                l, u = slice(0, 3), slice(3, 5)
                D[np.arange(3), np.arange(3)] += D[u, l].sum(axis=0)
                #the kept populations solve the condensed system
                P_cond = D[l, l] + (np.diag(out["loss"][t, d]) - out["gains"][t, d])
                P_cond[np.arange(3), np.arange(3)] += data["dir_ion"][t, d, :3, 0]
                recomb = out["recombination"][t, d] + sum(data[k][t, d, :3] for k in
                                                         ("recomb_rad", "recomb_diel", "recomb_three_body", "recomb_cx"))
                np.testing.assert_allclose(np.linalg.solve(P_cond, recomb), x[:3], rtol=1e-9)
                #all the ionization of the full system, from the kept shells and as parent coupling
                ionization = (out["ionization"][t, d, :, 0] + data["dir_ion"][t, d, :3, 0]) @ x[:3] + out["parent_coupling"][t, d, 0]
                np.testing.assert_allclose(ionization, I @ x, rtol=1e-9)

    def test_indirect_parts_recovered(self):
        # With all shells kept, the indirect parts are the projected minus the direct ones.
        s = bundle(2, indirect_gain=1., indirect_ion=1., indirect_recomb=1.)
        with tempfile.TemporaryDirectory() as tmp:
            adf17 = read_adf17(write_adf17(os.path.join(tmp, "c.dat"), {1: s, 3: bundle(3)}))["input_file"]
            adf18 = read_adf18(write_adf18(os.path.join(tmp, "e.dat")))
        group = next(g for g in indirect_couplings(adf17, adf18) if g["spin"] == 1)
        np.testing.assert_array_equal(group["shells"], SHELLS)
        np.testing.assert_allclose(group["gains"], np.moveaxis(s["indirect"]["gains"], (2, 3), (0, 1)), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(group["ionization"][..., 0], np.moveaxis(s["indirect"]["ionization"], (1, 2), (0, 1)),
                                   rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(group["recombination"], np.moveaxis(s["indirect"]["recombination"], (1, 2), (0, 1)),
                                   rtol=1e-6, atol=1e-20)


class ColradpyProjectionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.adf18 = write_adf18(os.path.join(self.dir, "exp.dat"))

    def files(self, singlet, triplet):
        return write_adf17(os.path.join(self.dir, "cbnm.dat"), {1: singlet, 3: triplet}), self.adf18

    def test_conservation(self):
        projection = self.files(bundle(4, indirect_gain=1., indirect_ion=1., indirect_recomb=1.),
                                bundle(5, indirect_gain=1., indirect_ion=1., indirect_recomb=1.))
        model = solve(projection)
        cr = model.data["cr_matrix"]["cr"]
        scale = np.abs(np.diagonal(cr, axis1=0, axis2=1)).max(axis=-1)
        self.assertLess((np.abs(cr.sum(axis=0)).max(axis=0) / scale).max(), 1e-13)
        reference = solve()
        self.assertFalse(np.allclose(model.data["processed"]["acd"], reference.data["processed"]["acd"], rtol=1e-6, atol=0))

    def test_recombination(self):
        # Without ionization all the recombined population ends in the ground: ACD grows by the
        # indirect recombination, summed over the levels with their weights (over the shells).
        singlet, triplet = bundle(6, indirect_recomb=1.), bundle(7, indirect_recomb=1.)
        options = dict(use_ionization=False, suppliment_with_ecip=False, use_recombination_three_body=False)
        model, reference = solve(self.files(singlet, triplet), **options), solve(**options)
        added = (singlet["indirect"]["recombination"] + triplet["indirect"]["recombination"]).sum(axis=0)[:, 0]
        acd = model.data["processed"]["acd"][0, 0, :, 0] - reference.data["processed"]["acd"][0, 0, :, 0]
        np.testing.assert_allclose(acd, added, rtol=1e-9)
        level = model.data["rates"]["projection"]["recombination"][:, 0, :, 0]
        np.testing.assert_allclose(level[0], 0.25 * singlet["indirect"]["recombination"][0, :, 0], rtol=1e-9)
        np.testing.assert_array_equal(level[[5, 7]], 0.)          #the 2p2 levels

    def test_ionization(self):
        # The levels of a shell ionize at the rate of the shell; the ground's adds to SCD directly.
        singlet, triplet = bundle(8, indirect_ion=1.), bundle(9)
        model, reference = solve(self.files(singlet, triplet)), solve()
        rates = model.data["rates"]["projection"]["ionization"][:, 0, :, 0]
        expected = singlet["indirect"]["ionization"][0, :, 0] / DENS[0]
        np.testing.assert_allclose(rates[0], expected, rtol=1e-9)
        np.testing.assert_allclose(rates[2], expected, rtol=1e-9)    #2s2p 1P, same shell
        scd = model.data["processed"]["scd"][0, 0, :, 0] - reference.data["processed"]["scd"][0, 0, :, 0]
        #the ground with its normalized population, and 2s2p 1P with its population
        processed = model.data["processed"]
        ground = 1 / (1 + processed["pops_no_norm"][:, 0, :, 0].sum(axis=0))
        level_1p = list(processed["excited_levels"]).index(2)
        np.testing.assert_allclose(scd, expected * (ground + processed["F"][level_1p, 0, :, 0]), rtol=1e-9)

    def test_transfers(self):
        # A transfer from shell m to shell n reaches the levels of n with their weights.
        singlet = bundle(10, indirect_gain=1.)
        model = solve(self.files(singlet, bundle(11)))
        couplings = model.data["rates"]["projection"]["couplings"][..., 0, 0]
        gains = singlet["indirect"]["gains"][..., 0, 0]
        self.assertAlmostEqual(couplings[0, 4] / gains[0, 1], 0.25, places=6)   #2s3s 1S -> 2s2 1S
        self.assertAlmostEqual(couplings[2, 4] / gains[0, 1], 0.75, places=6)   #2s3s 1S -> 2s2p 1P
        self.assertEqual(couplings[0, 2], 0.)                                    #same shell
        self.assertEqual(couplings[1, 4], 0.)                                    #another spin system

    def test_pairs_and_balance(self):
        projection = self.files(bundle(12, indirect_recomb=1., indirect_ion=1.), bundle(13, indirect_recomb=1.))
        temps = TEMPS_K / 11604.5
        grid = solve(projection)
        pairs = solve(projection, temps=temps, dens=DENS, temp_dens_pair=True)
        for key in ("scd", "acd"):
            np.testing.assert_allclose(pairs.data["processed"][key][..., 1],
                                       grid.data["processed"][key][..., 1, 1], rtol=1e-9)
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ion = ionization_balance([BE0, str(INPUT / "cpb03_ls#be1.dat")], np.array([0]), temps, DENS,
                                     projection=[projection, None])
        np.testing.assert_allclose(ion.data["cr_data"]["gcrs"]["0"]["acd"], grid.data["processed"]["acd"], rtol=1e-9)

    def test_level_order(self):
        # adf18 records number the levels as the adf04 file; levels out of energy order are sorted
        # by the reader, and the records follow them.
        text = Path(BE0).read_text()
        text = text.replace("(3)0( 1.0)    52081.0", "(3)0( 1.0)    55081.0")   #2s3s 3S above 2s3s 1S
        path = os.path.join(self.dir, "cpb03_ls#be0.dat")
        Path(path).write_text(text)
        triplet = bundle(18, indirect_recomb=1.)
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = colradpy(path, np.array([0]), TEMPS_K / 11604.5, DENS,
                             projection=self.files(bundle(19), triplet))
            model.make_projection()
        order = model.data["atomic"]["file_order"]
        self.assertEqual(list(order[3:5]), [4, 3])                 #file levels 5 and 4
        sorted_level = list(order).index(3)                        #the 2s3s 3S of the file
        recombination = model.data["rates"]["projection"]["recombination"][:, 0, 0, 0]
        weight = 3 / (3 + 9 + 15)                                  #2s3s 3S, 2s3p 3P, 2s3d 3D
        self.assertAlmostEqual(recombination[sorted_level] / triplet["indirect"]["recombination"][1, 0, 0], weight, places=4)
        self.assertEqual(model.data["atomic"]["S"][sorted_level], 3)

    def test_unreliable_condensed_data(self):
        # Recombination of +x and -x into two shells at the lowest temperature, as cbnm96 files have
        # there: that temperature is left out of the recombination only, with a warning when the
        # user temperatures need it.
        singlet = bundle(20, indirect_recomb=1., indirect_ion=1.)
        singlet["rhs"][0, 0, :] += 1e-6                              #n=2 at 1e4 K
        singlet["rhs"][1, 0, :] -= 1e-6                              #n=3
        projection = self.files(singlet, bundle(21))

        def project(temps):
            with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(io.StringIO()):
                warnings.simplefilter("always")
                model = colradpy(BE0, np.array([0]), temps, DENS, projection=projection)
                model.make_projection()
            return model.data["rates"]["projection"], [str(w.message) for w in caught if "round-off" in str(w.message)]

        rates, messages = project(TEMPS_K / 11604.5)
        self.assertEqual(len(messages), 1)
        for text in ("parent 1 (2S) and spin system 1", "negative indirect recombination", "at 0.862 eV"):
            self.assertIn(text, messages[0])
        #the indirect recombination is the same at every temperature, so the value of the next one
        #is the value without the artifact
        np.testing.assert_allclose(rates["recombination"][0, 0, 0, :],
                                   0.25 * singlet["indirect"]["recombination"][0, 0, :], rtol=1e-6)
        np.testing.assert_allclose(rates["ionization"][0, 0, 0, :],
                                   singlet["indirect"]["ionization"][0, 0, :] / DENS, rtol=1e-6)
        _, messages = project(TEMPS_K[1:] / 11604.5)
        self.assertEqual(messages, [])

    def test_errors(self):
        flagged = write_adf17(os.path.join(self.dir, "c93.dat"), {1: bundle(14), 3: bundle(15)},
                              switches="IEDMAT = 0   IECION = 0   IETREC = 0   IEDREC = 0   IERREC = 0   IEXREC = 0   "
                                       "IERSYS = 0   IEFPRS = 0   IEFPRE = 1")
        with self.assertRaisesRegex(NotImplementedError, "fractional parentage"):
            solve((flagged, self.adf18))
        with self.assertRaisesRegex(ValueError, "adf17 file of condensed matrices and the adf18"):
            colradpy(BE0, np.array([0]), np.array([10.]), np.array([1e10]), projection=(flagged,))
        other = Path(self.adf18).read_text().replace("cpb03_ls#be0.dat", "cpb03_ls#be1.dat")
        path = os.path.join(self.dir, "other.dat")
        Path(path).write_text(other)
        with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("always")
            model = colradpy(BE0, np.array([0]), TEMPS_K / 11604.5, DENS,
                             projection=(self.files(bundle(16), bundle(17))[0], path))
            model.make_projection()
        self.assertTrue(any("maps the levels of cpb03_ls#be1.dat" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()

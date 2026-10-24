"""Generate theory-manual figures and checks from the development worktree."""

import ast
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.linalg import expm


THEORY = Path(__file__).resolve().parents[1]  # docs/source/theory
ROOT = THEORY.parents[2]
OUT = THEORY / "figures"
BLUE, ORANGE, AQUA, GREY = "#2a6fb0", "#d9822b", "#2f9e8f", "#555555"
sys.path.insert(0, str(ROOT))


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def tex_number(value):
    mantissa, exponent = f"{value:.4e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def examples():
    branch = git("branch", "--show-current")
    if branch != "development":
        raise RuntimeError(f"The manual describes development; checked out branch is {branch!r}.")

    from colradpy import colradpy
    from colradpy.solve_matrix_exponential import (
        solve_matrix_exponential, solve_matrix_exponential_source,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 9, "legend.fontsize": 7.5,
        "axes.linewidth": 0.6, "lines.linewidth": 1.2,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })

    # Analytic two-level illustration, independent of atomic data.
    density = np.logspace(8, 16, 240)
    a, upward, downward = 1e7, 1e-8, 3e-8
    excited = density * upward / (a + density * (upward + downward))
    pec = a * upward / (a + density * (upward + downward))
    np.savetxt(OUT / "two_level.csv", np.c_[density, excited, pec], delimiter=",",
               header="density_cm-3,excited_fraction,pec_cm3_s-1", comments="")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.45), constrained_layout=True)
    axes[0].semilogx(density, excited, color=BLUE)
    axes[0].axhline(upward / (upward + downward), color=GREY, ls="--", lw=0.8)
    axes[0].set(xlabel=r"$n_e$ (cm$^{-3}$)", ylabel=r"$N_1/(N_0+N_1)$")
    axes[1].loglog(density, pec, color=ORANGE)
    axes[1].set(xlabel=r"$n_e$ (cm$^{-3}$)", ylabel=r"PEC (cm$^3$ s$^{-1}$)")
    fig.savefig(OUT / "two_level.pdf")
    plt.close(fig)

    # Be II XCD compared with a direct Schur-complement reduction.
    be = colradpy(str(ROOT / "examples/input/cpb03_ls#be1.dat"),
                  np.array([0, 1]), np.array([5., 20.]), np.array([1e12, 1e14]))
    be.solve_cr()
    matrix = be.data["cr_matrix"]["cr"][:, :, 0, 0]
    bound = len(be.data["atomic"]["energy"])
    excited_rows = be.data["processed"]["excited_levels"]
    parents = np.arange(bound, matrix.shape[0])
    reduced = matrix[np.ix_(parents, parents)] - matrix[np.ix_(parents, excited_rows)] @ np.linalg.solve(
        matrix[np.ix_(excited_rows, excited_rows)], matrix[np.ix_(excited_rows, parents)]
    )
    xcd_reference = reduced[0, 1] / 1e12
    xcd_model = be.data["processed"]["xcd"][1, 0, 0, 0]
    np.testing.assert_allclose(xcd_model, xcd_reference, rtol=1e-10, atol=0.)

    # Equal-rate chain: its defective eigenbasis requires a robust exponential.
    chain = np.array([[-1., 0., 0.], [1., -1., 0.], [0., 1., 0.]])
    initial = np.array([1., 0., 0.])
    times = np.linspace(0., 5., 151)
    exact = np.array([np.exp(-times), times * np.exp(-times),
                      1. - (1. + times) * np.exp(-times)])
    model = solve_matrix_exponential(chain[:, :, None, None], initial, times)[0][:, :, 0, 0]
    independent = np.stack([expm(chain * t) @ initial for t in times], axis=1)
    np.testing.assert_allclose(independent, exact, atol=1e-13)
    np.testing.assert_allclose(model, exact, atol=1e-12)
    np.savetxt(OUT / "chain.csv", np.c_[times, exact.T, model.T], delimiter=",",
               header="time_s,exact_0,exact_1,exact_2,development_0,development_1,development_2",
               comments="")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.45), constrained_layout=True)
    for level, color in enumerate((BLUE, ORANGE, AQUA)):
        axes[0].plot(times, exact[level], color=color, label=f"level {level}")
        axes[0].plot(times, model[level], color=color, ls="--", lw=0.8)
    axes[0].set(xlabel="Time (s)", ylabel="Population")
    axes[0].legend(frameon=False)
    axes[1].plot(times, exact.sum(axis=0), color=GREY, label="analytic")
    axes[1].plot(times, model.sum(axis=0), color=ORANGE, ls="--", label="development")
    axes[1].set(xlabel="Time (s)", ylabel="Total population")
    axes[1].legend(frameon=False)
    fig.savefig(OUT / "chain.pdf")
    plt.close(fig)

    slow = np.array([[-1e-12, 0.], [1e-12, 0.]])[:, :, None, None]
    slow_model = solve_matrix_exponential_source(
        slow, np.array([1., 0.]), np.zeros(2), np.array([1e12])
    )[0][:, 0, 0, 0]
    np.testing.assert_allclose(slow_model, [np.exp(-1), 1 - np.exp(-1)], atol=1e-12)
    results = {
        "chain_max_population_error": float(np.max(np.abs(model - exact))),
        "chain_development_t1": model[:, 30].tolist(),
        "chain_exact_t1": exact[:, 30].tolist(),
        "slow_development_t1e12": slow_model.tolist(),
        "slow_exact_t1e12": [float(np.exp(-1)), float(1 - np.exp(-1))],
        "be1_xcd_development_cm3_s": float(xcd_model),
        "be1_xcd_schur_cm3_s": float(xcd_reference),
        "be1_density_cm-3": 1e12,
        "be1_temperature_eV": 5.,
        "be1_metastables": [0, 1],
    }
    (OUT / "checks.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    table = [r"\begin{tabular}{lrr}\toprule Quantity & Development & Reference\\\midrule",
             f"Chain: $N_1(1)$ & {model[1,30]:.6f} & {exact[1,30]:.6f}" + r"\\",
             f"Slow mode: $N_0(10^{{12}})$ & {slow_model[0]:.6f} & {np.exp(-1):.6f}" + r"\\",
             r"Be II: XCD$[1,0]$ & " + tex_number(xcd_model) + " & " + tex_number(xcd_reference) + r"\\",
             r"\bottomrule\end{tabular}"]
    (OUT / "results.tex").write_text("\n".join(table) + "\n", encoding="utf-8")

    source_files = sorted((ROOT / "colradpy").glob("*.py"))
    check_inputs = [
        Path(__file__).resolve(),
        ROOT / "examples/input/cpb03_ls#be1.dat",
    ]
    documentation_files = [
        THEORY / "ColRadPy_theory_manual.tex",
        THEORY / "scripts/build_theory_manual.py",
        THEORY.parent / "theory_manual.rst",
        ROOT / "pyproject.toml",
    ]
    provenance = {
        "git_branch": branch,
        "git_head": git("rev-parse", "HEAD"),
        "working_tree_changes": git("status", "--short", "--", "colradpy"),
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
        "source_import": str(Path(sys.modules["colradpy"].__file__).relative_to(ROOT)),
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_files},
        "check_input_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in check_inputs},
        "documentation_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in documentation_files},
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    source_map = {
        "colradpy/colradpy_class.py": [
            "populate_data", "make_ioniz_from_reduced_ionizrates", "suppliment_with_ecip",
            "make_cx_rates_from_file", "make_recombination_rates_from_file",
            "make_three_body_recombination", "make_electron_excitation_rates",
            "populate_cr_matrix", "solve_quasi_static", "solve_time_dependent",
            "split_pec_multiplet", "shift_j_res_energy_to_nist", "solve_cr",
            "make_nat_broad", "make_dopp_broad", "write_pecs_adf15", "get_sxb", "write_sxb_adf13",
        ],
        "colradpy/ionization_balance_class.py": [
            "populate_ion_matrix", "solve_no_source", "solve_source", "solve_time_independent",
            "metastable_fractions", "collapse_metastables",
        ],
        "colradpy/solve_matrix_exponential.py": [
            "solve_matrix_exponential", "eval_matrix_exponential_solution",
            "solve_matrix_exponential_steady_state", "solve_matrix_exponential_source",
        ],
        "colradpy/energy_balance.py": [
            "solve", "plasma_energy_transfer_frequency", "neutral_energy_transfer_frequency",
            "coulomb_logarithm", "debye_length",
        ],
        "colradpy/read_fac.py": ["read_fac", "_fold_autoionization", "fac_cross_sections"],
        "colradpy/convolve_eedf.py": ["rate_coefficients", "resonance_rate_coefficients", "maxwellian"],
    }
    rows = [r"\begingroup\footnotesize",
            r"\begin{longtable}{>{\raggedright\arraybackslash}p{6.3cm}>{\raggedright\arraybackslash}p{5.9cm}r}",
            r"\toprule File & Entry point & Line\\\midrule\endfirsthead",
            r"\toprule File & Entry point & Line\\\midrule\endhead"]
    for filename, names in source_map.items():
        tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
        for index, name in enumerate(names):
            matches = [node.lineno for node in ast.walk(tree)
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one {name} in {filename}, found {len(matches)}")
            label = rf"\code{{{filename}}}" if index == 0 else "same"
            rows.append(label + rf" & \routine{{{name}}} & {matches[0]}" + r"\\")
    rows.extend([r"\bottomrule", r"\end{longtable}", r"\endgroup"])
    (OUT / "source_map.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"branch": branch, "head": provenance["git_head"], "results": results}, indent=2))


if __name__ == "__main__":
    examples()

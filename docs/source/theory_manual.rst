Theory manual
=============

:download:`Read the theory manual (PDF) <theory/ColRadPy_theory_manual.pdf>`
or edit its :download:`LaTeX source <theory/ColRadPy_theory_manual.tex>`.

The manual describes the ``development`` branch. Its ten main
chapters follow atomic processes through the level-rate matrix, stationary and
transient populations, quasistatic reduction, effective charge-state kinetics
and observable emission. Worked examples appear beside their derivations.
Population responses, normalization and retained versus total ion fractions
are explained together before the effective coefficients are used.

Later chapters develop high-shell projection, atomic-data preparation
(including FAC), charge-state kinetic-energy evolution, and applicability
checks. Appendices provide array and file conventions, numerical algorithms,
the ADAS production chain, the source-code map and evidence/reproduction paths.
Limitations remain beside the relevant theory, including two-photon spectra,
relativistic equilibrium consistency, rate filtering and the charge-state
layout required by the energy solver. Published Be/Mo results are
distinguished from the bundled checks. Blue boxes explain implementation
choices and orange boxes state limitations.

Regenerate checks and figures
-----------------------------

From the repository root, with the ``development`` branch checked out::

    python docs/source/theory/scripts/theory_examples.py
    python docs/source/theory/scripts/build_theory_manual.py

The generator imports the current working-tree code and records the branch,
HEAD, changed package source files, source hashes, and runtime versions in
``docs/source/theory/figures/provenance.json``. It compares an equal-rate chain, a
slow transition, and bundled Be II XCD against analytic or direct matrix
solutions. It also hashes the generator, bundled Be II input, manual sources,
build script and package metadata.
The PDF builder requires ``pdflatex`` and the packages listed in
the TeX preamble. It runs the compiler three times to resolve references.

These checks cover selected invariants. They are not a complete atomic-data
validation or an ADAS benchmark.

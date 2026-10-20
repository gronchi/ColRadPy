Theory manual
=============

:download:`Read the theory manual (PDF) <theory/ColRadPy_theory_manual.pdf>`
or edit its :download:`LaTeX source <theory/ColRadPy_theory_manual.tex>`.

The manual describes the local ``development`` working tree. It covers
atomic rates, CR matrices, quasistatic elimination, PECs and GCR
coefficients, transient and stationary solutions, ionization and energy
balance, feeding-channel diagnostics, S/XB ratios, wavelengths, multiplet
splitting and profiles. The paper's ADAS comparison and Be/Mo examples are
identified as published results, separate from the checks rerun here.

The layout follows the EZSSS theory manual: Latin Modern fonts, A4 pages,
numbered equations, blue implementation boxes, orange caveat boxes, and
blue/orange/aqua figures.

Regenerate checks and figures
-----------------------------

From the repository root, with the ``development`` branch checked out::

    python docs/source/theory/scripts/theory_examples.py
    python docs/source/theory/scripts/build_theory_manual.py

The generator imports the current working-tree code and records the branch,
HEAD, changed source files, source hashes, and runtime versions in
``docs/source/theory/figures/provenance.json``. It compares an equal-rate chain, a
slow transition, and bundled Be II XCD against analytic or direct matrix
solutions. The PDF builder requires ``pdflatex`` and the packages listed in
the TeX preamble. It runs the compiler three times to resolve references.

These checks cover selected invariants. They are not a complete atomic-data
validation or an ADAS benchmark.

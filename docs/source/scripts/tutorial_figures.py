"""Run the Python code blocks of the tutorial in order and save the figures it shows.

The ``.. code-block:: python`` blocks of docs/source/tutorial.rst run in one shared namespace, as a
reader would run them, from a temporary copy of examples/input so that the files the blocks write
stay out of the repository. The figures a block makes are saved, in order, to the
``.. figure:: images/...`` directives that follow it before the next code block; figures without a
directive (those of the built-in plotting methods) are closed. A block preceded by the comment
``.. tutorial: not run`` needs data that are not bundled and is skipped. The figures are saved at
300 dpi, and a matplotlib deprecation warning stops the run, so that the tutorial keeps to the
current matplotlib API.

Run from the repository root:  python docs/source/scripts/tutorial_figures.py
"""

from pathlib import Path
import os
import re
import shutil
import sys
import tempfile
import textwrap
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
TUTORIAL = REPO / "docs" / "source" / "tutorial.rst"
IMAGES = TUTORIAL.parent / "images"
SKIP = ".. tutorial: not run"


def sections(text):
    """[(code, figures)]: each python block with the images of the figure directives after it."""
    lines = text.expandtabs(8).splitlines()
    blocks, skip, i = [], False, 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == SKIP:
            skip = True
        elif re.match(r"\.\. code-block:: *python\s*$", line):
            i += 1
            while i < len(lines) and re.match(r"\s+:\w+:", lines[i]):      # directive options
                i += 1
            body = []
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith((" ", "\t"))):
                body.append(lines[i])
                i += 1
            blocks.append([textwrap.dedent("\n".join(body)), [], skip])
            skip = False
            continue
        elif line.startswith(".. figure::") and blocks:
            blocks[-1][1].append(line.split("::", 1)[1].strip())
        i += 1
    return blocks


def main():
    blocks = sections(TUTORIAL.read_text(encoding="utf-8"))
    namespace = {"__name__": "__tutorial__"}
    work = Path(tempfile.mkdtemp(prefix="colradpy-tutorial-"))
    shutil.copytree(REPO / "examples" / "input", work / "examples" / "input")
    sys.path.insert(0, str(REPO))
    here = os.getcwd()
    os.chdir(work)
    saved = []
    warnings.filterwarnings("error", category=matplotlib.MatplotlibDeprecationWarning)
    try:
        for number, (code, figures, skip) in enumerate(blocks, 1):
            if skip:
                print(f"block {number}: not run")
                continue
            before = set(plt.get_fignums())
            start = time.time()
            exec(compile(code, f"tutorial block {number}", "exec"), namespace)
            new = [n for n in plt.get_fignums() if n not in before]
            print(f"block {number}: {time.time() - start:.1f} s, {len(new)} figure(s), "
                  f"{len(figures)} figure directive(s)")
            if len(new) < len(figures):
                raise RuntimeError(f"block {number} makes {len(new)} figures for {figures}")
            for fig_number, image in zip(new, figures):
                path = TUTORIAL.parent / image
                plt.figure(fig_number).savefig(path, dpi=300)
                saved.append(path)
            plt.close("all")
    finally:
        os.chdir(here)
        shutil.rmtree(work, ignore_errors=True)
    print("saved:", ", ".join(p.name for p in saved))


if __name__ == "__main__":
    main()

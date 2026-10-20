"""Compile the standalone manual without leaving LaTeX auxiliary files next to it."""

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


THEORY = Path(__file__).resolve().parents[1]  # docs/source/theory
NAME = "ColRadPy_theory_manual"


def build(compiler):
    executable = shutil.which(compiler)
    if executable is None:
        raise SystemExit(f"LaTeX compiler not found: {compiler}")
    with tempfile.TemporaryDirectory(prefix="colradpy-theory-latex-") as directory:
        for _ in range(3):
            result = subprocess.run(
                [executable, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
                 "-output-directory=" + directory, NAME + ".tex"],
                cwd=THEORY, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace",
            )
            if result.returncode:
                print(result.stdout)
                raise SystemExit(result.returncode)
        log = (Path(directory) / (NAME + ".log")).read_text(errors="replace")
        if "There were undefined references" in log or "There were undefined citations" in log:
            raise SystemExit("Unresolved LaTeX references or citations; PDF was not replaced.")
        for line in log.splitlines():
            if "Overfull" in line or "Warning:" in line:
                print(line)
        destination = THEORY / (NAME + ".pdf")
        shutil.copyfile(Path(directory) / (NAME + ".pdf"), destination)
        print(f"Built {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdflatex", default="pdflatex", help="LaTeX executable or path")
    build(parser.parse_args().pdflatex)

"""MkDocs hook: regenerate example plots into docs/plots/ before each build."""

import subprocess
import sys
from pathlib import Path


def on_pre_build(config):
    out_dir = Path(config["docs_dir"]) / "plots"
    out_dir.mkdir(exist_ok=True)
    subprocess.run(
        [sys.executable, "plot_example.py", "--out", str(out_dir)],
        check=True,
    )

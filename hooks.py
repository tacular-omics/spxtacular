"""MkDocs hooks.

- Regenerate example plots into docs/plots/ before each build.
- Serve the repo-root llms.txt and llms-full.txt at the site root.
"""

import shutil
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


LLMS_FILES = ("llms.txt", "llms-full.txt")


def on_post_build(config, **kwargs):
    root = Path(config["config_file_path"]).parent
    site_dir = Path(config["site_dir"])
    for name in LLMS_FILES:
        shutil.copy2(root / name, site_dir / name)

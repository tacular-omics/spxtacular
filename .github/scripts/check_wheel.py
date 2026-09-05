"""Exercise a built wheel in an isolated environment outside the checkout."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

PORTABLE_TESTS = (
    "test_maintenance.py",
    "test_core_extra.py",
    "test_combine.py",
    "test_scoring.py",
    "test_similarity.py",
    "test_json_serialization.py",
    "test_chromatogram.py",
    "test_peaklist.py",
    "test_msp.py",
    "test_scientific_reference.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path)
    parser.add_argument("--lowest", action="store_true")
    args = parser.parse_args()
    wheels = list(args.dist.resolve().glob("*.whl"))
    if len(wheels) != 1:
        parser.error("the distribution directory must contain exactly one wheel")
    repo = Path(__file__).resolve().parents[2]
    config = tomllib.loads((repo / "pyproject.toml").read_text())

    with tempfile.TemporaryDirectory(prefix="spxtacular-wheel-") as temporary:
        root = Path(temporary)
        environment = root / "env"
        subprocess.run(["uv", "venv", "--python", sys.executable, str(environment)], check=True)
        python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        install = ["uv", "pip", "install", "--python", str(python), "--only-binary=:all:", str(wheels[0])]
        if args.lowest:
            # Name runtime dependencies directly so lowest-direct applies to
            # them, rather than just to the already selected wheel.
            install += ["--resolution", "lowest-direct", *config["project"]["dependencies"]]
        subprocess.run(install, check=True)
        subprocess.run(
            [str(python), "-I", str(repo / ".github/scripts/check_minimal_install.py")], cwd=root, check=True
        )
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), "pytest>=9,<10", "jsonschema>=4.25,<5"], check=True
        )
        # Record what was actually tested, especially in the lower-bound job.
        versions = subprocess.check_output(
            [
                str(python),
                "-I",
                "-c",
                "import importlib.metadata as m, json\n"
                "print(json.dumps({d.metadata['Name']: d.version for d in m.distributions()}, sort_keys=True))",
            ],
            text=True,
        )
        print(json.dumps(json.loads(versions), indent=2), flush=True)
        subprocess.run(["uv", "pip", "check", "--python", str(python)], check=True)
        scratch_tests = root / "tests"
        scratch_tests.mkdir()
        for name in (*PORTABLE_TESTS, "__init__.py", "data.py"):
            shutil.copy2(repo / "tests" / name, scratch_tests / name)
        shutil.copytree(repo / "tests/reference", scratch_tests / "reference")
        # No project config or editable checkout can affect this test run.
        (root / "pytest.ini").write_text("[pytest]\nfilterwarnings = error\n", encoding="utf-8")
        subprocess.run(
            [str(python), "-I", "-m", "pytest", "-q", *[str(scratch_tests / name) for name in PORTABLE_TESTS]],
            cwd=root,
            check=True,
        )


if __name__ == "__main__":
    main()

"""The optional interoperability stacks must stay lazy at package import time."""

import subprocess
import sys

import pytest


@pytest.mark.slow  # subprocess interpreter start + package import, ~2 s
def test_package_import_does_not_import_interop_dependencies() -> None:
    code = (
        "import sys; import spxtacular; assert 'matchms' not in sys.modules; assert 'spectrum_utils' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)

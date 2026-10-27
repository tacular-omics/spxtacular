"""Run the Python examples in ``docs/`` and ``README.md``.

Every fenced ``python`` block on a page is executed in page order, in one shared
namespace, so a page's ``Setup`` block defines the names the later blocks use -- the
same way a reader works through the page. A block that is *not* meant to be executed
says so in its fence title (see :data:`NOT_EXECUTED`), which also renders as a label
above the block, so a reader can tell which blocks they can paste.

Examples run in a scratch directory that links the shipped fixtures under
``tests/data``, so files an example writes (``spectrum.html``, ``scan_001.npz``, ...)
never land in the checkout. ``Figure.show()`` is stubbed and matplotlib is forced onto
its non-interactive backend, because a test run has no browser and no display.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pytest_examples import CodeExample, EvalExample, find_examples

ROOT = Path(__file__).parent.parent

# The exact set of reasons a block may give for not being executed. Keeping it closed
# stops "not executed" from becoming a way to park a broken example.
NOT_EXECUTED = frozenset(
    {
        "signature only - not executed",
        "needs your own data file - not executed",
        "needs network access - not executed",
        "needs the Thermo .NET runtime - not executed",
    }
)

# Fixtures an example may open, linked into the scratch directory under tests/data.
FIXTURES = ("example.mzML", "example.mzML.gz", "example_dda.d", "example_dia.d", "example_prm.d", "mzspeclib")

# Modules a page's runnable examples need; the page is skipped when one is missing, so
# the test is meaningful on an install without the optional extras.
REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "getting-started.md": ("mzmlpy", "spectrl"),
    "readers.md": ("mzmlpy", "tdfpy"),
    "scoring.md": ("mzmlpy",),
    "spectrum.md": ("mzmlpy", "spectrl"),
    "visualization.md": ("matplotlib", "kaleido", "mzmlpy", "tdfpy"),
    "api.md": ("matplotlib", "kaleido", "matchms", "spectrum_utils", "spectrl"),
    "README.md": ("spectrl",),
}

PAGES = [*sorted((ROOT / "docs").glob("*.md")), ROOT / "README.md"]


def _examples(page: Path) -> list[CodeExample]:
    return list(find_examples(page))


def _title(example: CodeExample) -> str:
    # pytest-examples lower-cases fence settings, so compare lower-cased titles.
    return example.prefix_settings().get("title", "")


_REASONS = frozenset(reason.lower() for reason in NOT_EXECUTED)


@pytest.fixture
def docs_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch working directory with ``tests/data`` beside it."""
    data = tmp_path / "tests" / "data"
    data.mkdir(parents=True)
    for name in FIXTURES:
        source = ROOT / "tests" / "data" / name
        target = data / name
        try:
            target.symlink_to(source, target_is_directory=source.is_dir())
        except OSError:  # Windows without developer mode
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def headless(monkeypatch: pytest.MonkeyPatch) -> None:
    """No browser and no display: draw figures, never show them."""
    import plotly.graph_objects as go

    monkeypatch.setattr(go.Figure, "show", lambda self, *args, **kwargs: None, raising=False)
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")


@pytest.fixture
def theme_state() -> Iterator[None]:
    """Plot-theme module state, restored after the page.

    ``set_plot_theme("dark")`` and ``set_palette(...)`` are what a reader wants to see on the
    page, but both write globals in :mod:`spxtacular.theme` that would otherwise outlive the
    test and recolour every later one.
    """
    from spxtacular import theme

    mode = theme._DEFAULT_MODE
    palettes = (theme._CATEGORICAL, theme._CHARGE_RAMP, theme._SEQUENTIAL)
    saved = [dict(palette) for palette in palettes]
    try:
        yield
    finally:
        theme._DEFAULT_MODE = mode
        for palette, original in zip(palettes, saved, strict=True):
            palette.clear()
            palette.update(original)


@pytest.mark.parametrize("page", PAGES, ids=lambda path: path.name)
@pytest.mark.timeout(300)
# matchms warns about its own optional hdf5plugin on import; nothing to do with the example.
@pytest.mark.filterwarnings("ignore:hdf5plugin is missing")
def test_page_examples(
    page: Path, eval_example: EvalExample, docs_cwd: Path, headless: None, theme_state: None
) -> None:
    """Every executable block on the page runs, in order, in one namespace."""
    for module in REQUIREMENTS.get(page.name, ()):
        pytest.importorskip(module)

    namespace: dict[str, Any] = {}
    for example in _examples(page):
        if _title(example) in _REASONS:
            continue
        namespace = eval_example.run(example, module_globals=namespace)


@pytest.mark.parametrize("page", PAGES, ids=lambda path: path.name)
def test_unexecuted_blocks_give_a_known_reason(page: Path) -> None:
    """A block that is not executed says why, in the wording this test knows."""
    for example in _examples(page):
        title = _title(example)
        if not title:
            continue
        assert title in _REASONS, f"{example}: unknown fence title {title!r}; expected one of {sorted(NOT_EXECUTED)}"

"""Render the spxtacular figure gallery in both backends.

Every figure is drawn once per backend and style and written as PNG and SVG::

    python docs/gallery/build.py                   # everything, into docs/gallery/out
    python docs/gallery/build.py --only annotated  # one figure
    python docs/gallery/build.py --backend matplotlib --out /tmp/gallery

The data is the tiny fetal-brain mzSpecLib fixture shipped with the tests,
plus a few synthetic spectra for the figure types it cannot show (profile
data, ion mobility, reporter ions, chromatograms). Plotly static export needs
kaleido (``pip install 'spxtacular[plotly-export]'``) and a Chrome/Chromium.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import peptacular as pt

import spxtacular as spx
from spxtacular.core import MsnSpectrum, Spectrum, SpectrumType
from spxtacular.mzspeclib import read_mzspeclib

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "tests" / "data" / "mzspeclib" / "fetal_brain_tiny.mzSpecLib.txt.gz"
ION_TYPES = ("b", "y")


def _library() -> list[Any]:
    return list(read_mzspeclib(LIBRARY))


def _psm() -> tuple[MsnSpectrum, str, list[Any]]:
    entry = _library()[0]
    peptide = str(entry.peptidoform)
    annotation = pt.parse(peptide)
    fragments = annotation.fragment(ion_types=ION_TYPES, charges=[1, 2])
    return entry.spectrum, peptide, fragments


def _replicate() -> MsnSpectrum:
    """A second acquisition of the same peptide when the fixture has one, else a jittered copy."""
    entries = _library()
    first = entries[0]
    for entry in entries[1:]:
        if str(entry.peptidoform) == str(first.peptidoform):
            return entry.spectrum
    rng = np.random.default_rng(7)
    s = first.spectrum
    return MsnSpectrum(
        mz=s.mz + rng.normal(0, 0.002, len(s.mz)),
        intensity=s.intensity * rng.uniform(0.6, 1.4, len(s.intensity)),
        ms_level=2,
        precursors=list(s.precursors or []),
    )


def _profile() -> Spectrum:
    rng = np.random.default_rng(3)
    mz = np.linspace(799.0, 803.5, 1800)
    inten = np.zeros_like(mz)
    for z, mono, top in ((2, 800.40, 1.0), (3, 800.95, 0.45)):
        for k, rel in enumerate((1.0, 0.85, 0.42, 0.15)):
            centre = mono + k * 1.00336 / z
            inten += top * rel * np.exp(-0.5 * ((mz - centre) / 0.012) ** 2)
    # A positive baseline, like a real detector: exact zeros between peaks make
    # the log-parabola apex fit in centroid() unstable.
    inten = inten * 1e6 + 5e3 + np.abs(rng.normal(0, 2e3, len(mz)))
    return Spectrum(mz=mz, intensity=inten, spectrum_type=SpectrumType.PROFILE)


def _im_spectrum() -> Spectrum:
    rng = np.random.default_rng(11)
    n = 160
    mz = np.sort(rng.uniform(300, 1300, n))
    inten = rng.lognormal(10, 1.2, n)
    im = 0.6 + (mz - 300) / 1000 * 0.7 + rng.normal(0, 0.05, n)
    return Spectrum(mz=mz, intensity=inten, im=im, spectrum_type=SpectrumType.CENTROID)


def _reporter_spectrum() -> Spectrum:
    rng = np.random.default_rng(5)
    empty = Spectrum(mz=np.array([100.0]), intensity=np.array([1.0]))
    reporter_mz = spx.extract_reporter_ions(empty, "TMT10").reporter_mz
    ratios = [1.0, 0.92, 0.55, 0.61, 0.3, 0.33, 0.98, 1.05, 0.12, 0.7]
    mz = [float(m) + rng.normal(0, 0.0004) for m in reporter_mz]
    inten = [r * 2e5 * rng.uniform(0.9, 1.1) for r in ratios]
    # An interfering peak and some noise, as in a real reporter region.
    mz += [126.5, 128.09, 129.95, 131.4]
    inten += [9e3, 2.2e4, 6e3, 1.1e4]
    order = np.argsort(mz)
    return Spectrum(mz=np.asarray(mz)[order], intensity=np.asarray(inten)[order], spectrum_type=SpectrumType.CENTROID)


def _chromatograms() -> list[spx.Chromatogram]:
    rt = np.linspace(1100, 1300, 400)
    out = []
    for label, apex, width, top in (("416.8757 (3+)", 1189.6, 4.5, 6.3e5), ("842.8869 (2+)", 1175.3, 6.0, 1.7e6)):
        inten = top * np.exp(-0.5 * ((rt - apex) / width) ** 2) + 2e3
        out.append(spx.Chromatogram(rt, inten, label=label))
    return out


def figures() -> dict[str, Callable[..., Any]]:
    spectrum, peptide, fragments = _psm()
    replicate = _replicate()
    decon = spectrum.deconvolute(charge_range=(1, 3))
    return {
        "spectrum": lambda **kw: spx.plot_spectrum(spectrum, **kw),
        "annotated": lambda **kw: spx.annotate_spectrum(
            spectrum, fragments, peptide=peptide, mass_error_panel=True, **kw
        ),
        "mirror": lambda **kw: spx.mirror_plot(
            replicate, spectrum, fragments=fragments, names=("query", "library"), similarity="cosine", **kw
        ),
        "mirror_decon": lambda **kw: spx.mirror_plot(spectrum, decon, **kw),
        "mass_error": lambda **kw: spx.mass_error_plot(spectrum, fragments, **kw),
        "coverage": lambda **kw: spx.sequence_coverage_plot(spectrum, peptide, fragments, **kw),
        "facet": lambda **kw: spx.facet_plot(spectrum, fragments=fragments, mirror_spectrum=replicate, **kw),
        "profile_centroid": lambda **kw: spx.profile_centroid_plot(
            _profile(), centroids=_profile().centroid(min_intensity="noise"), **kw
        ),
        "ion_mobility": lambda **kw: spx.plot_spectrum(_im_spectrum(), color="im", **kw),
        "reporter": lambda **kw: spx.reporter_ion_plot(_reporter_spectrum(), "TMT10", **kw),
        "chromatogram": lambda **kw: spx.plot_chromatogram(_chromatograms(), **kw),
        "composed": lambda **kw: spx.compose_figure(
            [
                spx.annotate_spectrum(spectrum, fragments, peptide=peptide, backend="spec", style=kw.get("style")),
                spx.mirror_plot(
                    replicate, spectrum, fragments=fragments, names=("query", "library"), backend="spec",
                    style=kw.get("style"),
                ),
                spx.mass_error_plot(spectrum, fragments, backend="spec", style=kw.get("style")),
                spx.reporter_ion_plot(_reporter_spectrum(), backend="spec", style=kw.get("style")),
            ],
            ncols=2,
            backend=kw["backend"],
            style=kw.get("style"),
        ),
    }  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    parser.add_argument("--only", nargs="*", help="figure names to render (default: all)")
    parser.add_argument("--backend", choices=["plotly", "matplotlib", "both"], default="both")
    parser.add_argument("--style", nargs="*", default=["paper", "screen"])
    parser.add_argument("--formats", nargs="*", default=["png", "svg"])
    args = parser.parse_args(argv)

    backends = ["plotly", "matplotlib"] if args.backend == "both" else [args.backend]
    catalogue = figures()
    names = args.only or list(catalogue)
    unknown = [n for n in names if n not in catalogue]
    if unknown:
        parser.error(f"unknown figures: {', '.join(unknown)}; choose from {', '.join(catalogue)}")
    args.out.mkdir(parents=True, exist_ok=True)
    failures = 0
    for name in names:
        for backend in backends:
            for style in args.style:
                started = time.perf_counter()
                try:
                    fig = catalogue[name](backend=backend, style=style)
                    for fmt in args.formats:
                        spx.save_figure(fig, args.out / f"{name}-{backend}-{style}.{fmt}")
                except Exception as exc:
                    failures += 1
                    print(f"FAIL {name} {backend} {style}: {type(exc).__name__}: {exc}", file=sys.stderr)
                    continue
                print(f"ok   {name:<17} {backend:<10} {style:<6} {time.perf_counter() - started:5.1f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

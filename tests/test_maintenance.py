"""Regression coverage for processing chains and metadata ownership."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from spxtacular import (
    SODIATED,
    DeconvolutionProvenance,
    MgfReader,
    Ms2Reader,
    MsnSpectrum,
    MspReader,
    Precursor,
    Spectrum,
    SpectrumType,
    write_mgf,
    write_ms2,
    write_msp,
)


def _sodium_spectrum() -> Spectrum:
    return Spectrum(
        mz=np.array([SODIATED.ion_mz(1000.0, 1)]),
        intensity=np.array([1.0]),
        charge=np.array([1]),
        deconvolution=DeconvolutionProvenance(
            isotope_model="peptide",
            ionization_model=SODIATED,
            charge_range=(1, 1),
            tolerance=10.0,
            tolerance_type="ppm",
            intensity_mode="total",
            min_intensity=0.0,
            min_score=0.0,
        ),
    )


def test_combine_differing_records_requires_conversion_first() -> None:
    first = _sodium_spectrum()
    assert first.deconvolution is not None
    second = first.update(deconvolution=replace(first.deconvolution, tolerance=20.0))
    with pytest.raises(ValueError, match="Decharge inputs separately"):
        Spectrum.combine([first, second])
    combined = Spectrum.combine([first.decharge(), second.decharge()])
    assert combined.is_decharged
    np.testing.assert_allclose(combined.mz, [1000.0, 1000.0], atol=1e-10)


def test_combine_shared_carrier_survives_json_and_decharge() -> None:
    first = _sodium_spectrum()
    combined = Spectrum.combine([first, first.copy()])
    restored = Spectrum.from_json(combined.to_json())
    np.testing.assert_allclose(restored.decharge().mz, [1000.0, 1000.0], atol=1e-10)


def test_combine_cannot_mix_mass_axes() -> None:
    charged = _sodium_spectrum()
    with pytest.raises(ValueError, match="neutral masses and m/z"):
        Spectrum.combine([charged, charged.decharge()])


def test_decharge_cannot_silently_drop_neutral_peaks() -> None:
    mixed = Spectrum(mz=np.array([1000.0, 501.0]), intensity=np.ones(2), charge=np.array([0, 2]))
    with pytest.raises(ValueError, match="mixture of neutral masses"):
        mixed.decharge(inplace=True)
    assert len(mixed) == 2


def test_combine_does_not_erase_state_with_empty_input() -> None:
    first = _sodium_spectrum()
    empty = Spectrum(mz=np.array([]), intensity=np.array([]))
    combined = Spectrum.combine([empty, first])
    np.testing.assert_allclose(combined.decharge().mz, [1000.0], atol=1e-10)


def test_combine_rejects_loss_of_negative_scan_polarity() -> None:
    negative = MsnSpectrum(
        mz=np.array([999.0]), intensity=np.ones(1), charge=np.ones(1, dtype=np.int32), polarity="negative"
    )
    with pytest.raises(ValueError, match="lose scan polarity"):
        Spectrum.combine([negative])


@pytest.mark.parametrize("width", [2, 3, 4, 5, 20])
@pytest.mark.parametrize("inplace", [False, True])
def test_flat_centroid_keeps_midpoint_height_and_mobility(width: int, inplace: bool) -> None:
    intensity = np.array([1.0, 5.0] + [9.0] * width + [5.0, 1.0])
    mz = np.arange(len(intensity), dtype=float) * 0.01 + 500.0
    mobility = np.arange(len(intensity), dtype=float) * 0.1
    original = Spectrum(mz=mz, intensity=intensity, im=mobility, spectrum_type="profile")
    result = original.centroid(inplace=inplace)
    np.testing.assert_allclose(result.mz, [(mz[2] + mz[width + 1]) / 2], atol=1e-12)
    np.testing.assert_array_equal(result.intensity, [9.0])
    np.testing.assert_array_equal(result.im, [mobility[(width + 3) // 2]])
    assert result.spectrum_type == SpectrumType.CENTROID
    if not inplace:
        np.testing.assert_array_equal(original.intensity, intensity)


@pytest.mark.parametrize("intensity", [[9, 9, 9, 5, 1], [1, 5, 9, 9, 9], [9, 9, 9, 9]])
def test_centroid_does_not_invent_a_peak_at_a_boundary(intensity: list[int]) -> None:
    spec = Spectrum(mz=np.arange(len(intensity), dtype=float), intensity=np.array(intensity))
    assert len(spec.centroid()) == 0


def test_gaussian_fit_has_no_symmetric_peak_bias() -> None:
    mz = 1000.0 + np.array([-0.04, -0.02, 0.0, 0.02, 0.04])
    intensity = 100.0 * np.exp(-0.5 * ((mz - 1000.0) / 0.03) ** 2)
    result = Spectrum(mz=mz, intensity=intensity).centroid()
    np.testing.assert_allclose(result.mz, [1000.0], atol=1e-11, rtol=0)
    np.testing.assert_allclose(result.intensity, [100.0], rtol=1e-12)


@pytest.mark.parametrize("method", ["max", "tic", "median"])
@pytest.mark.parametrize("inplace", [False, True])
def test_filter_then_normalize_recomputes(method, inplace: bool) -> None:
    original = Spectrum(mz=np.array([100.0, 200.0]), intensity=np.array([3.0, 1.0])).normalize(method)
    filtered = original.filter(min_mz=150.0, inplace=inplace)
    assert filtered.normalized is None
    np.testing.assert_allclose(filtered.normalize(method).intensity, [1.0])


def test_combine_then_normalize_recomputes_tic() -> None:
    first = Spectrum(mz=np.array([100.0]), intensity=np.ones(1)).normalize("tic")
    combined = Spectrum.combine([first, first.copy()])
    assert combined.normalized is None
    assert combined.normalize("tic").intensity.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("inplace", [False, True])
def test_rounding_then_normalize_recomputes(inplace: bool) -> None:
    spec = Spectrum(mz=np.array([100.1, 100.2]), intensity=np.ones(2)).normalize()
    rounded = spec.round_mz(inplace=inplace)
    assert rounded.normalized is None
    np.testing.assert_allclose(rounded.normalize().intensity, [1.0])


def test_rounding_neutral_masses_preserves_axis_through_transport() -> None:
    neutral = _sodium_spectrum().decharge().round_mz()
    restored = Spectrum.from_json(neutral.to_json())
    assert restored.is_decharged
    with pytest.warns(UserWarning, match="already decharged"):
        np.testing.assert_allclose(restored.decharge().mz, [1000.0])


@pytest.mark.parametrize("operation", ["copy", "filter", "sort", "normalize", "merge", "round_mz", "update"])
def test_transforms_own_their_precursor_list(operation: str) -> None:
    precursor = Precursor(mz=500.0, intensity=1.0, charge=2, is_monoisotopic=True)
    supplied = [precursor]
    original = MsnSpectrum(mz=np.array([100.0]), intensity=np.ones(1), precursors=supplied)
    result = getattr(original, operation)()
    result.precursors.clear()
    assert original.precursors == [precursor]
    assert supplied == [precursor]


@pytest.mark.parametrize(
    ("writer", "reader", "suffix"),
    [(write_mgf, MgfReader, ".mgf"), (write_ms2, Ms2Reader, ".ms2"), (write_msp, MspReader, ".msp")],
)
@pytest.mark.parametrize("gzip", [False, True])
def test_peaklist_write_failure_preserves_destination_and_cleans_temp(
    tmp_path: Path, writer, reader, suffix, gzip
) -> None:
    path = tmp_path / ("peaks" + suffix + (".gz" if gzip else ""))
    spec = Spectrum(mz=np.array([100.0]), intensity=np.ones(1))
    writer(spec, path)
    before = path.read_bytes()

    def broken_input():
        yield spec
        raise RuntimeError("source interrupted")

    with pytest.raises(RuntimeError, match="source interrupted"):
        writer(broken_input(), path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]

    writer(reader(path), path)
    np.testing.assert_array_equal(reader(path)[0].mz, spec.mz)


@pytest.mark.parametrize(("writer", "suffix"), [(write_mgf, ".mgf"), (write_ms2, ".ms2"), (write_msp, ".msp")])
def test_peaklist_rejects_neutral_mass_axis(tmp_path: Path, writer, suffix: str) -> None:
    destination = tmp_path / ("peaks" + suffix)
    with pytest.raises(ValueError, match="neutral masses"):
        writer(_sodium_spectrum().decharge(), destination)
    assert not destination.exists()

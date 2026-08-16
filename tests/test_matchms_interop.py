"""Interoperability tests for matchms."""

import numpy as np
import pytest

matchms = pytest.importorskip("matchms")

from matchms import Spectrum as MatchmsSpectrum  # noqa: E402
from matchms.filtering import normalize_intensities  # noqa: E402

from spxtacular import MsnSpectrum, Precursor, Spectrum, SpectrumType, from_matchms, to_matchms  # noqa: E402


def _rich_msn() -> MsnSpectrum:
    return MsnSpectrum(
        mz=np.array([300.0, 100.0, 200.0]),
        intensity=np.array([30.0, 10.0, 20.0]),
        charge=np.array([3, 1, 2], dtype=np.int32),
        im=np.array([1.3, 1.1, 1.2]),
        iso_score=np.array([0.3, 0.1, 0.2]),
        spectrum_type=SpectrumType.DECONVOLUTED,
        denoised="mad",
        normalized="tic",
        scan_number=42,
        ms_level=2,
        native_id="controllerType=0 scan=42",
        im_type="ook0",
        rt=125.5,
        injection_time=12.0,
        total_ion_current=60.0,
        mz_range=(50.0, 1500.0),
        im_range=(0.5, 1.5),
        polarity="positive",
        analyzer="orbitrap",
        collision_energy=28.0,
        activation_type="HCD",
        precursors=[Precursor(mz=500.25, intensity=8000.0, charge=2, im=1.05, is_monoisotopic=True)],
        isolation_mz_range=(499.0, 501.0),
    )


def test_to_matchms_sorts_and_populates_conventional_metadata() -> None:
    converted = to_matchms(_rich_msn(), extra_metadata={"smiles": "CCO"})
    np.testing.assert_array_equal(converted.peaks.mz, [100.0, 200.0, 300.0])
    np.testing.assert_array_equal(converted.peaks.intensities, [10.0, 20.0, 30.0])
    assert converted.get("id") == "controllerType=0 scan=42"
    assert converted.get("precursor_mz") == pytest.approx(500.25)
    assert converted.get("charge") == 2
    assert converted.get("retention_time") == pytest.approx(125.5)
    assert converted.get("ionmode") == "positive"
    assert converted.get("smiles") == "CCO"
    assert isinstance(converted.get("spxtacular_metadata"), str)


def test_matchms_round_trip_preserves_rich_spectrum() -> None:
    restored = from_matchms(to_matchms(_rich_msn()))
    assert isinstance(restored, MsnSpectrum)
    np.testing.assert_array_equal(restored.mz, [100.0, 200.0, 300.0])
    np.testing.assert_array_equal(restored.intensity, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(restored.charge, [1, 2, 3])
    np.testing.assert_array_equal(restored.im, [1.1, 1.2, 1.3])
    np.testing.assert_array_equal(restored.iso_score, [0.1, 0.2, 0.3])
    assert restored.scan_number == 42
    assert restored.native_id == "controllerType=0 scan=42"
    assert restored.rt == pytest.approx(125.5)
    assert restored.im_range == pytest.approx((0.5, 1.5))
    assert restored.precursors is not None
    assert restored.precursors[0].mz == pytest.approx(500.25)
    assert restored.precursors[0].is_monoisotopic is True


def test_matchms_filtered_subset_realigns_per_peak_arrays() -> None:
    converted = to_matchms(_rich_msn())
    filtered = MatchmsSpectrum(
        mz=converted.peaks.mz[[0, 2]],
        intensities=converted.peaks.intensities[[0, 2]],
        metadata=converted.metadata,
        metadata_harmonization=False,
    )
    restored = from_matchms(filtered)
    np.testing.assert_array_equal(restored.mz, [100.0, 300.0])
    np.testing.assert_array_equal(restored.charge, [1, 3])
    np.testing.assert_array_equal(restored.im, [1.1, 1.3])
    np.testing.assert_array_equal(restored.iso_score, [0.1, 0.3])


def test_real_matchms_filter_preserves_payload_and_uses_processed_intensity() -> None:
    converted = to_matchms(_rich_msn())
    normalized = normalize_intensities(converted)
    restored = from_matchms(normalized)
    assert restored.intensity.max() == pytest.approx(1.0)
    np.testing.assert_array_equal(restored.charge, [1, 2, 3])


def test_matchms_changed_mz_drops_unalignable_extension_arrays() -> None:
    converted = to_matchms(_rich_msn())
    rounded = MatchmsSpectrum(
        mz=converted.peaks.mz + 0.1,
        intensities=converted.peaks.intensities,
        metadata=converted.metadata,
        metadata_harmonization=False,
    )
    with pytest.warns(UserWarning, match="could not be realigned"):
        restored = from_matchms(rounded)
    assert restored.charge is None
    assert restored.im is None
    assert restored.iso_score is None


def test_foreign_matchms_spectrum_maps_to_msn() -> None:
    foreign = MatchmsSpectrum(
        mz=np.array([100.0, 200.0]),
        intensities=np.array([10.0, 20.0]),
        metadata={
            "id": "library-entry-1",
            "precursor_mz": 500.2,
            "charge": 2,
            "retention_time": 30.0,
            "ionmode": "positive",
        },
        metadata_harmonization=False,
    )
    converted = from_matchms(foreign)
    assert isinstance(converted, MsnSpectrum)
    assert converted.ms_level == 2
    assert converted.native_id == "library-entry-1"
    assert converted.rt == pytest.approx(30.0)
    assert converted.precursors is not None
    assert converted.precursors[0].charge == 2


def test_foreign_matchms_pepmass_sequence_maps_to_precursor() -> None:
    foreign = MatchmsSpectrum(
        mz=np.array([100.0]),
        intensities=np.array([10.0]),
        metadata={"pepmass": [500.2, 9000.0], "charge": "2+"},
        metadata_harmonization=False,
    )
    converted = from_matchms(foreign)
    assert isinstance(converted, MsnSpectrum)
    assert converted.precursors is not None
    assert converted.precursors[0].mz == pytest.approx(500.2)
    assert converted.precursors[0].charge == 2


def test_foreign_plain_matchms_spectrum_maps_to_plain_spectrum() -> None:
    foreign = MatchmsSpectrum(
        mz=np.array([100.0]),
        intensities=np.array([10.0]),
        metadata={},
        metadata_harmonization=False,
    )
    converted = from_matchms(foreign)
    assert type(converted) is Spectrum


def test_payload_can_be_ignored() -> None:
    converted = to_matchms(_rich_msn())
    restored = from_matchms(converted, prefer_spxtacular_metadata=False)
    assert isinstance(restored, MsnSpectrum)
    assert restored.charge is None
    assert restored.precursors is not None
    assert restored.precursors[0].mz == pytest.approx(500.25)

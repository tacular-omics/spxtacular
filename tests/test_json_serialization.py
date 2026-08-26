"""Versioned JSON transport tests for spectra and chromatograms."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from jsonschema import Draft202012Validator, ValidationError, validate

from spxtacular import (
    Chromatogram,
    MsnSpectrum,
    Precursor,
    Reader,
    Spectrum,
    get_json_schema,
    write_mgf,
    write_ms2,
    write_msp,
)
from spxtacular.ionization import PROTONATED, DeconvolutionProvenance

DATA_DIR = Path(__file__).parent / "data"


def _provenance() -> DeconvolutionProvenance:
    return DeconvolutionProvenance(
        isotope_model="peptide",
        ionization_model=PROTONATED,
        charge_range=(1, 5),
        tolerance=10.0,
        tolerance_type="ppm",
        intensity_mode="total",
        min_intensity=0.0,
        min_score=0.0,
    )


def _plain_spectrum() -> Spectrum:
    return Spectrum(
        mz=np.array([101.071, 129.102, 175.119]),
        intensity=np.array([1204.0, 8431.0, 4250.0]),
        charge=np.array([1, -1, 2], dtype=np.int32),
        im=np.array([0.91, 1.02, 1.13]),
        iso_score=np.array([0.8, 0.0, 0.92]),
        spectrum_type="deconvoluted",
        denoised="mad",
        normalized="max",
        deconvolution=_provenance(),
    )


def _msn_spectrum() -> MsnSpectrum:
    return MsnSpectrum(
        mz=np.array([101.071, 129.102, 175.119]),
        intensity=np.array([1204.0, 8431.0, 4250.0]),
        charge=np.array([1, 1, 2], dtype=np.int32),
        im=np.array([0.91, 1.02, 1.13]),
        iso_score=np.array([0.8, 0.7, 0.92]),
        spectrum_type="centroid",
        scan_number=np.int32(1452),  # ty: ignore[invalid-argument-type]
        ms_level=np.int32(2),  # ty: ignore[invalid-argument-type]
        native_id="controllerType=0 controllerNumber=1 scan=1452",
        im_type="ook0",
        rt=np.float64(582.31),
        injection_time=np.float32(23.5),  # ty: ignore[invalid-argument-type]
        total_ion_current=np.float64(1.2e8),
        mz_range=(100.0, 2000.0),
        im_range=(0.5, 1.7),
        polarity="positive",
        resolution=np.float64(60000.0),
        analyzer="vendor:future-analyzer",
        ramp_time=np.float32(100.0),  # ty: ignore[invalid-argument-type]
        collision_energy=np.float32(28.0),  # ty: ignore[invalid-argument-type]
        activation_type="vendor:future-activation",
        precursors=[
            Precursor(
                mz=523.2764,
                intensity=321.0,
                charge=2,
                im=1.04,
                iso_score=0.97,
                is_monoisotopic=True,
            ),
            Precursor(
                mz=617.3201,
                intensity=0.0,
                charge=None,
                im=None,
                iso_score=None,
                is_monoisotopic=None,
            ),
        ],
        isolation_mz_range=(522.8, 523.8),
        isolation_im_range=(0.95, 1.1),
    )


class TestSpectrumDictionary:
    def test_plain_spectrum_round_trip(self) -> None:
        spectrum = _plain_spectrum()

        payload = spectrum.to_dict()
        restored = Spectrum.from_dict(payload)

        assert payload["schema"] == "spxtacular.spectrum"
        assert payload["schema_version"] == 1
        assert payload["kind"] == "spectrum"
        assert type(restored) is Spectrum
        assert restored == spectrum

    def test_msn_spectrum_round_trip_dispatches_from_base_class(self) -> None:
        spectrum = _msn_spectrum()

        payload = spectrum.to_dict()
        restored = Spectrum.from_dict(payload)

        assert payload["kind"] == "msn_spectrum"
        assert type(restored) is MsnSpectrum
        assert restored == spectrum

    def test_msn_class_round_trip(self) -> None:
        spectrum = _msn_spectrum()
        restored = MsnSpectrum.from_dict(spectrum.to_dict())

        assert type(restored) is MsnSpectrum
        assert restored == spectrum

    def test_output_uses_only_json_native_values(self) -> None:
        payload = _msn_spectrum().to_dict()

        assert type(payload["metadata"]["scan_number"]) is int
        assert type(payload["metadata"]["rt"]) is float
        assert type(payload["arrays"]["charge"][0]) is int
        json.dumps(payload, allow_nan=False)

    def test_empty_spectrum_and_null_arrays_round_trip(self) -> None:
        spectrum = Spectrum(mz=np.array([]), intensity=np.array([]))

        payload = spectrum.to_dict()
        restored = Spectrum.from_dict(payload)

        assert payload["arrays"] == {
            "mz": [],
            "intensity": [],
            "charge": None,
            "im": None,
            "iso_score": None,
        }
        assert restored == spectrum

    def test_input_payload_is_not_mutated(self) -> None:
        payload = _msn_spectrum().to_dict()
        original = json.loads(json.dumps(payload))

        Spectrum.from_dict(payload)

        assert payload == original


class TestSpectrumJson:
    @pytest.mark.parametrize("indent", [None, 2])
    def test_json_round_trip(self, indent: int | None) -> None:
        spectrum = _msn_spectrum()

        encoded = spectrum.to_json(indent=indent)
        restored = Spectrum.from_json(encoded)

        assert restored == spectrum
        if indent is None:
            assert "\n" not in encoded
        else:
            assert "\n" in encoded

    @pytest.mark.parametrize("container", [bytes, bytearray])
    def test_json_accepts_utf8_bytes(self, container) -> None:
        spectrum = _plain_spectrum()
        encoded = container(spectrum.to_json().encode())

        assert Spectrum.from_json(encoded) == spectrum

    def test_malformed_json_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid JSON"):
            Spectrum.from_json("{not-json}")

    def test_non_standard_nan_token_is_rejected(self) -> None:
        encoded = _plain_spectrum().to_json().replace("101.071", "NaN", 1)

        with pytest.raises(ValueError, match="non-standard numeric constant NaN"):
            Spectrum.from_json(encoded)

    def test_duplicate_object_key_is_rejected(self) -> None:
        encoded = (
            _plain_spectrum()
            .to_json()
            .replace(
                '{"schema":',
                '{"schema":"duplicate","schema":',
                1,
            )
        )

        with pytest.raises(ValueError, match="duplicate object key 'schema'"):
            Spectrum.from_json(encoded)

    def test_non_finite_array_is_rejected_on_output(self) -> None:
        spectrum = Spectrum(mz=np.array([100.0]), intensity=np.array([float("inf")]))

        with pytest.raises(ValueError, match=r"arrays.intensity\[0\] must be finite"):
            spectrum.to_dict()

    def test_non_finite_metadata_is_rejected_on_output(self) -> None:
        spectrum = _msn_spectrum()
        spectrum.rt = float("nan")

        with pytest.raises(ValueError, match=r"metadata\.rt must be finite"):
            spectrum.to_dict()

    def test_unicode_metadata_round_trip(self) -> None:
        spectrum = _msn_spectrum()
        spectrum.native_id = "样品质谱 scan=1452"

        encoded = spectrum.to_json()

        assert "样品质谱" in encoded
        assert Spectrum.from_json(encoded) == spectrum

    def test_wrong_input_type_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="str, bytes, or bytearray"):
            Spectrum.from_json({})  # ty: ignore[invalid-argument-type]


class TestSpectrumValidation:
    def test_unsupported_version_is_rejected(self) -> None:
        payload = _plain_spectrum().to_dict()
        payload["schema_version"] = 2

        with pytest.raises(ValueError, match=r"Unsupported spxtacular\.spectrum schema version 2"):
            Spectrum.from_dict(payload)

    def test_boolean_version_is_rejected(self) -> None:
        payload = _plain_spectrum().to_dict()
        payload["schema_version"] = True

        with pytest.raises(ValueError, match="Unsupported"):
            Spectrum.from_dict(payload)

    def test_wrong_schema_is_rejected(self) -> None:
        payload = _plain_spectrum().to_dict()
        payload["schema"] = "example.spectrum"

        with pytest.raises(ValueError, match=r"Expected schema 'spxtacular\.spectrum'"):
            Spectrum.from_dict(payload)

    def test_invalid_kind_is_rejected(self) -> None:
        payload = _plain_spectrum().to_dict()
        payload["kind"] = "unknown"

        with pytest.raises(ValueError, match=r"payload\.kind"):
            Spectrum.from_dict(payload)

    def test_msn_constructor_rejects_plain_payload(self) -> None:
        with pytest.raises(ValueError, match="requires kind='msn_spectrum'"):
            MsnSpectrum.from_dict(_plain_spectrum().to_dict())

    @pytest.mark.parametrize("field", ["schema", "schema_version", "kind", "arrays", "metadata"])
    def test_missing_envelope_field_is_rejected(self, field: str) -> None:
        payload = _plain_spectrum().to_dict()
        del payload[field]

        with pytest.raises(ValueError, match="missing required field"):
            Spectrum.from_dict(payload)

    def test_unknown_envelope_field_is_rejected(self) -> None:
        payload = _plain_spectrum().to_dict()
        payload["future"] = True

        with pytest.raises(ValueError, match="unknown field"):
            Spectrum.from_dict(payload)

    @pytest.mark.parametrize("key", [1, None])
    def test_non_string_envelope_key_is_rejected(self, key: Any) -> None:
        payload: dict[Any, Any] = _plain_spectrum().to_dict()
        payload[key] = "invalid"

        with pytest.raises(TypeError, match="payload keys must be strings"):
            Spectrum.from_dict(payload)

    def test_missing_array_field_is_rejected(self) -> None:
        payload = _plain_spectrum().to_dict()
        del payload["arrays"]["im"]

        with pytest.raises(ValueError, match=r"payload\.arrays is missing"):
            Spectrum.from_dict(payload)

    def test_required_array_cannot_be_null(self) -> None:
        payload = _plain_spectrum().to_dict()
        payload["arrays"]["mz"] = None

        with pytest.raises(ValueError, match="cannot be null"):
            Spectrum.from_dict(payload)

    def test_non_json_array_is_rejected(self) -> None:
        payload = _plain_spectrum().to_dict()
        payload["arrays"]["mz"] = (100.0, 200.0)

        with pytest.raises(TypeError, match="JSON array or null"):
            Spectrum.from_dict(payload)

    def test_non_numeric_peak_value_is_rejected(self) -> None:
        payload = _plain_spectrum().to_dict()
        payload["arrays"]["mz"][0] = "100.0"

        with pytest.raises(TypeError, match=r"payload\.arrays\.mz\[0\] must be a number"):
            Spectrum.from_dict(payload)

    def test_fractional_charge_is_rejected(self) -> None:
        payload = _plain_spectrum().to_dict()
        payload["arrays"]["charge"][0] = 1.5

        with pytest.raises(TypeError, match=r"payload\.arrays\.charge\[0\] must be an integer"):
            Spectrum.from_dict(payload)

    def test_mismatched_array_lengths_are_rejected(self) -> None:
        payload = _plain_spectrum().to_dict()
        payload["arrays"]["intensity"].pop()

        with pytest.raises(ValueError, match="mz and intensity must have same length"):
            Spectrum.from_dict(payload)

    def test_unknown_metadata_field_is_rejected(self) -> None:
        payload = _plain_spectrum().to_dict()
        payload["metadata"]["future"] = True

        with pytest.raises(ValueError, match=r"payload\.metadata contains unknown"):
            Spectrum.from_dict(payload)

    def test_invalid_range_length_is_rejected(self) -> None:
        payload = _msn_spectrum().to_dict()
        payload["metadata"]["mz_range"] = [100.0]

        with pytest.raises(ValueError, match="must contain exactly two numbers"):
            Spectrum.from_dict(payload)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [("scan_number", -1, "nonnegative"), ("ms_level", 0, "positive")],
    )
    def test_invalid_scan_identifiers_are_rejected(self, field: str, value: int, message: str) -> None:
        payload = _msn_spectrum().to_dict()
        payload["metadata"][field] = value

        with pytest.raises(ValueError, match=message):
            Spectrum.from_dict(payload)

    def test_invalid_precursor_field_is_rejected(self) -> None:
        payload = _msn_spectrum().to_dict()
        payload["metadata"]["precursors"][0]["charge"] = 2.5

        with pytest.raises(TypeError, match="charge must be an integer"):
            Spectrum.from_dict(payload)

    def test_boolean_numeric_metadata_is_rejected(self) -> None:
        payload = _msn_spectrum().to_dict()
        payload["metadata"]["rt"] = True

        with pytest.raises(TypeError, match="rt must be a number"):
            Spectrum.from_dict(payload)

    def test_unknown_deconvolution_field_is_rejected(self) -> None:
        payload = _plain_spectrum().to_dict()
        payload["metadata"]["deconvolution"]["future"] = True

        with pytest.raises(ValueError, match="unknown field"):
            Spectrum.from_dict(payload)

    def test_noncanonical_deconvolution_integer_is_rejected(self) -> None:
        payload = _plain_spectrum().to_dict()
        payload["metadata"]["deconvolution"]["max_isotope_gaps"] = 0.0

        with pytest.raises(TypeError, match="max_isotope_gaps must be an integer"):
            Spectrum.from_dict(payload)


class TestChromatogramJson:
    def test_round_trip(self) -> None:
        chromatogram = Chromatogram(
            rt=np.array([10.0, 20.0, 30.0]),
            intensity=np.array([100.0, 500.0, 250.0]),
            label="m/z 523.2764",
            mz=523.2764,
            tolerance=20.0,
            tolerance_type="ppm",
            meta={"aggregate": "sum", "targets": np.array([523.2764])},
        )

        payload = chromatogram.to_dict()
        restored = Chromatogram.from_json(chromatogram.to_json())

        assert payload["schema"] == "spxtacular.chromatogram"
        assert payload["kind"] == "chromatogram"
        np.testing.assert_array_equal(restored.rt, chromatogram.rt)
        np.testing.assert_array_equal(restored.intensity, chromatogram.intensity)
        assert restored.label == chromatogram.label
        assert restored.mz == chromatogram.mz
        assert restored.tolerance == chromatogram.tolerance
        assert restored.tolerance_type == chromatogram.tolerance_type
        assert restored.meta == {"aggregate": "sum", "targets": [523.2764]}

    def test_empty_chromatogram_round_trip(self) -> None:
        chromatogram = Chromatogram(rt=np.array([]), intensity=np.array([]))
        restored = Chromatogram.from_dict(chromatogram.to_dict())

        assert len(restored) == 0
        assert restored.label == ""

    def test_mismatched_lengths_are_rejected(self) -> None:
        payload = Chromatogram(rt=np.array([1.0]), intensity=np.array([2.0])).to_dict()
        payload["arrays"]["rt"].append(2.0)

        with pytest.raises(ValueError, match="same length"):
            Chromatogram.from_dict(payload)

    def test_unknown_metadata_field_is_rejected(self) -> None:
        payload = Chromatogram(rt=np.array([1.0]), intensity=np.array([2.0])).to_dict()
        payload["metadata"]["future"] = True

        with pytest.raises(ValueError, match=r"payload\.metadata contains unknown"):
            Chromatogram.from_dict(payload)

    def test_non_string_label_is_rejected(self) -> None:
        payload = Chromatogram(rt=np.array([1.0]), intensity=np.array([2.0])).to_dict()
        payload["metadata"]["label"] = 7

        with pytest.raises(TypeError, match="label must be a string"):
            Chromatogram.from_dict(payload)

    def test_non_finite_metadata_value_is_rejected(self) -> None:
        chromatogram = Chromatogram(
            rt=np.array([1.0]),
            intensity=np.array([2.0]),
            meta={"bad": float("inf")},
        )

        with pytest.raises(ValueError, match=r"metadata\.meta\.bad must be finite"):
            chromatogram.to_dict()

    def test_non_string_metadata_key_is_rejected(self) -> None:
        chromatogram = Chromatogram(
            rt=np.array([1.0]),
            intensity=np.array([2.0]),
            meta={1: "bad"},
        )

        with pytest.raises(TypeError, match="keys must be strings"):
            chromatogram.to_dict()


class TestReaderAndPersistenceCompatibility:
    def test_mgf_reader_spectrum_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "example.mgf"
        path.write_text(
            "BEGIN IONS\n"
            "TITLE=scan-7\n"
            "SCANS=7\n"
            "RTINSECONDS=12.5\n"
            "PEPMASS=523.2764 1000\n"
            "CHARGE=2+\n"
            "101.071 1204\n"
            "175.119 4250\n"
            "END IONS\n"
        )
        from spxtacular import Reader

        with Reader(path) as reader:
            spectrum = next(iter(reader.ms2))

        restored = Spectrum.from_json(spectrum.to_json())
        assert restored == spectrum

    @pytest.mark.parametrize(
        ("suffix", "writer"),
        [(".mgf", write_mgf), (".ms2", write_ms2), (".msp", write_msp)],
    )
    def test_peak_list_reader_round_trip(self, tmp_path: Path, suffix: str, writer) -> None:
        path = writer([_msn_spectrum()], tmp_path / f"example{suffix}")

        with Reader(path) as reader:
            spectrum = next(iter(reader.ms2))

        assert Spectrum.from_json(spectrum.to_json()) == spectrum

    def test_mzml_reader_spectrum_round_trip(self) -> None:
        pytest.importorskip("mzmlpy")
        from spxtacular import Reader

        with Reader(DATA_DIR / "example.mzML") as reader:
            spectrum = next(iter(reader.ms2))

        restored = Spectrum.from_json(spectrum.to_json())
        assert restored == spectrum

    @pytest.mark.parametrize(
        ("path", "level"),
        [
            (DATA_DIR / "example_dda.d", 1),
            (DATA_DIR / "example_dda.d", 2),
            (DATA_DIR / "example_dia.d", 2),
            (DATA_DIR / "example_prm.d", 2),
        ],
    )
    def test_bruker_reader_spectrum_round_trip(self, path: Path, level: int) -> None:
        pytest.importorskip("tdfpy")

        with Reader(path) as reader:
            spectra = reader.ms1 if level == 1 else reader.ms2
            spectrum = next(iter(spectra))

        assert Spectrum.from_json(spectrum.to_json()) == spectrum

    def test_npz_persistence_remains_compatible(self, tmp_path: Path) -> None:
        spectrum = _msn_spectrum()
        path = tmp_path / "spectrum.npz"

        spectrum.save(path)
        restored = MsnSpectrum.load(path)

        assert restored == spectrum


def test_packaged_json_schemas_are_valid_json() -> None:
    schema_package = files("spxtacular.schemas")
    spectrum_schema = json.loads(schema_package.joinpath("spectrum-v1.schema.json").read_text())
    chromatogram_schema = json.loads(schema_package.joinpath("chromatogram-v1.schema.json").read_text())

    assert spectrum_schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert spectrum_schema["properties"]["schema_version"] == {"const": 1}
    assert chromatogram_schema["properties"]["schema_version"] == {"const": 1}
    Draft202012Validator.check_schema(spectrum_schema)
    Draft202012Validator.check_schema(chromatogram_schema)


def test_spectrum_payloads_validate_against_packaged_schema() -> None:
    schema = get_json_schema("spectrum")

    validate(_plain_spectrum().to_dict(), schema)
    validate(_msn_spectrum().to_dict(), schema)


def test_schema_rejects_incomplete_deconvolution_provenance() -> None:
    payload = _plain_spectrum().to_dict()
    del payload["metadata"]["deconvolution"]["ionization_model"]

    with pytest.raises(ValidationError, match="ionization_model"):
        validate(payload, get_json_schema("spectrum"))


def test_chromatogram_payload_validates_against_packaged_schema() -> None:
    chromatogram = Chromatogram(
        rt=np.array([1.0, 2.0]),
        intensity=np.array([10.0, 20.0]),
        meta={"source": "TIC"},
    )

    validate(chromatogram.to_dict(), get_json_schema("chromatogram"))


@pytest.mark.parametrize("kind", ["spectrum", "chromatogram"])
def test_get_json_schema_returns_fresh_copy(kind: str) -> None:
    first = get_json_schema(kind)  # ty: ignore[invalid-argument-type]
    second = get_json_schema(kind)  # ty: ignore[invalid-argument-type]

    first["title"] = "changed"
    assert second["title"] != "changed"


def test_get_json_schema_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unknown JSON Schema kind"):
        get_json_schema("unknown")  # ty: ignore[invalid-argument-type]

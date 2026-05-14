import base64
import binascii
import gzip
import struct
import zlib
from collections.abc import Generator
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .core import Spectrum


def compress_with_method(data_bytes: bytes, method: str) -> bytes:
    """Compress data using specified method"""
    if method == "gzip":
        return gzip.compress(data_bytes, compresslevel=9)
    elif method == "zlib":
        return zlib.compress(data_bytes, level=zlib.Z_BEST_COMPRESSION)
    elif method == "brotli":
        try:
            import brotli  # type: ignore

            return brotli.compress(data_bytes, quality=11)
        except ImportError:
            raise ImportError("brotli library not available. Install with: pip install brotli") from None
    else:
        raise ValueError(f"Unknown compression method: {method}")


def decompress_with_method(data_bytes: bytes, method: str) -> bytes:
    """Decompress data using specified method"""
    if method == "gzip":
        return gzip.decompress(data_bytes)
    elif method == "zlib":
        return zlib.decompress(data_bytes)
    elif method == "brotli":
        try:
            import brotli  # type: ignore

            return brotli.decompress(data_bytes)
        except ImportError:
            raise ImportError("brotli library not available") from None
    else:
        raise ValueError(f"Unknown compression method: {method}")


def _float_to_hex(f: float) -> str:
    return format(struct.unpack("!I", struct.pack("!f", f))[0], "08x")


def _hex_to_float(s: str) -> float:
    return struct.unpack("!f", struct.pack("!I", int(s, 16)))[0]


def _encode_leading_zero(lz: int) -> str:
    if 0 <= lz < 16:
        return hex(lz)[-1]
    raise ValueError(f"Leading zero count {lz} out of range [0-15]")


def _decode_leading_zero(lz: str) -> int:
    return int(lz, 16)


def _hex_delta(a: str, b: str) -> str:
    diff = int(a, 16) - int(b, 16)
    return format(diff & 0xFFFFFFFF, "08x")


def _hex_delta_rev(a: str, b: str) -> str:
    diff = int(a, 16) + int(b, 16)
    return format(diff & 0xFFFFFFFF, "08x")


def _count_leading_zeros(s: str) -> int:
    return len(s) - len(s.lstrip("0"))


def _delta_encode_single_string(vals: NDArray[np.float64]) -> str:
    """Delta-encode a float64 array into a compact hex string.

    Values are quantised to float32 (big-endian, viewed as uint32). The first
    value is stored verbatim; subsequent values are stored as the unsigned
    wrap-around delta from the previous uint32. Each chunk has its leading
    zeros stripped and the per-chunk zero count is appended (reversed) at the
    end as a packed 4-bit run — see ``_delta_decode_single_string`` for the
    inverse.
    """
    if vals.size == 0:
        return ""

    u32 = vals.astype(">f4").view(">u4")
    initial_hex = f"{u32[0]:08x}"
    initial_hex_loops = _count_leading_zeros(initial_hex)

    deltas = np.diff(u32)  # unsigned subtraction wraps around

    if deltas.size == 0:
        return initial_hex.lstrip("0") + _encode_leading_zero(initial_hex_loops)

    delta_bytes = deltas.astype(">u4").tobytes()
    all_hex = binascii.hexlify(delta_bytes).decode("ascii")

    n = deltas.size
    chunks = [all_hex[i * 8 : (i + 1) * 8] for i in range(n)]
    stripped = [c.lstrip("0") for c in chunks]
    hex_delta_str = initial_hex.lstrip("0") + "".join(stripped)

    leading_zeros = [8 - len(s) for s in stripped]
    leading_zero_str = _encode_leading_zero(initial_hex_loops) + "".join(
        _encode_leading_zero(lz) for lz in leading_zeros
    )
    return hex_delta_str + leading_zero_str[::-1]


def _delta_decode_single_string(s: str) -> Generator[float, None, None]:
    if not s:
        return

    initial_lz = _decode_leading_zero(s[-1])
    initial_hex = "0" * initial_lz + s[: 8 - initial_lz]
    s = s[8 - initial_lz : -1]

    yield _hex_to_float(initial_hex)
    curr_value_int = int(initial_hex, 16)

    while s:
        lz = _decode_leading_zero(s[-1])
        hex_diff = "0" * lz + s[: 8 - lz]
        diff_int = int(hex_diff, 16)
        curr_value_int = (curr_value_int + diff_int) & 0xFFFFFFFF
        yield struct.unpack("!f", struct.pack("!I", curr_value_int))[0]
        s = s[8 - lz : -1]


def _hex_encode(intensities: NDArray[np.float64]) -> str:
    """Hex-encode a float64 array as big-endian float32 bytes."""
    if intensities.size == 0:
        return ""
    return binascii.hexlify(intensities.astype(">f4").tobytes()).decode("ascii")


def _hex_decode(s: str) -> Generator[float, None, None]:
    """Decode a hex string of big-endian float32s back to float64s."""
    if not s:
        return
    try:
        arr = np.frombuffer(bytes.fromhex(s), dtype=">f4").astype(np.float64)
        yield from arr
    except ValueError:
        for i in range(0, len(s), 8):
            yield _hex_to_float(s[i : i + 8])


def _encode_charges(charges: NDArray[np.int32] | None) -> str:
    """Encode charges as a single hex digit per peak.

    0 → '0' (decodes to None — missing/decharged), 1-14 → '1'-'e', and the
    singleton sentinel -1 → 'f'. Charge state 15 is unsupported to leave room
    for the singleton encoding; it is rare in practice (MS rarely sees z>10).
    """
    if charges is None or charges.size == 0:
        return ""

    if np.any((charges < -1) | (charges > 14)):
        bad = charges[(charges < -1) | (charges > 14)]
        raise ValueError(f"Charge {bad[0]} out of range (supported: -1, 0..14)")

    encoded = np.where(charges == -1, 15, charges)
    chars = np.array(list("0123456789abcdef"))
    return "".join(chars[encoded])


def _decode_charges(s: str) -> Generator[int | None, None, None]:
    """Decode charges from compact string.

    '0' → None (missing/decharged), '1'..'e' → 1..14, 'f' → -1 (singleton).
    """
    if not s:
        return

    for char in s:
        val = int(char, 16)
        if val == 0:
            yield None
        elif val == 15:
            yield -1
        else:
            yield val


def _encode_binary_payload(
    mz_str: str,
    intensity_str: str,
    charge_str: str = "",
    im_str: str = "",
    iso_score_str: str = "",
) -> bytes:
    """Encode mz, intensity, charge, im, and iso_score data into binary payload.

    The mz/intensity/charge/im chunks are always emitted (matching the
    pre-iso_score wire format). The iso_score chunk is appended only when
    non-empty, so payloads without iso_score stay byte-identical to old output
    and old decoders can still read them.
    """
    mz_bytes = mz_str.encode("ascii")
    intensity_bytes = intensity_str.encode("ascii")
    charge_bytes = charge_str.encode("ascii")
    im_bytes = im_str.encode("ascii")

    payload = (
        struct.pack("!I", len(mz_bytes))
        + mz_bytes
        + struct.pack("!I", len(intensity_bytes))
        + intensity_bytes
        + struct.pack("!I", len(charge_bytes))
        + charge_bytes
        + struct.pack("!I", len(im_bytes))
        + im_bytes
    )

    if iso_score_str:
        iso_score_bytes = iso_score_str.encode("ascii")
        payload += struct.pack("!I", len(iso_score_bytes)) + iso_score_bytes

    return payload


def _decode_binary_payload(payload: bytes) -> tuple[str, str, str, str, str]:
    """Decode binary payload into (mz, intensity, charge, im, iso_score) strings.

    The last three chunks are optional for backward compatibility with older
    payloads written before the corresponding fields were supported.
    """
    offset = 0

    def read_chunk(offset: int) -> tuple[str, int]:
        if len(payload) < offset + 4:
            if offset == len(payload):
                return "", offset
            raise ValueError("Invalid binary payload: too short")
        length = struct.unpack("!I", payload[offset : offset + 4])[0]
        offset += 4
        if len(payload) < offset + length:
            raise ValueError("Invalid binary payload: truncated data")
        data = payload[offset : offset + length].decode("ascii")
        offset += length
        return data, offset

    mz_str, offset = read_chunk(offset)
    intensity_str, offset = read_chunk(offset)

    charge_str = ""
    if offset < len(payload):
        charge_str, offset = read_chunk(offset)

    im_str = ""
    if offset < len(payload):
        im_str, offset = read_chunk(offset)

    iso_score_str = ""
    if offset < len(payload):
        iso_score_str, offset = read_chunk(offset)

    return mz_str, intensity_str, charge_str, im_str, iso_score_str


def compress_spectra(
    spectrum: "Spectrum",
    url_safe: bool = False,
    mz_precision: int | None = None,
    intensity_precision: int | None = None,
    im_precision: int | None = None,
    iso_score_precision: int | None = None,
    compression: str = "gzip",
) -> str:
    """Compress spectrum data with configurable precision and compression."""
    # Validate precision inputs
    for name, val in [
        ("mz", mz_precision),
        ("intensity", intensity_precision),
        ("im", im_precision),
        ("iso_score", iso_score_precision),
    ]:
        if val is not None and (not isinstance(val, int) or val < 0):
            raise ValueError(f"{name}_precision must be non-negative integer or None")

    if compression not in ["gzip", "zlib", "brotli"]:
        raise ValueError("compression must be 'gzip', 'zlib', or 'brotli'")

    mzs = spectrum.mz
    intensities = spectrum.intensity
    charges = spectrum.charge
    ims = spectrum.im
    iso_scores = spectrum.iso_score

    if mz_precision is not None:
        mzs = np.round(mzs, mz_precision)

    if intensity_precision is not None:
        intensities = np.round(intensities, intensity_precision)

    if im_precision is not None and ims is not None:
        ims = np.round(ims, im_precision)

    if iso_score_precision is not None and iso_scores is not None:
        iso_scores = np.round(iso_scores, iso_score_precision)

    mz_str = _delta_encode_single_string(mzs) if mzs.size > 0 else ""
    intensity_str = _hex_encode(intensities) if intensities.size > 0 else ""
    charge_str = _encode_charges(charges) if charges is not None else ""
    im_str = _hex_encode(ims) if ims is not None else ""
    iso_score_str = _hex_encode(iso_scores) if iso_scores is not None else ""

    binary_payload = _encode_binary_payload(mz_str, intensity_str, charge_str, im_str, iso_score_str)
    compressed_bytes = compress_with_method(binary_payload, compression)

    compression_flag = {"gzip": "G", "zlib": "Z", "brotli": "R"}[compression]

    if url_safe:
        encoded = base64.urlsafe_b64encode(compressed_bytes).decode("ascii")
        return "U" + compression_flag + encoded
    else:
        encoded = base64.b85encode(compressed_bytes).decode("ascii")
        return "B" + compression_flag + encoded


def decompress_spectra(
    compressed_str: str,
) -> "Spectrum":
    """Decompress spectra data. Returns Spectrum object."""
    from .core import Spectrum

    if not compressed_str:
        raise ValueError("compressed_str cannot be empty")

    if not isinstance(compressed_str, str):
        raise ValueError("compressed_str must be a string")

    if len(compressed_str) < 3:
        raise ValueError("Invalid compressed string format")

    encoding_flag = compressed_str[0]
    compression_flag = compressed_str[1]
    encoded_data = compressed_str[2:]

    if encoding_flag not in ["U", "B"]:
        raise ValueError(f"Unknown encoding method: {encoding_flag}")

    if compression_flag not in ["G", "Z", "R"]:
        raise ValueError(f"Unknown compression method: {compression_flag}")

    compression_scheme = {"G": "gzip", "Z": "zlib", "R": "brotli"}[compression_flag]

    if encoding_flag == "U":
        compressed_bytes = base64.urlsafe_b64decode(encoded_data)
    else:
        compressed_bytes = base64.b85decode(encoded_data)

    binary_payload = decompress_with_method(compressed_bytes, compression_scheme)
    mz_str, intensity_str, charge_str, im_str, iso_score_str = _decode_binary_payload(binary_payload)

    mz = np.fromiter(_delta_decode_single_string(mz_str), dtype=float) if mz_str else np.array([], dtype=float)
    intensity = np.fromiter(_hex_decode(intensity_str), dtype=float) if intensity_str else np.array([], dtype=float)

    charge = None
    if charge_str:
        # Charges can be None, so we decode to list first, then handle None -> 0 conversion for numpy array
        # or keep as None if user expects list? But Spectrum usually desires numpy arrays.
        # Assuming 0 is used for missing charge in numpy array context often.
        decoded_charges = list(_decode_charges(charge_str))
        charge = np.array([c if c is not None else 0 for c in decoded_charges], dtype=int)

    im = None
    if im_str:
        im = np.fromiter(_hex_decode(im_str), dtype=float)

    iso_score = None
    if iso_score_str:
        iso_score = np.fromiter(_hex_decode(iso_score_str), dtype=float)

    return Spectrum(mz=mz, intensity=intensity, charge=charge, im=im, iso_score=iso_score)

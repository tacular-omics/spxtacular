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
            raise ImportError("brotli library not available. Install with: pip install brotli")
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
            raise ImportError("brotli library not available")
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
    if vals.size == 0:
        return ""

    # Convert to float32 (big-endian), then view as uint32 (big-endian)
    u32 = vals.astype(">f4").view(">u4")

    # Initial value
    initial_val = u32[0]
    initial_hex = f"{initial_val:08x}"

    # We need to count leading zeros.
    # Instead of string manipulation which requires looping, can we compute it?
    # leading zeros in 8-char hex string:
    # 0 = 00000000 (8)
    # 1 = 00000001 (7)
    # ...
    # This is effectively counting leading zero nibbles.
    # We can do this with value thresholds vectorized.

    # However, we need to strip zeros to create the variable length string.
    # To do this purely in numpy (no python loop for string join) is very hard efficiently
    # because of the variable length output.
    # But we can at least avoid format().

    # Initial value processing
    initial_hex_loops = _count_leading_zeros(initial_hex)  # Keep using helper for scalar, it's fast enough

    # Calculate deltas (unsigned subtraction with wraparound)
    deltas = np.diff(u32)

    if deltas.size == 0:
        return initial_hex.lstrip("0") + _encode_leading_zero(initial_hex_loops)

    # Convert deltas to big-endian bytes
    # deltas is uint32. tobytes() gives 4 bytes per int.
    # binascii.hexlify gives 8 hex chars.
    delta_bytes = deltas.astype(">u4").tobytes()
    all_hex = binascii.hexlify(delta_bytes).decode("ascii")

    # Now we have one giant string of 8-char hex segments.
    # To strip leading zeros, we iterate.
    # It might be faster to iterate on the ints than string slicing if we want to avoid formatting?
    # No, formatting is slow. Slicing pre-computed hex string is fast.

    # Split into 8-char chunks
    # Note: Using a list comp over range is standard python speed.
    n = deltas.size
    chunks = [all_hex[i * 8 : (i + 1) * 8] for i in range(n)]

    # Strip zeros.
    # Python lstrip is C-optimized.
    stripped = [c.lstrip("0") for c in chunks]
    hex_delta_str = initial_hex.lstrip("0") + "".join(stripped)

    # Count leading zeros for suffix
    # 8 chars - len(stripped). If stripped is empty, it was "00000000", len 0 -> 8 zeros.
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

    # Decode initial value
    # Optimized: Use struct or numpy? Single value struct is fine.
    # Keeping original logic for single values helper
    initial_val_float = _hex_to_float(initial_hex)
    yield initial_val_float

    curr_value_int = int(initial_hex, 16)

    while s:
        lz = _decode_leading_zero(s[-1])
        hex_fragment = s[: 8 - lz]
        hex_diff = "0" * lz + hex_fragment

        diff_int = int(hex_diff, 16)

        # Reverse delta: curr = prev + diff
        # We need 32-bit wrapping addition behavior
        curr_value_int = (curr_value_int + diff_int) & 0xFFFFFFFF

        # Convert int back to float
        # struct.unpack("!f", struct.pack("!I", curr_value_int))[0]
        # Or optimization:
        # yield _int_to_float(curr_value_int)
        # But we don't have _int_to_float helper exposed nicely. _hex_to_float does string->float.
        # Let's just reconstruct hex string to use existing helper or use struct directly.

        yield struct.unpack("!f", struct.pack("!I", curr_value_int))[0]

        s = s[8 - lz : -1]


def _hex_encode(intensities: NDArray[np.float64]) -> str:
    if intensities.size == 0:
        return ""
    # use binascii for super fast hex encoding
    # >f4 is big-endian float32
    return binascii.hexlify(intensities.astype(">f4").tobytes()).decode("ascii")


def _hex_decode(s: str) -> Generator[float, None, None]:
    if not s:
        return
    # Numpy optimization
    try:
        b = bytes.fromhex(s)
        # >f4 is big-endian float32
        arr = np.frombuffer(b, dtype=">f4").astype(np.float64)
        for x in arr:
            yield x
    except ValueError:
        # Fallback if weird length (though fromhex handles even length)
        # The previous implementation processed in chunks of 8 chars.
        # bytes.fromhex requires even number of digits. 8 chars is safe.
        # If len(s) is not multiple of 8? The format implies it should be.
        # Let's assume valid input for performance optimization.
        for i in range(0, len(s), 8):
            yield _hex_to_float(s[i : i + 8])


def _encode_charges(charges: NDArray[np.int32] | None) -> str:
    """Encode charges as a compact string. None/0 is encoded as 0, values 1-15 as hex digits."""
    if charges is None or charges.size == 0:
        return ""

    # Ensure we are working with ints.
    # Replace None (0) with 0. If it's a numpy array of ints, 0 remains 0.
    # Assume charges is already int32 array where 0 represents None/Uncharged.

    # Check boundaries
    if np.any((charges < 0) | (charges > 15)):
        offending = charges[(charges < 0) | (charges > 15)]
        raise ValueError(f"Charge {offending[0]} out of range [0-15]")

    # Create mapping array for fast lookup?
    # Or just format. Formating ints is fast.
    # Actually, charges are 0-15.
    # Hex string: 0, 1, ..., 9, a, b, c, d, e, f.
    # This matches standard hex() output without '0x' for single digit 0-9?
    # No, hex(10) is '0xa'.
    # We want 'a'.
    # We can use format.

    # Vectorized approach:
    # Use formatted string on the whole array? No.
    # Map integers 0-15 to characters "0123456789abcdef"
    chars = np.array(list("0123456789abcdef"))
    return "".join(chars[charges])


def _decode_charges(s: str) -> Generator[int | None, None, None]:
    """Decode charges from compact string. '0' decodes to None, other hex digits to int."""
    if not s:
        return

    for char in s:
        val = int(char, 16)
        if val == 0:
            yield None
        else:
            yield val


def _encode_binary_payload(mz_str: str, intensity_str: str, charge_str: str = "", im_str: str = "") -> bytes:
    """Encode mz, intensity, charge, and im data into binary payload."""
    mz_bytes = mz_str.encode("ascii")
    intensity_bytes = intensity_str.encode("ascii")
    charge_bytes = charge_str.encode("ascii")
    im_bytes = im_str.encode("ascii")

    return (
        struct.pack("!I", len(mz_bytes))
        + mz_bytes
        + struct.pack("!I", len(intensity_bytes))
        + intensity_bytes
        + struct.pack("!I", len(charge_bytes))
        + charge_bytes
        + struct.pack("!I", len(im_bytes))
        + im_bytes
    )


def _decode_binary_payload(payload: bytes) -> tuple[str, str, str, str]:
    """Decode binary payload into mz, intensity, charge, and im strings."""
    offset = 0

    def read_chunk(offset):
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
        try:
            charge_str, offset = read_chunk(offset)
        except ValueError:
            raise

    im_str = ""
    if offset < len(payload):
        im_str, offset = read_chunk(offset)

    return mz_str, intensity_str, charge_str, im_str


def compress_spectra(
    spectrum: "Spectrum",
    url_safe: bool = False,
    mz_precision: int | None = None,
    intensity_precision: int | None = None,
    im_precision: int | None = None,
    compression: str = "gzip",
) -> str:
    """Compress spectrum data with configurable precision and compression."""
    # Validate precision inputs
    for name, val in [("mz", mz_precision), ("intensity", intensity_precision), ("im", im_precision)]:
        if val is not None and (not isinstance(val, int) or val < 0):
            raise ValueError(f"{name}_precision must be non-negative integer or None")

    if compression not in ["gzip", "zlib", "brotli"]:
        raise ValueError("compression must be 'gzip', 'zlib', or 'brotli'")

    mzs = spectrum.mz
    intensities = spectrum.intensity
    charges = spectrum.charge
    ims = spectrum.ion_mobility

    if mz_precision is not None:
        mzs = np.round(mzs, mz_precision)

    if intensity_precision is not None:
        intensities = np.round(intensities, intensity_precision)

    if im_precision is not None and ims is not None:
        ims = np.round(ims, im_precision)

    mz_str = _delta_encode_single_string(mzs) if mzs.size > 0 else ""
    intensity_str = _hex_encode(intensities) if intensities.size > 0 else ""
    charge_str = _encode_charges(charges) if charges is not None else ""
    im_str = _hex_encode(ims) if ims is not None else ""

    binary_payload = _encode_binary_payload(mz_str, intensity_str, charge_str, im_str)
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
    mz_str, intensity_str, charge_str, im_str = _decode_binary_payload(binary_payload)

    mzs = np.fromiter(_delta_decode_single_string(mz_str), dtype=float) if mz_str else np.array([], dtype=float)
    intensities = np.fromiter(_hex_decode(intensity_str), dtype=float) if intensity_str else np.array([], dtype=float)

    charges = None
    if charge_str:
        # Charges can be None, so we decode to list first, then handle None -> 0 conversion for numpy array
        # or keep as None if user expects list? But Spectrum usually desires numpy arrays.
        # Assuming 0 is used for missing charge in numpy array context often.
        decoded_charges = list(_decode_charges(charge_str))
        charges = np.array([c if c is not None else 0 for c in decoded_charges], dtype=int)

    ims = None
    if im_str:
        ims = np.fromiter(_hex_decode(im_str), dtype=float)

    return Spectrum(mz=mzs, intensity=intensities, charge=charges, ion_mobility=ims)

    return Spectrum(mz=mzs, intensity=intensities, charge=charges, ion_mobility=ims)

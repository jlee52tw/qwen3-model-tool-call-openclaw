#!/usr/bin/env python3
"""
NVMe Direct I/O for Windows
============================

Bypasses the OS page cache when writing/reading intermediate tensor data
to NVMe storage. On a memory-constrained system (96 GB RAM with 79+ GB
used by NNCF + FP16 model), regular file I/O pollutes the page cache
and can trigger additional memory pressure or page-file thrashing.

Uses Windows CreateFileW with FILE_FLAG_NO_BUFFERING + FILE_FLAG_WRITE_THROUGH
and VirtualAlloc for sector-aligned buffers (4096-byte alignment).

Falls back to standard file I/O on non-Windows or if Direct I/O fails.

File format (save_array / load_array):
  [HEADER: 4096 bytes, sector-aligned]
    8B  magic   "NVMEDUMP"
    4B  version  1
    4B  dtype_len
    NB  dtype string (e.g. "float32")
    4B  ndim
    ndim*8B  shape (int64 each)
    8B  data_size (actual byte count before padding)
    <zero-padded to SECTOR_SIZE>
  [DATA: sector-aligned]
    raw numpy bytes, zero-padded to SECTOR_SIZE

Usage:
    from nvme_direct_io import save_array, load_array

    save_array("D:/nvme-temp/stats_0.bin", numpy_array)
    arr = load_array("D:/nvme-temp/stats_0.bin")
"""

import os
import sys
import struct
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

# ── Constants ──────────────────────────────────────────────────────────────────

SECTOR_SIZE = 4096
HEADER_MAGIC = b"NVMEDUMP"
HEADER_VERSION = 1

# ── Platform detection ─────────────────────────────────────────────────────────

_USE_DIRECT_IO = sys.platform == "win32"

if _USE_DIRECT_IO:
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.windll.kernel32

    # CreateFileW constants
    _GENERIC_READ = 0x80000000
    _GENERIC_WRITE = 0x40000000
    _FILE_SHARE_READ = 0x00000001
    _CREATE_ALWAYS = 2
    _OPEN_EXISTING = 3
    _FILE_FLAG_NO_BUFFERING = 0x20000000
    _FILE_FLAG_WRITE_THROUGH = 0x80000000
    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    # VirtualAlloc constants
    _MEM_COMMIT = 0x1000
    _MEM_RESERVE = 0x2000
    _MEM_RELEASE = 0x8000
    _PAGE_READWRITE = 0x04


# ── Internal helpers ───────────────────────────────────────────────────────────

def _align_up(size: int, alignment: int) -> int:
    """Round up size to the next multiple of alignment."""
    return (size + alignment - 1) & ~(alignment - 1)


def _make_header(array: np.ndarray) -> bytes:
    """Create a fixed-size header (SECTOR_SIZE bytes) for a numpy array."""
    dtype_str = str(array.dtype).encode("ascii")
    shape = array.shape
    data_size = array.nbytes

    header = bytearray(SECTOR_SIZE)
    offset = 0

    # Magic (8 bytes)
    struct.pack_into("8s", header, offset, HEADER_MAGIC)
    offset += 8

    # Version (4 bytes)
    struct.pack_into("<I", header, offset, HEADER_VERSION)
    offset += 4

    # Dtype string length + string
    struct.pack_into("<I", header, offset, len(dtype_str))
    offset += 4
    header[offset:offset + len(dtype_str)] = dtype_str
    offset += len(dtype_str)

    # Number of dimensions
    struct.pack_into("<I", header, offset, len(shape))
    offset += 4

    # Shape elements (int64 each)
    for dim in shape:
        struct.pack_into("<q", header, offset, dim)
        offset += 8

    # Actual data size in bytes
    struct.pack_into("<Q", header, offset, data_size)

    max_header_payload = 8 + 4 + 4 + 256 + 4 + 32 * 8 + 8  # generous max
    if offset > SECTOR_SIZE:
        raise ValueError(f"Header exceeds {SECTOR_SIZE} bytes (ndim={len(shape)}, dtype={dtype_str})")

    return bytes(header)


def _parse_header(data: bytes) -> Tuple[np.dtype, tuple, int, int]:
    """Parse header → (dtype, shape, data_offset, data_size)."""
    offset = 0

    magic = struct.unpack_from("8s", data, offset)[0]
    offset += 8
    if magic != HEADER_MAGIC:
        raise ValueError(f"Invalid magic: {magic!r}, expected {HEADER_MAGIC!r}")

    version = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    if version != HEADER_VERSION:
        raise ValueError(f"Unsupported version: {version}")

    dtype_len = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    dtype_str = data[offset:offset + dtype_len].decode("ascii")
    offset += dtype_len

    ndim = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    shape = tuple(struct.unpack_from("<q", data, offset + i * 8)[0] for i in range(ndim))
    offset += ndim * 8

    data_size = struct.unpack_from("<Q", data, offset)[0]

    return np.dtype(dtype_str), shape, SECTOR_SIZE, data_size


# ── Windows Direct I/O ────────────────────────────────────────────────────────

def _write_direct(path: str, payload: bytes) -> bool:
    """Write payload using Windows Direct I/O. Returns True on success."""
    if not _USE_DIRECT_IO:
        return False

    try:
        aligned_size = _align_up(len(payload), SECTOR_SIZE)

        # Allocate sector-aligned buffer via VirtualAlloc
        buf = _kernel32.VirtualAlloc(
            None, aligned_size, _MEM_COMMIT | _MEM_RESERVE, _PAGE_READWRITE
        )
        if not buf:
            return False

        try:
            # Copy payload into aligned buffer
            ctypes.memmove(buf, payload, len(payload))
            # Zero-pad remainder for sector alignment
            if aligned_size > len(payload):
                ctypes.memset(buf + len(payload), 0, aligned_size - len(payload))

            # Open with NO_BUFFERING + WRITE_THROUGH to bypass page cache
            handle = _kernel32.CreateFileW(
                str(path),
                _GENERIC_WRITE,
                0,  # exclusive access during write
                None,
                _CREATE_ALWAYS,
                _FILE_FLAG_NO_BUFFERING | _FILE_FLAG_WRITE_THROUGH,
                None,
            )
            if handle == _INVALID_HANDLE_VALUE:
                return False

            try:
                bytes_written = wintypes.DWORD(0)
                success = _kernel32.WriteFile(
                    handle, buf, aligned_size,
                    ctypes.byref(bytes_written), None
                )
                return bool(success)
            finally:
                _kernel32.CloseHandle(handle)
        finally:
            _kernel32.VirtualFree(buf, 0, _MEM_RELEASE)

    except Exception:
        return False


def _read_direct(path: str) -> Optional[bytes]:
    """Read entire file using Windows Direct I/O. Returns None on failure."""
    if not _USE_DIRECT_IO:
        return None

    try:
        file_size = os.path.getsize(path)
        aligned_size = _align_up(file_size, SECTOR_SIZE)

        buf = _kernel32.VirtualAlloc(
            None, aligned_size, _MEM_COMMIT | _MEM_RESERVE, _PAGE_READWRITE
        )
        if not buf:
            return None

        try:
            handle = _kernel32.CreateFileW(
                str(path),
                _GENERIC_READ,
                _FILE_SHARE_READ,
                None,
                _OPEN_EXISTING,
                _FILE_FLAG_NO_BUFFERING,
                None,
            )
            if handle == _INVALID_HANDLE_VALUE:
                return None

            try:
                bytes_read = wintypes.DWORD(0)
                success = _kernel32.ReadFile(
                    handle, buf, aligned_size,
                    ctypes.byref(bytes_read), None
                )
                if not success:
                    return None
                # Return only the real data (trim zero-padding)
                return ctypes.string_at(buf, file_size)
            finally:
                _kernel32.CloseHandle(handle)
        finally:
            _kernel32.VirtualFree(buf, 0, _MEM_RELEASE)

    except Exception:
        return None


# ── Public API ─────────────────────────────────────────────────────────────────

def save_array(path: str, array: np.ndarray) -> None:
    """
    Save a numpy array to disk, bypassing the OS page cache on Windows.

    On non-Windows or if Direct I/O fails, falls back to standard file I/O.
    The array is stored with a self-describing header (dtype, shape) so that
    load_array() can reconstruct it without extra metadata.

    :param path: File path to write.
    :param array: Numpy array to save. Must be contiguous.
    """
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Ensure contiguous memory layout
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)

    header = _make_header(array)
    data = array.tobytes()
    payload = header + data

    if not _write_direct(path, payload):
        # Fallback to standard I/O (works on all platforms)
        with open(path, "wb") as f:
            f.write(payload)


def load_array(path: str) -> np.ndarray:
    """
    Load a numpy array from disk, bypassing the OS page cache on Windows.

    :param path: File path to read (must have been saved by save_array).
    :return: Numpy array with original dtype and shape.
    """
    path = str(path)

    raw = _read_direct(path)
    if raw is None:
        with open(path, "rb") as f:
            raw = f.read()

    dtype, shape, data_offset, data_size = _parse_header(raw)
    data = raw[data_offset:data_offset + data_size]
    return np.frombuffer(data, dtype=dtype).copy().reshape(shape)


def save_bytes(path: str, data: bytes) -> None:
    """
    Save raw bytes to disk with Direct I/O bypass.

    :param path: File path to write.
    :param data: Raw bytes to save.
    """
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if not _write_direct(path, data):
        with open(path, "wb") as f:
            f.write(data)


def load_bytes(path: str) -> bytes:
    """
    Load raw bytes from disk with Direct I/O bypass.

    :param path: File path to read.
    :return: File contents as bytes.
    """
    path = str(path)

    raw = _read_direct(path)
    if raw is None:
        with open(path, "rb") as f:
            raw = f.read()
    return raw


def is_direct_io_available() -> bool:
    """Check if Direct I/O is available on this platform."""
    return _USE_DIRECT_IO


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print(f"Platform: {sys.platform}")
    print(f"Direct I/O available: {is_direct_io_available()}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with various dtypes and shapes
        test_cases = [
            np.random.randn(4096).astype(np.float32),
            np.random.randn(32, 4096).astype(np.float16),
            np.random.randn(3, 256, 256).astype(np.float64),
            np.array([1, 2, 3], dtype=np.int64),
            np.zeros((1,), dtype=np.float32),
        ]

        for i, arr in enumerate(test_cases):
            fpath = os.path.join(tmpdir, f"test_{i}.bin")
            save_array(fpath, arr)
            loaded = load_array(fpath)

            assert loaded.dtype == arr.dtype, f"dtype mismatch: {loaded.dtype} vs {arr.dtype}"
            assert loaded.shape == arr.shape, f"shape mismatch: {loaded.shape} vs {arr.shape}"
            assert np.array_equal(loaded, arr), f"data mismatch for test case {i}"

            file_size = os.path.getsize(fpath)
            print(f"  Test {i}: {arr.dtype} {arr.shape} → {file_size} bytes ✓")

        # Test raw bytes
        raw_data = b"Hello, NVMe Direct I/O!"
        raw_path = os.path.join(tmpdir, "test_raw.bin")
        save_bytes(raw_path, raw_data)
        loaded_raw = load_bytes(raw_path)
        assert loaded_raw[:len(raw_data)] == raw_data
        print(f"  Raw bytes test: {len(raw_data)} bytes ✓")

    print("\nAll tests passed!")

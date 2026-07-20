"""Safely download the OodleUE Windows DLL when no local runtime is available.

The ZIP layout and exact-entry extraction mirror CUE4Parse's OodleHelper:
https://github.com/FabianFG/CUE4Parse/blob/master/CUE4Parse/Compression/OodleHelper.cs
"""

import io
import os
import tempfile
import urllib.request
import zipfile


OODLE_DLL_NAME = "oodle-data-shared.dll"
OODLE_RELEASE_URL = (
    "https://github.com/WorkingRobot/OodleUE/releases/latest/download/"
    "clang-cl-x64-release.zip"
)
OODLE_ZIP_ENTRY = f"bin/{OODLE_DLL_NAME}"

_MAX_DOWNLOAD_SIZE = 64 * 1024 * 1024
_MAX_DLL_SIZE = 32 * 1024 * 1024
_MIN_DLL_SIZE = 64 * 1024
_PE_MACHINE_AMD64 = 0x8664


class OodleDownloadError(RuntimeError):
    """Raised when the Oodle runtime cannot be safely downloaded or installed."""


def _download_zip(url, timeout, urlopen):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "AFoP-Mesh-Tool-Oodle-Helper"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) > _MAX_DOWNLOAD_SIZE:
                raise OodleDownloadError("Oodle archive is larger than the allowed limit")
            payload = response.read(_MAX_DOWNLOAD_SIZE + 1)
    except OodleDownloadError:
        raise
    except Exception as error:
        raise OodleDownloadError(f"Could not download the Oodle archive: {error}") from error
    if len(payload) > _MAX_DOWNLOAD_SIZE:
        raise OodleDownloadError("Oodle archive is larger than the allowed limit")
    return payload


def _extract_oodle_dll(archive_data):
    try:
        with zipfile.ZipFile(io.BytesIO(archive_data), "r") as archive:
            matches = [
                info for info in archive.infolist()
                if info.filename.replace("\\", "/") == OODLE_ZIP_ENTRY
            ]
            if len(matches) != 1:
                raise OodleDownloadError(
                    f"The archive must contain exactly one {OODLE_ZIP_ENTRY} entry"
                )
            info = matches[0]
            if info.is_dir() or not (_MIN_DLL_SIZE <= info.file_size <= _MAX_DLL_SIZE):
                raise OodleDownloadError("The Oodle DLL has an unexpected size")
            dll_data = archive.read(info)
    except OodleDownloadError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise OodleDownloadError(f"Could not read the Oodle archive: {error}") from error
    if len(dll_data) != info.file_size:
        raise OodleDownloadError("The extracted Oodle DLL size does not match the ZIP entry")
    return dll_data


def _validate_windows_x64_dll(dll_data):
    if len(dll_data) < _MIN_DLL_SIZE or dll_data[:2] != b"MZ":
        raise OodleDownloadError("The downloaded Oodle file is not a Windows DLL")
    pe_offset = int.from_bytes(dll_data[0x3C:0x40], "little")
    if pe_offset < 0x40 or pe_offset + 6 > len(dll_data):
        raise OodleDownloadError("The downloaded Oodle DLL has an invalid PE header")
    if dll_data[pe_offset:pe_offset + 4] != b"PE\0\0":
        raise OodleDownloadError("The downloaded Oodle DLL has an invalid PE signature")
    machine = int.from_bytes(dll_data[pe_offset + 4:pe_offset + 6], "little")
    if machine != _PE_MACHINE_AMD64:
        raise OodleDownloadError("The downloaded Oodle DLL is not an x64 build")


def ensure_oodle_dll(addon_directory=None, url=OODLE_RELEASE_URL, timeout=30, urlopen=None):
    """Return the add-on DLL path, downloading and atomically installing it if absent."""
    addon_directory = addon_directory or os.path.dirname(os.path.abspath(__file__))
    destination = os.path.join(addon_directory, OODLE_DLL_NAME)
    if os.path.isfile(destination):
        return destination

    archive_data = _download_zip(url, timeout, urlopen or urllib.request.urlopen)
    dll_data = _extract_oodle_dll(archive_data)
    _validate_windows_x64_dll(dll_data)

    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{OODLE_DLL_NAME}.",
            suffix=".download_tmp",
            dir=addon_directory,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(dll_data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    except OSError as error:
        raise OodleDownloadError(f"Could not install the Oodle DLL: {error}") from error
    finally:
        if temporary is not None:
            try:
                os.remove(temporary)
            except OSError:
                pass
    return destination


__all__ = (
    "OODLE_DLL_NAME",
    "OODLE_RELEASE_URL",
    "OodleDownloadError",
    "ensure_oodle_dll",
)

"""STF/DDS conversion and Blender image loading for material imports."""

from __future__ import annotations

import struct

import bpy

from ..log import logger


_STF_DXGI = {
    0x04: 87, 0x05: 2, 0x07: 10, 0x0A: 11, 0x0D: 29, 0x0E: 28,
    0x17: 49, 0x19: 56, 0x1B: 61, 0x1C: 72, 0x1D: 71, 0x1E: 75,
    0x1F: 74, 0x20: 78, 0x21: 77, 0x45: 80, 0x46: 81, 0x47: 83,
    0x48: 84, 0x49: 95, 0x4A: 96, 0x4B: 98, 0x4C: 99,
}
_BLOCK_BYTES = {
    71: 8, 72: 8, 74: 16, 75: 16, 77: 16, 78: 16, 80: 8, 81: 8,
    83: 16, 84: 16, 95: 16, 96: 16, 98: 16, 99: 16,
}
_BYTES_PER_PIXEL = {2: 16, 10: 8, 11: 8, 28: 4, 29: 4, 49: 2, 56: 2, 61: 1, 87: 4}


def _mip_entries(data):
    descriptor = struct.unpack_from("<Q", data, 4)[0]
    count = ((descriptor >> 56) & 0x0F) + 1
    if 12 + count * 4 > len(data):
        raise ValueError("truncated STF mip table")
    entries = []
    for index in range(count):
        value = struct.unpack_from("<I", data, 12 + index * 4)[0]
        entries.append(((value >> 23) & 0x3F, (value & 0x7FFFFF) * 16))
    return descriptor, entries


def _build_dds(width, height, dxgi, payload):
    header = bytearray(148)
    header[:4] = b"DDS "
    struct.pack_into("<I", header, 4, 124)
    struct.pack_into("<I", header, 8, 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000)
    struct.pack_into("<I", header, 12, height)
    struct.pack_into("<I", header, 16, width)
    struct.pack_into("<I", header, 20, len(payload))
    struct.pack_into("<I", header, 28, 1)
    struct.pack_into("<I", header, 76, 32)
    struct.pack_into("<I", header, 80, 0x4)
    header[84:88] = b"DX10"
    struct.pack_into("<I", header, 108, 0x1000)
    struct.pack_into("<I", header, 128, dxgi)
    struct.pack_into("<I", header, 132, 3)  # D3D10_RESOURCE_DIMENSION_TEXTURE2D
    struct.pack_into("<I", header, 140, 1)
    return bytes(header) + payload


def texture_to_dds(data, asset):
    """Turn an extracted STF texture into a one-mip, Blender-readable DDS."""
    if data[:4] == b"DDS ":
        return data
    if data[:4] != b"STF\x02":
        raise ValueError("texture has neither STF nor DDS magic")
    descriptor, entries = _mip_entries(data)
    dxgi = _STF_DXGI.get(descriptor & 0xFF)
    width = (descriptor >> 10) & 0x7FFF
    height = (descriptor >> 25) & 0x7FFF
    if dxgi is None or not (0 < width <= 16384 and 0 < height <= 16384):
        raise ValueError("unsupported STF texture descriptor")

    slice_sizes = [item.decompressed_size for item in asset.data_slices]
    header_size = len(data) - sum(slice_sizes)
    resident = -1
    for index, (_mips, expected_size) in enumerate(entries[:len(slice_sizes)]):
        if abs(slice_sizes[index] - expected_size) > 16:
            break
        resident = index
    if resident < 0:
        # Some old/stub descriptors disagree with their slices.  The largest
        # slice is still preferable to silently importing an empty texture.
        resident = max(range(len(slice_sizes)), key=slice_sizes.__getitem__)

    mip_level = sum(entry[0] for entry in entries[resident + 1:])
    mip_width = max(width >> mip_level, 1)
    mip_height = max(height >> mip_level, 1)
    offset = header_size + sum(slice_sizes[:resident])
    resident_data = data[offset:offset + slice_sizes[resident]]
    if dxgi in _BLOCK_BYTES:
        needed = (
            max((mip_width + 3) // 4, 1)
            * max((mip_height + 3) // 4, 1)
            * _BLOCK_BYTES[dxgi]
        )
    else:
        needed = mip_width * mip_height * _BYTES_PER_PIXEL[dxgi]
    if needed > len(resident_data):
        raise ValueError("resident texture mip is truncated")
    return _build_dds(mip_width, mip_height, dxgi, resident_data[:needed])


def _load_image(path, logical_path, non_color=False, alpha_mode=None):
    """Load one game texture, isolating material-specific alpha handling."""
    alpha_override = alpha_mode or ""
    existing = next(
        (
            image for image in bpy.data.images
            if image.get("afop_asset_path") == logical_path
            and image.get("afop_alpha_mode_override", "") == alpha_override
        ),
        None,
    )
    if existing is not None:
        image = existing
    else:
        # Preserve Blender's normal path-based reuse unless this material needs
        # its own alpha mode. If path reuse finds an overridden copy, isolate
        # the ordinary image as well so Body can remain Straight.
        image = bpy.data.images.load(path, check_existing=alpha_mode is None)
        if alpha_mode is None and image.get("afop_alpha_mode_override", ""):
            image = bpy.data.images.load(path, check_existing=False)
    image["afop_asset_path"] = logical_path
    if alpha_mode is not None:
        image["afop_alpha_mode_override"] = alpha_mode
        try:
            image.alpha_mode = alpha_mode
        except (AttributeError, TypeError, ValueError):
            logger.warning("Could not set alpha mode %s for %s", alpha_mode, logical_path)
    if non_color:
        try:
            image.colorspace_settings.name = "Non-Color"
        except (AttributeError, TypeError):
            logger.warning("Could not set Non-Color space for %s", logical_path)
    return image

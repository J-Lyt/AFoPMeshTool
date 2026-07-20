"""Encoder and transactional rewriter for MMB secondary meshlet sections.

The secondary stream used by ``lod_info_type == 2`` duplicates the geometry
needed by Snowdrop's meshlet path.  It therefore has to be rebuilt whenever a
primary LOD's positions, skin, normals, or topology change.
"""

from collections import Counter
import math
from struct import pack, unpack

import numpy as np


MAX_VERTICES = 64
MAX_TRIANGLES = 64
_Q23 = 8388607.0


def _morton_order(centroids):
    values = np.asarray(centroids, dtype=np.float64)
    minimum = values.min(axis=0)
    extent = np.maximum(values.max(axis=0) - minimum, 1e-9)
    quantized = np.clip(
        ((values - minimum) / extent * 1023.0).astype(np.int64), 0, 1023)

    def spread(value):
        value &= 0x3FF
        value = (value | (value << 16)) & 0x30000FF
        value = (value | (value << 8)) & 0x300F00F
        value = (value | (value << 4)) & 0x30C30C3
        value = (value | (value << 2)) & 0x9249249
        return value

    codes = (spread(quantized[:, 0])
             | (spread(quantized[:, 1]) << 1)
             | (spread(quantized[:, 2]) << 2))
    return np.argsort(codes, kind='stable')


def build_meshlets(positions, triangles):
    """Partition triangles into spatially coherent <=64v/<=64t meshlets."""
    points = np.asarray(positions, dtype=np.float64)
    faces = [tuple(int(value) for value in triangle) for triangle in triangles]
    if not faces:
        return []
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("meshlet positions must be a finite Nx3 array")
    if any(index < 0 or index >= len(points)
           for triangle in faces for index in triangle):
        raise ValueError("meshlet triangle references an invalid vertex")

    face_array = np.asarray(faces, dtype=np.int64)
    centroids = points[face_array].mean(axis=1)
    vertex_faces = {}
    for face_index, triangle in enumerate(faces):
        for vertex in triangle:
            vertex_faces.setdefault(vertex, []).append(face_index)

    used = bytearray(len(faces))
    order = _morton_order(centroids)
    order_cursor = 0
    remaining = len(faces)
    result = []

    def next_unused():
        nonlocal order_cursor
        while order_cursor < len(faces) and used[int(order[order_cursor])]:
            order_cursor += 1
        return int(order[order_cursor]) if order_cursor < len(faces) else None

    while remaining:
        seed = next_unused()
        if seed is None:
            raise ValueError("meshlet partition ended before all triangles were consumed")
        local_vertices = {}
        source_faces = []
        frontier = []

        def attach(face_index):
            nonlocal remaining
            for vertex in faces[face_index]:
                if vertex not in local_vertices:
                    local_vertices[vertex] = len(local_vertices)
            source_faces.append(face_index)
            used[face_index] = 1
            remaining -= 1
            for vertex in faces[face_index]:
                frontier.extend(candidate for candidate in vertex_faces[vertex]
                                if not used[candidate])

        attach(seed)
        while (len(source_faces) < MAX_TRIANGLES
               and len(local_vertices) < MAX_VERTICES):
            added = False
            while frontier:
                candidate = frontier.pop()
                if used[candidate]:
                    continue
                extra = sum(vertex not in local_vertices
                            for vertex in faces[candidate])
                if len(local_vertices) + extra > MAX_VERTICES:
                    continue
                attach(candidate)
                added = True
                break
            if added:
                continue
            candidate = next_unused()
            if candidate is None:
                break
            extra = sum(vertex not in local_vertices
                        for vertex in faces[candidate])
            if len(local_vertices) + extra > MAX_VERTICES:
                break
            attach(candidate)

        vertices = list(local_vertices)
        local_faces = [tuple(local_vertices[index] for index in faces[source])
                       for source in source_faces]
        result.append((vertices, local_faces))

    rebuilt = Counter(
        tuple(vertices[index] for index in triangle)
        for vertices, local_faces in result for triangle in local_faces)
    if rebuilt != Counter(faces):
        raise ValueError("meshlet partition did not preserve triangle coverage")
    return result


def _global_scale(points):
    maximum = float(np.abs(points).max()) if len(points) else 1.0
    desired = max(maximum * (_Q23 / 8388352.0) * 1.0001, 6.1e-5)
    scale = float(np.float16(min(desired, 65504.0)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("meshlet position scale is not representable as float16")
    while maximum * (_Q23 / scale) > 8388352.0 and scale < 65504.0:
        next_scale = float(np.nextafter(
            np.float16(scale), np.float16(np.inf), dtype=np.float16))
        if next_scale == scale:
            break
        scale = next_scale
    if maximum * (_Q23 / scale) > 8388352.0:
        raise ValueError("meshlet positions exceed the 24-bit anchor range")
    return scale


def _encode_positions(points, scale):
    integer_points = np.asarray(points, dtype=np.float64) * (_Q23 / scale)
    center = (integer_points.min(axis=0) + integer_points.max(axis=0)) * 0.5
    anchor = np.clip(
        np.round(center / 256.0).astype(np.int64) * 256,
        -32768 * 256, 32767 * 256)
    residual = integer_points - anchor
    encoded = np.zeros((len(points), 4), dtype=np.int16)
    for index, delta in enumerate(residual):
        shift = 0
        while (shift < 31
               and float(np.abs(np.round(delta / float(1 << shift))).max())
               > 32767.0):
            shift += 1
        values = np.round(delta / float(1 << shift)).astype(np.int64)
        if np.any(values < -32768) or np.any(values > 32767):
            raise ValueError("meshlet position residual exceeds int16 range")
        encoded[index, :3] = values.astype(np.int16)
        encoded[index, 3] = shift
    return encoded, anchor


def _decode_positions(encoded, anchor, scale):
    encoded = np.asarray(encoded, dtype=np.int64)
    result = np.empty((len(encoded), 3), dtype=np.float64)
    shifts = encoded[:, 3] & 31
    for axis in range(3):
        result[:, axis] = (
            (encoded[:, axis] * (1 << shifts)) + anchor[axis])
        result[:, axis] *= scale / _Q23
    return result


def _pack_record(vertex_offset, vertex_count, triangle_offset,
                 triangle_count, anchor, scale):
    if (vertex_count > 0x1FF or triangle_count > 0x1FF
            or vertex_offset > 0x7FFFFF or triangle_offset > 0x7FFFFF):
        raise ValueError("meshlet record exceeds its packed field range")
    first = (vertex_offset & 0x7FFFFF) | (vertex_count << 23)
    second = (triangle_offset & 0x7FFFFF) | (triangle_count << 23)
    words = [first & 0xFFFF, first >> 16, second & 0xFFFF, second >> 16]
    anchors = [(int(value) >> 8) & 0xFFFF for value in anchor]
    scale_word = unpack('<H', pack('<e', scale))[0]
    return pack('<8H', *words, *anchors, scale_word)


def _encode_snorm(value, bits):
    maximum = (1 << (bits - 1)) - 1
    return (int(round(max(-1.0, min(1.0, value)) * maximum))
            & ((1 << bits) - 1))


def _encode_unorm(value, bits):
    maximum = (1 << bits) - 1
    return int(round(max(0.0, min(1.0, value)) * maximum)) & maximum


def _culling_record(points, face_normals, scale, skinned, primary_bone):
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) * 0.5
    radius = float(np.linalg.norm(points - center, axis=1).max())
    packed_center = (
        _encode_snorm(center[0] / scale, 10)
        | (_encode_snorm(center[1] / scale, 10) << 10)
        | (_encode_snorm(center[2] / scale, 10) << 20))
    if skinned:
        flags = 0x5  # frustum + occlusion; bounds follow the dominant bone
        packed_radius = (_encode_unorm(radius / (2.0 * scale), 10)
                         | (flags << 20))
        return pack('<4I', packed_center, int(primary_bone), 0, packed_radius)

    flags = 0x7  # frustum + backface + occlusion
    packed_radius = _encode_unorm(radius / (2.0 * scale), 10)
    cone = 0
    normals = np.asarray(face_normals, dtype=np.float64)
    if len(normals):
        lengths = np.linalg.norm(normals, axis=1)
        normals = normals[lengths > 1e-9]
        if len(normals):
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            average = normals.mean(axis=0)
            length = np.linalg.norm(average)
            if length > 1e-6:
                axis = -average / length
                minimum_dot = float((normals @ (-axis)).min())
                cutoff = math.sqrt(max(0.0, 1.0 - minimum_dot ** 2))
                cone = (_encode_snorm(axis[0], 10)
                        | (_encode_snorm(axis[1], 10) << 10)
                        | (_encode_snorm(axis[2], 10) << 20))
                packed_radius |= _encode_unorm(cutoff, 10) << 10
    packed_radius |= flags << 20
    return pack('<4I', packed_center, 0, cone, packed_radius)


def _validate_rows(rows, count, label):
    if rows is None:
        return 0
    if len(rows) != count:
        raise ValueError(f"{label} row count does not match the vertex count")
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError(f"{label} rows do not have a uniform stride")
    return widths.pop()


def encode_section(positions, triangles, vertex_payload=None,
                   normal_payload=None, skinned=False, vertex_skin=None):
    """Encode one complete secondary section and return it with its metadata."""
    points = np.asarray(positions, dtype=np.float64)
    faces = [tuple(int(value) for value in triangle) for triangle in triangles]
    if not len(points) or not faces:
        raise ValueError("cannot regenerate an empty meshlet LOD")
    payload_size = _validate_rows(vertex_payload, len(points), "vertex payload")
    normal_size = _validate_rows(normal_payload, len(points), "normal payload")
    if skinned and (vertex_skin is None or len(vertex_skin) != len(points)):
        raise ValueError("skinned meshlet input is missing per-vertex skin data")

    meshlets = build_meshlets(points, faces)
    scale = _global_scale(points)
    records = bytearray()
    culling = bytearray()
    vertices = bytearray()
    normals = bytearray()
    indices = bytearray()
    vertex_offset = 0
    triangle_offset = 0

    for global_vertices, local_faces in meshlets:
        local_points = points[global_vertices]
        encoded, anchor = _encode_positions(local_points, scale)
        decoded = _decode_positions(encoded, anchor, scale)
        # Each residual component is rounded after its per-vertex power-of-two
        # shift, so the admissible error follows the largest encoded shift.
        maximum_shift = int((encoded[:, 3] & 31).max())
        tolerance = max(
            (1 << maximum_shift) * scale / _Q23 * 0.501, 1e-6)
        if float(np.abs(decoded - local_points).max()) > tolerance:
            raise ValueError("meshlet position quantization exceeded its tolerance")

        for local_index, global_index in enumerate(global_vertices):
            vertices += pack('<4h', *(int(value) for value in encoded[local_index]))
            if payload_size:
                vertices += bytes(vertex_payload[global_index])
            if normal_size:
                normals += bytes(normal_payload[global_index])
        for triangle in local_faces:
            if any(index < 0 or index >= len(global_vertices) for index in triangle):
                raise ValueError("meshlet local triangle index is outside its vertex table")
            word = (triangle[0] | (triangle[1] << 10)
                    | (triangle[2] << 20))
            indices += pack('<I', word)

        primary_bone = 0
        if skinned:
            accumulated = {}
            for global_index in global_vertices:
                for bone, weight in vertex_skin[global_index].items():
                    accumulated[int(bone)] = accumulated.get(int(bone), 0.0) + weight
            if accumulated:
                primary_bone = max(accumulated, key=accumulated.get)
        face_normals = [
            np.cross(local_points[b] - local_points[a],
                     local_points[c] - local_points[a])
            for a, b, c in local_faces]
        records += _pack_record(
            vertex_offset, len(global_vertices), triangle_offset,
            len(local_faces), anchor, scale)
        culling += _culling_record(
            local_points, face_normals, scale, skinned, primary_bone)
        vertex_offset += len(global_vertices)
        triangle_offset += len(local_faces)

    section = bytes(records + culling + vertices + normals + indices)
    expected = (len(meshlets) * 32
                + vertex_offset * (8 + payload_size + normal_size)
                + triangle_offset * 4)
    if len(section) != expected or triangle_offset != len(faces):
        raise ValueError("generated meshlet section failed its size validation")
    return {
        'section': section,
        'meshlet_count': len(meshlets),
        'duplicated_vertex_count': vertex_offset,
        'triangle_count': triangle_offset,
        'vertex_stride': 8 + payload_size,
        'normal_stride': normal_size,
    }


def descriptor_fields(encoded, absolute_offset, virtual_base):
    meshlet_count = encoded['meshlet_count']
    vertex_count = encoded['duplicated_vertex_count']
    triangle_count = encoded['triangle_count']
    first = int(virtual_base)
    descriptor_end = first + meshlet_count * 32
    vertex_end = descriptor_end + vertex_count * encoded['vertex_stride']
    normal_end = vertex_end + vertex_count * encoded['normal_stride']
    size = normal_end - first + triangle_count * 4
    if size != len(encoded['section']):
        raise ValueError("meshlet descriptor does not cover the generated section")
    values = (first, meshlet_count, descriptor_end, vertex_end,
              normal_end, int(absolute_offset), size)
    if any(value < 0 or value > 0xFFFFFFFF for value in values):
        raise ValueError("meshlet descriptor field is outside uint32 range")
    return values


def _section_entries(file_data, asset):
    entries = []
    for mesh_index, mesh in enumerate(asset.meshes):
        if mesh.lod_info_type != 2:
            continue
        for lod in mesh.lods:
            descriptor_offset = lod.start_offset + lod.lod_field_offset + 36
            if descriptor_offset < 0 or descriptor_offset + 28 > len(file_data):
                raise ValueError(
                    f"'{mesh.name}' LOD{lod.index} meshlet descriptor is outside the file")
            fields = unpack('<7I', file_data[descriptor_offset:descriptor_offset + 28])
            if fields[6] == 0:
                continue
            entries.append({
                'key': (mesh_index, lod.index),
                'mesh_name': mesh.name,
                'descriptor_offset': descriptor_offset,
                'fields': fields,
            })
    entries.sort(key=lambda entry: entry['fields'][5])
    return entries


def validate_region(file_data, asset):
    """Validate descriptor arithmetic and contiguous secondary-stream tiling."""
    entries = _section_entries(file_data, asset)
    if not entries:
        return {'section_count': 0, 'stream_start': len(file_data)}
    main_end = max(
        lod.data_offset + lod.data_size
        for mesh in asset.meshes for lod in mesh.lods if lod.data_size > 0)
    cursor = main_end
    for entry in entries:
        f0, f1, f2, f3, f4, f5, f6 = entry['fields']
        name = f"{entry['mesh_name']} LOD{entry['key'][1]}"
        if f5 != cursor:
            raise ValueError(f"{name} meshlet section does not tile contiguously")
        if not (f0 <= f2 <= f3 <= f4 <= f0 + f6):
            raise ValueError(f"{name} meshlet descriptor offsets are invalid")
        if f2 - f0 != f1 * 32:
            raise ValueError(f"{name} meshlet descriptor table has the wrong size")
        if (f0 + f6 - f4) % 4:
            raise ValueError(f"{name} meshlet triangle region is not uint32-aligned")
        if f5 + f6 > len(file_data):
            raise ValueError(f"{name} meshlet section is truncated")
        cursor += f6
    if cursor != len(file_data):
        raise ValueError("meshlet stream does not terminate at end-of-file")
    return {'section_count': len(entries), 'stream_start': main_end}


def rebuild_region(file_data, asset, regenerated):
    """Replace selected sections, preserve all others, and re-tile transactionally.

    ``regenerated`` maps ``(mesh_index, lod_index)`` to an ``encode_section``
    result.  Every requested key must name an existing secondary section.
    """
    original = bytes(file_data)
    validation = validate_region(original, asset)
    entries = _section_entries(original, asset)
    available = {entry['key'] for entry in entries}
    missing = set(regenerated) - available
    if missing:
        raise ValueError(f"meshlet regeneration requested unknown LOD(s): {sorted(missing)}")
    if not regenerated:
        return original

    stream_start = validation['stream_start']
    region = bytearray()
    descriptor_patches = {}
    cursor = stream_start
    for entry in entries:
        old_fields = entry['fields']
        encoded = regenerated.get(entry['key'])
        if encoded is None:
            section = original[old_fields[5]:old_fields[5] + old_fields[6]]
            fields = (*old_fields[:5], cursor, old_fields[6])
        else:
            section = encoded['section']
            fields = descriptor_fields(encoded, cursor, old_fields[0])
        descriptor_patches[entry['descriptor_offset']] = pack('<7I', *fields)
        region += section
        cursor += len(section)

    output = bytearray(original[:stream_start])
    output += region
    for offset, replacement in descriptor_patches.items():
        output[offset:offset + 28] = replacement
    validate_region(bytes(output), asset)
    return bytes(output)

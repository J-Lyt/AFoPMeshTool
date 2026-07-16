"""Snowdrop .mmb format model and parser."""

import io
from struct import pack, unpack

from mathutils import Vector

from .binary_io import bp, br
from .file_utils import CopyFile
from .log import logger

# Known values:
#   0.0     -> float32   : (8 bytes/vert, unquantized)
#   4095.0  -> compact   : (uint16 % 4096) / 4095
#   4096.0  -> wide      : signed int16 / 4096
#   32767.0 -> int16_norm: signed int16 / 32767
#
# A block is only accepted as a divisor table when every entry matches one of
# these values, so genuine hashes are never mistaken for divisors.
_UV_KNOWN_DIVISORS = (0.0, 4095.0, 4096.0, 32767.0)

# Vertex formats currently observed in stock stream-0 declarations. They cover
# POSITION, BLENDWEIGHT, and BLENDINDICES for the declaration-driven reader and
# writer; unrecognised formats fail before any row bytes are changed.
_VERTEX_FORMAT_LAYOUTS = {
    5: ('f', 4, 16), 6: ('f', 3, 12), 8: ('f', 2, 8),
    11: ('h', 4, 8), 14: ('B', 4, 4), 15: ('B', 4, 4),
    18: ('H', 2, 4), 19: ('h', 2, 4), 60: ('H', 4, 8),
    61: ('b', 4, 4),
}


def _parse_vertex_declaration(raw_values):
    """Decode declaration fields and track byte offsets within each stream."""
    offsets = {0: 0, 1: 0}
    offsets_known = {0: True, 1: True}
    elements = []
    for value in raw_values:
        stream = (value >> 31) & 1
        vertex_format = value & 0xFFFF
        layout = _VERTEX_FORMAT_LAYOUTS.get(vertex_format)
        offset = offsets[stream] if offsets_known[stream] else None
        size = layout[2] if layout else None
        elements.append({
            'stream': stream,
            'set': (value >> 24) & 0x7F,
            'semantic': (value >> 16) & 0xFF,
            'format': vertex_format,
            'offset': offset,
            'size': size,
        })
        if layout is None:
            offsets_known[stream] = False
        elif offsets_known[stream]:
            offsets[stream] += size
    return elements


def _read_vertex_format(buffer, offset, vertex_format):
    """Read raw components for one declared vertex element."""
    layout = _VERTEX_FORMAT_LAYOUTS.get(vertex_format)
    if layout is None:
        raise ValueError(f"unsupported vertex format {vertex_format}")
    code, count, size = layout
    if offset is None or offset < 0 or offset + size > len(buffer):
        raise ValueError(f"vertex format {vertex_format} is outside its stream row")
    return list(unpack(f'<{count}{code}', buffer[offset:offset + size]))


def _pack_vertex_format(vertex_format, values):
    """Pack raw components for one declared vertex element."""
    layout = _VERTEX_FORMAT_LAYOUTS.get(vertex_format)
    if layout is None:
        raise ValueError(f"unsupported vertex format {vertex_format}")
    code, count, _size = layout
    values = list(values[:count]) + [0] * max(0, count - len(values))
    if code != 'f':
        limits = {
            'b': (-128, 127), 'B': (0, 255),
            'h': (-32768, 32767), 'H': (0, 65535),
        }
        low, high = limits[code]
        values = [max(low, min(high, int(round(value)))) for value in values]
    return pack(f'<{count}{code}', *values)


def _weight_format_scale(vertex_format):
    """Return the normalized full-scale value for a declared weight format."""
    if vertex_format == 14:  # uint8x4
        return 255
    if vertex_format in (11, 19):  # signed int16x4/x2
        return 32767
    raise ValueError(f"unsupported BLENDWEIGHT format {vertex_format}")

def _uv_divisor_candidates(raw4_list):
    """Decode a list of 4-byte chunks to float32 divisors, or return None if any
    value is not a recognised divisor (i.e. the block is not a divisor table)."""
    out = []
    for _b in raw4_list:
        if len(_b) != 4:
            return None
        _v = unpack('<f', _b)[0]
        if _v not in _UV_KNOWN_DIVISORS:
            return None
        out.append(_v)
    return out

def _encoding_from_divisor(div):
    """Map a divisor float to an encoding name, or None if not a usable divisor."""
    if div == 0.0:
        return 'float32'
    if div == 4096.0:
        return 'wide'
    if div == 32767.0:
        return 'int16_norm'
    if div == 4095.0:
        return 'compact'
    return None

def _resolve_uv_encoding(divisor, probe_plausible_f32, compact_ok):
    """Returns the encoding for one UV set: 'float32', 'wide', 'compact', or 'int16_norm'.
    Divisor table is authoritative; float32 is double-checked via byte probe to guard
    against a stray entry producing NaN geometry. Falls back to probe + magnitude
    heuristic when no table is present."""
    if divisor is not None:
        _enc = _encoding_from_divisor(divisor)
        if _enc == 'float32':
            return 'float32' if probe_plausible_f32 else 'int16_norm'
        if _enc is not None:
            return _enc
    if probe_plausible_f32:
        return 'float32'
    return 'compact' if compact_ok else 'int16_norm'

class Asset:
    def __init__(self):
        self.magic = ""
        self.version = 0
        self.size = 0
    def parse(self,f):
        self.magic = br.string(f,3)
        self.version = br.uint8(f)
        self.size = br.uint32(f)
        if self.version >= 15:
            f.seek(4, 1)  # 4-byte header skip for v15+

class SkeletalMeshAsset(Asset):
    class Mesh:
        class LOD:
            def __init__(self, parent_mesh,index):
                self.start_offset = 0
                self.parent_mesh:SkeletalMeshAsset.Mesh = parent_mesh
                self.index = index
                self.vertex_count = 0
                self.index_count = 0
                self.size_a = 0
                self.vertex_data_offset_a = 0
                self.vertex_data_offset_b = 0
                self.face_block_offset = 0
                self.data_offset = 0
                self.data_size = 0
                self.data_offset_file_pos = 0   # file position of the data_offset uint32 field
                self.blender_obj_name = ""  # set at import time; used by exporter to find the object
                self.is_header_lod = False
                self.vertex_end_bytes = None
                self.normals_end_bytes = None
                self.faces_end_bytes = None
                self.lod_unk = 0  # v11 only: unknown uint32
                # Primary face-index width encoded by size_a:
                #   size_a * 2 == face_block_offset -> uint16
                #   size_a * 4 == face_block_offset -> uint32
                # None means the LOD is empty or its header is inconsistent.
                self.index_width_bytes = None
                # exported vc when _write_mod_file used the cloth slot-preserving layout
                self.exported_slot_identity = 0
                # appended slot -> mmb_vertex_order source, from the export-time
                # vertex set (includes seam-split duplicates the Blender mesh lacks)
                self.exported_append_sources = {}
                # sim budget reuse: orphaned slots rewritten with new geometry
                self.exported_sim_reused = set()
                # sim slots whose position changed vs source (a sim MOVE)
                self.exported_sim_moved = set()
                # sim tri slots present in the CURRENT mesh (excl. phantoms)
                self.exported_sim_valid_tris = None
                # true when this export appended beyond the source SIM budget
                self.exported_sim_grown = False
                # first appended slot index (== original vc) of the last slot export
                self.exported_append_base = 0

            @property
            def lod_field_offset(self):
                """Extra byte offset applied to every LOD header field after vc. v11 has an extra uint32"""
                return 4 if self.parent_mesh.parent_sk_mesh.version == 11 else 0

            def parse(self, f):
                self.start_offset = f.tell()
                self.vertex_count = br.uint32(f)
                if self.parent_mesh.parent_sk_mesh.version == 11:
                    self.lod_unk = br.uint32(f)
                self.index_count = br.uint32(f)
                self.size_a = br.uint32(f)  # face_block_offset divided by index width
                self.vertex_data_offset_a = br.uint32(f)
                self.vertex_data_offset_b = br.uint32(f)
                self.face_block_offset = br.uint32(f)
                if (self.index_count > 0 and self.face_block_offset > 0
                        and self.size_a * 4 == self.face_block_offset
                        and self.size_a * 2 != self.face_block_offset):
                    self.index_width_bytes = 4
                elif (self.index_count > 0 and self.face_block_offset > 0
                        and self.size_a * 2 == self.face_block_offset):
                    self.index_width_bytes = 2
                else:
                    self.index_width_bytes = None
                self.data_offset_file_pos = f.tell()
                self.data_offset = br.uint32(f)
                self.data_size = br.uint32(f)
                lod_screen_size = br.float(f)
                if self.data_offset < self.parent_mesh.parent_sk_mesh.size:
                    self.is_header_lod = True
                    logger.debug("LOD %d is stored in the header", self.index)

            def write(self, f):
                f.seek(self.start_offset)
                f.write(bp.uint32(self.vertex_count))
                if self.parent_mesh.parent_sk_mesh.version == 11:
                    f.write(bp.uint32(self.lod_unk))
                f.write(bp.uint32(self.index_count))
                width = self.index_width_bytes
                if width not in (2, 4):
                    width = 4 if self.parent_mesh.index_width_flag == 1 else 2
                self.size_a = self.face_block_offset // width
                f.write(bp.uint32(self.size_a))
                f.write(bp.uint32(self.vertex_data_offset_a))
                f.write(bp.uint32(self.vertex_data_offset_b))
                f.write(bp.uint32(self.face_block_offset))
                f.write(bp.uint32(self.data_offset))
                f.write(bp.uint32(self.data_size))

            def get_vertex_positions(self, raw_mesh_file):
                mesh = self.parent_mesh
                stride = mesh.vertex_stride
                elements = mesh.elements(semantic=0, stream=0)
                if len(elements) != 1:
                    raise ValueError(
                        f"'{mesh.name}' must declare exactly one stream-0 POSITION element")
                element = elements[0]
                vertex_format = element['format']
                if vertex_format not in (5, 6, 11) or element['offset'] is None:
                    raise ValueError(
                        f"'{mesh.name}' uses unsupported POSITION format {vertex_format}")
                raw_mesh_file.seek(self.vertex_data_offset_a)
                block = raw_mesh_file.read(self.vertex_count * stride)
                if len(block) != self.vertex_count * stride:
                    raise ValueError(
                        f"'{mesh.name}' LOD{self.index} vertex block is truncated")
                vertices = []
                for vertex_index in range(self.vertex_count):
                    row_start = vertex_index * stride
                    values = _read_vertex_format(
                        block, row_start + element['offset'], vertex_format)
                    if vertex_format == 11:
                        scale = values[3]
                        vertices.append(tuple(
                            values[axis] / 32767.0 * scale for axis in range(3)))
                    else:  # float3/float4 POSITION
                        vertices.append(tuple(values[:3]))
                return vertices

            def get_declared_vertex_positions(self, raw_mesh_file):
                """Compatibility name for the now declaration-driven position reader."""
                return self.get_vertex_positions(raw_mesh_file)

            def get_bone_weights(self, raw_mesh_file):
                mesh = self.parent_mesh
                stride = mesh.vertex_stride
                weight_elements = sorted(
                    mesh.elements(semantic=2, stream=0),
                    key=lambda element: (element['set'], element['offset']))
                index_elements = sorted(
                    mesh.elements(semantic=3, stream=0),
                    key=lambda element: (element['set'], element['offset']))
                for element in weight_elements:
                    _weight_format_scale(element['format'])
                for element in index_elements:
                    if element['format'] not in (15, 18, 60):
                        raise ValueError(
                            f"'{mesh.name}' uses unsupported BLENDINDICES format "
                            f"{element['format']}")
                raw_mesh_file.seek(self.vertex_data_offset_a)
                block = raw_mesh_file.read(self.vertex_count * stride)
                if len(block) != self.vertex_count * stride:
                    raise ValueError(
                        f"'{mesh.name}' LOD{self.index} vertex block is truncated")

                bone_weights = []
                for vertex_index in range(self.vertex_count):
                    row_start = vertex_index * stride
                    weights = []
                    indices = []
                    for element in weight_elements:
                        values = _read_vertex_format(
                            block, row_start + element['offset'], element['format'])
                        scale = _weight_format_scale(element['format'])
                        weights.extend(value / scale for value in values)
                    for element in index_elements:
                        indices.extend(int(value) for value in _read_vertex_format(
                            block, row_start + element['offset'], element['format']))
                    combined = {}
                    if not weight_elements:
                        if indices:
                            combined[indices[0]] = 1.0
                    else:
                        for weight, index in zip(weights, indices):
                            if weight > 0.0:
                                combined[index] = combined.get(index, 0.0) + weight
                    bone_weights.append(combined)
                return bone_weights

            def get_triangles(self,raw_mesh_file):
                """
                Seeks to Lod.face_block_offset and reads all triangle indices.
                :param raw_mesh_file: file that is exported by SkeletalMeshAsset.Mesh.extract_mesh_file()
                :return: a List of Tuples containing 3 vertex indices to form a triangle.
                """
                tris = []
                f = raw_mesh_file
                f.seek(self.face_block_offset)
                use_uint32 = self.index_width_bytes == 4
                if self.index_width_bytes is None and self.index_count > 0:
                    # Malformed/ambiguous legacy fallback. Valid stock files encode
                    # the width deterministically through size_a.
                    peek_bytes = f.read(16)
                    f.seek(self.face_block_offset)
                    if len(peek_bytes) >= 16:
                        import struct as _struct
                        hi_words = [_struct.unpack('<H', peek_bytes[i:i + 2])[0]
                                    for i in range(2, 16, 4)]
                        use_uint32 = all(v == 0 for v in hi_words)
                    logger.warning(
                        "%s LOD%d has an ambiguous index-width header; using the "
                        "legacy byte probe (%s)",
                        self.parent_mesh.name, self.index,
                        "uint32" if use_uint32 else "uint16")
                for i in range(int(self.index_count/3)):
                    if use_uint32:
                        f1 = br.uint32(f)
                        f2 = br.uint32(f)
                        f3 = br.uint32(f)
                    else:
                        f1 = br.uint16(f)
                        f2 = br.uint16(f)
                        f3 = br.uint16(f)
                    tris.append((f1,f2,f3))
                return tris

            def get_normals(self,raw_mesh_file):
                normals = []
                stride = self.parent_mesh.normals_stride
                color_in_normals = getattr(self.parent_mesh, 'color_in_normals', True)
                color_count = self.parent_mesh.color_count if color_in_normals else 0
                f = raw_mesh_file
                f.seek(self.vertex_data_offset_b)
                v = Vector((0.0,0.0,1.0))
                self._all_w_zero = True # If all w values are zero, the normals store non-surface data (e.g. VAT).
                self._tangent_space = False # Normals stored in tangent space: x axis non-negative across all verts.
                _raw_x_min = 1.0
                for i in range(self.vertex_count):
                    stride_start = f.tell()
                    if self.parent_mesh.normal_type == 0:
                        # Layout: color(4*cc) | tangent(4) | normal(4) | UV(4*uv_count)
                        f.seek(4 * color_count, 1)  # skip color (0 bytes if no color)
                        raw_x = br.int8_norm(f)
                        if raw_x < _raw_x_min:
                            _raw_x_min = raw_x
                        x = raw_x * -1
                        y = br.int8_norm(f)
                        z = br.int8_norm(f)
                        w = br.int8(f)
                        if w != 0:
                            self._all_w_zero = False
                        v = Vector((x*w, y*w, z*w)).normalized()
                        v.negate()  # TODO not sure about this
                    elif self.parent_mesh.normal_type == 1:
                        # Layout: color(4*cc) | normal(12f) | tangent(12f) | sign(4f) | UV
                        f.seek(4 * color_count, 1)  # skip colors before float normal (0 bytes if no color)
                        self._all_w_zero = False
                        raw_x = br.float(f)
                        if raw_x < _raw_x_min:
                            _raw_x_min = raw_x
                        x = raw_x * -1
                        y = br.float(f)
                        z = br.float(f)
                        v = Vector((x,y,z)).normalized()
                    f.seek(stride_start + stride)
                    normals.append(v)
                # If raw x never goes below -0.05, the x axis is non-negative i.e. tangent-space encoding.
                # Cannot decode to object-space normals without the tangent basis; fall back to computed.
                if _raw_x_min > -0.05:
                    self._tangent_space = True
                return normals

            def get_uvs(self,raw_mesh_file, index=0):
                """
                Seeks to Lod.vertex_data_offset_b and reads all UV data.
                :param raw_mesh_file: file that is exported by SkeletalMeshAsset.Mesh.extract_mesh_file()
                :return: a List of Tuples containing 2 floats as UV coordinates.
                """
                uvs = []
                stride = self.parent_mesh.normals_stride
                color_in_normals = getattr(self.parent_mesh, 'color_in_normals', True)
                color_count = self.parent_mesh.color_count if color_in_normals else 0
                normal_type = self.parent_mesh.normal_type
                f = raw_mesh_file
                f.seek(self.vertex_data_offset_b)

                # --- UV encoding detection ---
                # Per-set encoding comes from the binary divisor table (uv_divisors).
                # Falls back to float32 probe then compact/int16_norm when mesh has no table.
                _divs = getattr(self.parent_mesh, 'uv_divisors', None)

                # Normal block before the UV region
                #  normal_type 0 :  8 bytes (tangent(4)+normal(4))
                #  normal_type 1 : 28 bytes (float normal(12)+tangent(12)+sign(4))
                #  Float32 UV sets are 8 bytes wide, so field offsets accumulate variable widths.
                if normal_type == 0:
                    _normal_block_size = 8
                else:
                    _normal_block_size = 12 + 12 + 4
                _color_prefix = 4 * color_count

                # UV sets are packed at the END of the normals stride. When the
                # divisor table is present, anchor the start offset from the stride
                # end (stride - total_uv_bytes) so meshes with non-standard normal
                # block sizes land correctly. Without a table, use the forward guess.
                _cur_off = _color_prefix + _normal_block_size
                if _divs:
                    _nuv = max(1, self.parent_mesh.uv_count)
                    _total_uv_bytes = sum(8 if d == 0.0 else 4 for d in _divs[:_nuv])
                    _cur_off = max(_cur_off, stride - _total_uv_bytes)
                _uv_field_off = _cur_off
                _target_enc = 'int16_norm'
                start_pos = f.tell()
                for _ui in range(index + 1):
                    _div = _divs[_ui] if (_divs is not None and _ui < len(_divs)) else None
                    _plausible_f32 = False
                    _compact_ok = True
                    if self.vertex_count > 0:
                        _pl = 0
                        for _vi in range(self.vertex_count):
                            f.seek(start_pos + _vi * stride + _cur_off)
                            _fv = unpack('<f', f.read(4))[0]
                            if _fv == _fv and (_fv == 0.0 or 1e-4 < abs(_fv) < 500):
                                _pl += 1
                            if _div is None:
                                # Both axes must be small for compact; large values
                                # on either axis rule it out (% 4096 would shred them).
                                f.seek(start_pos + _vi * stride + _cur_off)
                                _ru = unpack('<h', f.read(2))[0]
                                _rv = unpack('<h', f.read(2))[0]
                                if abs(_ru) > 8191 or abs(_rv) > 8191:
                                    _compact_ok = False
                        _plausible_f32 = (_pl / self.vertex_count) > 0.90
                    _enc = _resolve_uv_encoding(_div, _plausible_f32, _compact_ok)
                    _uv_field_off = _cur_off
                    _target_enc = _enc
                    _cur_off += 8 if _enc == 'float32' else 4
                f.seek(start_pos)

                _is_float32 = (_target_enc == 'float32')
                use_wide = (_target_enc == 'wide')
                use_compact = (_target_enc == 'compact')

                for i in range(self.vertex_count):
                    f.seek(start_pos + i * stride + _uv_field_off)
                    if _is_float32:
                        u = br.float(f)
                        v = br.float(f)
                    elif use_wide:
                        _ru = unpack('<H', f.read(2))[0]
                        _rv = unpack('<H', f.read(2))[0]
                        u = ((_ru ^ 32768) - 32768) / 4096.0
                        v = ((_rv ^ 32768) - 32768) / 4096.0
                    elif use_compact:
                        u = br.uv_unorm_u(f)
                        v = br.uv_unorm_v(f)
                    else:
                        u = br.int16_norm(f)
                        v = br.int16_norm(f)
                    uvs.append((u,v))

                # Record which encoding was used for this UV set so callers can report it without re-running detection.
                if _is_float32:
                    _encoding_name = 'Float32'
                elif use_wide:
                    _encoding_name = 'Wide'
                elif use_compact:
                    _encoding_name = 'Compact'
                else:
                    _encoding_name = 'Int16'
                if not hasattr(self, 'last_uv_encodings'):
                    self.last_uv_encodings = {}
                self.last_uv_encodings[index] = _encoding_name

                return uvs

            def get_color(self,raw_mesh_file, index=0):
                """
                Seeks to Lod.vertex_data_offset_b and reads all Color data.
                :param raw_mesh_file: file that is exported by SkeletalMeshAsset.Mesh.extract_mesh_file()
                :return: a List of Tuples containing 4 floats as RGBA coordinates.
                """

                # If color is not present in the normals stride, return white.
                if not getattr(self.parent_mesh, 'color_in_normals', True):
                    return [(1.0, 1.0, 1.0, 1.0)] * self.vertex_count

                colors = []
                stride = self.parent_mesh.normals_stride
                normal_type = self.parent_mesh.normal_type
                f = raw_mesh_file
                f.seek(self.vertex_data_offset_b)
                for i in range(self.vertex_count):
                    stride_start = f.tell()
                    if normal_type == 0:
                        # color(4*cc) | normal | tangent | UV
                        f.seek(index * 4, 1)
                    elif normal_type == 1:
                        # color(4*cc) | normal(12) | tangent(12) | sign(4) | UV
                        f.seek(index * 4, 1)
                    r = br.uint8_norm(f)
                    g = br.uint8_norm(f)
                    b = br.uint8_norm(f)
                    a = br.uint8_norm(f)
                    f.seek(stride_start + stride)
                    colors.append((r,g,b,a))
                return colors

            def write_vertex_position(self,file,pos=(0.0,0.0,0.0),scale=None):
                """
                Writes a single vertex position into file at the current position and skips to the end of stride.
                :param file: file to write on
                :param pos: (x,y,z) world-space position in the same units get_vertex_positions returns.
                :param scale: the per-vertex w value read from the original file.
                              For format 11: world = int16_norm(raw) * w,
                              so the inverse is raw = round(world / w * 32767), packed as int16.
                              If None or 0, the vertex is skipped (zero-displacement override).
                :return:
                """
                f = file
                stride = self.parent_mesh.vertex_stride
                stride_start = f.tell()
                elements = self.parent_mesh.elements(semantic=0, stream=0)
                if len(elements) != 1 or elements[0]['offset'] is None:
                    raise ValueError(
                        f"'{self.parent_mesh.name}' has no writable POSITION declaration")
                element = elements[0]
                vertex_format = element['format']
                f.seek(stride_start + element['offset'])
                if vertex_format == 11:
                    max_abs = max(abs(value) for value in pos)
                    if scale is None or scale == 0 or max_abs >= abs(scale):
                        scale = max(1, int(max_abs) + 1)
                    if abs(scale) > 32767:
                        raise ValueError(
                            f"'{self.parent_mesh.name}' position scale exceeds int16 range")
                    values = [value / scale * 32767.0 for value in pos]
                    f.write(_pack_vertex_format(11, [*values, scale]))
                elif vertex_format == 6:
                    f.write(_pack_vertex_format(6, pos))
                elif vertex_format == 5:
                    f.write(_pack_vertex_format(5, [*pos, 1.0]))
                else:
                    raise ValueError(
                        f"'{self.parent_mesh.name}' uses unsupported POSITION format "
                        f"{vertex_format}")
                f.seek(stride_start + stride)

        def __init__(self,parent_sk_mesh, index=0):
            self.parent_sk_mesh:SkeletalMeshAsset = parent_sk_mesh
            self.index = index
            self.name = ""
            self.name_offset = 0        # byte offset of the uint16 length prefix in the source file
            self.name_length = 0        # original byte count of the name field (from the uint16 prefix)
            self.mesh_bounds_offset = 0 # 48-byte AABB/radius block after the mesh name
            self.pending_rename_new = ""  # staged mesh rename — applied to _MOD copy on next export
            self.zeroed_out_in_session = False # True if mesh was zeroed out this session
            self.zeroed_out_in_mmb = False     # True if mesh was zeroed out in the mmb
            self.lod_count = 0
            self.lod_info_type = 0
            self.lods = []
            self.vertex_stride = 0
            self.normals_stride = 0
            self.mesh_bones = {}
            self.mesh_bone_file_offsets = []  # file offset of each slot's 2-byte skeleton index field
            self.pending_bone_remaps = {}     # Applied on export
            self.pending_bone_additions = []  # Appended to bone table on export
            self.u_count_offset = 0           # file offset of the 2-byte u_count field
            self.bone_table_end_offset = 0    # file offset immediately after the last bone slot
            self.color_count = 0
            self.uv_count = 0
            self.normal_type = 0 # 0:int8_norm 1:floats
            self.color_in_normals = True # False when color_count not in normals stride
            self.position_type = 0 # compatibility: 0=int16 POSITION, 1=float POSITION
            self.vertex_elements = [] # decoded declaration metadata
            # Per-mesh tail. The first uint32 is the engine's primary index-width
            # flag (0=u16, 1=u32); the next three are aggregate mesh counts.
            self.index_width_flag_offset = 0
            self.index_width_flag = 0
            self.total_vertex_count_offset = 0
            self.total_vertex_count = 0
            self.total_index_count_offset = 0
            self.total_index_count = 0
            self.total_data_size_offset = 0
            self.total_data_size = 0
            self.tail_extra_u32 = None
        def elements(self, semantic, stream=None, semantic_set=None):
            """Return declaration elements matching a semantic and optional stream/set."""
            return [element for element in self.vertex_elements
                    if element['semantic'] == semantic
                    and (stream is None or element['stream'] == stream)
                    and (semantic_set is None or element['set'] == semantic_set)]
        def influence_capacity(self):
            """Usable stream-0 skinning influences declared for one vertex."""
            weights = self.elements(semantic=2, stream=0)
            indices = self.elements(semantic=3, stream=0)
            index_capacity = sum(
                _VERTEX_FORMAT_LAYOUTS[element['format']][1]
                for element in indices)
            if not weights:
                return 1 if index_capacity else 0
            weight_capacity = sum(
                _VERTEX_FORMAT_LAYOUTS[element['format']][1]
                for element in weights)
            return min(weight_capacity, index_capacity)
        def parse(self, f):
            version = self.parent_sk_mesh.version

            self.name_offset = f.tell()
            _nlen = unpack('<H', f.read(2))[0]
            self.name_length = _nlen
            self.name = br.string(f, _nlen).rstrip('\x00')
            self.mesh_bounds_offset = f.tell()
            f.seek(48, 1)  # AABB, center and culling radii
            f.seek(1, 1)
            if version == 11:
                f.seek(1, 1) # declaration flag
                x_count = br.uint16(f)
            else:
                x_count = br.uint8(f)
                f.seek(1, 1) # declaration flag
            declaration_raw = [br.uint32(f) for _ in range(x_count)]
            self.vertex_elements = _parse_vertex_declaration(declaration_raw)
            self.u_count_offset = f.tell()
            u_count = br.uint16(f)
            for b in range(u_count):
                matrix = br.matrix_4x4(f)
                offset = f.tell()
                bone_index = br.uint16(f)
                self.mesh_bone_file_offsets.append(offset)
                self.mesh_bones[bone_index] = matrix
            self.bone_table_end_offset = f.tell()

            # --- Pre-LOD section ---
            # root_bone_index: present only when u_count > 0, and absent for v12 entirely.
            # lod_info_type: present for v14,v15/v16/v17 always; absent for v11/v12/v13.
            #   v11/v12:        no root_bone_index, no lod_info_type
            #   v13, u>0:       root_bone_index[1]  lod_info_type[1]
            #   v13, u==0:      (no root_bone_index, no lod_info_type)
            #   v14, u>0:       root_bone_index[2]  lod_info_type[1]
            #   v14, u==0:      lod_info_type[1]
            #   v15/16/17, u>0: root_bone_index[2]  lod_info_type[1]
            #   v15/16/17, u==0: lod_info_type[1]
            if u_count > 0 and version not in (11, 12):
                if version == 13:
                    f.seek(1, 1)  # v13: 1-byte root_bone_index
                else:             # v14, v15, v16, v17
                    f.seek(2, 1)  # v14+: 2-byte root_bone_index
                lod_info_type = br.uint8(f)
            else:
                if version in (11, 12, 13):
                    lod_info_type = 0  # v11/v12/v13 have no lod_info_type byte
                else:                  # v14, v15, v16, v17 with u_count == 0
                    lod_info_type = br.uint8(f)

            self.lod_info_type = lod_info_type
            self.lod_count = br.uint8(f)
            f.seek(4, 1)  # unknown 4 bytes before LOD list

            # --- LOD list ---
            # v11: Each LOD is 40 bytes.
            # v12-v17: Each LOD is 36 bytes.
            # lod_info_type == 2: 28 extra bytes per LOD (v15/v16/v17).
            #   These hold a second per-LOD data block table (meshlet-style GPU data
            #   stored after all primary LOD blocks) as 7 uint32s:
            #     [0] cumulative offset of this LOD's block within the mesh's
            #         reverse-ordered second-section data (like voa's base)
            #     [1] meshlet descriptor count (32 bytes each)
            #     [2..4] cumulative end offsets of descriptor/vertex/normal sub-blocks
            #     [5] ABSOLUTE file offset of this LOD's second block
            #     [6] size of this LOD's second block
            #   Field [5] is the only absolute offset and must be shifted whenever
            #   bytes are inserted/removed anywhere before it in the file.
            for l in range(self.lod_count):
                lod = self.LOD(self, l)
                lod.parse(f)
                if lod_info_type == 2:
                    f.seek(28, 1)
                self.lods.append(lod)

            # --- Tail section: UV hashes, color hashes, strides ---
            # UV hashes are 4 bytes each on all versions.
            #
            # v11 layout: unk(4)  unk(4)  uv_count  uv_hashes  vs  ns (no color_count)
            # v12/v13/v15 layout: uv_count  uv_hashes  unk[4]  color_count  color_hashes
            # v16/v17 layout: uv_count  uv_hashes  color_count  color_hashes  unk[4]  count_c  c_data
            if version == 11:
                f.seek(8,1) # skip unk

            self.uv_count = br.uint8(f)

            # uv_divisors: divisor table read from file (see _UV_KNOWN_DIVISORS).
            # count_c/color_count is the authoritative UV set count when all entries
            # are valid divisors -- the header uv_count can undercount (e.g. some meshes have uv_count=1 in
            # the header but count_c=2 with valid divisors for both UV sets).
            self.uv_divisors = None
            if version == 11:
                # In v11 this block stores one float32 divisor per UV set, not
                # hashes. Retain recognised values so each set is decoded with
                # its actual scale; unknown values keep the existing fallback.
                _div_raw = [f.read(4) for _ in range(self.uv_count)]
                _candidates = _uv_divisor_candidates(_div_raw)
                if _candidates is not None:
                    self.uv_divisors = _candidates
                self.color_count = 0 # v11 does not store color_count; always 0
            elif version in (16, 17):
                f.seek(4 * self.uv_count, 1)
                self.color_count = br.uint8(f)
                f.seek(4 * self.color_count, 1)
                f.seek(4, 1) # unk after color (v16/v17)
                count_c = br.uint8(f)
                _div_raw = [f.read(4) for _ in range(count_c)]
                _candidates = _uv_divisor_candidates(_div_raw)
                if _candidates is not None:
                    self.uv_divisors = _candidates
                    self.uv_count = count_c
            else:
                f.seek(4 * self.uv_count, 1)
                f.seek(4, 1) # unk before color (v12/v13/v15)
                self.color_count = br.uint8(f)
                _div_raw = [f.read(4) for _ in range(self.color_count)]
                _candidates = _uv_divisor_candidates(_div_raw)
                if _candidates is not None:
                    self.uv_divisors = _candidates
                    self.uv_count = self.color_count

            self.vertex_stride = br.uint16(f)
            self.normals_stride = br.uint16(f)

            # --- Color-in-normals detection ---
            # color_count is declared in the header but may not be present in the
            # normals stride (e.g. meshes store color_count=1 in the header
            # but allocate no color bytes in the normals block).
            # Detection: if the normals stride is large enough to hold the normal
            # block, all UV sets and color, then color IS in the normals stride.
            _nb_with_color = self.normals_stride - 4 * self.color_count - 4 * self.uv_count
            _nb_without_color = self.normals_stride - 4 * self.uv_count
            self.color_in_normals = (_nb_with_color >= 8)
            # --- Normal type detection ---
            # normals_base = normals_stride - 4*uv_count - 4*color_count
            # normals_base > 8 -> normal_type 1 or 2 (float normals)
            # normals_base <= 8 -> normal_type 0 (int8_norm, 4-byte block)
            # normal_type 1: float normal(12) + tangent(12) + sign(4) = 28 bytes
            _normals_base = self.normals_stride - 4 * self.uv_count - 4 * self.color_count
            if _normals_base >= 28:
                self.normal_type = 1  # float normal(12) + tangent(12) + sign(4) = 28 bytes fixed
            else:
                self.normal_type = 0  # int8_norm normal(4) + tangent(4) = 8 bytes fixed
            # Layout for both types: color(4*cc) first, then normal block, then UV(4*uv)

            # Retained as a compatibility attribute for cloth/export code, but
            # now derived from the authoritative stream-0 declaration.
            _position_elements = self.elements(semantic=0, stream=0)
            _position_format = (_position_elements[0]['format']
                                if len(_position_elements) == 1 else None)
            if _position_format == 11:
                self.position_type = 0
            elif _position_format in (5, 6):
                self.position_type = 1
            else:
                raise ValueError(
                    f"'{self.name}' uses unsupported POSITION declaration "
                    f"{_position_format}")

            logger.debug(
                "%s: vertex stride=%d, normals stride=%d, UVs=%d, colors=%d, "
                "normal type=%d",
                self.name, self.vertex_stride, self.normals_stride,
                self.uv_count, self.color_count, self.normal_type)

            # --- Per-mesh tail ---
            # Corpus-confirmed across v11-v17:
            #   [index_width_flag, total_vc, total_ic, total_data]
            # v17 carries one additional unknown uint32 which is preserved.
            self.index_width_flag_offset = f.tell()
            self.index_width_flag = br.uint32(f)
            self.total_vertex_count_offset = f.tell()
            self.total_vertex_count = br.uint32(f)
            self.total_index_count_offset = f.tell()
            self.total_index_count = br.uint32(f)
            self.total_data_size_offset = f.tell()
            self.total_data_size = br.uint32(f)
            if version == 17:
                self.tail_extra_u32 = br.uint32(f)

            widths = {lod.index_width_bytes for lod in self.lods
                      if lod.index_width_bytes in (2, 4)}
            expected_flag = 1 if widths == {4} else 0 if widths == {2} else None
            if len(widths) > 1:
                logger.debug(
                    "%s contains mixed index widths; remaining uint16 LOD "
                    "index buffers will be widened to uint32",
                    self.name)
            elif expected_flag is not None and self.index_width_flag != expected_flag:
                logger.warning(
                    "%s index-width flag is %d but its LOD headers encode uint%d",
                    self.name, self.index_width_flag, 32 if expected_flag else 16)
        def extract_mesh_file(self,f):
            """
            Creates a file gathering the raw data of all Lods the Mesh.
            :param f: combined mmb file that contains the header and data.
            :return: Path to the extracted raw_mesh file.
            """
            extract_file = io.BytesIO()
            for lod in reversed(self.lods):
                CopyFile(f, extract_file, lod.data_offset, lod.data_size)
            return extract_file

    class Bone:
        def __init__(self,f):
            self.name = br.name(f)
            self.matrix = br.matrix_4x4(f)
            self.parent_index = br.uint16(f)

    def __init__(self):
        super().__init__()
        self.name = ""
        self.bone_count = 0
        self.bones = []
        self.mesh_count = 0
        self.meshes = []
        self.pending_file_rename_old = ""  # staged file rename — applied on next export
        self.pending_file_rename_new = ""
    def parse(self,f):
        super().parse(f)
        if self.version not in (11, 12, 13, 14, 15, 16, 17):
            raise Exception(f'Unsupported .mmb version: {self.version}. Supported versions: 11, 12, 13, 14, 15, 16, 17.')
        self.bone_count = br.uint32(f)
        for b in range(self.bone_count):
            self.bones.append(self.Bone(f))
        self.mesh_count = br.uint32(f)
        for m in range(self.mesh_count):
            mesh = self.Mesh(self, index=m)
            mesh.parse(f)
            self.meshes.append(mesh)

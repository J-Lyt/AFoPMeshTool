"""Snowdrop .mmb format model and parser."""

import io
from struct import unpack

from mathutils import Vector

from .binary_io import bp, br
from .file_utils import CopyFile

# Known values:
#   0.0     -> float32   : (8 bytes/vert, unquantized)
#   4095.0  -> compact   : (uint16 % 4096) / 4095
#   4096.0  -> wide      : signed int16 / 4096
#   32767.0 -> int16_norm: signed int16 / 32767
#
# A block is only accepted as a divisor table when every entry matches one of
# these values, so genuine hashes are never mistaken for divisors.
_UV_KNOWN_DIVISORS = (0.0, 4095.0, 4096.0, 32767.0)

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
                self.data_start = 0  # offset within mesh_file BytesIO where this LOD's block begins
                self.lod_unk = 0  # v11 only: unknown uint32
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
                self.size_a = br.uint32(f)  # seems to be face_block_offset divided by 2
                self.vertex_data_offset_a = br.uint32(f)
                self.vertex_data_offset_b = br.uint32(f)
                self.face_block_offset = br.uint32(f)
                self.data_offset_file_pos = f.tell()
                self.data_offset = br.uint32(f)
                self.data_size = br.uint32(f)
                lod_screen_size = br.float(f)
                if self.data_offset < self.parent_mesh.parent_sk_mesh.size:
                    self.is_header_lod = True
                    print("Lod ", self.index, "is in header.")

            def write(self, f):
                f.seek(self.start_offset)
                f.write(bp.uint32(self.vertex_count))
                if self.parent_mesh.parent_sk_mesh.version == 11:
                    f.write(bp.uint32(self.lod_unk))
                f.write(bp.uint32(self.index_count))
                f.write(bp.uint32(int(self.face_block_offset / 2)))
                f.write(bp.uint32(self.vertex_data_offset_a))
                f.write(bp.uint32(self.vertex_data_offset_b))
                f.write(bp.uint32(self.face_block_offset))
                f.write(bp.uint32(self.data_offset))
                f.write(bp.uint32(self.data_size))

            def gather_extra_bytes(self, f):
                offset = f.tell()
                f.seek(self.data_offset)
                real_vertex_size = self.vertex_count * self.parent_mesh.vertex_stride
                extra_bytes_size = self.vertex_data_offset_b - self.vertex_data_offset_a - real_vertex_size
                f.seek(real_vertex_size, 1)
                print(f.tell())
                self.vertex_end_bytes = f.read(extra_bytes_size)
                print("Vertex Extra Bytes:", self.vertex_end_bytes)

                real_normals_size = self.vertex_count * self.parent_mesh.normals_stride
                extra_bytes_size = self.face_block_offset - self.vertex_data_offset_b - real_normals_size
                f.seek(real_normals_size, 1)
                print(f.tell())
                self.normals_end_bytes = f.read(extra_bytes_size)
                print("Normal Extra Bytes:", self.normals_end_bytes)

                real_face_size = self.index_count * 2
                size_without_face = self.face_block_offset - self.vertex_data_offset_a
                extra_bytes_size = self.data_size - size_without_face - real_face_size
                f.seek(real_face_size, 1)
                print(f.tell())
                self.faces_end_bytes = f.read(extra_bytes_size)
                print("Face Extra Bytes:", self.faces_end_bytes)

                f.seek(offset)

            def get_vertex_positions(self, raw_mesh_file):
                vertices = []
                mesh = self.parent_mesh
                stride = mesh.vertex_stride
                f = raw_mesh_file
                f.seek(self.vertex_data_offset_a)
                pos = (0.0,0.0,0.0)
                for v in range(self.vertex_count):
                    stride_start = f.tell()
                    if mesh.position_type == 0:
                        x = br.int16_norm(f)
                        y = br.int16_norm(f)
                        z = br.int16_norm(f)
                        w = br.int16(f)
                        pos = (x*w,y*w,z*w)
                    elif mesh.position_type == 1:
                        x = br.float(f)
                        y = br.float(f)
                        z = br.float(f)
                        pos = (x,y,z)
                    f.seek(stride_start + stride)
                    vertices.append(pos)
                return vertices

            def get_bone_weights(self, raw_mesh_file):
                bone_weights = []
                stride = self.parent_mesh.vertex_stride
                f = raw_mesh_file
                # Detect stride-32 layout
                stride32_layout = None
                if stride == 32 and self.vertex_count > 0:
                    n_slots = len(self.parent_mesh.mesh_bones)
                    f.seek(self.vertex_data_offset_a + 8)
                    peek = f.read(24)
                    w8_u16 = [unpack('<H', peek[i * 2:i * 2 + 2])[0] for i in range(8)]
                    if sum(w8_u16) == 32767:
                        stride32_layout = 'A'
                    else:
                        c12_idx = list(peek[12:24])
                        if n_slots <= 256 and all(0 <= x < n_slots for x in c12_idx):
                            stride32_layout = 'C'
                        else:
                            stride32_layout = 'B'

                f.seek(self.vertex_data_offset_a)
                for v in range(self.vertex_count):
                    iw = {}
                    stride_start = f.tell()
                    if stride == 12:
                        # 4x uint8 bone slot indices, no weight bytes - Weight 1.0 on first index
                        f.seek(8, 1)
                        indices = [br.uint8(f) for _ in range(4)]
                        iw[indices[0]] = 1.0
                    elif stride == 16:
                        if self.parent_mesh.position_type == 1:
                            # Float XYZ position (12b) + 1x uint8 bone slot index + 3b padding.
                            # Same as stride=12 (weight 1.0 on first index) but with float positions.
                            f.seek(12, 1)
                            index = br.uint8(f)
                            iw[index] = 1.0
                        else:
                            # Int16 position (8b) + 4x uint8_norm weights + 4x uint8 indices
                            f.seek(8, 1)
                            weight_count = 4
                            weights = []
                            for w in range(weight_count):
                                weight = br.uint8_norm(f)
                                if weight > 0.0:
                                    weights.append(weight)
                            for i in range(weight_count):
                                if i < len(weights):
                                    iw[br.uint8(f)] = weights[i]
                                else:
                                    f.seek(1, 1)
                    elif stride == 20:
                        pos_skip = 12 if self.parent_mesh.position_type == 1 else 8
                        f.seek(pos_skip, 1)
                        remaining = stride - pos_skip
                        index_count = 4
                        weight_count = (remaining - index_count) // 2
                        weights = [br.uint16(f) / 32767.0 for _ in range(weight_count)]
                        indices = [br.uint8(f) for _ in range(index_count)]
                        for i in range(weight_count):
                            if weights[i] > 0.0:
                                iw[indices[i]] = weights[i]
                    elif stride == 32:
                        f.seek(8, 1)
                        if stride32_layout == 'A':
                            # Layout A: 8x uint16 weights, 8x uint8 indices
                            w8 = [br.uint16(f) / 32767.0 for _ in range(8)]
                            idx8 = [br.uint8(f) for _ in range(8)]
                            for i in range(8):
                                if w8[i] > 0.0:
                                    iw[idx8[i]] = iw.get(idx8[i], 0.0) + w8[i]
                        elif stride32_layout == 'C':
                            # Layout C: 12x uint8 weights, 12x uint8 indices
                            w12 = [br.uint8(f) for _ in range(12)]
                            idx12 = [br.uint8(f) for _ in range(12)]
                            for i in range(12):
                                wt = w12[i] / 255.0
                                if wt > 0.0:
                                    iw[idx12[i]] = iw.get(idx12[i], 0.0) + wt
                        else:
                            # Layout B: 6x uint8 weights, pad2, 6x uint16 indices, pad4
                            weights = [br.uint8_norm(f) for _ in range(6)]
                            f.seek(2, 1)  # skip 2 padding bytes
                            indices = [br.uint16(f) for _ in range(6)]
                            for i in range(6):
                                if weights[i] > 0.0:
                                    iw[indices[i]] = iw.get(indices[i], 0.0) + weights[i]
                    elif stride == 36:
                        f.seek(12, 1)
                        weights = [br.uint16(f) / 32767.0 for _ in range(8)]
                        indices = [br.uint8(f) for _ in range(8)]
                        for i in range(8):
                            if weights[i] > 0.0:
                                iw[indices[i]] = weights[i]
                    elif stride == 40:
                        f.seek(8, 1)
                        weights = [br.uint16(f) / 32767.0 for _ in range(8)]
                        indices = [br.uint16(f) for _ in range(8)]
                        for i in range(8):
                            if weights[i] > 0.0:
                                iw[indices[i]] = weights[i]
                    elif stride == 44:
                        pos_skip = 12 if self.parent_mesh.position_type == 1 else 8
                        f.seek(pos_skip, 1)
                        weight_count = 12
                        weights = [br.uint8_norm(f) for _ in range(weight_count)]
                        indices = [br.uint16(f) for _ in range(weight_count)]
                        for i in range(weight_count):
                            if weights[i] > 0.0:
                                iw[indices[i]] = weights[i]
                    else:
                        pos_length = 12 if self.parent_mesh.position_type == 1 else 8
                        f.seek(pos_length, 1)
                        weight_count = int((stride - pos_length) / 2)
                        weights = []
                        for w in range(weight_count):
                            weight = br.uint8_norm(f)
                            if weight > 0.0:
                                weights.append(weight)
                        for i in range(weight_count):
                            if i < len(weights):
                                iw[br.uint8(f)] = weights[i]
                            else:
                                f.seek(1, 1)
                    f.seek(stride_start + stride)
                    bone_weights.append(iw)
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
                use_uint32 = False
                if self.index_count > 0:
                    # Detect uint32 indices
                    if self.size_a == self.face_block_offset // 4 and self.size_a != self.face_block_offset // 2:
                        use_uint32 = True
                    else:
                        peek_bytes = f.read(16)
                        f.seek(self.face_block_offset)
                        if len(peek_bytes) >= 16:
                            import struct as _struct
                            hi_words = [_struct.unpack('<H', peek_bytes[i:i + 2])[0]
                                        for i in range(2, 16, 4)]
                            use_uint32 = all(v == 0 for v in hi_words)
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

            def get_normals_size(self):
                if self.parent_mesh.normal_type == 0:
                    return 8
                else:
                    return 28

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
                              For position_type 0: world = int16_norm(raw) * w,
                              so the inverse is raw = round(world / w * 32767), packed as int16.
                              If None or 0, the vertex is skipped (zero-displacement override).
                :return:
                """
                f = file
                x = pos[0]
                y = pos[1]
                z = pos[2]
                stride = self.parent_mesh.vertex_stride
                stride_start = f.tell()
                if self.parent_mesh.position_type == 0:
                    if scale is None or scale == 0:
                        f.seek(stride_start + stride)
                        return
                    # Auto-expand scale if vertex exceeds original bounds (e.g. after scaling up)
                    max_abs = max(abs(x), abs(y), abs(z))
                    if max_abs >= abs(scale):
                        scale = min(32767, int(max_abs) + 1)
                    f.write(bp.int16_norm(max(-1.0, min(1.0, x / scale))))
                    f.write(bp.int16_norm(max(-1.0, min(1.0, y / scale))))
                    f.write(bp.int16_norm(max(-1.0, min(1.0, z / scale))))
                    f.write(bp.int16(scale))
                elif self.parent_mesh.position_type == 1:
                    f.write(bp.float(x))
                    f.write(bp.float(y))
                    f.write(bp.float(z))
                f.seek(stride_start + stride)

        def __init__(self,parent_sk_mesh, index=0):
            self.parent_sk_mesh:SkeletalMeshAsset = parent_sk_mesh
            self.index = index
            self.name = ""
            self.name_offset = 0        # byte offset of the uint16 length prefix in the source file
            self.name_length = 0        # original byte count of the name field (from the uint16 prefix)
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
            self.position_type = 0 # 0:int16_norm 1:floats
            self.mesh_file = None  # BytesIO of reversed ordered LOD mesh data (used by create_mesh_file)
        def parse(self, f):
            version = self.parent_sk_mesh.version

            self.name_offset = f.tell()
            _nlen = unpack('<H', f.read(2))[0]
            self.name_length = _nlen
            self.name = br.string(f, _nlen).rstrip('\x00')
            f.seek(48, 1)  # some kind of matrix
            f.seek(1, 1)
            if version == 11:
                f.seek(1, 1) # skip x_count
                f.seek(4 * br.uint16(f), 1)
            else:
                x_count = br.uint8(f)
                f.seek(1 + 4 * x_count, 1)
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
            #   v14, u>0:       root_bone_index[1]  lod_info_type[1]
            #   v14, u==0:      lod_info_type[1]
            #   v15/16/17, u>0: root_bone_index[2]  lod_info_type[1]
            #   v15/16/17, u==0: lod_info_type[1]
            if u_count > 0 and version not in (11, 12):
                if version in (13, 14):
                    f.seek(1, 1)  # v13/v14: 1-byte root_bone_index
                else:             # v15, v16, v17
                    f.seek(2, 1)  # v15+: 2-byte root_bone_index
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
            f.seek(4 * self.uv_count, 1)

            # uv_divisors: divisor table read from file (see _UV_KNOWN_DIVISORS).
            # count_c/color_count is the authoritative UV set count when all entries
            # are valid divisors -- the header uv_count can undercount (e.g. some meshes have uv_count=1 in
            # the header but count_c=2 with valid divisors for both UV sets).
            self.uv_divisors = None
            if version == 11:
                self.color_count = 0 # v11 does not store color_count; always 0
            elif version in (16, 17):
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

            # --- Position type detection ---
            # Uses the same normals_base formula:
            #   normals_base == 28 -> float positions (3 × float32, 12 bytes)
            #   normals_base == 12 -> int16 positions (4 × int16 x/y/z/scale, 8 bytes)
            #   other              -> default to int16
            # Override: vertex_stride in (32,40) is always int16;
            #           vertex_stride in (28,36) is always float.
            # stride=44 is NOT overridden — some (e.g. gear upperbody) are float (nb=28)
            #   while others (e.g. head, nb=8) are correctly int16 via the formula.
            if self.vertex_stride in (32, 40):
                self.position_type = 0  # int16
            elif self.vertex_stride in (28, 36):
                self.position_type = 1  # float
            elif self.vertex_stride == 12:
                self.position_type = 1 if u_count == 0 else 0
            elif _normals_base >= 28:
                self.position_type = 1  # float
            else:
                self.position_type = 0  # int16

            print(f'\nName = {self.name}'
                  f'\nVertex Stride: {self.vertex_stride}'
                  f'\nNormals Stride: {self.normals_stride}'
                  f'\nUV Count: {self.uv_count}'
                  f'\nColor Count: {self.color_count}'
                  f'\nNormal Type: {self.normal_type}')

            # --- Post-stride skip ---
            # v17 has 4 extra bytes here compared to other versions
            if version == 17:
                f.seek(20, 1)
            else:
                f.seek(16, 1)
        def extract_mesh_file(self,f):
            """
            Creates a file gathering the raw data of all Lods the Mesh.
            :param f: combined mmb file that contains the header and data.
            :return: Path to the extracted raw_mesh file.
            """
            extract_file = io.BytesIO()
            for lod in reversed(self.lods):
                # print(f'Copy at {lod.data_offset} size {lod.data_size}')
                CopyFile(f, extract_file, lod.data_offset, lod.data_size)
            # print(f'Total size: {sum(lod.data_size for lod in self.lods)}')
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
            raise Exception(f'Unsupported .mmb version: {self.version}. Supported versions: 11, 12, 13, 15, 16, 17.')
        self.bone_count = br.uint32(f)
        for b in range(self.bone_count):
            self.bones.append(self.Bone(f))
        self.mesh_count = br.uint32(f)
        for m in range(self.mesh_count):
            mesh = self.Mesh(self, index=m)
            mesh.parse(f)
            self.meshes.append(mesh)

    def clear(self):
        self.bones = []
        self.meshes = []

    def get_sorted_lods(self):
        lod_map = {}
        for m in self.meshes:
            for l in m.lods:
                lod_map[l] = l.data_offset
        sorted_lods = {k: v for k, v in sorted(lod_map.items(), key=lambda item: item[1])}
        return sorted_lods

    def get_mesh_data_start_offset(self):
        lods = self.get_sorted_lods()
        first_lod = next(iter(lods))
        print("Mesh Data Start Offset = ", first_lod.data_offset)
        return first_lod.data_offset

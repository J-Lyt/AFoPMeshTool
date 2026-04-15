# Original author: AlexPo
# Modified by: JasperZebra — Avatar: Frontiers of Pandora (.mmb version 13) support
# Further modified — multi-version support: v12, v13, v15, v16, v17
#   - Added v12/v15/v16 parsing (multi-version support)
#   - Fixed pre-LOD section: root_bone_index and lod_info_type are version/u_count conditional
#   - Fixed tail section: UV hashes are 4 bytes on all versions; unk field order differs v12-15 vs v16-17
#   - Formula-based position type detection: normals_base = ns - 4*uv - 4*col; 28->float, 12->int16
#   - All fixes are version-conditional (additive only — existing versions unchanged)

bl_info = {
    "name": "AFoP Mesh Tool",
    "author": "JasperZebra, J-Lyt",
    "location": "Scene Properties > AFoP Mesh Tool Panel",
    "version": (0, 1, 39),
    "blender": (5, 0, 0),
    "description": "Imports skeletal meshes from AFoP .mmb files. Supports versions 12, 13, 15, 16, 17.",
    "category": "Import-Export"
    }

import shutil
import os
# Delete __pycache__ on load to prevent stale cache issues
_cache_dir = os.path.join(os.path.dirname(__file__), "__pycache__")
try:
    if os.path.exists(_cache_dir):
        shutil.rmtree(_cache_dir)
except OSError:
    pass

import bpy
import bmesh
from struct import unpack, pack
import numpy as np
import math
from mathutils import Matrix, Euler, Vector
from pathlib import Path
import io
import urllib.request
import threading
import re
import copy
import operator

# Auto-update
_RAW_URL = "https://raw.githubusercontent.com/J-Lyt/AFoPMeshTool/fullExport/__init__.py"
_BONE_JSON_URL = "https://raw.githubusercontent.com/J-Lyt/AFoPMeshTool/fullExport/bone_matrices.json"
_BONE_JSON_FILENAME = "bone_matrices.json"
_update_status = None   # None = not checked, "up_to_date", or "vX.X.X available"
_update_error  = None   # set if network fetch failed

def _get_bone_json_path():
    """Return the path for bone_matrices.json (same folder as plugin file)"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), _BONE_JSON_FILENAME)

def _download_bone_json():
    """Download bone_matrices.json to the plugin folder"""
    try:
        req = urllib.request.urlopen(_BONE_JSON_URL, timeout=30)
        data = req.read()
        dest = _get_bone_json_path()
        with open(dest, 'wb') as f:
            f.write(data)
        return True, None
    except Exception as e:
        return False, str(e)

def _check_bone_json():
    """If bone_matrices.json is missing, download it silently in a background thread."""
    if not os.path.isfile(_get_bone_json_path()):
        def _download_thread():
            ok, err = _download_bone_json()
            if not ok:
                print(f"[AFoPMT] Failed to download bone_matrices.json: {err}")
            else:
                print("[AFoPMT] bone_matrices.json downloaded successfully")
        threading.Thread(target=_download_thread, daemon=True).start()

def _fetch_remote_version():
    """Fetch remote __init__.py and return version tuple, or None on failure."""
    try:
        req = urllib.request.urlopen(_RAW_URL, timeout=8)
        text = req.read(4096).decode("utf-8", errors="ignore")
        m = re.search(r'"version"\s*:\s*\((\d+),\s*(\d+),\s*(\d+)\)', text)
        if m:
            return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except Exception:
        pass
    return None

def _check_update_thread():
    global _update_status, _update_error
    remote = _fetch_remote_version()
    if remote is None:
        _update_error = "Could not reach update server."
        return
    local = bl_info["version"]
    if remote > local:
        _update_status = f"v{remote[0]}.{remote[1]}.{remote[2]} available"
    else:
        _update_status = "up_to_date"


class ByteReader:
    @staticmethod
    def int8(f):
        b = f.read(1)
        i = unpack('<b', b)[0]
        return i
    @staticmethod
    def bool(f):
        b = f.read(1)
        i = unpack('<b', b)[0]
        if i == 0:
            return False
        elif i == 1:
            return True
        else:
            raise Exception("Byte at {v} wasn't a boolean".format(v=f.tell()))
    @staticmethod
    def uint8(f):
        b = f.read(1)
        i = unpack('<B', b)[0]
        return i
    @staticmethod
    def int16(f):
        return unpack('<h', f.read(2))[0]
    @staticmethod
    def uint16(f):
        b = f.read(2)
        i = unpack('<H', b)[0]
        return i
    @staticmethod
    def hash(f):
        b = f.read(8)
        return b
    @staticmethod
    def guid(f):
        return f.read(16)
    @staticmethod
    def int32(f):
        b = f.read(4)
        i = unpack('<i',b)[0]
        return i
    @staticmethod
    def uint32(f):
        b = f.read(4)
        i = unpack('<I',b)[0]
        return i
    @staticmethod
    def uint64(f):
        b = f.read(8)
        i = unpack('<Q',b)[0]
        return i
    @staticmethod
    def int64(f):
        b = f.read(8)
        i = unpack('<q', b)[0]
        return i
    @staticmethod
    def string(f,length):
        b = f.read(length)
        return "".join(chr(x) for x in b)
    @staticmethod
    def name(f):
        return br.string(f,br.uint16(f))
    @staticmethod
    def path(f):
        b = f.read(4)
        length = unpack('<i', b)[0]
        b = f.read(length)
        return "".join(chr(x) for x in b)
    @staticmethod
    def hashtext(f):
        b = f.read(4)
        length = unpack('<i', b)[0]
        f.seek(4,1)
        b = f.read(length)
        return "".join(chr(x) for x in b)
    @staticmethod
    def float(f):
        b = f.read(4)
        fl = unpack('<f',b)[0]
        return fl
    @staticmethod
    def vector3(f):
        b = f.read(12)
        return unpack('<fff', b)
    @staticmethod
    def dvector3(f):
        #double vector 3
        b = f.read(24)
        return unpack('<ddd', b)
    @staticmethod
    def vector4(f):
        b = f.read(16)
        return unpack('<ffff', b)
    @staticmethod
    def int16_norm(f):
        i = unpack('<H', f.read(2))[0]
        v = i ^ 2**15
        v -= 2**15
        v /= 2**15 - 1
        return v
    @staticmethod
    def uv_unorm_u(f):
        return (unpack('<H', f.read(2))[0] % 4096) / 4095.0
    @staticmethod
    def uv_unorm_v(f):
        return (unpack('<H', f.read(2))[0] % 4096) / 4095.0
    @staticmethod
    def uint16_norm(f):
        int16 = unpack('<H', f.read(2))[0]
        return int16 / 2 ** 16
    @staticmethod
    def uint8_norm(f):
        uint8 = unpack('<B', f.read(1))[0]
        maxint = (2 ** 8)-1
        return uint8 / maxint
    @staticmethod
    def int8_norm(f):
        int8 = unpack('<B', f.read(1))[0]
        v = int8 ^ 2**7
        v -= 2**7
        v /= 2**7 -1
        return v
    @staticmethod
    def X10Y10Z10W2_normalized(f):
        i = unpack('<I', f.read(4))[0]  # get 32bits of data

        x = i >> 0
        x = ((x & 0x3FF) ^ 512) - 512

        y = i >> 10
        y = ((y & 0x3FF) ^ 512) - 512

        z = i >> 20
        z = ((z & 0x3FF) ^ 512) - 512

        w = i >> 30
        w = w & 0x1

        vectorLength = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        # # print(x,y,z)
        if vectorLength != 0:
            x /= vectorLength
            y /= vectorLength
            z /= vectorLength
        return [x, y, z, w]
    @staticmethod
    def matrix_4x4(f):
        row1 = []
        row2 = []
        row3 = []
        row4 = []
        for i in range(4):
            for c in range(4):
                value = br.float(f)
                if c == 0:
                    row1.append(value)
                if c == 1:
                    row2.append(value)
                if c == 2:
                    row3.append(value)
                if c == 3:
                    row4.append(value)
        # print(Matrix((row1,row2,row3,row4)))
        matrix = Matrix((row1, row2, row3, row4))#.inverted()
        return matrix
class BytePacker:
    @staticmethod
    def int8(v):
        return pack('<b', v)
    @staticmethod
    def uint8(v):
        return pack('<B', v)
    @staticmethod
    def uint8_norm(v):
        if 0.0 <= v <= 1.0:
            i = max(0, min(int(v * ((2 ** 8)-1)), 255))
        else:
            raise Exception("Couldn't normalize value as uint8Norm, "
                            "it wasn't between 0.0 and 1.0. Unknown max value."
                            +str(v))
        return pack('<B', i)
    @staticmethod
    def int16(v):
        return pack('<h', v)
    @staticmethod
    def uint16(v):
        return pack('<H', v)
    @staticmethod
    def int16_norm(v):
        # print(v)
        if -1.0 <= v <= 1.0:
            # if v >= 0:
            #     v = int(abs(v) * (2 ** 15))
            # else:
            #     v = 2 ** 16 - int(abs(v) * (2 ** 15))
            v = max(min(int(v * (2 ** 15)), 32767), -32768)
        else:
            raise Exception("Couldn't normalize value as int16Norm, it wasn't between -1.0 and 1.0. Unknown max value.")
        return pack('<h', v)
    @staticmethod
    def uv_unorm_u(v):
        return pack('<H', max(0, min(4095, int(round(v * 4095)))))
    @staticmethod
    def uv_unorm_v(v):
        return pack('<H', max(0, min(4095, int(round(v * 4095)))))
    @staticmethod
    def uint16_norm(v):
        if 0.0 < v < 1.0:
            i = v * (2 ** 16) - 1
            i = int(i)
        else:
            raise Exception("Couldn't normalize value as uint16Norm, it wasn't between -1.0 and 1.0. Unknown max value.")
        return pack('<H', i)
    @staticmethod
    def float16(v):
        f32 = np.float32(v)
        f16 = f32.astype(np.float16)
        b16 = f16.tobytes()
        return b16
    @staticmethod
    def int32(v):
        return pack('<i', v)
    @staticmethod
    def uint32(v):
        return pack('<I', v)
    @staticmethod
    def uint64(v):
        return pack('<Q', v)
    @staticmethod
    def int64(v):
        return pack('<q', v)
    @staticmethod
    def float(v):
        return pack('<f', v)
    @staticmethod
    def X10Y10Z10W2(x,y,z,w):
        if x >= 0:
            x = int(abs(x) * 2 ** 9)
        else:
            x = 2**10 - int(abs(x) * 2 ** 9)
        if y >= 0:
            y = int(abs(y) * 2 ** 9)
        else:
            y = 2**10 - int(abs(y) * 2 ** 9)
        if z >= 0:
            z = int(abs(z) * 2 ** 9)
        else:
            z = 2**10 - int(abs(z) * 2 ** 9)


        w = int(w)


        x = (abs(x) & 0x3FF)
        y = (abs(y) & 0x3FF) << 10
        z = (abs(z) & 0x3FF) << 20
        w = (abs(w) & 0x3) << 30

        v = x | y | z | w
        return pack("<I", v)

br = ByteReader
bp = BytePacker

def CopyFile(read,write,offset,size,buffer_size=500000):
    read.seek(offset)
    chunks = size // buffer_size
    for o in range(chunks):
        write.write(read.read(buffer_size))
    write.write(read.read(size%buffer_size))
def get_merged_mmb(mmb):
    files = []
    if str(mmb).endswith("mmb"):
        files.append(mmb)
    else:
        i = 0
        while True:
            current_file = f"{str(mmb)[:-1]}{i}"
            if os.path.isfile(current_file):
                files.append(current_file)
                i += 1
            else:
                break

    f = io.BytesIO()
    for file_dir in files:
        with open(file_dir, 'rb') as file:
            f.write(file.read())
    return f

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

            def parse(self, f):
                self.start_offset = f.tell()
                self.vertex_count = br.uint32(f)
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
                pos_length = 0 #size of vertex position in stride
                f = raw_mesh_file
                f.seek(self.vertex_data_offset_a)
                for v in range(self.vertex_count):
                    iw = {}
                    if stride == 20:
                        f.seek(8, 1)
                        weight_count = 4
                        weights = [br.uint16(f) / 32767.0 for _ in range(weight_count)]
                        indices = [br.uint8(f) for _ in range(weight_count)]
                        for i in range(weight_count):
                            if weights[i] > 0.0:
                                iw[indices[i]] = weights[i]
                    elif stride == 32:
                        # Layout: 8×uint16 weights | 8×uint8 indices
                        # Two sub-variants detected by whether all 8 uint16 values sum to 32767:
                        #   sum == 32767: all 8 slots are real bone weights (e.g. body mesh)
                        #   sum != 32767: only slots 0-3 are real weights; slots 4-7 are
                        #                secondary index data, not bone weights (e.g. banshee)
                        f.seek(8, 1)
                        weights = [br.uint16(f) / 32767.0 for _ in range(8)]
                        indices = [br.uint8(f) for _ in range(8)]
                        weight_slots = range(8) if sum(int(w * 32767) for w in weights) == 32767 else range(4)
                        for i in weight_slots:
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
                        if stride == 24:
                            pos_length = 8
                        elif stride == 28:
                            pos_length = 12
                        else:
                            pos_length = 8
                        f.seek(pos_length,1)
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
                                f.seek(1,1)
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
                    peek_bytes = f.read(16)
                    f.seek(self.face_block_offset)
                    if len(peek_bytes) >= 16:
                        import struct as _struct
                        hi_words = [_struct.unpack('<H', peek_bytes[i:i+2])[0]
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
                color_count = self.parent_mesh.color_count
                f = raw_mesh_file
                f.seek(self.vertex_data_offset_b)
                v = Vector((0.0,0.0,1.0))
                self._all_w_zero = True # If all w values are zero, the normals store non-surface data (e.g. VAT).
                self._tangent_space = False # Normals stored in tangent space: x axis non-negative across all verts.
                _raw_x_min = 1.0
                for i in range(self.vertex_count):
                    stride_start = f.tell()
                    if self.parent_mesh.normal_type == 0:
                        # Layout: color(4*color_count) | tangent(4) | normal(4) | UV(4*uv_count)
                        f.seek(4 * color_count, 1)  # skip color
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
                        f.seek(4 * color_count, 1)  # skip colors before float normal
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
                color_count = self.parent_mesh.color_count
                normal_type = self.parent_mesh.normal_type
                f = raw_mesh_file
                f.seek(self.vertex_data_offset_b)

                # Detect whether UVs are compact or int16_norm.
                if self.vertex_count > 0:
                    if normal_type == 0:
                        uv_field_off = 4 * color_count + 4 + 4 + index * 4
                    else:  # normal_type == 1
                        uv_field_off = 4 * color_count + 12 + 12 + 4 + index * 4
                    start_pos = f.tell()
                    use_compact = True
                    for i in range(self.vertex_count):
                        f.seek(start_pos + i * stride + uv_field_off)
                        raw_u = unpack('<h', f.read(2))[0]  # signed
                        if abs(raw_u) > 8191:
                            use_compact = False
                            break
                    f.seek(start_pos)

                for i in range(self.vertex_count):
                    stride_start = f.tell()
                    if normal_type == 0:
                        # color(4*cc) | normal(4) | tangent(4) | UV(4*uv)
                        f.seek(4 * color_count + 4 + 4, 1)
                    elif normal_type == 1:
                        # color(4*cc) | normal(12) | tangent(12) | sign(4) | UV(4*uv)
                        f.seek(4 * color_count + 12 + 12 + 4, 1)
                    f.seek(index * 4, 1)  # skip previous uv sets
                    if use_compact:
                        u = br.uv_unorm_u(f)
                        v = br.uv_unorm_v(f)
                    else:
                        u = br.int16_norm(f)
                        v = br.int16_norm(f)
                    f.seek(stride_start + stride)
                    uvs.append((u,v))
                return uvs

            def get_color(self,raw_mesh_file, index=0):
                """
               Seeks to Lod.vertex_data_offset_b and reads all Color data.
               :param raw_mesh_file: file that is exported by SkeletalMeshAsset.Mesh.extract_mesh_file()
               :return: a List of Tuples containing 4 floats as RGBA coordinates.
               """
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
                    f.write(bp.int16_norm(x / scale))
                    f.write(bp.int16_norm(y / scale))
                    f.write(bp.int16_norm(z / scale))
                    f.write(bp.int16(scale))
                elif self.parent_mesh.position_type == 1:
                    f.write(bp.float(x))
                    f.write(bp.float(y))
                    f.write(bp.float(z))
                f.seek(stride_start + stride)

            def write_bone_weights(self, file, mmb_index, blender_weights):
                """
                Writes bone weights for one vertex into file at the correct stride offset.
                :param file: file to write on
                :param mmb_index: which vertex slot in the file to write
                :param blender_weights: bone indices already converted from Blender vertex group names back to mmb
                                        mesh-bone indices.
                """
                stride = self.parent_mesh.vertex_stride
                f = file

                pos_len = 12 if self.parent_mesh.position_type == 1 else 8

                # Max bone influences supported by this stride
                if stride == 20:
                    max_bones = 4
                elif stride in (32, 36, 40):
                    max_bones = 8
                elif stride == 44:
                    max_bones = 12
                else:
                    max_bones = int((stride - pos_len) / 2)

                # Sort by weight descending, keep top max_bones
                # Do not normalise weights
                sorted_w = sorted(blender_weights.items(), key=lambda kv: kv[1], reverse=True)[:max_bones]
                # Clamp each weight to [0.0, 1.0]
                sorted_w = [(idx, max(0.0, min(1.0, w))) for idx, w in sorted_w]
                active_count = len(sorted_w)

                f.seek(self.data_offset + mmb_index * stride + pos_len)

                if stride == 20:
                    for _, w in sorted_w:
                        f.write(bp.uint16(int(round(w * 32767))))
                    for _ in range(max_bones - active_count):
                        f.write(bp.uint16(0))
                    for idx, _ in sorted_w:
                        f.write(bp.uint8(idx))

                elif stride in (32, 36):
                    for _, w in sorted_w:
                        f.write(bp.uint16(int(round(w * 32767))))
                    for _ in range(max_bones - active_count):
                        f.write(bp.uint16(0))
                    for idx, _ in sorted_w:
                        f.write(bp.uint8(idx))

                elif stride == 40:
                    for _, w in sorted_w:
                        f.write(bp.uint16(int(round(w * 32767))))
                    for _ in range(max_bones - active_count):
                        f.write(bp.uint16(0))
                    for idx, _ in sorted_w:
                        f.write(bp.uint16(idx))

                elif stride == 44:
                    for _, w in sorted_w:
                        f.write(bp.uint8(int(round(w * 255))))
                    for _ in range(max_bones - active_count):
                        f.write(bp.uint8(0))
                    for idx, _ in sorted_w:
                        f.write(bp.uint16(idx))
                    for _ in range(max_bones - active_count):
                        f.write(bp.uint16(0))

                else:
                    for _, w in sorted_w:
                        f.write(bp.uint8(int(round(w * 255))))
                    for _ in range(max_bones - active_count):
                        f.write(bp.uint8(0))
                    for idx, _ in sorted_w:
                        f.write(bp.uint8(idx))
            
        def __init__(self,parent_sk_mesh, index=0):
            self.parent_sk_mesh:SkeletalMeshAsset = parent_sk_mesh
            self.index = index
            self.name = ""
            self.name_offset = 0        # byte offset of the uint16 length prefix in the source file
            self.name_length = 0        # original byte count of the name field (from the uint16 prefix)
            self.pending_rename_new = ""  # staged mesh rename — applied to _MOD copy on next export
            self.zeroed_out_in_session = False  # True if Zero Out was used this session
            self.lod_count = 0
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
            self.position_type = 0 # 0:int16_norm 1:floats
            self.mesh_file = None  # BytesIO of reversed ordered LOD mesh data (used by create_mesh_file)
        def parse(self, f):
            self.name_offset = f.tell()
            _nlen = unpack('<H', f.read(2))[0]
            self.name_length = _nlen
            self.name = br.string(f, _nlen).rstrip('\x00')
            f.seek(48, 1)  # some kind of matrix
            f.seek(1, 1)
            x_count = br.uint8(f)
            f.seek(1, 1)
            f.seek(4 * x_count, 1)
            self.u_count_offset = f.tell()
            u_count = br.uint16(f)
            for b in range(u_count):
                matrix = br.matrix_4x4(f)
                offset = f.tell()
                bone_index = br.uint16(f)
                self.mesh_bone_file_offsets.append(offset)
                self.mesh_bones[bone_index] = matrix
            self.bone_table_end_offset = f.tell()

            version = self.parent_sk_mesh.version

            # --- Pre-LOD section ---
            # root_bone_index: present only when u_count > 0, and absent for v12 entirely.
            # lod_info_type: present for v15/v16/v17 always; absent for v12/v13.
            #   v12:            no root_bone_index, no lod_info_type
            #   v13, u>0:       root_bone_index[1]  lod_info_type[1]
            #   v13, u==0:      (no root_bone_index, no lod_info_type)
            #   v15/16/17, u>0: root_bone_index[2]  lod_info_type[1]
            #   v15/16/17, u==0: lod_info_type[1]
            if u_count > 0 and version != 12:
                if version == 13:
                    f.seek(1, 1)  # v13: 1-byte root_bone_index
                else:             # v15, v16, v17
                    f.seek(2, 1)  # v15+: 2-byte root_bone_index
                lod_info_type = br.uint8(f)
            else:
                if version in (12, 13):
                    lod_info_type = 0  # v12/v13 have no lod_info_type byte
                else:                  # v15, v16, v17 with u_count == 0
                    lod_info_type = br.uint8(f)

            self.lod_count = br.uint8(f)
            f.seek(4, 1)  # unknown 4 bytes before LOD list

            # --- LOD list ---
            # Each LOD is 36 bytes.
            # lod_info_type == 2: 28 extra bytes per LOD (v15/v16/v17)
            for l in range(self.lod_count):
                lod = self.LOD(self,l)
                lod.parse(f)
                if lod_info_type == 2:
                    f.seek(28, 1)
                self.lods.append(lod)

            # --- Tail section: UV hashes, color hashes, strides ---
            # UV hashes are 4 bytes each on all versions.
            #
            # v12/v13/v15 layout:  uv_count  uv_hashes  unk[4]  color_count  color_hashes
            # v16/v17 layout:      uv_count  uv_hashes  color_count  color_hashes  unk[4]  count_c  c_data
            self.uv_count = br.uint8(f)
            f.seek(4 * self.uv_count, 1)

            if version in (16, 17):
                self.color_count = br.uint8(f)
                f.seek(4 * self.color_count, 1)
                f.seek(4, 1)               # unk after color (v16/v17)
                count_c = br.uint8(f)
                f.seek(4 * count_c, 1)
            else:
                f.seek(4, 1)               # unk before color (v12/v13/v15)
                self.color_count = br.uint8(f)
                f.seek(4 * self.color_count, 1)

            self.vertex_stride = br.uint16(f)
            self.normals_stride = br.uint16(f)

            # --- Normal type detection ---
            # normals_base = normals_stride - 4*uv_count - 4*color_count
            # normals_base > 8 -> normal_type 1 or 2 (float normals)
            # normals_base <= 8 -> normal_type 0 (int8_norm, 4-byte block)
            # normal_type 1: float normal(12) + tangent(12) + sign(4) = 28 bytes
            _normals_base = self.normals_stride - 4 * self.uv_count - 4 * self.color_count
            if _normals_base == 28:
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
        if self.version not in (12, 13, 15, 16, 17):
            raise Exception(f'Unsupported .mmb version: {self.version}. Supported versions: 12, 13, 15, 16, 17.')
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

class BlenderMeshImporter:
    @staticmethod
    def find_or_create_collection(name):
        index = bpy.data.collections.find(name)
        if index != -1:
            return bpy.data.collections[index]
        else:
            collection = bpy.data.collections.new(name)
            bpy.context.scene.collection.children.link(collection)
            return collection


    @staticmethod
    def import_mesh(file, skeletal_mesh:SkeletalMeshAsset, mesh:SkeletalMeshAsset.Mesh, lod_index = 0):
        # Extract raw mesh file
        raw_mesh_file = mesh.extract_mesh_file(file)

        # Create Mesh and Object
        obj_name = f'{mesh.name}_LOD{lod_index}'
        obj_data = bpy.data.meshes.new(obj_name)
        obj = bpy.data.objects.new(obj_name, obj_data)
        collection = BMI.find_or_create_collection(skeletal_mesh.name)
        collection.objects.link(obj)

        # Create BMesh
        bm = bmesh.new()
        bm.from_mesh(obj_data)

        lod = mesh.lods[lod_index]
        lod.blender_obj_name = obj.name  # record actual Blender name (may differ from obj_name if Blender de-duped it)

        # Get any extra bytes between the end of vertex/normal data and the next block.
        # These are preserved on export to maintain correct vob/fb offsets.
        real_vert_size = lod.vertex_count * mesh.vertex_stride
        real_normal_size = lod.vertex_count * mesh.normals_stride
        vert_extra = (lod.vertex_data_offset_b - lod.vertex_data_offset_a) - real_vert_size
        normal_extra = (lod.face_block_offset - lod.vertex_data_offset_b) - real_normal_size
        if vert_extra > 0:
            raw_mesh_file.seek(lod.vertex_data_offset_a + real_vert_size)
            lod.vertex_end_bytes = raw_mesh_file.read(vert_extra)
        else:
            lod.vertex_end_bytes = b''
        if normal_extra > 0:
            raw_mesh_file.seek(lod.vertex_data_offset_b + real_normal_size)
            lod.normals_end_bytes = raw_mesh_file.read(normal_extra)
        else:
            lod.normals_end_bytes = b''

        # Import vertices
        verts = lod.get_vertex_positions(raw_mesh_file)
        for v in verts:
            bmv = bm.verts.new()
            v_co = (v[0]*-1,v[1],v[2])
            bmv.co = v_co
        bm.verts.ensure_lookup_table()
        # Import triangles
        triangles = lod.get_triangles(raw_mesh_file)
        for tris in triangles:
            face_vertices = []
            for v_index in tris:
                tv = bm.verts[v_index]
                face_vertices.append(tv)
            # some meshes store intentional duplicate triangles; skip them.
            if bm.faces.get(face_vertices) is not None:
                continue
            bm_face = bm.faces.new(face_vertices)
            bm_face.normal_flip() #this is required because the *-1 on x vertex co flips the mesh normals
        bm.to_mesh(obj_data)
        bm.free()

        # Store original MMB vertex index as a named int attribute.
        # This survives separate/join operations and vertex reordering, allowing
        # write_normals to look up the correct original tangent/nw for each vertex
        # without relying on raw-position matching.
        orig_idx_attr = obj_data.attributes.new(name="mmb_vertex_order", type='INT', domain='POINT')
        for vi in range(len(obj_data.vertices)):
            orig_idx_attr.data[vi].value = vi

        bm = bmesh.new()
        bm.from_mesh(obj_data)
        bm.faces.ensure_lookup_table()
        # Import UVs
        for uv_index in range(mesh.uv_count):
            uvs = lod.get_uvs(raw_mesh_file,uv_index)
            uv_layer = bm.loops.layers.uv.new(f'UVMap_{uv_index}')
            # Detect UV convention
            v_vals = [uvs[i][1] for i in range(len(uvs))]
            u_vals = [uvs[i][0] for i in range(len(uvs))]
            v_min, v_max = min(v_vals), max(v_vals)
            u_min, u_max = min(u_vals), max(u_vals)
            centred_v = v_max > 0 and v_min < 0 and abs(v_min + v_max) < 0.15
            centred_u = u_min < -0.1 and abs(u_min + u_max) < 0.15
            for finder, face in enumerate(bm.faces):
                for lindex, loop in enumerate(face.loops):
                    v_index = loop.vert.index
                    u, v = uvs[v_index][0], uvs[v_index][1]
                    u_out = (u + 1) / 2.0 if centred_u else u
                    v_out = v + 0.5 if centred_v else v * -1 + 1
                    loop[uv_layer].uv = (u_out, v_out)

        bm.to_mesh(obj_data)
        bm.free()
        obj_data.update()

        # Store per-UV-layer centred flags as mesh attributes so export can
        # reverse the correct transform without re-detecting from Blender values.
        for uv_index in range(mesh.uv_count):
            uvs = lod.get_uvs(raw_mesh_file, uv_index)
            v_vals = [uvs[i][1] for i in range(len(uvs))]
            u_vals = [uvs[i][0] for i in range(len(uvs))]
            centred_v = min(v_vals) < 0 and max(v_vals) > 0 and abs(min(v_vals) + max(v_vals)) < 0.15
            centred_u = min(u_vals) < -0.1 and abs(min(u_vals) + max(u_vals)) < 0.15
            cu_attr = obj_data.attributes.new(name=f'mmb_uv{uv_index}_centred_u', type='INT', domain='POINT')
            cv_attr = obj_data.attributes.new(name=f'mmb_uv{uv_index}_centred_v', type='INT', domain='POINT')
            for vi in range(len(obj_data.vertices)):
                cu_attr.data[vi].value = 1 if centred_u else 0
                cv_attr.data[vi].value = 1 if centred_v else 0

        # Import Colors
        # Written directly to obj_data.attributes (POINT domain, FLOAT_COLOR type) rather
        # than via a BMesh float_color layer, because the BMesh vertex-layer -> to_mesh
        # round-trip corrupts float_color values, causing them to appear white in the viewport.
        for color_index in range(mesh.color_count):
            colors = lod.get_color(raw_mesh_file, color_index)
            attr_name = f"Color_{color_index}"
            attr = obj_data.attributes.new(name=attr_name, type='FLOAT_COLOR', domain='POINT')
            for vi, (r, g, b, a) in enumerate(colors):
                attr.data[vi].color = (r, g, b, a)

        # Import Normals
        # Some VAT meshes store animation-texture lookup data in the normals stream
        # rather than actual surface normals (e.g. geckofish: all ny=0, nz=0).
        # Detect degenerate normals (all pointing the same direction) and fall back to computing normals.
        computed_normals = lod.get_normals(raw_mesh_file)
        degenerate = False
        if computed_normals:
            first = computed_normals[0]
            sample = computed_normals[1:min(len(computed_normals), 256)]

            all_w_zero = getattr(lod, '_all_w_zero', False)
            tangent_space = getattr(lod, '_tangent_space', False)
            all_same_dir = all(abs(n.dot(first)) > 0.999 for n in sample)
            zero_count = sum(1 for n in computed_normals if n.length < 0.1)
            mostly_zero = zero_count > len(computed_normals) * 0.5

            # Compare normals in file against the meshes own face normals.
            # Create a temporary mesh and measure the average dot between each stored normal and the face it belongs to.
            # Valid normals should match with their faces (avg_dot typically between 0.4 and 0.9).
            import bmesh as _bmesh
            bm_check = _bmesh.new()
            bm_check.from_mesh(obj_data)
            bm_check.faces.ensure_lookup_table()
            dot_sum = 0.0
            dot_count = 0
            for face in bm_check.faces:
                face.normal_update()
                fn = face.normal
                for loop in face.loops:
                    vi = loop.vert.index
                    if vi < len(computed_normals):
                        sn = computed_normals[vi]
                        if sn.length > 0.1:
                            dot_sum += fn.dot(sn)
                            dot_count += 1
                    if dot_count >= 4096:
                        break
                if dot_count >= 4096:
                    break
            bm_check.free()
            avg_dot = (dot_sum / dot_count) if dot_count > 0 else 0.0
            # Stored normals that don't match with face normals (avg_dot < 0.1)
            # or normals that point consistently opposite to faces (avg_dot < -0.1) are degenerate.
            bad_correlation = abs(avg_dot) < 0.1

            degenerate = all_w_zero or all_same_dir or mostly_zero or bad_correlation

            if not degenerate:
                print(f"[AFoPMT] {obj.name}: Importing normals from file")
                obj_data.normals_split_custom_set_from_vertices(computed_normals)
            else:
                reasons = []
                if all_w_zero: reasons.append("All w=0 (e.g. VAT Data)")
                if all_same_dir: reasons.append("All Same Direction")
                if mostly_zero: reasons.append(f"Mostly Zero ({zero_count}/{len(computed_normals)})")
                if bad_correlation: reasons.append(f"No Face Correlation (avg_dot={avg_dot:.3f})")
                print(f"[AFoPMT] {obj.name}: Degenerate Normals Detected ({', '.join(reasons)}) - Computing Normals")

        if degenerate:
            BME._compute_normals_for_object(obj)

        # Import Bone Weights
        weights = lod.get_bone_weights(raw_mesh_file)
        mesh_bones = list(mesh.mesh_bones.keys())
        # Only create vertex groups for bones this mesh references
        for real_bone_index in mesh_bones:
            if real_bone_index < len(skeletal_mesh.bones):
                bone_name = skeletal_mesh.bones[real_bone_index].name
                if obj.vertex_groups.get(bone_name) is None:
                    obj.vertex_groups.new(name=bone_name)

        for v_index in range(lod.vertex_count):
            v_bone_weights = weights[v_index]
            for bone_index in v_bone_weights.keys():
                if bone_index < len(mesh_bones):
                    real_bone_index = mesh_bones[bone_index] # Convert mesh bone index to skeleton bone index
                    bone_name = skeletal_mesh.bones[real_bone_index].name
                    obj.vertex_groups[bone_name].add([v_index], v_bone_weights[bone_index], "ADD")
        return obj

    @staticmethod
    def find_or_create_skeleton(skeletal_mesh:SkeletalMeshAsset):
        index = bpy.data.objects.find(skeletal_mesh.name)
        if index != -1:
            return bpy.data.objects[index]
        else:
            return BMI.import_skeleton(skeletal_mesh)

    @staticmethod
    def import_skeleton(skeletal_mesh:SkeletalMeshAsset):
        _armature = bpy.data.armatures.new(skeletal_mesh.name)
        _obj = bpy.data.objects.new(skeletal_mesh.name, _armature)
        collection = BMI.find_or_create_collection(skeletal_mesh.name)
        collection.objects.link(_obj)
        bpy.context.view_layer.objects.active = _obj
        bpy.ops.object.mode_set(mode='EDIT')
        for i,b in enumerate(skeletal_mesh.bones):
            bone = _armature.edit_bones.new(b.name)
            parent_index = b.parent_index
            if b.parent_index == 65535:
                parent_index = -1
            bone.parent = _armature.edit_bones[parent_index] if parent_index != -1 else None
            bone.tail = Vector([0.0,0.0,0.1])
            parent_matrix = Matrix()
            if bone.parent:
                parent_matrix = bone.parent.matrix
            bone.matrix = parent_matrix @ b.matrix
        scale_matrix = Matrix().Scale(-1.0, 4, Vector((1.0, 0.0, 0.0)))
        _armature.transform(scale_matrix)
        bpy.ops.object.mode_set(mode='OBJECT')
        return _obj
    @staticmethod
    def parent_obj_to_armature(obj, armature, mesh):
        # Skip the armature modifier for meshes with no bone slots.
        if not mesh.mesh_bones:
            obj.parent = armature
            return
        obj.modifiers.new(name='Armature', type='ARMATURE')
        obj.modifiers['Armature'].object = armature
        obj.parent = armature
    @staticmethod
    def rotate_model(obj,armature):
        # armature.rotation_euler[0] = math.radians(90)
        # bpy.ops.object.transform_apply(rotation = True)
        rot = Euler(map(math.radians,(90,0,0)),'XYZ')
        mat = rot.to_matrix().to_4x4()
        armature.matrix_world = mat
        bpy.ops.object.transform_apply(rotation=True)

class BlenderMeshExporter:
    @staticmethod
    def _compute_normals_for_object(obj):
        """
        Compute smooth angle-weighted normals for obj and store them as custom split normals.
        Merge vertices by distance, compute angle-weighted face normals on the merged mesh, then transfer
        them back to the original (possibly split) vertex layout.
        """
        obj_data = obj.data
        merge_dist = 1e-4

        bm_weld = bmesh.new()
        bm_weld.from_mesh(obj_data)
        bmesh.ops.remove_doubles(bm_weld, verts=bm_weld.verts, dist=merge_dist)
        bm_weld.faces.ensure_lookup_table()
        bm_weld.verts.ensure_lookup_table()

        weld_normals = [Vector((0.0, 0.0, 0.0)) for _ in range(len(bm_weld.verts))]
        for face in bm_weld.faces:
            face.normal_update()
            for loop in face.loops:
                angle = loop.calc_angle()
                weld_normals[loop.vert.index] += face.normal * angle

        def _key(co):
            return (round(co.x, 4), round(co.y, 4), round(co.z, 4))

        pos_to_normal = {}
        for wv in bm_weld.verts:
            n = weld_normals[wv.index]
            length = n.length
            pos_to_normal[_key(wv.co)] = n / length if length > 1e-8 else Vector((0.0, 0.0, 1.0))

        bm_weld.free()

        bm_orig = bmesh.new()
        bm_orig.from_mesh(obj_data)
        bm_orig.verts.ensure_lookup_table()

        smooth_normals = []
        for ov in bm_orig.verts:
            smooth_normals.append(pos_to_normal.get(_key(ov.co), Vector((0.0, 0.0, 1.0))))

        bm_orig.free()

        obj_data.normals_split_custom_set_from_vertices(smooth_normals)
        obj_data.update()

    @staticmethod
    def _triangulate_object(obj, compute_normals=False):
        """
        Triangulate all faces of obj in-place using the Triangulate modifier with
        Keep Normals enabled, then optionally recompute smooth normals.
        """
        import bpy as _bpy

        mod = obj.modifiers.new(name="_tri_export", type='TRIANGULATE')
        mod.keep_custom_normals = True
        mod.quad_method = 'BEAUTY'
        mod.ngon_method = 'BEAUTY'

        dg = _bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(dg)
        me = _bpy.data.meshes.new_from_object(eval_obj)

        obj.data = me
        obj.modifiers.remove(mod)
        obj.data.update()

        if compute_normals:
            BME._compute_normals_for_object(obj)

    @staticmethod
    def find_object_by_name(name=""):
        return bpy.data.objects.get(name, None)
    @staticmethod
    def _write_mod_file(edited_lod_index_per_mesh: dict, out_path: str):
        """
        For each mesh/LOD pair in edited_lod_index_per_mesh, the vertex, normal,
        and face data are rewritten from the current Blender mesh. All other LODs
        and the full file header are copied verbatim from the source file.

        If the new data fits within the original block it is written in-place.
        If the new data is larger, the file is restructured: bytes are inserted at
        the end of the edited LOD's block, and all affected offsets are updated.

        :param edited_lod_index_per_mesh: dict mapping mesh_index -> lod_index to rewrite.
               A value of -1 means copy all LODs verbatim (no Blender mesh needed).
        :param out_path: destination file path to write.
        """
        SWOMT = bpy.context.scene.SWOMT
        src_path = SWOMT.AssetPath
        export_normals = SWOMT.export_normals or _vert_count_changed()

        # If a _MOD file already exists, use it as the source so successive exports accumulate.
        mod_candidate = os.path.splitext(src_path)[0] + "_MOD.mmb"
        if os.path.isfile(mod_candidate):
            src_path = mod_candidate

        with open(src_path, 'rb') as src:
            file_data = bytearray(src.read())

        # Sync lod.data_offset values from the source file (_MOD or original).
        for m in asset.meshes:
            for lod in m.lods:
                fp = lod.data_offset_file_pos
                lod.data_offset = unpack('<I', file_data[fp:fp+4])[0]

        # Read asset.size (bytes 4-8) - boundary between header and streaming sections
        asset_size = unpack('<I', file_data[4:8])[0]

        for mesh_index, lod_index in edited_lod_index_per_mesh.items():
            if lod_index < 0:
                continue
            mesh = asset.meshes[mesh_index]
            if lod_index >= len(mesh.lods):
                continue
            lod = mesh.lods[lod_index]

            obj = BME.find_object_by_name(mesh.name + f"_LOD{lod_index}")
            if obj is None:
                continue

            new_vert_count  = len(obj.data.vertices)
            new_index_count = len(obj.data.polygons) * 3

            # Build new data blocks
            verts_buf = io.BytesIO()
            BME.write_vertices(verts_buf, mesh, lod_index)
            if lod.vertex_end_bytes:
                verts_buf.write(lod.vertex_end_bytes)

            norms_buf = io.BytesIO()
            if export_normals:
                BME.write_normals(norms_buf, mesh, lod_index)
                if lod.normals_end_bytes:
                    norms_buf.write(lod.normals_end_bytes)
            else:
                # Preserve the original normals bytes from the source file unchanged.
                orig_higher_size = sum(
                    mesh.lods[li].data_size
                    for li in range(lod_index + 1, len(mesh.lods))
                )
                with open(SWOMT.AssetPath, 'rb') as orig_src:
                    orig_file_data = orig_src.read()
                orig_data_offset = unpack('<I', orig_file_data[lod.data_offset_file_pos:lod.data_offset_file_pos + 4])[0]
                file_vob_orig = unpack('<I', orig_file_data[lod.start_offset + 16:lod.start_offset + 20])[0]
                file_fb_orig  = unpack('<I', orig_file_data[lod.start_offset + 20:lod.start_offset + 24])[0]
                orig_intra_vob = file_vob_orig - orig_higher_size
                orig_intra_fb  = file_fb_orig  - orig_higher_size
                orig_normals_size = orig_intra_fb - orig_intra_vob
                orig_abs_vob = orig_data_offset + orig_intra_vob
                norms_buf.write(orig_file_data[orig_abs_vob:orig_abs_vob + orig_normals_size])

            faces_buf = io.BytesIO()
            BME.write_triangles(faces_buf, mesh, lod_index)
            if lod.faces_end_bytes:
                faces_buf.write(lod.faces_end_bytes)

            vd = verts_buf.getvalue()
            nd = norms_buf.getvalue()
            fd = faces_buf.getvalue()

            # higher_size: sum of data_size for LODs with index -> lod_index in this mesh.
            higher_size = sum(
                mesh.lods[li].data_size
                for li in range(lod_index + 1, len(mesh.lods))
            )

            # Intra-block positions (relative to lod.data_offset).
            # These are preserved from the original and must NOT be included in new_data_size.
            intra_voa = lod.vertex_data_offset_a - higher_size

            # New intra-block positions based on actual buffer sizes.
            new_intra_vob = intra_voa + len(vd)
            new_intra_fb  = new_intra_vob + len(nd)

            # new_data_size is the only writable region (vd+nd+fd), excluding the preserved prefix bytes.
            file_lod_data_size = unpack('<I', file_data[lod.start_offset + 28:lod.start_offset + 32])[0]
            orig_data_size   = file_lod_data_size - intra_voa  # writable region in source

            # Preserve any trailing bytes after face data (e.g. fa7f sentinel padding).
            orig_trailing_size = orig_data_size - len(vd) - len(nd) - len(fd)
            if orig_trailing_size > 0:
                trailing_abs = lod.data_offset + intra_voa + len(vd) + len(nd) + len(fd)
                trailing_bytes = bytes(file_data[trailing_abs:trailing_abs + orig_trailing_size])
            else:
                trailing_bytes = b''
                orig_trailing_size = 0

            new_data_size    = len(vd) + len(nd) + len(fd) + len(trailing_bytes)
            delta = new_data_size - orig_data_size  # bytes added (negative = shrink)

            # Absolute positions in file_data
            abs_voa     = lod.data_offset + intra_voa
            new_abs_vob = lod.data_offset + new_intra_vob
            new_abs_fb  = lod.data_offset + new_intra_fb
            # Insertion point is at the end of the full block (including prefix)
            insert_at   = lod.data_offset + file_lod_data_size

            if delta > 0:
                # Growing: insert delta bytes into file_data at insert_at
                file_data[insert_at:insert_at] = b'\x00' * delta

                # Update data_offset for every LOD in every mesh whose block starts after the insertion point
                for other_mesh in asset.meshes:
                    for other_lod in other_mesh.lods:
                        if other_lod.data_offset > lod.data_offset:
                            fp = other_lod.data_offset_file_pos
                            old_val = unpack('<I', file_data[fp:fp+4])[0]
                            file_data[fp:fp+4] = pack('<I', old_val + delta)

                # Update asset.size if the edited LOD lives in the header section
                if lod.data_offset < asset_size:
                    asset_size += delta
                    file_data[4:8] = pack('<I', asset_size)

                # Update voa/vob/fb for lower-indexed LODs in this mesh.
                for li in range(0, lod_index):
                    other_lod = mesh.lods[li]
                    so = other_lod.start_offset
                    for field_off in (12, 16, 20):  # voa, vob, fb
                        old = unpack('<I', file_data[so+field_off:so+field_off+4])[0]
                        file_data[so+field_off:so+field_off+4] = pack('<I', old + delta)

            elif delta < 0:
                # Shrinking: zero the freed bytes (no structural change needed)
                shrink_start = lod.data_offset + intra_voa + new_data_size
                file_data[shrink_start:insert_at] = b'\x00' * (-delta)

            # Write the new vertex/normal/face data at their correct positions.
            abs_voa     = lod.data_offset + intra_voa
            new_abs_vob = lod.data_offset + new_intra_vob
            new_abs_fb  = lod.data_offset + new_intra_fb

            file_data[abs_voa    :abs_voa     + len(vd)] = vd
            file_data[new_abs_vob:new_abs_vob + len(nd)] = nd
            file_data[new_abs_fb :new_abs_fb  + len(fd)] = fd
            if trailing_bytes:
                trail_abs = new_abs_fb + len(fd)
                file_data[trail_abs:trail_abs + len(trailing_bytes)] = trailing_bytes

            # Patch the edited LOD's own header fields.
            so = lod.start_offset
            new_vob_for_header = higher_size + new_intra_vob
            new_fb_for_header  = higher_size + new_intra_fb
            # data_size in the header = intra_voa (prefix) + writable region
            new_file_data_size = intra_voa + new_data_size
            file_data[so +  0: so +  4] = pack('<I', new_vert_count)
            file_data[so +  4: so +  8] = pack('<I', new_index_count)
            file_data[so +  8: so + 12] = pack('<I', new_fb_for_header // 2)  # size_a
            # voa (so+12) is unchanged
            file_data[so + 16: so + 20] = pack('<I', new_vob_for_header)      # vob
            file_data[so + 20: so + 24] = pack('<I', new_fb_for_header)       # fb
            # data_offset (so+24) is unchanged for the edited LOD itself
            file_data[so + 28: so + 32] = pack('<I', new_file_data_size)      # data_size

            # Update the in-memory LOD so re-import in the same session uses correct values.
            lod.vertex_count         = new_vert_count
            lod.index_count          = new_index_count
            lod.vertex_data_offset_b = new_vob_for_header
            lod.face_block_offset    = new_fb_for_header
            lod.data_size            = new_file_data_size

            # For other LODs: update data_offset and reversed-buffer offsets
            if delta != 0:
                for other_mesh in asset.meshes:
                    for other_lod in other_mesh.lods:
                        if other_lod.data_offset > lod.data_offset:
                            # Read the patched value from file_data
                            fp = other_lod.data_offset_file_pos
                            other_lod.data_offset = unpack('<I', file_data[fp:fp+4])[0]

                for li in range(0, lod_index):
                    other_lod = mesh.lods[li]
                    other_lod_so = other_lod.start_offset
                    other_lod.vertex_data_offset_a = unpack('<I', file_data[other_lod_so+12:other_lod_so+16])[0]
                    other_lod.vertex_data_offset_b = unpack('<I', file_data[other_lod_so+16:other_lod_so+20])[0]
                    other_lod.face_block_offset    = unpack('<I', file_data[other_lod_so+20:other_lod_so+24])[0]

        with open(out_path, 'wb') as out:
            out.write(file_data)


    @staticmethod
    def _apply_header_patches(file_path: str, mesh, skeletal_mesh: SkeletalMeshAsset, operator=None):
        """
        Apply all staged header-level patches to an already-written mod file:
        bone remaps, bone additions, mesh rename, and zero-out.
        """
        # --- Bone remaps ---
        if mesh.pending_bone_remaps:
            with open(file_path, 'rb+') as f:
                mesh_bones_list = list(mesh.mesh_bones.items())
                for slot_idx, (new_skel_idx, new_matrix) in mesh.pending_bone_remaps.items():
                    if slot_idx < len(mesh.mesh_bone_file_offsets):
                        index_offset = mesh.mesh_bone_file_offsets[slot_idx]
                        matrix_offset = index_offset - 64
                        f.seek(matrix_offset)
                        f.write(pack('<16f', *new_matrix))
                        f.seek(index_offset)
                        f.write(pack('<H', new_skel_idx))
            new_mesh_bones = {}
            for slot_idx, (old_skel_idx, matrix) in enumerate(mesh_bones_list):
                remap = mesh.pending_bone_remaps.get(slot_idx)
                if remap is not None:
                    new_skel_idx, new_matrix = remap
                    new_mesh_bones[new_skel_idx] = new_matrix
                else:
                    new_mesh_bones[old_skel_idx] = matrix
            mesh.mesh_bones = new_mesh_bones
            mesh.pending_bone_remaps = {}

        # Bone additions
        if mesh.pending_bone_additions:
            n = len(mesh.pending_bone_additions)
            insert_at = mesh.bone_table_end_offset
            inserted_bytes = n * 66  # 64-byte matrix + 2-byte index per slot

            new_slot_bytes = b''
            for new_skel_idx, new_matrix in mesh.pending_bone_additions:
                new_slot_bytes += pack('<16f', *new_matrix)
                new_slot_bytes += pack('<H', new_skel_idx)

            with open(file_path, 'rb') as f:
                file_data = bytearray(f.read())
            file_data[insert_at:insert_at] = new_slot_bytes

            old_u = unpack('<H', file_data[mesh.u_count_offset:mesh.u_count_offset + 2])[0]
            file_data[mesh.u_count_offset:mesh.u_count_offset + 2] = pack('<H', old_u + n)

            for other_mesh in skeletal_mesh.meshes:
                for lod in other_mesh.lods:
                    if lod.data_offset_file_pos > insert_at:
                        field_pos = lod.data_offset_file_pos + inserted_bytes
                    else:
                        field_pos = lod.data_offset_file_pos
                    old_val = unpack('<I', file_data[field_pos:field_pos + 4])[0]
                    if old_val > 0:
                        file_data[field_pos:field_pos + 4] = pack('<I', old_val + inserted_bytes)
                    lod.data_offset += inserted_bytes
                    lod.data_offset_file_pos = field_pos

            with open(file_path, 'wb') as f:
                f.write(file_data)

            for i in range(len(mesh.mesh_bone_file_offsets)):
                if mesh.mesh_bone_file_offsets[i] >= insert_at:
                    mesh.mesh_bone_file_offsets[i] += inserted_bytes
            for si, (new_skel_idx, new_matrix) in enumerate(mesh.pending_bone_additions):
                new_index_offset = insert_at + si * 66 + 64
                mesh.mesh_bone_file_offsets.append(new_index_offset)
                mesh.mesh_bones[new_skel_idx] = new_matrix
            mesh.bone_table_end_offset = insert_at + inserted_bytes
            mesh.pending_bone_additions = []

        # Zero out all LODs
        if mesh.zeroed_out_in_session:
            with open(file_path, 'rb+') as f:
                for lod in mesh.lods:
                    for v in range(lod.vertex_count):
                        f.seek(lod.data_offset + v * mesh.vertex_stride)
                        lod.write_vertex_position(f, pos=(0.0, 0.0, 0.0),
                                                  scale=None if mesh.position_type == 1 else 1)

        # Mesh name rename
        if mesh.pending_rename_new:
            padded = (b'\x00' * (mesh.name_length - len(mesh.pending_rename_new))
                      + mesh.pending_rename_new.encode('utf-8'))
            try:
                with open(file_path, 'rb+') as f:
                    f.seek(mesh.name_offset + 2)
                    f.write(padded)
            except Exception as e:
                if operator:
                    operator.report({'ERROR'}, f"Failed to patch mesh name in mod file: {e}")

    @staticmethod
    def get_vertex_blend_indices(vertex):
        """
        Returns a dictionary of the vertex normalized, sorted and truncated 8 weights.
        """
        group_weights = {}

        # Gather bone groups
        for vg in vertex.groups:
            if vg.weight > 0.0:
                group_weights[vg.group] = vg.weight

        # Normalize
        total_weight = 0
        for k in group_weights.keys():
            total_weight += group_weights[k]
        if total_weight == 0:
            raise Exception("Vertex {v} has no weight".format(v=vertex.index))
        normalizer = 1 / total_weight
        for gw in group_weights:
            group_weights[gw] *= normalizer

        # Sort Weights
        sorted_weights = sorted(group_weights.items(), key=operator.itemgetter(1), reverse=True)

        # Truncate Weights
        trunc_weights = sorted_weights[0:8]
        return trunc_weights

    @staticmethod
    def convert_coordinate(co):
        return Vector((co[0] * -1, co[1], co[2]))

    @staticmethod
    def write_vertices(file, mesh:SkeletalMeshAsset.Mesh, lod_index=0):
        f = file
        obj = BME.find_object_by_name(mesh.name + f"_LOD{lod_index}")
        lod: SkeletalMeshAsset.Mesh.LOD = mesh.lods[lod_index]
        if obj:
            data = obj.data
            bm = bmesh.new()
            bm.from_mesh(data)
            bm.verts.ensure_lookup_table()
            stride = mesh.vertex_stride
            pos_length = 8 if mesh.position_type == 0 else 12
            print(stride, mesh.position_type)
            mesh_bones = list(mesh.mesh_bones.keys())  # skeleton bone indices in mesh-slot order

            # Map: bone name -> mesh slot index
            name_to_mesh_slot = {}
            for slot, skel_idx in enumerate(mesh_bones):
                if skel_idx < len(asset.bones):
                    name_to_mesh_slot[asset.bones[skel_idx].name] = slot

            # Map: Blender vertex group index -> mesh slot index
            vgroup_to_mesh_slot = {}
            for vg in obj.vertex_groups:
                if vg.name in name_to_mesh_slot:
                    vgroup_to_mesh_slot[vg.index] = name_to_mesh_slot[vg.name]

            # For int16 positions, read per-vertex w scale from the original source file.
            SWOMT = bpy.context.scene.SWOMT
            # Use MOD file as source if it exists, matching _write_mod_file behaviour.
            src_path = SWOMT.AssetPath
            mod_candidate = os.path.splitext(src_path)[0] + "_MOD.mmb"
            if os.path.isfile(mod_candidate):
                src_path = mod_candidate

            if mesh.position_type == 0:
                original_w = []
                # Compute the correct absolute start of the vertex array.
                # lod.data_offset is the start of the whole LOD block; the vertex
                # positions begin at vertex_data_offset_a which is stored relative
                # to the mesh block.  Using lod.data_offset + 6 directly reads from
                # the wrong place (face buffer or another LOD), producing garbage
                # scale values and causing in-game vertex stretching.
                higher_size_w = sum(
                    mesh.lods[li].data_size
                    for li in range(lod_index + 1, len(mesh.lods))
                )
                intra_voa_w = lod.vertex_data_offset_a - higher_size_w
                abs_voa_w_read = lod.data_offset + intra_voa_w
                with open(src_path, 'rb') as src:
                    src.seek(abs_voa_w_read + 6)  # skip x,y,z int16s to reach w
                    for _ in range(lod.vertex_count):
                        original_w.append(unpack('<h', src.read(2))[0])
                        src.seek(stride - 2, 1)
                if len(data.vertices) != lod.vertex_count:
                    # Replacement mesh: derive a scale from the new mesh's actual
                    # extents so every coordinate fits in [-scale, scale] without
                    # overflow or silent skipping.  Averaging the original w values
                    # is unreliable when the replacement is a different size/position.
                    if data.vertices:
                        max_coord = max(
                            max(abs(v.co.x), abs(v.co.y), abs(v.co.z))
                            for v in data.vertices
                        )
                    else:
                        max_coord = 1.0
                    fallback_scale = max(1, int(max_coord) + 1)
                    original_w = [fallback_scale] * len(data.vertices)
            else:
                original_w = [None] * len(data.vertices)

            # Read original weight+index bytes per vertex when vert count is unchanged and export_weights is unchecked.
            export_weights = bpy.context.scene.SWOMT.export_weights or _vert_count_changed()
            weight_bytes_per_vert = stride - pos_length

            # Compute abs_voa once for weight reading
            higher_size_w = sum(
                mesh.lods[li].data_size
                for li in range(lod_index + 1, len(mesh.lods))
            )
            intra_voa_w = lod.vertex_data_offset_a - higher_size_w
            abs_voa_w   = lod.data_offset + intra_voa_w

            orig_weight_bytes = None
            # Per-vertex data needed when export_weights=True and stride=32
            orig_stride32_data = None
            # Per-vertex original slot indices for the else branch (strides 12, 16, 24 etc...)
            orig_slot_indices = None

            if not export_weights and weight_bytes_per_vert > 0:
                orig_weight_bytes = []
                with open(src_path, 'rb') as src:
                    for vi in range(lod.vertex_count):
                        src.seek(abs_voa_w + vi * stride + pos_length)
                        orig_weight_bytes.append(src.read(weight_bytes_per_vert))
            elif export_weights and len(data.vertices) == lod.vertex_count:
                if stride == 32:
                    orig_stride32_data = []
                    with open(src_path, 'rb') as src:
                        for vi in range(lod.vertex_count):
                            src.seek(abs_voa_w + vi * stride + pos_length)
                            all_w = unpack('<8H', src.read(16))  # 8 uint16 weights
                            all_i = list(src.read(8))            # 8 uint8 indices
                            orig_stride32_data.append((all_w, all_i))
                elif stride not in (20, 36, 40, 44):
                    # else branch strides (12, 16 etc): wc = (stride - pos_length) / 2
                    # weights are uint8, indices are uint8
                    wc = int(weight_bytes_per_vert / 2)
                    orig_slot_indices = []
                    with open(src_path, 'rb') as src:
                        for vi in range(lod.vertex_count):
                            src.seek(abs_voa_w + vi * stride + pos_length + wc)
                            orig_slot_indices.append(list(src.read(wc)))

            for vi, v in enumerate(bm.verts):
                stride_start = f.tell()

                # Write position
                lod.write_vertex_position(
                    f,
                    pos=BME.convert_coordinate(v.co),
                    scale=original_w[vi],
                )
                f.seek(stride_start + pos_length)

                # Write bone weights - use original bytes if vert count is unchanged
                if orig_weight_bytes is not None:
                    f.write(orig_weight_bytes[vi])
                    f.seek(stride_start + stride)
                    continue

                # Gather bone weights
                vertex = data.vertices[v.index]
                raw_weights = []
                for vge in vertex.groups:
                    slot = vgroup_to_mesh_slot.get(vge.group)
                    if slot is not None and vge.weight > 0.0:
                        raw_weights.append((slot, vge.weight))
                raw_weights.sort(key=lambda x: x[1], reverse=True)

                # Write bone weights
                if stride == 20:
                    max_bones = 4
                    sw = raw_weights[:max_bones]
                    for _, w in sw:
                        f.write(bp.uint16(max(0, min(int(round(w * 32767)), 32767))))
                    for _ in range(max_bones - len(sw)):
                        f.write(bp.uint16(0))
                    for s, _ in sw:
                        f.write(bp.uint8(s))
                    for _ in range(max_bones - len(sw)):
                        f.write(bp.uint8(0))

                elif stride == 32:
                    vert_count_matches = len(data.vertices) == lod.vertex_count
                    if orig_stride32_data is not None and vert_count_matches:
                        # Vert count unchanged: preserve exact slot order from source.
                        orig_all_w, orig_all_i = orig_stride32_data[vi]
                        weight_slots = range(8) if sum(orig_all_w) == 32767 else range(4)
                        remaining = {s: w for s, w in raw_weights}
                        # Write weights for active slots from Blender groups
                        for slot in weight_slots:
                            bone = orig_all_i[slot]
                            w = remaining.pop(bone, 0.0)
                            f.write(bp.uint16(max(0, min(int(round(w * 32767)), 32767))))
                        # Remaining uint16 slots written verbatim (secondary index data or zeros)
                        for slot in range(len(weight_slots), 8):
                            f.write(bp.uint16(orig_all_w[slot]))
                        # All 8 uint8 indices verbatim
                        for bone in orig_all_i:
                            f.write(bp.uint8(bone))
                    else:
                        # Vert count changed or no source data: write up to 8 slots by weight desc.
                        sw = raw_weights[:8]
                        for _, w in sw:
                            f.write(bp.uint16(max(0, min(int(round(w * 32767)), 32767))))
                        for _ in range(8 - len(sw)):
                            f.write(bp.uint16(0))
                        for s, _ in sw:
                            f.write(bp.uint8(s))
                        for _ in range(8 - len(sw)):
                            f.write(bp.uint8(0))

                elif stride == 36:
                    max_bones = 8
                    sw = raw_weights[:max_bones]
                    for _, w in sw:
                        f.write(bp.uint16(max(0, min(int(round(w * 32767)), 32767))))
                    for _ in range(max_bones - len(sw)):
                        f.write(bp.uint16(0))
                    for s, _ in sw:
                        f.write(bp.uint8(s))
                    for _ in range(max_bones - len(sw)):
                        f.write(bp.uint8(0))

                elif stride == 40:
                    max_bones = 8
                    sw = raw_weights[:max_bones]
                    for _, w in sw:
                        f.write(bp.uint16(max(0, min(int(round(w * 32767)), 32767))))
                    for _ in range(max_bones - len(sw)):
                        f.write(bp.uint16(0))
                    for s, _ in sw:
                        f.write(bp.uint16(s))
                    for _ in range(max_bones - len(sw)):
                        f.write(bp.uint16(0))

                elif stride == 44:
                    max_bones = 12
                    sw = raw_weights[:max_bones]
                    for i in range(max_bones):
                        if i < len(sw):
                            f.write(bp.uint8(max(0, min(int(round(sw[i][1] * 255)), 255))))
                        else:
                            f.write(bp.uint8(0))
                    for i in range(max_bones):
                        if i < len(sw):
                            f.write(bp.uint16(sw[i][0]))
                        else:
                            f.write(bp.uint16(0))

                else:
                    wc = int((stride - pos_length) / 2)
                    sw = raw_weights[:wc]
                    if orig_slot_indices is not None:
                        # Preserve original slot structure
                        remaining = {s: w for s, w in raw_weights}
                        slot_idxs = orig_slot_indices[vi]
                        for slot_bone in slot_idxs:
                            w = remaining.pop(slot_bone, 0.0)
                            f.write(bp.uint8(max(0, min(int(round(w * 255)), 255))))
                        for slot_bone in slot_idxs:
                            f.write(bp.uint8(slot_bone))
                    else:
                        # No original indices available - write Blender weights
                        for i in range(wc):
                            if i < len(sw):
                                f.write(bp.uint8(max(0, min(int(round(sw[i][1] * 255)), 255))))
                            else:
                                f.write(bp.uint8(0))
                        for i in range(wc):
                            if i < len(sw):
                                f.write(bp.uint8(sw[i][0]))
                            else:
                                f.write(bp.uint8(0))

                # Advance to next vertex slot
                f.seek(stride_start + stride)
            bm.free()



    @staticmethod
    def write_normals(file, mesh:SkeletalMeshAsset.Mesh, lod_index=0):
        f = file
        obj = BME.find_object_by_name(mesh.name + f"_LOD{lod_index}")
        lod: SkeletalMeshAsset.Mesh.LOD = mesh.lods[lod_index]
        if obj:
            data = obj.data

            # Save custom split normals before calc_tangents() - it resets loop normals
            # to geometry-derived values, discarding any custom normals on the mesh.
            saved_loop_normals = None
            if data.has_custom_normals:
                saved_loop_normals = [l.normal.copy() for l in data.loops]

            bm = bmesh.new()
            bm.from_mesh(data)
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            data.loops.data.calc_tangents()

            # Restore custom normals after calc_tangents() reset them
            if saved_loop_normals is not None:
                data.normals_split_custom_set(saved_loop_normals)

            # Gather per-vertex normal, tangent, bitangent-sign, and UV data from loops
            NTB = [(Vector((1.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0)), 1.0)] * len(data.vertices)
            UVs = [[[(0.0, 0.0)] * len(data.vertices)] for _ in range(mesh.uv_count)]
            # Build per-UV-layer data
            uv_layers = bm.loops.layers.uv
            all_uvs = [[(0.0, 0.0)] * len(data.vertices) for _ in range(mesh.uv_count)]

            # Detect UV Convention
            uv_centred_u = []
            uv_centred_v = []
            for ui in range(mesh.uv_count):
                # Prefer attributes
                cu_attr = data.attributes.get(f'mmb_uv{ui}_centred_u')
                cv_attr = data.attributes.get(f'mmb_uv{ui}_centred_v')
                if cu_attr is not None and cv_attr is not None:
                    uv_centred_u.append(bool(cu_attr.data[0].value))
                    uv_centred_v.append(bool(cv_attr.data[0].value))
                elif ui < len(uv_layers):
                    # Fallback
                    u_vals = [loop[uv_layers[ui]].uv[0] for bface in bm.faces for loop in bface.loops]
                    v_vals = [loop[uv_layers[ui]].uv[1] for bface in bm.faces for loop in bface.loops]
                    u_min, u_max = min(u_vals), max(u_vals)
                    v_min, v_max = min(v_vals), max(v_vals)
                    uv_centred_u.append(u_min < -0.05 and abs((u_min + u_max) / 2.0) < 0.15)
                    uv_centred_v.append(v_min < -0.05 and abs((v_min + v_max) / 2.0 - 0.5) < 0.15)
                else:
                    uv_centred_u.append(False)
                    uv_centred_v.append(False)

            for bface in bm.faces:
                for loop in bface.loops:
                    vi = loop.vert.index
                    for ui in range(mesh.uv_count):
                        if ui < len(uv_layers):
                            u_bl = loop[uv_layers[ui]].uv[0]
                            v_bl = loop[uv_layers[ui]].uv[1]
                            u = u_bl * 2.0 - 1.0 if uv_centred_u[ui] else u_bl
                            v = v_bl - 0.5 if uv_centred_v[ui] else 1 - v_bl
                            all_uvs[ui][vi] = (u, v)

            for l in data.loops:
                flip = -1.0 if l.bitangent_sign == -1 else 1.0
                NTB[l.vertex_index] = (l.normal, l.tangent, flip)

            color_layer = bm.verts.layers.float_color.get("Color_0")

            if mesh.normal_type == 0:
                # int8_norm format:
                #   4 bytes: normal as (x, y, z) int8_norm + w int8 sign (always -1)
                #   4 bytes: tangent packed as X10Y10Z10W2
                #   color_count * 4 bytes: vertex colors as uint8_norm RGBA
                #   uv_count * 4 bytes: UVs as pairs of int16_norm

                # Read original nw and tangent bytes from source for all vertices.
                SWOMT = bpy.context.scene.SWOMT
                # Always read nw and tangent bytes from the original asset file.
                higher_size = sum(
                    mesh.lods[li].data_size
                    for li in range(lod_index + 1, len(mesh.lods))
                )
                vert_count_unchanged = len(data.vertices) == lod.vertex_count
                with open(SWOMT.AssetPath, 'rb') as orig_src:
                    orig_file_bytes = orig_src.read()
                orig_do  = unpack('<I', orig_file_bytes[lod.data_offset_file_pos:lod.data_offset_file_pos+4])[0]
                orig_vob = unpack('<I', orig_file_bytes[lod.start_offset+16:lod.start_offset+20])[0]
                orig_voa = unpack('<I', orig_file_bytes[lod.start_offset+12:lod.start_offset+16])[0]
                abs_vob_src = orig_do + (orig_vob - higher_size)
                abs_voa_src = orig_do + (orig_voa - higher_size)
                orig_vc_src = unpack('<I', orig_file_bytes[lod.start_offset:lod.start_offset+4])[0]
                ns = mesh.normals_stride
                vs = mesh.vertex_stride

                # Read all original nw, tangent, and color bytes.
                orig_nw       = []
                orig_tangents = []
                orig_colors   = []  # list of raw 4*color_count byte chunks per vertex
                orig_uvs      = []  # list of raw 4*uv_count byte chunks per vertex
                orig_trailing = []  # any extra bytes after color+normal+tangent+UVs

                # color(4*cc) | normal(4) | tangent(4) | UV
                cc = mesh.color_count
                uv_off_in_stride = 4 * cc + 4 + 4
                written_per_vert = 4 * cc + 4 + 4 + 4 * mesh.uv_count
                trailing_per_vert = ns - written_per_vert
                for ni in range(orig_vc_src):
                    off = abs_vob_src + ni * ns
                    normal_off = off + 4 * cc
                    tangent_off = normal_off + 4
                    uv_off = tangent_off + 4
                    orig_nw.append(unpack('<b', orig_file_bytes[normal_off + 3:normal_off + 4])[0])
                    orig_tangents.append(orig_file_bytes[tangent_off:tangent_off + 4])
                    if mesh.color_count > 0:
                        orig_colors.append(orig_file_bytes[off:off + 4 * mesh.color_count])
                    if mesh.uv_count > 0:
                        orig_uvs.append(orig_file_bytes[uv_off:uv_off + 4 * mesh.uv_count])
                    if trailing_per_vert > 0:
                        trail_off = off + written_per_vert
                        orig_trailing.append(orig_file_bytes[trail_off:trail_off + trailing_per_vert])
                # Detect compact UVs
                uv_compact = False
                if mesh.uv_count > 0 and orig_vc_src > 0:
                    uv_compact = True
                    for ni in range(orig_vc_src):
                        raw_u = unpack('<h', orig_file_bytes[abs_vob_src + ni * ns + uv_off_in_stride: abs_vob_src + ni * ns + uv_off_in_stride + 2])[0]
                        if abs(raw_u) > 8191:
                            uv_compact = False
                            break
                print(f"[AFoPMT] {mesh.name} LOD{lod_index} (nt=0): uv_compact={uv_compact} uv_off={uv_off_in_stride} ns={ns} cc={cc} first_raw_u={unpack('<h', orig_file_bytes[abs_vob_src + uv_off_in_stride : abs_vob_src + uv_off_in_stride + 2])[0] if orig_vc_src > 0 else 'N/A'}")

                # Build per-vertex source index using mmb_vertex_order attribute if present
                orig_idx_attr = data.attributes.get("mmb_vertex_order")
                if orig_idx_attr is not None:
                    orig_idx_for_vi = {vi: orig_idx_attr.data[vi].value
                                       for vi in range(len(data.vertices))}
                else:
                    orig_idx_for_vi = None

                # Build position -> source index map as fallback for meshes
                orig_pos_to_ni = {}
                if orig_idx_for_vi is None:
                    for ni in range(orig_vc_src):
                        pos_key = orig_file_bytes[abs_voa_src + ni*vs : abs_voa_src + ni*vs + 6]
                        if pos_key not in orig_pos_to_ni:
                            orig_pos_to_ni[pos_key] = ni

                for vi, v in enumerate(data.vertices):
                    normal  = NTB[v.index][0]
                    tangent = NTB[v.index][1]
                    flip    = NTB[v.index][2]

                    # Determine source vertex index for nw/tangent lookup
                    if orig_idx_for_vi is not None:
                        # Use stored original index directly (survives separate/join)
                        orig_vi = orig_idx_for_vi.get(vi, vi)
                        src_vi = min(orig_vi, orig_vc_src - 1)
                    else:
                        # Fallback
                        src_vi = min(vi, orig_vc_src - 1)

                    read_orig_nw = vert_count_unchanged

                    # Layout: color(4*color_count) | normal(4) | tangent(4) | UV(4*uv_count)

                    # Write vertex color: preserve original bytes unless Export Vertex Colors is on.
                    if mesh.color_count > 0:
                        if SWOMT.export_vertex_colors and color_layer is not None:
                            # Export from Blender color layer
                            vertex_color = bm.verts[v.index][color_layer]
                            for c in vertex_color:
                                f.write(bp.uint8_norm(c))
                        elif orig_colors and src_vi < len(orig_colors):
                            # Preserve original bytes verbatim
                            f.write(orig_colors[src_vi])
                        else:
                            # No source - write zeros
                            f.write(b'\x00' * (4 * mesh.color_count))

                    # Write normal as int8 (scale to [-127,127], flip x)
                    def clamp_i8(val):
                        return max(-127, min(127, int(round(val * 127))))
                    f.write(bp.int8(clamp_i8(normal[0] * -1)))
                    f.write(bp.int8(clamp_i8(normal[1])))
                    f.write(bp.int8(clamp_i8(normal[2])))
                    nw_val = orig_nw[src_vi] if read_orig_nw else -1
                    f.write(bp.int8(nw_val))

                    # Write tangent
                    if orig_tangents:
                        f.write(orig_tangents[src_vi])
                    else:
                        f.write(bp.X10Y10Z10W2(tangent[0] * -1, tangent[1], tangent[2], max(0, int(flip))))

                    # Write UVs
                    if mesh.uv_count > 0:
                        if SWOMT.export_uvs:
                            for ui in range(mesh.uv_count):
                                u, v_uv = all_uvs[ui][v.index]
                                if uv_compact:
                                    f.write(bp.uv_unorm_u(max(0.0, min(1.0, u))))
                                    f.write(bp.uv_unorm_v(max(0.0, min(1.0, v_uv))))
                                else:
                                    f.write(bp.int16_norm(max(-1.0, min(1.0, u))))
                                    f.write(bp.int16_norm(max(-1.0, min(1.0, v_uv))))
                        elif orig_uvs and src_vi < len(orig_uvs):
                            f.write(orig_uvs[src_vi])
                        else:
                            f.write(b'\x00' * (4 * mesh.uv_count))

                    # Preserve any trailing bytes
                    if orig_trailing and src_vi < len(orig_trailing):
                        f.write(orig_trailing[src_vi])
                    elif trailing_per_vert > 0:
                        f.write(b'\x00' * trailing_per_vert)

            else:
                # float format (normal_type == 1 or 2).
                # normal_type 1: color(4*cc) | normal(12f) | tangent(12f) | sign(4f) | UV(4*uv)
                SWOMT = bpy.context.scene.SWOMT
                higher_size = sum(
                    mesh.lods[li].data_size
                    for li in range(lod_index + 1, len(mesh.lods))
                )
                vert_count_unchanged = len(data.vertices) == lod.vertex_count
                with open(SWOMT.AssetPath, 'rb') as orig_src:
                    orig_file_bytes = orig_src.read()
                orig_do  = unpack('<I', orig_file_bytes[lod.data_offset_file_pos:lod.data_offset_file_pos+4])[0]
                orig_vob = unpack('<I', orig_file_bytes[lod.start_offset+16:lod.start_offset+20])[0]
                abs_vob_src = orig_do + (orig_vob - higher_size)
                orig_vc_src = unpack('<I', orig_file_bytes[lod.start_offset:lod.start_offset+4])[0]
                ns = mesh.normals_stride

                orig_colors = []
                orig_uvs    = []
                # color(4*cc) | normal(12) | tangent(12) | sign(4) | UV
                color_off_in_stride = 0
                uv_off_in_stride    = 4 * mesh.color_count + 12 + 12 + 4
                # Detect compact UV encoding
                uv_compact = False
                if mesh.uv_count > 0 and orig_vc_src > 0:
                    uv_compact = True
                    for ni in range(orig_vc_src):
                        raw_u = unpack('<h', orig_file_bytes[abs_vob_src + ni*ns + uv_off_in_stride : abs_vob_src + ni*ns + uv_off_in_stride + 2])[0]
                        if abs(raw_u) > 8191:
                            uv_compact = False
                            break
                for ni in range(orig_vc_src):
                    off = abs_vob_src + ni * ns
                    if mesh.color_count > 0:
                        orig_colors.append(orig_file_bytes[off + color_off_in_stride:off + color_off_in_stride + 4 * mesh.color_count])
                    if mesh.uv_count > 0:
                        orig_uvs.append(orig_file_bytes[off + uv_off_in_stride:off + uv_off_in_stride + 4 * mesh.uv_count])

                orig_idx_attr = data.attributes.get("mmb_vertex_order")
                orig_idx_for_vi = ({vi: orig_idx_attr.data[vi].value for vi in range(len(data.vertices))}
                                   if orig_idx_attr is not None else None)

                for vi, v in enumerate(data.vertices):
                    normal  = NTB[v.index][0]
                    tangent = NTB[v.index][1]
                    v_flip  = NTB[v.index][2]
                    src_vi  = min(orig_idx_for_vi.get(vi, vi) if orig_idx_for_vi else vi, orig_vc_src - 1)

                    # Colors first
                    if mesh.color_count > 0:
                        if SWOMT.export_vertex_colors and color_layer is not None:
                            vertex_color = bm.verts[v.index][color_layer]
                            for c in vertex_color:
                                f.write(bp.uint8_norm(c))
                        elif orig_colors and src_vi < len(orig_colors):
                            f.write(orig_colors[src_vi])
                        else:
                            f.write(b'\x00' * (4 * mesh.color_count))

                    f.write(bp.float(normal[0] * -1))
                    f.write(bp.float(normal[1]))
                    f.write(bp.float(normal[2]))
                    f.write(bp.float(tangent[0] * -1))
                    f.write(bp.float(tangent[1]))
                    f.write(bp.float(tangent[2]))
                    f.write(bp.float(v_flip))

                    # UVs
                    if mesh.uv_count > 0:
                        if SWOMT.export_uvs:
                            for ui in range(mesh.uv_count):
                                u, v_uv = all_uvs[ui][v.index]
                                if uv_compact:
                                    f.write(bp.uv_unorm_u(max(0.0, min(1.0, u))))
                                    f.write(bp.uv_unorm_v(max(0.0, min(1.0, v_uv))))
                                else:
                                    f.write(bp.int16_norm(max(-1.0, min(1.0, u))))
                                    f.write(bp.int16_norm(max(-1.0, min(1.0, v_uv))))
                        elif orig_uvs and src_vi < len(orig_uvs):
                            f.write(orig_uvs[src_vi])
                        else:
                            f.write(b'\x00' * (4 * mesh.uv_count))

            bm.free()


    @staticmethod
    def write_triangles(file, mesh:SkeletalMeshAsset.Mesh, lod_index=0):
        f = file
        obj = BME.find_object_by_name(mesh.name + f"_LOD{lod_index}")
        lod: SkeletalMeshAsset.Mesh.LOD = mesh.lods[lod_index]
        if obj:
            data = obj.data
            bm = bmesh.new()
            bm.from_mesh(data)
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            for p in data.polygons:
                f.write(bp.uint16(p.vertices[0]))
                f.write(bp.uint16(p.vertices[2]))
                f.write(bp.uint16(p.vertices[1]))
            bm.free()

    @staticmethod
    def copy_previous_mesh_data(source_path, file, mesh_index=0, lod_index=0):
        sorted_lods = asset.get_sorted_lods()
        with open(source_path, 'rb') as source:
            for lod in sorted_lods:
                print(lod.data_offset, lod.data_size, lod.vertex_count)
                if lod.index == lod_index and lod.parent_mesh.index == mesh_index:
                    return
                source.seek(lod.data_offset)
                file.write(source.read(lod.data_size))

    @staticmethod
    def create_mesh_file(mesh_index=0, lod_index=-1):
        """
        Creates a BytesIO of the reverse-sorted LODs for a mesh using the edited
        Blender mesh for the given lod_index (or unedited data for all others).
        :return: Mesh object with updated LOD offset/size fields and mesh_file set.
        """
        SWOMT = bpy.context.scene.SWOMT
        file = SWOMT.AssetPath
        mesh_file = io.BytesIO()
        offset_diff = 0
        mesh = asset.meshes[mesh_index]
        modded_mesh: SkeletalMeshAsset.Mesh = copy.deepcopy(asset.meshes[mesh_index])

        with open(file, 'rb') as source:
            for lod in reversed(asset.meshes[mesh_index].lods):
                current_modded_lod = modded_mesh.lods[lod.index]
                current_modded_lod.parent_mesh = modded_mesh
                new_vertex_data_offset_a = mesh_file.tell()
                current_modded_lod.data_start = mesh_file.tell() # start of this LOD's block in mesh_file
                if lod.index == lod_index:
                    print("Edited LOD")
                    obj = BME.find_object_by_name(mesh.name + f"_LOD{lod_index}")
                    current_modded_lod.vertex_count = len(obj.data.vertices)
                    current_modded_lod.index_count = len(obj.data.polygons) * 3
                    print("New Vertex count: ", len(obj.data.vertices))
                    print("New indices count: ", len(obj.data.polygons) * 3)
                    # Vertices
                    current_modded_lod.vertex_data_offset_a = mesh_file.tell()
                    BME.write_vertices(mesh_file, mesh, lod_index)
                    if lod.vertex_end_bytes is not None:
                        mesh_file.write(lod.vertex_end_bytes)
                    # Normals
                    current_modded_lod.vertex_data_offset_b = mesh_file.tell()
                    BME.write_normals(mesh_file, mesh, lod_index)
                    if lod.normals_end_bytes is not None:
                        mesh_file.write(lod.normals_end_bytes)
                    # Indices
                    current_modded_lod.face_block_offset = mesh_file.tell()
                    BME.write_triangles(mesh_file, mesh, lod_index)
                    if lod.faces_end_bytes is not None:
                        mesh_file.write(lod.faces_end_bytes)
                else:
                    # Unedited lod - copy from source file
                    source.seek(lod.data_offset)
                    current_modded_lod.vertex_data_offset_a = new_vertex_data_offset_a
                    offset_diff = new_vertex_data_offset_a - lod.vertex_data_offset_a
                    current_modded_lod.vertex_data_offset_b = lod.vertex_data_offset_b + offset_diff
                    current_modded_lod.face_block_offset = lod.face_block_offset + offset_diff
                    mesh_file.write(source.read(lod.data_size))

                current_modded_lod.data_size = mesh_file.tell() - new_vertex_data_offset_a
                print("\n", lod.index, lod.data_offset, lod.data_size, lod.vertex_count)
                print("Vertex_data_offset_a = ", new_vertex_data_offset_a, "   was  ", lod.vertex_data_offset_a,
                      "  diff = ", offset_diff)
                print("Mesh_File data start : ", new_vertex_data_offset_a, "Data Size : ", current_modded_lod.data_size)
        modded_mesh.mesh_file = mesh_file
        return modded_mesh

asset : SkeletalMeshAsset = None
BMI = BlenderMeshImporter
BME = BlenderMeshExporter

@bpy.app.handlers.persistent
def _on_load_post(filepath, *args, **kwargs):
    """Resets the asset when a .blend file is loaded, then re-loads it from the AssetPath if the file still exists."""
    global asset
    asset = None
    try:
        for scene in bpy.data.scenes:
            path = scene.SWOMT.get("AssetPath", "")
            if not path or not os.path.isfile(path):
                continue
            try:
                with open(path, 'rb') as f:
                    sk_mesh = SkeletalMeshAsset()
                    sk_mesh.parse(f)
                    sk_mesh.name = Path(path).stem
                    asset = sk_mesh
                print(f"[AFoPMT] Loaded '{sk_mesh.name}' from '{path}'")
            except Exception as e:
                print(f"[AFoPMT] Failed to Load '{path}': {e}")
            break
    except Exception as e:
        print(f"[AFoPMT] _on_load_post error: {e}")

def _auto_load_mmb(self, context):
    path = self.AssetPath
    if not path or not os.path.isfile(path):
        return
    try:
        with open(path, 'rb') as file:
            sk_mesh = SkeletalMeshAsset()
            sk_mesh.parse(file)
            sk_mesh.name = Path(path).stem
            global asset
            asset = sk_mesh
    except Exception as e:
        print(f"MMB auto-load failed: {e}")

def _vert_count_changed():
    """Return True if any imported LOD Blender object has a different vert count than the MMB."""
    if asset is None:
        return False
    for m in asset.meshes:
        for li, lod in enumerate(m.lods):
            if lod.vertex_count == 0:
                continue
            obj_name = lod.blender_obj_name or f"{m.name}_LOD{li}"
            obj = bpy.data.objects.get(obj_name)
            if obj is not None and len(obj.data.vertices) != lod.vertex_count:
                return True
    return False

def _on_compute_normals_on_export_update(self, context):
    """Auto-enable export_normals when compute_normals_on_export is checked."""
    if self.compute_normals_on_export:
        self.export_normals = True

def _on_export_normals_update(self, context):
    """Auto-uncheck compute_normals_on_export, export_vertex_colors and export_uvs when export_normals is unchecked."""
    if not self.export_normals:
        if self.compute_normals_on_export:
            self.compute_normals_on_export = False
        if self.export_vertex_colors:
            self.export_vertex_colors = False
        if self.export_uvs:
            self.export_uvs = False

def _on_export_vertex_colors_update(self, context):
    """Auto-enable export_normals when export_vertex_colors is checked."""
    if self.export_vertex_colors:
        self.export_normals = True

def _on_export_uvs_update(self, context):
    """Auto-enable export_normals when export_uvs is checked."""
    if self.export_uvs:
        self.export_normals = True

class SWOMTSettings(bpy.types.PropertyGroup):
    AssetPath: bpy.props.StringProperty(name="Asset Path", update=_auto_load_mmb)
    mesh_expanded: bpy.props.BoolVectorProperty(size=32, default=tuple([False]*32))
    bone_slots_expanded: bpy.props.BoolVectorProperty(size=32, default=tuple([False]*32))
    compute_normals_on_export: bpy.props.BoolProperty(
        name="Compute Normals on Export",
        default=False,
        description="Recompute normals on export.",
        update=_on_compute_normals_on_export_update,
    )
    export_normals: bpy.props.BoolProperty(
        name="Export Normals",
        default=True,
        description="Write normals into the exported file. When unchecked, the original normals from the .mmb are preserved. Automatically forced on when vert count has changed.",
        update=_on_export_normals_update,
    )
    export_weights: bpy.props.BoolProperty(
        name="Export Weights",
        default=True,
        description="Write bone weights into the exported file. When unchecked, the original weights from the .mmb are preserved. Automatically forced on when vert count has changed.",
    )
    export_vertex_colors: bpy.props.BoolProperty(
        name="Export Vertex Colors",
        default=True,
        description="Write vertex colors from Blender into the exported file. When unchecked, the original vertex colors from the .mmb are preserved.",
        update=_on_export_vertex_colors_update,
    )

    export_uvs: bpy.props.BoolProperty(
        name="Export UVs",
        default=True,
        description="Write UV coordinates from Blender into the exported file. When unchecked, the original UVs from the .mmb are preserved.",
        update=_on_export_uvs_update,
    )

    export_options_expanded: bpy.props.BoolProperty(
        name="Export Options",
        default=True,
    )

# OPERATORS #
class BrowseMMBFile(bpy.types.Operator):
    """Load a .mmb file"""
    bl_idname = "object.browse_mmb_file"
    bl_label = "Import .mmb"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.mmb", options={'HIDDEN'})

    def invoke(self, context, event):
        current = context.scene.SWOMT.AssetPath
        if current:
            self.filepath = current
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        context.scene.SWOMT.AssetPath = self.filepath
        return {'FINISHED'}


class LoadMMB(bpy.types.Operator):
    """Reads data from base .mmb file"""
    bl_idname = "object.load_mmb"
    bl_label = "Load"

    def execute(self,context):
        SWOMT = context.scene.SWOMT
        with open(SWOMT.AssetPath, 'rb') as file:
            sk_mesh = SkeletalMeshAsset()
            sk_mesh.parse(file)
            sk_mesh.name = Path(SWOMT.AssetPath).stem
            global asset
            asset = sk_mesh

        return {'FINISHED'}

class ImportLOD(bpy.types.Operator):
    """Imports the given LOD"""
    bl_idname = 'object.import_lod'
    bl_label = 'Import'

    mesh_index: bpy.props.IntProperty()
    lod_index: bpy.props.IntProperty()

    @classmethod
    def poll(cls,context):
        return asset is not None

    def execute(self,context):
        sk_mesh = asset
        mesh = sk_mesh.meshes[self.mesh_index]
        lod = mesh.lods[self.lod_index]
        SWOMT = context.scene.SWOMT
        merged_mmb = get_merged_mmb(SWOMT["AssetPath"])
        obj = BMI.import_mesh(merged_mmb,
                              skeletal_mesh=sk_mesh,
                              mesh=mesh,
                              lod_index=lod.index)
        is_new_armature = bpy.data.objects.find(sk_mesh.name) == -1
        armature = BMI.find_or_create_skeleton(sk_mesh)
        BMI.parent_obj_to_armature(obj, armature, mesh)
        if is_new_armature:
            BMI.rotate_model(obj,armature)
        return {'FINISHED'}

class ExportLOD(bpy.types.Operator):
    """Exports the given LOD"""
    bl_idname = 'object.export_lod'
    bl_label = 'Export'

    mesh_index: bpy.props.IntProperty()
    lod_index: bpy.props.IntProperty()

    @classmethod
    def poll(cls,context):
        return asset is not None

    def execute(self,context):
        if bpy.context.active_object and bpy.context.active_object.mode == 'EDIT':
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        with bpy.context.temp_override(window=window, area=area):
                            bpy.ops.object.mode_set(mode='OBJECT')
                        break

        SWOMT = context.scene.SWOMT
        src_path = SWOMT.AssetPath
        mod_file = os.path.splitext(src_path)[0] + "_MOD.mmb"

        # Triangulate the Blender mesh before export so all faces are tris
        mesh = asset.meshes[self.mesh_index]
        lod  = mesh.lods[self.lod_index]
        lod_obj_name = lod.blender_obj_name or f"{mesh.name}_LOD{self.lod_index}"
        tri_obj = BME.find_object_by_name(lod_obj_name)
        if tri_obj:
            BME._triangulate_object(tri_obj, compute_normals=context.scene.SWOMT.compute_normals_on_export)

        try:
            BME._write_mod_file(
                edited_lod_index_per_mesh={self.mesh_index: self.lod_index},
                out_path=mod_file,
            )
        except Exception as e:
            self.report({'ERROR'}, f"Export failed: {e}")
            return {'CANCELLED'}

        # Apply header-level patches for every mesh
        for mesh in asset.meshes:
            BME._apply_header_patches(mod_file, mesh, asset, operator=self)

        # Apply staged file rename
        if asset.pending_file_rename_new:
            new_file = str(Path(mod_file).parent / (asset.pending_file_rename_new + '.mmb'))
            try:
                os.replace(mod_file, new_file)
            except Exception as e:
                self.report({'ERROR'}, f"Failed to rename mod file: {e}")
                return {'FINISHED'}
            self.report({'INFO'}, f"Exported -> {os.path.basename(new_file)}")

        return {'FINISHED'}

class RenameMesh(bpy.types.Operator):
    """Rename a mesh and patch its name in the .mmb file in place"""
    bl_idname = "object.rename_mesh"
    bl_label = "Rename Mesh"

    mesh_index: bpy.props.IntProperty()

    @classmethod
    def poll(cls, context):
        return asset is not None

    def invoke(self, context, event):
        old_name = asset.meshes[self.mesh_index].name
        # Re-register a Scene property with maxlen set to the original name length.
        # Blender enforces maxlen natively in the text widget — this is the only
        # reliable way to hard-lock the input length dynamically.
        bpy.types.Scene.mmb_rename_input = bpy.props.StringProperty(
            name="New Name",
            maxlen=len(old_name),
        )
        context.scene.mmb_rename_input = old_name
        # Center the dialog on screen
        context.window.cursor_warp(context.window.width // 2, context.window.height // 2)
        return context.window_manager.invoke_props_dialog(self, width=380)

    def draw(self, context):
        mesh = asset.meshes[self.mesh_index]
        layout = self.layout
        layout.label(text=f"Original: {mesh.name}")
        layout.label(text=f"Max length: {len(mesh.name)} characters")
        layout.prop(context.scene, "mmb_rename_input", text="New Name")

    def execute(self, context):
        mesh = asset.meshes[self.mesh_index]
        old_name = mesh.name
        new_name = context.scene.mmb_rename_input.strip()

        if len(new_name) == 0:
            self.report({'ERROR'}, "Name cannot be empty.")
            return {'CANCELLED'}

        if len(new_name) > len(old_name):
            self.report({'ERROR'}, f"Name too long. Max {len(old_name)} characters.")
            return {'CANCELLED'}

        # Stage the rename — original .mmb is never touched.
        # The patch is written to the _MOD copy when the user exports a LOD.
        mesh.pending_rename_new = new_name
        mesh.name = new_name

        # Rename any already-imported Blender objects so Export can still find them
        for li, lod in enumerate(mesh.lods):
            old_obj_name = lod.blender_obj_name if lod.blender_obj_name else f"{old_name}_LOD{li}"
            new_obj_name = f"{new_name}_LOD{li}"
            obj = bpy.data.objects.get(old_obj_name)
            if obj is not None:
                obj.name = new_obj_name
                if obj.data is not None:
                    obj.data.name = new_obj_name
                lod.blender_obj_name = obj.name  # use obj.name in case Blender de-duped it

        return {'FINISHED'}


_bone_matrices = None

def _load_bone_matrices():
    """
    Load bone_matrices.json from the same directory as the plugin file
    """
    global _bone_matrices
    if _bone_matrices is not None:
        return _bone_matrices
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bone_matrices.json")
    if not os.path.isfile(json_path):
        return None
    try:
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            _bone_matrices = json.load(f)
        print(f"[AFoPMT] Loaded {len(_bone_matrices)} bone matrices from bone_matrices.json")
        return _bone_matrices
    except Exception as e:
        print(f"[AFoPMT] Failed to load bone_matrices.json: {e}")
        return None


def _read_donor_matrix(donor_path, target_bone_name, mesh_name):
    """
    Search a donor MMB file for a mesh bone slot that maps to target_bone_name
    """
    try:
        donor_mmb = get_merged_mmb(donor_path)
        f = donor_mmb
        f.seek(0)
        br.string(f, 3)
        version = br.uint8(f)
        f.seek(4, 1)
        if version >= 15:
            f.seek(4, 1)

        bone_count = br.uint32(f)
        donor_bone_index = None
        for i in range(bone_count):
            nlen = unpack('<H', f.read(2))[0]
            name = br.string(f, nlen)
            f.seek(64, 1)
            f.seek(2, 1)
            if name == target_bone_name:
                donor_bone_index = i

        if donor_bone_index is None:
            return None

        mesh_count = br.uint32(f)
        fallback_matrix = None
        for mi in range(mesh_count):
            nlen = unpack('<H', f.read(2))[0]
            dname = br.string(f, nlen).rstrip('\x00')
            f.seek(48, 1); f.seek(1, 1)
            x_count = br.uint8(f); f.seek(1, 1); f.seek(4 * x_count, 1)
            u_count = br.uint16(f)
            slots = []
            for b in range(u_count):
                mat = unpack('<16f', f.read(64))
                idx = br.uint16(f)
                slots.append((idx, mat))
            for idx, mat in slots:
                if idx == donor_bone_index:
                    if dname == mesh_name:
                        return mat
                    if fallback_matrix is None:
                        fallback_matrix = mat
            if version not in (12, 13, 15, 16, 17):
                break
            if u_count > 0 and version != 12:
                f.seek(1 if version == 13 else 2, 1)
                lod_info_type = br.uint8(f)
            else:
                lod_info_type = 0 if version in (12, 13) else br.uint8(f)
            lod_count = br.uint8(f); f.seek(4, 1)
            for _ in range(lod_count):
                f.seek(36, 1)
                if lod_info_type == 2:
                    f.seek(28, 1)
            uv_count = br.uint8(f); f.seek(4 * uv_count, 1)
            if version in (16, 17):
                color_count = br.uint8(f); f.seek(4 * color_count, 1)
                f.seek(4, 1); count_c = br.uint8(f); f.seek(4 * count_c, 1)
            else:
                f.seek(4, 1); color_count = br.uint8(f); f.seek(4 * color_count, 1)
            f.seek(4, 1)
            f.seek(20 if version == 17 else 16, 1)
        return fallback_matrix
    except Exception as e:
        print(f"[AFoPMT] Donor Matrix Error: {e}")
        return None


def _bone_search_cb(self, context, edit_text):
    """filters skeleton bone names by typed text"""
    if asset is None:
        return []
    edit_lower = edit_text.lower()
    return [
        b.name
        for b in asset.bones
        if edit_lower in b.name.lower()
    ]


class RemapMeshBone(bpy.types.Operator):
    """
    Remap a mesh bone slot to a different bone.
    By default the inverse bind matrix is looked up from bone_matrices.json.
    Uncheck Auto to supply a donor MMB file instead.
    """
    bl_idname = "object.remap_mesh_bone"
    bl_label = "Remap Bone Slot"

    mesh_index: bpy.props.IntProperty()
    slot_index: bpy.props.IntProperty()

    new_bone_name: bpy.props.StringProperty(
        name="New Bone",
        description="Skeleton bone to remap this slot to",
        search=_bone_search_cb,
        search_options={'SORT'},
    )
    use_auto: bpy.props.BoolProperty(
        name="Auto",
        description="Use bone_matrices.json for the inverse bind matrix (Recommended). Uncheck to select a donor MMB file instead.",
        default=True,
    )
    donor_path: bpy.props.StringProperty(
        name="Donor MMB",
        description="An MMB file whose mesh already references the new bone",
        subtype="FILE_PATH",
    )

    @classmethod
    def poll(cls, context):
        return asset is not None

    def invoke(self, context, event):
        mesh = asset.meshes[self.mesh_index]
        mesh_bones_list = list(mesh.mesh_bones.keys())
        current_skel_idx = mesh_bones_list[self.slot_index]
        self.new_bone_name = asset.bones[current_skel_idx].name if current_skel_idx < len(asset.bones) else ""
        self.use_auto = True
        self.donor_path = ""
        context.window.cursor_warp(context.window.width // 2, context.window.height // 2)
        return context.window_manager.invoke_props_dialog(self, width=450)

    def draw(self, context):
        mesh = asset.meshes[self.mesh_index]
        mesh_bones_list = list(mesh.mesh_bones.keys())
        current_skel_idx = mesh_bones_list[self.slot_index]
        current_name = asset.bones[current_skel_idx].name if current_skel_idx < len(asset.bones) else str(current_skel_idx)
        layout = self.layout
        layout.label(text=f"Mesh: {mesh.name}   Slot: {self.slot_index}   Current: {current_name}")
        layout.separator()
        layout.prop(self, "new_bone_name", text="New Bone", icon="BONE_DATA")
        layout.separator()
        layout.prop(self, "use_auto")
        if self.use_auto:
            matrices = _load_bone_matrices()
            if matrices is None:
                layout.label(text="bone_matrices.json not found in plugin folder", icon="ERROR")
                layout.label(text="Ensure plugin was installed correctly")
        else:
            layout.label(text="Donor MMB - an MMB file whose mesh already uses the new bone:", icon="FILE")
            layout.prop(self, "donor_path", text="")

    def execute(self, context):
        mesh = asset.meshes[self.mesh_index]
        new_name = self.new_bone_name.strip()

        if not new_name:
            self.report({'ERROR'}, "Bone name cannot be empty")
            return {'CANCELLED'}

        new_skel_idx = next((i for i, b in enumerate(asset.bones) if b.name == new_name), None)
        if new_skel_idx is None:
            self.report({'ERROR'}, f"Bone '{new_name}' not found in skeleton")
            return {'CANCELLED'}

        mesh_bones_list = list(mesh.mesh_bones.keys())
        if self.slot_index >= len(mesh_bones_list):
            self.report({'ERROR'}, "Slot index out of range")
            return {'CANCELLED'}

        old_skel_idx = mesh_bones_list[self.slot_index]
        old_name = asset.bones[old_skel_idx].name if old_skel_idx < len(asset.bones) else str(old_skel_idx)

        if old_skel_idx == new_skel_idx:
            self.report({'INFO'}, "Slot already maps to that bone")
            return {'FINISHED'}

        if new_skel_idx in mesh_bones_list:
            self.report({'ERROR'}, f"Slot already has '{new_name}' at position {mesh_bones_list.index(new_skel_idx)}")
            return {'CANCELLED'}

        # Get the inverse bind matrix from either the JSON file or a donor mmb
        new_matrix = None
        if self.use_auto:
            matrices = _load_bone_matrices()
            if matrices is None:
                self.report({'ERROR'},
                    "bone_matrices.json not found in the plugin folder. "
                    "Ensure plugin was installed correctly.")
                return {'CANCELLED'}
            new_matrix = matrices.get(new_name)
            if new_matrix is None:
                self.report({'ERROR'},
                    f"'{new_name}' not found in bone_matrices.json. "
                    f"Uncheck Auto and supply a donor MMB instead.")
                return {'CANCELLED'}
            new_matrix = tuple(new_matrix)
        else:
            donor_path = self.donor_path.strip()
            if not donor_path or not os.path.isfile(donor_path):
                self.report({'ERROR'}, "Please select a valid donor MMB file.")
                return {'CANCELLED'}
            new_matrix = _read_donor_matrix(donor_path, new_name, mesh.name)
            if new_matrix is None:
                self.report({'ERROR'},
                    f"Donor file found but '{new_name}' is not referenced by any mesh in it. "
                    f"Choose a donor MMB whose mesh already uses that bone.")
                return {'CANCELLED'}

        # Stage both the skeleton index AND the matrix for export
        mesh.pending_bone_remaps[self.slot_index] = (new_skel_idx, new_matrix)

        # Update mesh.mesh_bones
        new_mesh_bones = {}
        for slot_i, (skel_idx, matrix) in enumerate(mesh.mesh_bones.items()):
            if slot_i == self.slot_index:
                new_mesh_bones[new_skel_idx] = new_matrix
            else:
                new_mesh_bones[skel_idx] = matrix
        mesh.mesh_bones = new_mesh_bones

        # Rename the vertex group on any already-imported objects
        for li, lod in enumerate(mesh.lods):
            obj_name = lod.blender_obj_name if lod.blender_obj_name else f"{mesh.name}_LOD{li}"
            obj = bpy.data.objects.get(obj_name)
            if obj is not None:
                vg = obj.vertex_groups.get(old_name)
                if vg is not None:
                    vg.name = new_name

        source = "bone_matrices.json" if self.use_auto else "donor MMB"
        self.report({'INFO'}, f"Slot {self.slot_index}: '{old_name}' to '{new_name}' via {source} (will patch on export)")
        return {'FINISHED'}

class AddMeshBone(bpy.types.Operator):
    """
    Add a new bone slot to this mesh's bone table.
    Appends a new 66-byte entry (matrix + skeleton index) to the bone table to the
    exported file and creates the matching vertex group in Blender.
    """
    bl_idname = "object.add_mesh_bone"
    bl_label = "Add Bone Slot"

    mesh_index: bpy.props.IntProperty()

    new_bone_name: bpy.props.StringProperty(
        name="New Bone",
        description="Bone to add as a new slot",
        search=_bone_search_cb,
        search_options={'SORT'},
    )
    use_auto: bpy.props.BoolProperty(
        name="Auto",
        description="Use bone_matrices.json for the inverse bind matrix (Recommended). Uncheck to select a donor MMB file instead.",
        default=True,
    )
    donor_path: bpy.props.StringProperty(
        name="Donor MMB",
        description="An MMB file whose mesh already references the new bone",
        subtype="FILE_PATH",
    )

    @classmethod
    def poll(cls, context):
        return asset is not None

    def invoke(self, context, event):
        self.new_bone_name = ""
        self.use_auto = True
        self.donor_path = ""
        context.window.cursor_warp(context.window.width // 2, context.window.height // 2)
        return context.window_manager.invoke_props_dialog(self, width=450)

    def draw(self, context):
        mesh = asset.meshes[self.mesh_index]
        layout = self.layout
        layout.label(text=f"Mesh: {mesh.name}   Current slots: {len(mesh.mesh_bones)}")
        layout.separator()
        layout.prop(self, "new_bone_name", text="New Bone", icon="BONE_DATA")
        layout.separator()
        layout.prop(self, "use_auto")
        if self.use_auto:
            matrices = _load_bone_matrices()
            if matrices is None:
                layout.label(text="bone_matrices.json not found in plugin folder.", icon="ERROR")
                layout.label(text="Ensure plugin was installed correctly")
        else:
            layout.label(text="Donor MMB - an MMB file whose mesh already uses the new bone:", icon="FILE")
            layout.prop(self, "donor_path", text="")

    def execute(self, context):
        mesh = asset.meshes[self.mesh_index]
        new_name = self.new_bone_name.strip()

        if not new_name:
            self.report({'ERROR'}, "Bone name cannot be empty.")
            return {'CANCELLED'}

        new_skel_idx = next((i for i, b in enumerate(asset.bones) if b.name == new_name), None)
        if new_skel_idx is None:
            self.report({'ERROR'}, f"Bone '{new_name}' not found in skeleton.")
            return {'CANCELLED'}

        if new_skel_idx in mesh.mesh_bones:
            self.report({'ERROR'}, f"'{new_name}' is already in this mesh's bone table.")
            return {'CANCELLED'}

        # Check it's not already staged for addition
        if any(idx == new_skel_idx for idx, _ in mesh.pending_bone_additions):
            self.report({'ERROR'}, f"'{new_name}' is already staged for addition.")
            return {'CANCELLED'}

        # Get the inverse bind matrix
        new_matrix = None
        if self.use_auto:
            matrices = _load_bone_matrices()
            if matrices is None:
                self.report({'ERROR'},
                    "bone_matrices.json not found in the plugin folder. "
                    "Ensure plugin was installed correctly.")
                return {'CANCELLED'}
            new_matrix = matrices.get(new_name)
            if new_matrix is None:
                self.report({'ERROR'},
                    f"'{new_name}' not found in bone_matrices.json. "
                    f"Uncheck Auto and supply a donor MMB instead.")
                return {'CANCELLED'}
            new_matrix = tuple(new_matrix)
        else:
            donor_path = self.donor_path.strip()
            if not donor_path or not os.path.isfile(donor_path):
                self.report({'ERROR'}, "Please select a valid donor MMB file.")
                return {'CANCELLED'}
            new_matrix = _read_donor_matrix(donor_path, new_name, mesh.name)
            if new_matrix is None:
                self.report({'ERROR'},
                    f"Donor file found but '{new_name}' is not referenced by any mesh in it. "
                    f"Choose a donor MMB whose mesh already uses that bone.")
                return {'CANCELLED'}

        # Stage the addition
        mesh.pending_bone_additions.append((new_skel_idx, new_matrix))

        # Update mesh.mesh_bones
        mesh.mesh_bones[new_skel_idx] = new_matrix

        # Create the vertex group on any already-imported Blender objects
        for li, lod in enumerate(mesh.lods):
            obj_name = lod.blender_obj_name if lod.blender_obj_name else f"{mesh.name}_LOD{li}"
            obj = bpy.data.objects.get(obj_name)
            if obj is not None and obj.vertex_groups.get(new_name) is None:
                obj.vertex_groups.new(name=new_name)

        source = "bone_matrices.json" if self.use_auto else "donor MMB"
        self.report({'INFO'}, f"'{new_name}' staged for addition via {source} (will patch on export)")
        return {'FINISHED'}


class ZeroOutMesh(bpy.types.Operator):
    """Zero out all vertex positions for all LODs of this mesh in the _MOD file"""
    bl_idname = "object.zero_out_mesh"
    bl_label = "Zero Out"

    mesh_index: bpy.props.IntProperty()

    @classmethod
    def poll(cls, context):
        return asset is not None

    def execute(self, context):
        mesh = asset.meshes[self.mesh_index]
        mesh.zeroed_out_in_session = True

        # Update any already-imported Blender objects in the viewport
        for li, lod in enumerate(mesh.lods):
            obj_name = lod.blender_obj_name if lod.blender_obj_name else f"{mesh.name}_LOD{li}"
            obj = bpy.data.objects.get(obj_name)
            if obj is not None:
                for vert in obj.data.vertices:
                    vert.co = (0.0, 0.0, 0.0)
                obj.data.update()

        self.report({'INFO'}, f"'{mesh.name}' hidden in viewport — will be zeroed in _MOD on Export.")
        return {'FINISHED'}


class RevertMesh(bpy.types.Operator):
    """Revert vertex positions to original for all LODs of this mesh in the _MOD file.
    Only works if Zero Out was used in this session."""
    bl_idname = "object.revert_mesh"
    bl_label = "Revert"

    mesh_index: bpy.props.IntProperty()

    @classmethod
    def poll(cls, context):
        return asset is not None

    def execute(self, context):
        mesh = asset.meshes[self.mesh_index]

        if not mesh.zeroed_out_in_session:
            self.report({'WARNING'}, "Mesh was not zeroed out in this session — nothing to revert.")
            return {'CANCELLED'}

        SWOMT = context.scene.SWOMT
        original_file = get_merged_mmb(SWOMT.AssetPath)

        # Update any already-imported Blender objects in the viewport
        raw_mesh_file = mesh.extract_mesh_file(original_file)
        for li, lod in enumerate(mesh.lods):
            obj_name = lod.blender_obj_name if lod.blender_obj_name else f"{mesh.name}_LOD{li}"
            obj = bpy.data.objects.get(obj_name)
            if obj is not None:
                verts = lod.get_vertex_positions(raw_mesh_file)
                for i, vert in enumerate(obj.data.vertices):
                    vert.co = (-verts[i][0], verts[i][1], verts[i][2])
                obj.data.update()

        mesh.zeroed_out_in_session = False
        self.report({'INFO'}, f"Reverted '{mesh.name}' ({len(mesh.lods)} LOD(s)) to original positions.")
        return {'FINISHED'}


class SelectMGraphObject(bpy.types.Operator):
    """Select the MGraphObject file and patch the mesh name inside it"""
    bl_idname = "object.select_mgraphobject"
    bl_label = "Select .mgraphobject"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.mgraphobject", options={'HIDDEN'})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        old_name = context.scene.get('_mmb_rename_old', '')
        new_name = context.scene.get('_mmb_rename_new', '')

        if not old_name or not new_name:
            self.report({'ERROR'}, "Rename info missing. Please rename the mesh again.")
            return {'CANCELLED'}

        try:
            with open(self.filepath, 'rb') as f:
                data = f.read()
        except Exception as e:
            self.report({'ERROR'}, f"Could not read file: {e}")
            return {'CANCELLED'}

        # The mgraphobject uses null-terminated strings stored as \x00value\x00.
        # We search for \x00MeshName_value\x00 so we only match the exact MeshName
        # field — not partial matches like g_res_torso_base_01_f which ends with
        # the same bytes as res_torso_base_01_f.
        #
        # For null-terminated strings, padding nulls go AFTER the new name
        # (new_name\x00 + padding), so the game reads the correct string and
        # stops at the first null terminator.
        old_pattern = b'\x00' + old_name.encode('utf-8') + b'\x00'
        padding = len(old_name) - len(new_name)
        new_pattern = b'\x00' + new_name.encode('utf-8') + b'\x00' + b'\x00' * padding

        count = data.count(old_pattern)
        if count == 0:
            self.report({'WARNING'}, f"MeshName '{old_name}' not found in selected file. Nothing patched.")
            return {'FINISHED'}

        patched = data.replace(old_pattern, new_pattern)

        backup_path = self.filepath + '.bak'
        try:
            import shutil
            shutil.copy2(self.filepath, backup_path)
        except Exception as e:
            self.report({'ERROR'}, f"Could not create backup: {e}")
            return {'CANCELLED'}

        try:
            with open(self.filepath, 'wb') as f:
                f.write(patched)
        except Exception as e:
            self.report({'ERROR'}, f"Could not write file: {e}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Patched {count} MeshName occurrence(s): '{old_name}' -> '{new_name}' in {os.path.basename(self.filepath)} (backup: {os.path.basename(backup_path)})")
        return {'FINISHED'}


class RenameMMBFile(bpy.types.Operator):
    """Rename the loaded .mmb file on disk and update path references in a mgraphobject"""
    bl_idname = "object.rename_mmb_file"
    bl_label = "Rename MMB File"

    @classmethod
    def poll(cls, context):
        return asset is not None

    def invoke(self, context, event):
        old_stem = Path(context.scene.SWOMT.AssetPath).stem
        bpy.types.Scene.mmb_file_rename_input = bpy.props.StringProperty(
            name="New Filename",
            maxlen=len(old_stem),
        )
        context.scene.mmb_file_rename_input = old_stem
        context.window.cursor_warp(context.window.width // 2, context.window.height // 2)
        return context.window_manager.invoke_props_dialog(self, width=380)

    def draw(self, context):
        old_stem = Path(context.scene.SWOMT.AssetPath).stem
        layout = self.layout
        layout.label(text=f"Original: {old_stem}")
        layout.label(text=f"Must be exactly {len(old_stem)} characters")
        layout.prop(context.scene, "mmb_file_rename_input", text="New Filename")

    def execute(self, context):
        SWOMT = context.scene.SWOMT
        old_path = Path(SWOMT.AssetPath)
        old_stem = old_path.stem
        new_stem = context.scene.mmb_file_rename_input.strip()

        if not new_stem:
            self.report({'ERROR'}, "Filename cannot be empty.")
            return {'CANCELLED'}
        if len(new_stem) != len(old_stem):
            self.report({'ERROR'}, f"New name must be exactly {len(old_stem)} characters.")
            return {'CANCELLED'}

        # Stage the file rename — original .mmb is never touched.
        # The _MOD copy is renamed on next export.
        asset.pending_file_rename_old = old_stem
        asset.pending_file_rename_new = new_stem

        # Stash for the mgraphobject file-reference patch (applied immediately)
        context.scene['_mmb_file_old_stem'] = old_stem
        context.scene['_mmb_file_new_stem'] = new_stem

        bpy.ops.object.select_mgraphobject_file_patch('INVOKE_DEFAULT')
        return {'FINISHED'}


class SelectMGraphObjectFilePatch(bpy.types.Operator):
    """Select the MGraphObject file and update all path references to the renamed .mmb"""
    bl_idname = "object.select_mgraphobject_file_patch"
    bl_label = "Select .mgraphobject"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.mgraphobject", options={'HIDDEN'})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        old_stem = context.scene.get('_mmb_file_old_stem', '')
        new_stem = context.scene.get('_mmb_file_new_stem', '')

        if not old_stem or not new_stem:
            self.report({'ERROR'}, "File rename info missing. Please rename the file again.")
            return {'CANCELLED'}

        try:
            with open(self.filepath, 'rb') as f:
                data = bytearray(f.read())
        except Exception as e:
            self.report({'ERROR'}, f"Could not read file: {e}")
            return {'CANCELLED'}

        # Only replace the .mmb file reference — leave .mreflex, .juice,
        # assetName, and any other occurrences of the stem untouched.
        old_b = (old_stem + '.mmb').encode('utf-8')
        new_b = (new_stem + '.mmb').encode('utf-8')

        count = data.count(old_b)
        if count == 0:
            self.report({'WARNING'}, f"'{old_stem}.mmb' not found in selected file. Nothing patched.")
            return {'FINISHED'}

        patched = data.replace(old_b, new_b)

        backup_path = self.filepath + '.bak'
        try:
            import shutil
            shutil.copy2(self.filepath, backup_path)
        except Exception as e:
            self.report({'ERROR'}, f"Could not create backup: {e}")
            return {'CANCELLED'}

        try:
            with open(self.filepath, 'wb') as f:
                f.write(patched)
        except Exception as e:
            self.report({'ERROR'}, f"Could not write file: {e}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Updated {count} .mmb reference(s): '{old_stem}.mmb' -> '{new_stem}.mmb' in {os.path.basename(self.filepath)} (backup: {os.path.basename(backup_path)})")
        return {'FINISHED'}

def _import_all_lods(context, lod_n):
    """Shared import logic for ImportAllLODNs operators."""
    sk_mesh = asset
    SWOMT = context.scene.SWOMT
    merged_mmb = get_merged_mmb(SWOMT["AssetPath"])
    is_new_armature = bpy.data.objects.find(sk_mesh.name) == -1
    armature = BMI.find_or_create_skeleton(sk_mesh)
    last_obj = None
    for mi, mesh in enumerate(sk_mesh.meshes):
        if len(mesh.lods) <= lod_n:
            continue
        lod = mesh.lods[lod_n]
        obj = BMI.import_mesh(merged_mmb,
                              skeletal_mesh=sk_mesh,
                              mesh=mesh,
                              lod_index=lod.index)
        BMI.parent_obj_to_armature(obj, armature, mesh)
        last_obj = obj
    if is_new_armature and last_obj is not None:
        BMI.rotate_model(last_obj, armature)
    return {'FINISHED'}

class ImportAllLOD0s(bpy.types.Operator):
    """Imports LOD0 for every mesh in the asset"""
    bl_idname = 'object.import_all_lod0s'
    bl_label = "Import All LOD0's"

    @classmethod
    def poll(cls, context):
        return asset is not None

    def execute(self, context):
        return _import_all_lods(context, 0)

class ImportAllLOD1s(bpy.types.Operator):
    """Imports LOD1 for every mesh in the asset"""
    bl_idname = 'object.import_all_lod1s'
    bl_label = "Import All LOD1's"

    @classmethod
    def poll(cls, context):
        return asset is not None

    def execute(self, context):
        return _import_all_lods(context, 1)

class ImportAllLOD2s(bpy.types.Operator):
    """Imports LOD2 for every mesh in the asset"""
    bl_idname = 'object.import_all_lod2s'
    bl_label = "Import All LOD2's"

    @classmethod
    def poll(cls, context):
        return asset is not None

    def execute(self, context):
        return _import_all_lods(context, 2)

class ImportAllLOD3s(bpy.types.Operator):
    """Imports LOD3 for every mesh in the asset"""
    bl_idname = 'object.import_all_lod3s'
    bl_label = "Import All LOD3's"

    @classmethod
    def poll(cls, context):
        return asset is not None

    def execute(self, context):
        return _import_all_lods(context, 3)

class ExportAllLODs(bpy.types.Operator):
    """Exports every LOD that has a Blender object in the scene"""
    bl_idname = 'object.export_all_lods'
    bl_label = "Export All LODs"

    @classmethod
    def poll(cls, context):
        return asset is not None

    def execute(self, context):
        if bpy.context.active_object and bpy.context.active_object.mode == 'EDIT':
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        with bpy.context.temp_override(window=window, area=area):
                            bpy.ops.object.mode_set(mode='OBJECT')
                        break

        SWOMT = context.scene.SWOMT
        src_path = SWOMT.AssetPath
        mod_file = os.path.splitext(src_path)[0] + "_MOD.mmb"

        # Triangulate every LOD object that exists in the scene
        for m in asset.meshes:
            for li, lod in enumerate(m.lods):
                lod_obj_name = lod.blender_obj_name or f"{m.name}_LOD{li}"
                tri_obj = BME.find_object_by_name(lod_obj_name)
                if tri_obj:
                    BME._triangulate_object(tri_obj, compute_normals=context.scene.SWOMT.compute_normals_on_export)

        # Export each LOD level in order (0 -> 3). _write_mod_file accumulates on
        # top of _MOD.mmb when it already exists, so each pass layers on top of
        # the previous one correctly.
        exported_any = False
        for lod_n in range(4):
            edited = {}
            for m in asset.meshes:
                if len(m.lods) <= lod_n:
                    continue
                lod = m.lods[lod_n]
                obj_name = lod.blender_obj_name or f"{m.name}_LOD{lod_n}"
                if BME.find_object_by_name(obj_name) is not None:
                    edited[m.index] = lod_n
            if not edited:
                continue
            try:
                BME._write_mod_file(edited_lod_index_per_mesh=edited, out_path=mod_file)
                exported_any = True
            except Exception as e:
                self.report({'ERROR'}, f"Export LOD{lod_n} failed: {e}")
                return {'CANCELLED'}

        if not exported_any:
            self.report({'WARNING'}, "No LOD objects found in scene to export.")
            return {'CANCELLED'}

        # Apply header-level patches once after all LODs are written
        for mesh in asset.meshes:
            BME._apply_header_patches(mod_file, mesh, asset, operator=self)

        if asset.pending_file_rename_new:
            new_file = str(Path(mod_file).parent / (asset.pending_file_rename_new + '.mmb'))
            try:
                os.replace(mod_file, new_file)
            except Exception as e:
                self.report({'ERROR'}, f"Failed to rename mod file: {e}")
                return {'FINISHED'}
            self.report({'INFO'}, f"Exported -> {os.path.basename(new_file)}")

        return {'FINISHED'}

# PANELS #
class SWOMTPanel(bpy.types.Panel):
    """Creates a Panel in the Scene Properties window"""
    bl_label = "AFoP Mesh Tool | Version {}.{}.{}".format(*bl_info["version"])
    bl_idname = "OBJECT_PT_swomtpanel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "scene"


    def draw(self,context):
        SWOMT = context.scene.SWOMT

        layout = self.layout

        # Update status bar
        if _update_status is None and _update_error is None:
            row = layout.row()
            row.operator("object.check_for_updates", text="Check for Updates", icon="FILE_REFRESH")
        elif _update_error:
            row = layout.row()
            row.label(text=f"Update check failed", icon="ERROR")
            row.operator("object.check_for_updates", text="Retry", icon="FILE_REFRESH")
        elif _update_status == "up_to_date":
            row = layout.row()
            row.label(text="Tool is up to date", icon="CHECKMARK")
            row.operator("object.check_for_updates", text="", icon="FILE_REFRESH")
        else:
            # update available
            box = layout.box()
            box.label(text=f"Update available: {_update_status}", icon="INFO")
            row = box.row()
            row.operator("object.apply_update", text="Update Now", icon="IMPORT")
            row.operator("object.check_for_updates", text="", icon="FILE_REFRESH")

        layout.separator()
        row = layout.row(align=True)
        row.prop(SWOMT, "AssetPath", text="Asset Path")
        row.operator("object.browse_mmb_file", text="", icon="FILE_FOLDER")

        layout.separator()
        row = layout.row()
        row.operator("object.compute_normals", text="Compute Normals", icon="NORMALS_FACE")
        row.operator("object.clear_custom_normals", text="Clear Normals", icon="REMOVE")

        layout.separator()
        row = layout.row()
        if asset:
            row.label(text=asset.name)
            row.operator("object.rename_mmb_file", text="Rename File")
            any_imported = any(
                len(m.lods) > lod_n and bpy.data.objects.get(
                    m.lods[lod_n].blender_obj_name if m.lods[lod_n].blender_obj_name else f"{m.name}_LOD{lod_n}"
                ) is not None
                for m in asset.meshes if m.lods
                for lod_n in range(len(m.lods))
            )
            gen_lods_row = layout.row()
            gen_lods_row.enabled = any_imported
            gen_lods_row.operator("object.generate_lods", text="Generate LODs", icon="MOD_DECIM")
            layout.separator()
            layout.label(text="Import", icon='IMPORT')
            imp_row = layout.row(align=True)
            for lod_n in range(4):
                any_lod_exists = any(len(m.lods) > lod_n for m in asset.meshes if m.lods)
                btn = imp_row.row(align=True)
                btn.enabled = any_lod_exists
                btn.operator(f"object.import_all_lod{lod_n}s", text=f"LOD{lod_n}")
            layout.separator()
            layout.label(text="Export", icon='EXPORT')
            exp_row = layout.row()
            exp_row.scale_y = 1.5
            exp_row.enabled = any_imported
            exp_row.operator("object.export_all_lods", text="Export All LODs")
            layout.row().operator("object.bake_parent_inverse", text="Bake Parent Inverse", icon="ORIENTATION_PARENT")

            # Export Options collapsible box
            forced = _vert_count_changed()
            box = layout.box()
            row = box.row()
            row.prop(SWOMT, "export_options_expanded",
                     icon='TRIA_DOWN' if SWOMT.export_options_expanded else 'TRIA_RIGHT',
                     icon_only=True, emboss=False)
            row.label(text="Export Options")
            if SWOMT.export_options_expanded:
                box.prop(SWOMT, "compute_normals_on_export")
                normals_row = box.row()
                if forced:
                    normals_row.enabled = False
                    normals_row.prop(SWOMT, "export_normals")
                    normals_row.label(text="", icon='LOCKED')
                else:
                    normals_row.prop(SWOMT, "export_normals")
                weights_row = box.row()
                if forced:
                    weights_row.enabled = False
                    weights_row.prop(SWOMT, "export_weights")
                    weights_row.label(text="", icon='LOCKED')
                else:
                    weights_row.prop(SWOMT, "export_weights")
                box.prop(SWOMT, "export_vertex_colors")
                box.prop(SWOMT, "export_uvs")
            if forced:
                warn_row = layout.row()
                warn_row.label(text="Tip: Transfer Weights from original mesh", icon='INFO')
            for mi, m in enumerate(asset.meshes):
                expanded = SWOMT.mesh_expanded[mi] if mi < 32 else True
                mesh_row = layout.row()
                mesh_box = mesh_row.box()
                name_split = mesh_box.split(factor=0.5)
                name_left = name_split.row()
                name_left.prop(SWOMT, "mesh_expanded", index=mi, text="",
                               icon='TRIA_DOWN' if expanded else 'TRIA_RIGHT', emboss=False)
                name_left.label(text=m.name, icon="MESH_ICOSPHERE")
                name_right = name_split.row()
                if expanded:
                    rename_op = name_right.operator("object.rename_mesh", text="Rename Mesh")
                    rename_op.mesh_index = mi
                    action_row = mesh_box.row()
                    zero_op = action_row.operator("object.zero_out_mesh", text="Remove Mesh")
                    zero_op.mesh_index = mi
                    revert_op = action_row.operator("object.revert_mesh", text="Revert Mesh")
                    revert_op.mesh_index = mi
                    for li,l in enumerate(m.lods):
                        row = mesh_box.row()
                        row.label(text = f"LOD{li} - {l.vertex_count}", icon = "CON_SIZELIKE")
                        lod_import_button = row.operator("object.import_lod")
                        lod_import_button.lod_index = li
                        lod_import_button.mesh_index = mi
                        obj_name = l.blender_obj_name if l.blender_obj_name else f"{m.name}_LOD{li}"
                        lod_obj = bpy.data.objects.get(obj_name)
                        lod_export_row = row.row()
                        lod_export_row.enabled = lod_obj is not None
                        lod_export_button = lod_export_row.operator("object.export_lod")
                        lod_export_button.lod_index = li
                        lod_export_button.mesh_index = mi
                    # Bone slot remap section
                    if m.mesh_bones:
                        bs_expanded = SWOMT.bone_slots_expanded[mi] if mi < 32 else True
                        bone_header = mesh_box.row()
                        bone_header.prop(SWOMT, "bone_slots_expanded", index=mi, text="",
                                         icon='TRIA_DOWN' if bs_expanded else 'TRIA_RIGHT', emboss=False)
                        bone_header.label(text="Bone Slots", icon="BONE_DATA")
                        add_op = bone_header.operator("object.add_mesh_bone", text="", icon="ADD")
                        add_op.mesh_index = mi
                        if bs_expanded:
                            bone_box = mesh_box.box()
                            mesh_bones_list = list(m.mesh_bones.keys())
                            added_indices = {idx for idx, _ in m.pending_bone_additions}
                            for si, skel_idx in enumerate(mesh_bones_list):
                                bone_name = asset.bones[skel_idx].name if skel_idx < len(asset.bones) else str(skel_idx)
                                is_added = skel_idx in added_indices
                                pending_remap = m.pending_bone_remaps.get(si)
                                if is_added:
                                    label = bone_name + " +"
                                elif pending_remap is not None:
                                    label = bone_name + " *"
                                else:
                                    label = bone_name
                                slot_row = bone_box.row()
                                slot_row.label(text=f"[{si}] {label}", icon="GROUP_BONE")
                                remap_op = slot_row.operator("object.remap_mesh_bone", text="Remap")
                                remap_op.mesh_index = mi
                                remap_op.slot_index = si
                                if is_added:
                                    slot_row.enabled = False

class ComputeNormals(bpy.types.Operator):
    """Compute normals for the selected mesh object (Original normals are preserved on export)"""
    bl_idname = "object.compute_normals"
    bl_label = "Compute Normals"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
                obj is not None and
                obj.type == 'MESH' and
                obj.select_get()
        )

    def execute(self, context):
        obj = context.active_object
        BME._compute_normals_for_object(obj)
        self.report({'INFO'}, f"Computed normals for {obj.name}.")
        return {'FINISHED'}

class ClearNormals(bpy.types.Operator):
    """Clear normals for the selected mesh object (Original normals are preserved on export)"""
    bl_idname = "object.clear_custom_normals"
    bl_label = "Clear Normals"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
                obj is not None and
                obj.type == 'MESH' and
                obj.select_get() and
                obj.data.has_custom_normals
        )

    def execute(self, context):
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
        return {'FINISHED'}

class CheckForUpdates(bpy.types.Operator):
    """Check GitHub for plugin updates"""
    bl_idname = "object.check_for_updates"
    bl_label = "Check for Updates"

    def execute(self, context):
        global _update_status, _update_error
        _update_status = None
        _update_error  = None
        threading.Thread(target=_check_update_thread, daemon=True).start()
        self.report({'INFO'}, "Checking for updates...")
        return {'FINISHED'}


class ApplyUpdate(bpy.types.Operator):
    bl_idname = "object.apply_update"
    bl_label = "Update Now"

    def execute(self, context):
        try:
            req = urllib.request.urlopen(_RAW_URL, timeout=30)
            new_code = req.read()
        except Exception as e:
            self.report({'ERROR'}, f"Download failed: {e}")
            return {'CANCELLED'}

        dest = os.path.abspath(__file__)
        try:
            with open(dest, 'wb') as f:
                f.write(new_code)
        except Exception as e:
            self.report({'ERROR'}, f"Could not write file: {e}")
            return {'CANCELLED'}

        # Update bone_matrices.json
        ok, err = _download_bone_json()

        global _update_status
        _update_status = None
        if not ok:
            self.report({'WARNING'}, f"Updated! Restart Blender to apply. (Failed to download bone_matrices.json): {err}")
        else:
            self.report({'INFO'}, "Updated! Restart Blender to apply.")
        return {'FINISHED'}


class GenerateLODs(bpy.types.Operator):
    """Generate LODs for all imported LOD0 meshes by decimating from LOD0"""
    bl_idname = 'object.generate_lods'
    bl_label = "Generate LODs"

    @classmethod
    def poll(cls, context):
        if asset is None:
            return False
        return any(
            bpy.data.objects.get(
                m.lods[0].blender_obj_name if m.lods and m.lods[0].blender_obj_name else f"{m.name}_LOD0"
            ) is not None
            for m in asset.meshes if m.lods
        )

    def execute(self, context):
        import bpy as _bpy
        import bmesh as _bmesh

        # Helper to find a VIEW_3D area for operator context overrides
        def get_view3d_override(obj):
            for window in _bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        return _bpy.context.temp_override(
                            window=window, area=area,
                            active_object=obj,
                            selected_objects=[obj],
                            object=obj,
                        )
            return None

        for mesh in asset.meshes:
            if not mesh.lods or len(mesh.lods) < 2:
                continue

            lod0_name = mesh.lods[0].blender_obj_name or f"{mesh.name}_LOD0"
            lod0_obj = bpy.data.objects.get(lod0_name)
            if lod0_obj is None:
                continue

            lod0_vc = len(lod0_obj.data.vertices)
            if lod0_vc == 0:
                continue

            # Find the collection(s) LOD0 belongs to (excluding the scene master collection)
            lod0_collections = [c for c in lod0_obj.users_collection
                                 if c != context.scene.collection]

            # Determine max weights per vertex from the mesh's vertex stride.
            stride = mesh.vertex_stride
            if stride == 20:
                max_weights = 4
            elif stride == 32:
                max_weights = 8   # 4 primary + 4 secondary
            elif stride == 36:
                max_weights = 8
            elif stride == 40:
                max_weights = 8
            elif stride == 44:
                max_weights = 12
            else:
                pos_length = 12 if mesh.position_type == 1 else 8
                max_weights = int((stride - pos_length) / 2)
            print(f"[AFoPMT] {mesh.name}: max weights per vertex = {max_weights} (stride={stride})")

            for lod_index in range(1, len(mesh.lods)):
                lod = mesh.lods[lod_index]
                orig_vc = lod.vertex_count
                if orig_vc == 0:
                    continue

                ratio = orig_vc / lod0_vc

                # Duplicate LOD0 as the base for this LOD
                new_obj = lod0_obj.copy()
                new_obj.data = lod0_obj.data.copy()
                new_obj.name = f"{mesh.name}_LOD{lod_index}"

                # Link into the same collections as LOD0 (or scene collection as fallback)
                if lod0_collections:
                    for col in lod0_collections:
                        col.objects.link(new_obj)
                else:
                    context.scene.collection.objects.link(new_obj)

                context.view_layer.objects.active = new_obj
                new_obj.select_set(True)

                # Merge by distance before decimating to clean up seam splits
                bm = _bmesh.new()
                bm.from_mesh(new_obj.data)
                _bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
                bm.to_mesh(new_obj.data)
                bm.free()
                new_obj.data.update()

                # Apply Decimate modifier
                dec = new_obj.modifiers.new(name="Decimate", type='DECIMATE')
                dec.ratio = max(0.001, min(1.0, ratio))

                dg = _bpy.context.evaluated_depsgraph_get()
                eval_obj = new_obj.evaluated_get(dg)
                me = _bpy.data.meshes.new_from_object(eval_obj)
                new_obj.modifiers.remove(dec)
                new_obj.data = me

                # Limit weights per vertex to match LOD0.
                override = get_view3d_override(new_obj)
                if override:
                    with override:
                        if _bpy.context.active_object and _bpy.context.active_object.mode != 'OBJECT':
                            _bpy.ops.object.mode_set(mode='OBJECT')
                        _bpy.ops.object.vertex_group_limit_total(
                            group_select_mode='ALL',
                            limit=max_weights
                        )

                # Compute smooth normals on the decimated mesh
                BME._compute_normals_for_object(new_obj)

                new_obj.select_set(False)

                # Parent to armature if LOD0 has one
                if lod0_obj.parent and lod0_obj.parent.type == 'ARMATURE':
                    new_obj.parent = lod0_obj.parent
                    new_obj.parent_type = lod0_obj.parent_type
                    new_obj.matrix_parent_inverse = lod0_obj.matrix_parent_inverse.copy()
                    for mod in lod0_obj.modifiers:
                        if mod.type == 'ARMATURE':
                            arm_mod = new_obj.modifiers.new(name=mod.name, type='ARMATURE')
                            arm_mod.object = mod.object
                            break

                self.report({'INFO'}, f"Generated {new_obj.name} ({len(new_obj.data.vertices)} verts, ratio {ratio:.3f})")

        return {'FINISHED'}

class BakeParentInverse(bpy.types.Operator):
    """Bakes the parent inverse transform into the active mesh vertices and resets it to identity.
    Select your replacement mesh, then click this before exporting.
    Equivalent to: manually clearing parent inverse, counter-rotating, and applying."""
    bl_idname = 'object.bake_parent_inverse'
    bl_label = 'Bake Parent Inverse'

    obj_name: bpy.props.StringProperty()

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        # Try obj_name first, fall back to view_layer active object
        obj = bpy.data.objects.get(self.obj_name) if self.obj_name else None
        if obj is None:
            obj = context.view_layer.objects.active
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "No mesh object is active. Select your replacement mesh first.")
            return {'CANCELLED'}

        pi = obj.matrix_parent_inverse
        if all(abs(pi[i][j] - (1.0 if i == j else 0.0)) < 1e-6 for i in range(4) for j in range(4)):
            self.report({'INFO'}, f"'{obj.name}' parent inverse is already identity — nothing to do.")
            return {'FINISHED'}

        # Apply the parent inverse into each vertex position.
        # This is the same as: Object > Apply > Parent Inverse (not available as a standard op).
        for v in obj.data.vertices:
            v.co = pi @ v.co

        # Reset matrix_parent_inverse to identity so future exports are clean.
        obj.matrix_parent_inverse = Matrix.Identity(4)

        obj.data.update()
        self.report({'INFO'}, f"Parent inverse baked into '{obj.name}' and reset to identity.")
        return {'FINISHED'}

classes=[SWOMTSettings,
         BrowseMMBFile,
         ComputeNormals,
         ClearNormals,
         ImportAllLOD0s,
         ImportAllLOD1s,
         ImportAllLOD2s,
         ImportAllLOD3s,
         ExportAllLODs,
         GenerateLODs,
         ZeroOutMesh,
         RevertMesh,
         RemapMeshBone,
         AddMeshBone,
         CheckForUpdates,
         ApplyUpdate,
         SWOMTPanel,
         LoadMMB,
         ImportLOD,
         ExportLOD,
         RenameMesh,
         SelectMGraphObject,
         RenameMMBFile,
         SelectMGraphObjectFilePatch,
         BakeParentInverse]

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.SWOMT = bpy.props.PointerProperty(type=SWOMTSettings)
    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)
    # Kick off background version check on startup
    threading.Thread(target=_check_update_thread, daemon=True).start()
    # Download bone_matrices.json if it is missing
    _check_bone_json()

def unregister():
    for c in classes:
        bpy.utils.unregister_class(c)
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)

if __name__ == "__main__":
    register()

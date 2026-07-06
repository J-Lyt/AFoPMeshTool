# Original author: AlexPo
bl_info = {
    "name": "AFoP Mesh Tool",
    "author": "JasperZebra, J-Lyt, SaintBaron",
    "location": "Scene Properties > AFoP Mesh Tool Panel",
    "version": (0, 1, 66),
    "blender": (5, 0, 0),
    "description": "Imports skeletal meshes from AFoP .mmb files. Supports versions 11-17.",
    "category": "Import-Export"
    }

import shutil
import os
import json
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
_RAW_URL = "https://raw.githubusercontent.com/J-Lyt/AFoPMeshTool/master/__init__.py"
_LOD_CFG_URL = "https://raw.githubusercontent.com/J-Lyt/AFoPMeshTool/master/lod_presets.cfg"
_LOD_CFG_FILENAME = "lod_presets.cfg"
_MMB_JSON_URL = "https://raw.githubusercontent.com/J-Lyt/AFoPMeshTool/master/mmb_lod_presets.json"
_MMB_JSON_FILENAME = "mmb_lod_presets.json"
_update_status = None   # None = not checked, "up_to_date", or "vX.X.X available"
_update_error  = None   # set if network fetch failed

def _download_data_file(url: str, filename: str):
    """Downloads a data file to the plugin folder."""
    try:
        req = urllib.request.urlopen(url, timeout=30)
        data = req.read()
        dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        with open(dest, 'wb') as f:
            f.write(data)
        return True, None
    except Exception as e:
        return False, str(e)

def _mod_file_output(src_path: str, overwrite: bool = False) -> str:
    """
    Determine the output file path with overwrite protection.

    - If overwrite is True, return 'src_path' directly (overwrite the loaded file).
    - If src_path already contains '_MOD' in its stem, return it directly. (overwrite it)
    - If '<stem>_MOD.mmb' doesn't exist, return it. If it does, increment: '<stem>_MOD1.mmb', '<stem>_MOD2.mmb', etc.
    """
    if overwrite:
        return src_path
    stem, _ = os.path.splitext(src_path)
    # If already a _MOD file, overwrite it
    if '_MOD' in os.path.basename(stem):
        return src_path
    base = stem + "_MOD.mmb"
    if not os.path.isfile(base):
        return base
    i = 1
    while True:
        candidate = f"{stem}_MOD{i}.mmb"
        if not os.path.isfile(candidate):
            return candidate
        i += 1

def _check_data_files():
    """If any data files are missing, download them in the background"""
    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    missing = []
    for url, filename in [
        (_LOD_CFG_URL,   _LOD_CFG_FILENAME),
        (_MMB_JSON_URL,  _MMB_JSON_FILENAME),
    ]:
        if not os.path.isfile(os.path.join(plugin_dir, filename)):
            missing.append((url, filename))
    if not missing:
        return
    def _download_thread():
        for url, filename in missing:
            ok, err = _download_data_file(url, filename)
            if not ok:
                print(f"[AFoPMT] Failed to download {filename}: {err}")
            else:
                print(f"[AFoPMT] {filename} downloaded successfully")
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

# --- UV encoding: binary divisor table ---
# Each UV set stores its dequantization divisor as a float32.
#
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
                self.lod_unk = 0  # v11 only: unknown uint32

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
                            # Int16 position (8b) + 2x uint8_norm weights + 2b pad + 2x uint16 bone indices
                            f.seek(8, 1)
                            w0 = br.uint8_norm(f)
                            w1 = br.uint8_norm(f)
                            f.seek(2, 1) # 2 padding bytes (always zero)
                            i0 = br.uint16(f)
                            i1 = br.uint16(f)
                            if w0 > 0.0:
                                iw[i0] = iw.get(i0, 0.0) + w0
                            if w1 > 0.0:
                                iw[i1] = iw.get(i1, 0.0) + w1
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
        uv_centred_flags = []
        for uv_index in range(mesh.uv_count):
            uvs = lod.get_uvs(raw_mesh_file,uv_index)
            uv_layer = bm.loops.layers.uv.new(f'UVMap_{uv_index}')
            # Detect UV convention
            v_vals = [uvs[i][1] for i in range(len(uvs))]
            u_vals = [uvs[i][0] for i in range(len(uvs))]
            v_min, v_max = min(v_vals), max(v_vals)
            centred_v = v_max > 0 and v_min < 0 and abs(v_min + v_max) < 0.15 and v_max < 0.6
            centred_u = False # Imported raw; game maps [-1,1] directly to texture width
            uv_centred_flags.append((centred_u, centred_v))
            for finder, face in enumerate(bm.faces):
                for lindex, loop in enumerate(face.loops):
                    v_index = loop.vert.index
                    u, v = uvs[v_index][0], uvs[v_index][1]
                    u_out = u
                    v_out = v + 0.5 if centred_v else v * -1 + 1
                    loop[uv_layer].uv = (u_out, v_out)

        bm.to_mesh(obj_data)
        bm.free()
        obj_data.update()

        if mesh.uv_count > 0:
            _enc_map = getattr(lod, 'last_uv_encodings', {})
            _enc_list = '/'.join(_enc_map.get(i, '?') for i in range(mesh.uv_count))
            print(f'UV Encoding: {mesh.uv_count} ({_enc_list})')

        # Store per-UV-layer centred flags as mesh attributes so export can
        # reverse the correct transform without re-detecting from Blender values.
        for uv_index, (centred_u, centred_v) in enumerate(uv_centred_flags):
            if centred_u:
                cu_attr = obj_data.attributes.get(f'mmb_uv{uv_index}_centred_u') or \
                          obj_data.attributes.new(name=f'mmb_uv{uv_index}_centred_u', type='INT', domain='POINT')
                for vi in range(len(cu_attr.data)):
                    cu_attr.data[vi].value = 1
            if centred_v:
                cv_attr = obj_data.attributes.get(f'mmb_uv{uv_index}_centred_v') or \
                          obj_data.attributes.new(name=f'mmb_uv{uv_index}_centred_v', type='INT', domain='POINT')
                for vi in range(len(cv_attr.data)):
                    cv_attr.data[vi].value = 1

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
        # Faces must be smooth-shaded for custom split normals to actually render smooth
        # (flat-shaded faces ignore per-vertex normals, giving harsh per-facet lighting).
        for poly in obj_data.polygons:
            poly.use_smooth = True
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
    def find_object_by_name(name=""):
        return bpy.data.objects.get(name, None)

    @staticmethod
    def _bake_parent_inverse(obj):
        # If obj has a non-identity parent inverse matrix, bake it into the duplicate mesh's
        # vertex positions and reset it to identity so the exporter reads
        # coordinates in the correct space regardless of how the object was
        # parented or re-parented in Blender.
        if obj is None or obj.type != 'MESH':
            return
        pi = obj.matrix_parent_inverse
        is_identity = all(
            abs(pi[i][j] - (1.0 if i == j else 0.0)) < 1e-6
            for i in range(4) for j in range(4)
        )
        if is_identity:
            return
        # Apply parent inverse into vertex positions
        for v in obj.data.vertices:
            v.co = pi @ v.co
        obj.data.update()

    @staticmethod
    def _triangulate_object(obj, compute_normals=False, split_seams=False):
        """
        Temporarily modifies obj.data to apply seam split, parent inverse, and triangulation.
        new_from_object is evaluated and then restores the original obj.data so mesh in Blender is untouched.
        """
        import bpy as _bpy
        import bmesh as _bmesh

        original_data = obj.data

        # Copy of the mesh data for seam split, parent inverse, and triangulation
        temp_data = original_data.copy()
        obj.data = temp_data
        try:
            bm = _bmesh.new()
            bm.from_mesh(obj.data)
            bm.edges.ensure_lookup_table()
            bm.verts.ensure_lookup_table()

            seam_edges = [e for e in bm.edges if e.seam]
            if seam_edges:
                print(f"[AFoPMT] seam_edges: True")
                _bmesh.ops.split_edges(bm, edges=seam_edges)
                bm.to_mesh(obj.data)
                obj.data.update()
            bm.free()

            BME._bake_parent_inverse(obj)

            mod = obj.modifiers.new(name="_tri_export", type='TRIANGULATE')
            mod.keep_custom_normals = True
            mod.quad_method = 'BEAUTY'
            mod.ngon_method = 'BEAUTY'

            dg = _bpy.context.evaluated_depsgraph_get()
            eval_obj = obj.evaluated_get(dg)
            me = _bpy.data.meshes.new_from_object(eval_obj)
            obj.modifiers.remove(mod)
        finally:
            # Always restore original_data and clean up temp_data
            obj.data = original_data
            _bpy.data.meshes.remove(temp_data)

        # Replace with original_data
        obj.data = me
        obj.data.update()

        if compute_normals:
            print(f"[AFoPMT] compute_normals: True")
            BME._compute_normals_for_object(obj)

    @staticmethod
    def _write_mod_file(edited_lod_index_per_mesh: dict, out_path: str, src_path: str = None):
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
        :param src_path: file to copy unedited data from. Defaults to SWOMT.AssetPath.
               ExportAllLODs passes the partially-written mod file here so each LOD
               level accumulates on top of the previous one.
        """
        SWOMT = bpy.context.scene.SWOMT
        if src_path is None:
            src_path = SWOMT.AssetPath

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
            if len(obj.data.vertices) == 0:
                continue  # zero-vert mesh - auto-zero section will zero positions in file_data

            new_vert_count  = len(obj.data.vertices)
            new_index_count = len(obj.data.polygons) * 3

            # The preserve-bytes path below copies a normals block sized for the
            # original count, so rebuild whenever this LOD's count changed.
            export_normals = SWOMT.export_normals or new_vert_count != lod.vertex_count

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
                file_vob_orig = unpack('<I', orig_file_data[lod.start_offset + lod.lod_field_offset + 16:lod.start_offset + lod.lod_field_offset + 20])[0]
                file_fb_orig = unpack('<I', orig_file_data[lod.start_offset + lod.lod_field_offset + 20:lod.start_offset + lod.lod_field_offset + 24])[0]
                orig_intra_vob = file_vob_orig - orig_higher_size
                orig_intra_fb  = file_fb_orig  - orig_higher_size
                orig_normals_size = orig_intra_fb - orig_intra_vob
                orig_abs_vob = orig_data_offset + orig_intra_vob
                norms_buf.write(orig_file_data[orig_abs_vob:orig_abs_vob + orig_normals_size])

            faces_buf = io.BytesIO()
            # Detect uint32 indices
            orig_sa_pre = unpack('<I', file_data[lod.start_offset + lod.lod_field_offset + 8: lod.start_offset + lod.lod_field_offset + 12])[0]
            orig_fb_pre = unpack('<I', file_data[lod.start_offset + lod.lod_field_offset + 20: lod.start_offset + lod.lod_field_offset + 24])[0]
            orig_uses_uint32_pre = (orig_fb_pre > 0 and orig_sa_pre == orig_fb_pre // 4)
            BME.write_triangles(faces_buf, mesh, lod_index, force_uint32=orig_uses_uint32_pre or new_vert_count > 65535)
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
            file_lod_data_size = unpack('<I', file_data[lod.start_offset + lod.lod_field_offset + 28:lod.start_offset + lod.lod_field_offset + 32])[0]
            orig_data_size   = file_lod_data_size - intra_voa  # writable region in source

            # Preserve any trailing bytes after face data (e.g. fa7f sentinel padding).
            if orig_uses_uint32_pre:
                # Must use ORIGINAL vd/nd/fd sizes to correctly locate trailing bytes.
                orig_vc = unpack('<I', file_data[lod.start_offset:lod.start_offset + 4])[0]
                orig_ic = unpack('<I', file_data[lod.start_offset + lod.lod_field_offset + 4:lod.start_offset + lod.lod_field_offset + 8])[0]
                orig_sa = unpack('<I', file_data[lod.start_offset + lod.lod_field_offset + 8:lod.start_offset + lod.lod_field_offset + 12])[0]
                orig_fb_hdr = unpack('<I', file_data[lod.start_offset + lod.lod_field_offset + 20:lod.start_offset + lod.lod_field_offset + 24])[0]
                orig_idx_size = 4 if (orig_fb_hdr > 0 and orig_sa == orig_fb_hdr // 4) else 2
                orig_vd_size = orig_vc * mesh.vertex_stride
                orig_nd_size = orig_vc * mesh.normals_stride
                orig_fd_size = orig_ic * orig_idx_size
            else:
                orig_vc = unpack('<I', file_data[lod.start_offset:lod.start_offset + 4])[0]
                orig_ic = unpack('<I', file_data[lod.start_offset + lod.lod_field_offset + 4:lod.start_offset + lod.lod_field_offset + 8])[0]
                orig_vd_size = orig_vc * mesh.vertex_stride
                orig_nd_size = orig_vc * mesh.normals_stride
                orig_fd_size = orig_ic * 2 # uint16 indices
            orig_trailing_size = orig_data_size - orig_vd_size - orig_nd_size - orig_fd_size
            if orig_trailing_size > 0:
                trailing_abs = lod.data_offset + intra_voa + orig_vd_size + orig_nd_size + orig_fd_size
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
            elif delta < 0:
                # Shrinking: remove the freed bytes from file_data
                shrink_start = lod.data_offset + intra_voa + new_data_size
                del file_data[shrink_start:insert_at]

            if delta != 0:
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

                # Update voa/vob/fb AND size_a for lower-indexed LODs in this mesh.
                # size_a must stay fb//2 (fb//4 for uint32 indices) or the mesh
                # corrupts in-game.
                for li in range(0, lod_index):
                    other_lod = mesh.lods[li]
                    so = other_lod.start_offset
                    fo = other_lod.lod_field_offset
                    old_fb = unpack('<I', file_data[so + fo + 20:so + fo + 24])[0]
                    old_sa = unpack('<I', file_data[so + fo + 8:so + fo + 12])[0]
                    for field_off in (12, 16, 20):  # voa, vob, fb
                        old = unpack('<I', file_data[so + fo + field_off:so + fo + field_off + 4])[0]
                        file_data[so + fo + field_off:so + fo + field_off + 4] = pack('<I', old + delta)
                    other_uses_uint32 = (old_fb > 0 and old_sa == old_fb // 4)
                    new_sa = (old_fb + delta) // 4 if other_uses_uint32 else (old_fb + delta) // 2
                    file_data[so + fo + 8:so + fo + 12] = pack('<I', new_sa)

                # Shift the second-section ABSOLUTE offsets - field [5] of the 28
                # extra LOD header bytes (see the LOD list notes in Mesh.parse),
                # located at data_offset field + 32. Those blocks live after all
                # primary LOD blocks, so resizing any LOD moves them. '>=' matters:
                # the first block starts exactly at insert_at when LOD0 grows.
                for other_mesh in asset.meshes:
                    if getattr(other_mesh, 'lod_info_type', 0) != 2:
                        continue
                    for other_lod in other_mesh.lods:
                        sp = other_lod.data_offset_file_pos + 32
                        old_val = unpack('<I', file_data[sp:sp + 4])[0]
                        if old_val >= insert_at:
                            file_data[sp:sp + 4] = pack('<I', old_val + delta)

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
            fo = lod.lod_field_offset
            new_vob_for_header = higher_size + new_intra_vob
            new_fb_for_header  = higher_size + new_intra_fb
            # data_size in the header = intra_voa (prefix) + writable region
            new_file_data_size = intra_voa + new_data_size
            # size_a: fb//2 for uint16, fb//4 for uint32.
            # Use uint32 if original mesh used it OR if vert count exceeds uint16 range.
            use_uint32_faces = orig_uses_uint32_pre or new_vert_count > 65535
            new_size_a = new_fb_for_header // 4 if use_uint32_faces else new_fb_for_header // 2
            file_data[so + 0: so + 4] = pack('<I', new_vert_count)
            file_data[so + fo + 4: so + fo + 8] = pack('<I', new_index_count)
            file_data[so + fo + 8: so + fo + 12] = pack('<I', new_size_a)  # size_a
            # voa (so+fo+12) is unchanged
            file_data[so + fo + 16: so + fo + 20] = pack('<I', new_vob_for_header)  # vob
            file_data[so + fo + 20: so + fo + 24] = pack('<I', new_fb_for_header)  # fb
            # data_offset (so+fo+24) is unchanged for the edited LOD itself
            file_data[so + fo + 28: so + fo + 32] = pack('<I', new_file_data_size)  # data_size

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
                    other_lod_fo = other_lod.lod_field_offset
                    other_lod.vertex_data_offset_a = unpack('<I', file_data[other_lod_so + other_lod_fo + 12:other_lod_so + other_lod_fo + 16])[0]
                    other_lod.vertex_data_offset_b = unpack('<I', file_data[other_lod_so + other_lod_fo + 16:other_lod_so + other_lod_fo + 20])[0]
                    other_lod.face_block_offset = unpack('<I', file_data[other_lod_so + other_lod_fo + 20:other_lod_so + other_lod_fo + 24])[0]

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

            # The bone table lives in the header section - grow asset.size to match.
            old_asset_size = unpack('<I', file_data[4:8])[0]
            file_data[4:8] = pack('<I', old_asset_size + inserted_bytes)

            for other_mesh in skeletal_mesh.meshes:
                for lod in other_mesh.lods:
                    if lod.data_offset_file_pos > insert_at:
                        field_pos = lod.data_offset_file_pos + inserted_bytes
                    else:
                        field_pos = lod.data_offset_file_pos
                    old_val = unpack('<I', file_data[field_pos:field_pos + 4])[0]
                    if old_val > 0:
                        file_data[field_pos:field_pos + 4] = pack('<I', old_val + inserted_bytes)
                    # Shift the second-section absolute offset (field [5], data_offset field
                    # + 32; see Mesh.parse) too.
                    if getattr(other_mesh, 'lod_info_type', 0) == 2:
                        sp = field_pos + 32
                        sec2_val = unpack('<I', file_data[sp:sp + 4])[0]
                        if sec2_val > 0:
                            file_data[sp:sp + 4] = pack('<I', sec2_val + inserted_bytes)
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
                        lod.write_vertex_position(f, pos=(0.0, 0.0, 0.0), scale=None if mesh.position_type == 1 else 1)

        # Mesh name rename
        if mesh.pending_rename_new:
            padded = (mesh.pending_rename_new.encode('utf-8')
                      + b'\x00' * (mesh.name_length - len(mesh.pending_rename_new)))
            try:
                with open(file_path, 'rb+') as f:
                    f.seek(mesh.name_offset + 2)
                    f.write(padded)
            except Exception as e:
                if operator:
                    operator.report({'ERROR'}, f"Failed to patch mesh name in mod file: {e}")

    @staticmethod
    def normalize_weights(raw_weights, max_bones):
        """
        Normalize so the weights sum to 1.0.
        Normalize before encoding to ensure encode_weights_u8/u16 only need to correct the rounding error.
        """
        sw = raw_weights[:max_bones]
        total = sum(w for _, w in sw)
        if total <= 0.0:
            return sw
        inv = 1.0 / total
        return [(s, w * inv) for s, w in sw]

    @staticmethod
    def encode_weights_u8(sw):
        """
        Encode pairs to uint8, then fix the integer sum to exactly 255.
        Applies one-unit adjustments iteratively, always picking the entry
        with the largest rounding error that can actually absorb the step
        without clamping.
        """
        encoded = [(s, int(round(w * 255))) for s, w in sw]
        diff = 255 - sum(e for _, e in encoded)
        if diff == 0:
            return encoded
        step = 1 if diff > 0 else -1
        for _ in range(abs(diff)):
            best_err = -1
            best_idx = -1
            for i, (s, e) in enumerate(encoded):
                if step == 1 and e >= 255:
                    continue
                if step == -1 and e <= 0:
                    continue
                err = abs((sw[i][1] * 255) - e)
                if err > best_err:
                    best_err = err
                    best_idx = i
            if best_idx == -1:
                break
            s, e = encoded[best_idx]
            encoded[best_idx] = (s, e + step)
        return encoded

    @staticmethod
    def encode_weights_u16(sw):
        """
        Encode pairs to uint16, then fix the integer sum to exactly 32767.
        Applies one-unit adjustments iteratively, always picking the entry
        with the largest rounding error that can actually absorb the step
        without clamping.
        """
        encoded = [(s, int(round(w * 32767))) for s, w in sw]
        diff = 32767 - sum(e for _, e in encoded)
        if diff == 0:
            return encoded
        step = 1 if diff > 0 else -1
        for _ in range(abs(diff)):
            best_err = -1
            best_idx = -1
            for i, (s, e) in enumerate(encoded):
                if step == 1 and e >= 32767:
                    continue
                if step == -1 and e <= 0:
                    continue
                err = abs((sw[i][1] * 32767) - e)
                if err > best_err:
                    best_err = err
                    best_idx = i
            if best_idx == -1:
                break
            s, e = encoded[best_idx]
            encoded[best_idx] = (s, e + step)
        return encoded

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
            mesh_bones = list(mesh.mesh_bones.keys())  # skeleton bone indices in mesh-slot order

            # Map: bone name -> mesh slot index
            name_to_mesh_slot = {}
            for slot, skel_idx in enumerate(mesh_bones):
                if skel_idx < len(asset.bones):
                    name_to_mesh_slot[asset.bones[skel_idx].name] = slot

            # Map: Blender vertex group index -> mesh slot index (matched by bone name)
            vgroup_to_mesh_slot = {}
            for vg in obj.vertex_groups:
                if vg.name in name_to_mesh_slot:
                    vgroup_to_mesh_slot[vg.index] = name_to_mesh_slot[vg.name]

            # Fallback: if no vertex groups matched by name (e.g. bone names differ between
            # models), attempt to match by the numeric suffix of the group name against the
            # mesh-slot index, then fall back to treating the group index itself as the mesh
            # slot.  This prevents weights from being silently dropped on export.
            if not vgroup_to_mesh_slot and obj.vertex_groups:
                print(f"[AFoPMT] WARNING: No vertex groups on '{obj.name}' matched bone names "
                      f"in mesh '{mesh.name}'. Attempting index-based fallback mapping.")
                import re as _re
                for vg in obj.vertex_groups:
                    # Try to parse a trailing integer from the group name (e.g. "Bone_7" -> 7)
                    m = _re.search(r'(\d+)$', vg.name)
                    if m:
                        slot = int(m.group(1))
                    else:
                        slot = vg.index
                    if slot < len(mesh_bones):
                        vgroup_to_mesh_slot[vg.index] = slot
                if vgroup_to_mesh_slot:
                    print(f"[AFoPMT] Index-based fallback produced {len(vgroup_to_mesh_slot)} "
                          f"group mappings for '{obj.name}'.")
                else:
                    print(f"[AFoPMT] ERROR: Index-based fallback also failed for '{obj.name}'. "
                          f"Weights will be zero. Check that vertex group names match skeleton "
                          f"bone names or contain a bone index suffix.")

            # For int16 positions, read per-vertex w scale from the original source file.
            # Always read from SWOMT.AssetPath using the original data_offset from the
            # file header. lod.data_offset may have been corrupted in-memory by a prior
            # mesh export in the same _write_mod_file call (ExportAllLODs exports all
            # meshes at a given LOD level in one call).
            SWOMT = bpy.context.scene.SWOMT
            src_path = SWOMT.AssetPath
            with open(src_path, 'rb') as _f:
                _f.seek(lod.data_offset_file_pos)
                orig_data_offset = unpack('<I', _f.read(4))[0]

            if mesh.position_type == 0:
                original_w = []
                higher_size_w = sum(
                    mesh.lods[li].data_size
                    for li in range(lod_index + 1, len(mesh.lods))
                )
                intra_voa_w = lod.vertex_data_offset_a - higher_size_w
                abs_voa_w_read = orig_data_offset + intra_voa_w
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

            # Read original weight+index bytes per vertex when THIS LOD's vert count
            # is unchanged (they are indexed by vertex index) and export_weights is unchecked.
            export_weights = (bpy.context.scene.SWOMT.export_weights
                              or len(data.vertices) != lod.vertex_count)
            weight_bytes_per_vert = stride - pos_length

            # Compute abs_voa once for weight reading (use orig_data_offset from file)
            higher_size_w = sum(
                mesh.lods[li].data_size
                for li in range(lod_index + 1, len(mesh.lods))
            )
            intra_voa_w = lod.vertex_data_offset_a - higher_size_w
            abs_voa_w   = orig_data_offset + intra_voa_w

            orig_weight_bytes = None
            # Per-vertex data needed when export_weights=True and stride=32
            orig_stride32_data = None
            # Per-vertex original index bytes for packed-uint8 strides (stride=16 int16, else-branch)
            # Used to preserve the original bone index in zero-weight padding slots, which the
            # game may use as secondary bone references independent of weight.
            orig_index_bytes = None
            # Detect stride=32 layout variant
            stride32_use_u16 = stride == 32 and len(mesh.mesh_bones) > 256

            # Layout C: 12x uint8 weights, 12x uint8 indices.
            stride32_layout_c = False
            if stride == 32 and not stride32_use_u16 and lod.vertex_count > 0:
                _n_slots = len(mesh.mesh_bones)
                with open(src_path, 'rb') as _src_lc:
                    _src_lc.seek(abs_voa_w + pos_length)
                    _peek24 = _src_lc.read(24)
                _a_w_check = unpack('<8H', _peek24[:16])
                if sum(_a_w_check) != 32767:
                    _c_i_check = list(_peek24[12:24])
                    if _n_slots <= 256 and all(0 <= x < _n_slots for x in _c_i_check):
                        stride32_layout_c = True

            if not export_weights and weight_bytes_per_vert > 0:
                orig_weight_bytes = []
                with open(src_path, 'rb') as src:
                    for vi in range(lod.vertex_count):
                        src.seek(abs_voa_w + vi * stride + pos_length)
                        orig_weight_bytes.append(src.read(weight_bytes_per_vert))
            elif export_weights:
                # Read the original weight/index data from the source file for stride=32
                # meshes, where the layout variant (A/B/C) and active slot count must be
                # preserved. Other strides always write weights sorted by descending weight.
                if stride == 32 and lod.vertex_count > 0:
                    orig_stride32_data = []
                    with open(src_path, 'rb') as src:
                        for vi in range(lod.vertex_count):
                            src.seek(abs_voa_w + vi * stride + pos_length)
                            if stride32_use_u16:
                                # Layout B
                                all_w = list(src.read(6))
                                src.seek(2, 1) # skip 2 padding bytes
                                all_i = list(unpack('<6H', src.read(12)))
                            elif stride32_layout_c:
                                # Layout C
                                all_w = list(src.read(12))
                                all_i = list(src.read(12))
                            else:
                                # Layout A
                                all_w = unpack('<8H', src.read(16))
                                all_i = list(src.read(8))
                            orig_stride32_data.append((all_w, all_i))
                elif stride not in (12, 20, 32, 36, 40, 44) and lod.vertex_count > 0:
                    # Packed uint8 strides (stride=16 int16, stride=24, and any other else-branch
                    # strides): read the original index bytes so zero-weight padding slots can
                    # preserve the original bone index rather than being zeroed out.
                    wc_idx = int(weight_bytes_per_vert / 2)
                    orig_index_bytes = []
                    with open(src_path, 'rb') as src:
                        for vi in range(lod.vertex_count):
                            src.seek(abs_voa_w + vi * stride + pos_length + wc_idx)
                            orig_index_bytes.append(src.read(wc_idx))

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
                raw_weights = []
                if weight_bytes_per_vert > 0:
                    vertex = data.vertices[v.index]
                    for vge in vertex.groups:
                        slot = vgroup_to_mesh_slot.get(vge.group)
                        if slot is not None and vge.weight > 0.0:
                            raw_weights.append((slot, vge.weight))
                    raw_weights.sort(key=lambda x: x[1], reverse=True)

                    if not raw_weights:
                        raise ValueError(
                            f"'{mesh.name}' LOD{lod_index} has vertices with no bone weights. "
                            f"Assign bone weights to all vertices before exporting."
                        )

                # Write bone weights
                #
                # *** If a new `elif stride == N:` is added to the branch below, or an
                # existing stride's index width (bp.uint8 vs bp.uint16) is changed,
                # update '_UINT8_INDEX_LIMITED_STRIDES/_UINT16_NON_LIMITED_STRIDES' to match.
                # Otherwise, the 256-slot-reuse logic in Add Bone Slots will stop catching meshes that need it. ***
                if weight_bytes_per_vert == 0:
                    pass
                elif stride == 12:
                    # 4x uint8 bone slot indices, no weight bytes - Weight 1.0 on first index
                    for _ in range(4):
                        f.write(bp.uint8(raw_weights[0][0] if raw_weights else 0))

                elif stride == 16:
                    if pos_length == 12:
                        # Float XYZ position (12b) + 1x uint8 bone slot index + 3b padding.
                        # Same as stride=12 (weight 1.0 on first index) but with float positions.
                        slot = raw_weights[0][0] if raw_weights else 0
                        f.write(bp.uint8(slot))
                        f.write(b'\x00\x00\x00')  # 3 padding bytes
                    else:
                        # Int16 position (8b) + 4x uint8_norm weights + 4x uint8 indices
                        wc = 4
                        sw = BME.normalize_weights(raw_weights, wc)
                        enc = BME.encode_weights_u8(sw)
                        for i in range(wc):
                            f.write(bp.uint8(enc[i][1] if i < len(enc) else 0))
                        orig_idx = orig_index_bytes[vi] if (orig_index_bytes is not None and vi < len(orig_index_bytes)) else None
                        for i in range(wc):
                            if i < len(enc):
                                f.write(bp.uint8(enc[i][0]))
                            else:
                                f.write(bp.uint8(orig_idx[i] if orig_idx is not None else 0))

                elif stride == 20:
                    max_bones = 4
                    sw = BME.normalize_weights(raw_weights, max_bones)
                    enc = BME.encode_weights_u16(sw)
                    for _, e in enc:
                        f.write(bp.uint16(e))
                    for _ in range(max_bones - len(enc)):
                        f.write(bp.uint16(0))
                    for s, _ in enc:
                        f.write(bp.uint8(s))
                    for _ in range(max_bones - len(enc)):
                        f.write(bp.uint8(0))

                elif stride == 32:
                    vert_count_matches = len(data.vertices) == lod.vertex_count
                    if orig_stride32_data is not None and vert_count_matches and vi < len(orig_stride32_data):
                        orig_all_w, orig_all_i = orig_stride32_data[vi]
                        if stride32_use_u16:
                            # Layout B
                            remaining = {s: w for s, w in BME.normalize_weights(raw_weights, 6)}
                            # fill zero-weight slots with any added-bone weights that original indices don't cover.
                            all_i = list(orig_all_i)
                            extra = sorted(remaining.keys() - set(all_i),
                                           key=lambda s: remaining[s], reverse=True)
                            for k in range(6):
                                if all_i[k] not in remaining and extra:
                                    all_i[k] = extra.pop(0)
                            # Pre-encode all 6 weights as a batch to fix rounding sum
                            ordered = [(all_i[k], remaining.pop(all_i[k], 0.0)) for k in range(6)]
                            enc = BME.encode_weights_u8(ordered)
                            for _, e in enc:
                                f.write(bp.uint8(e))
                            f.write(b'\x00\x00') # 2 padding bytes
                            # Write 6 uint16 indices
                            for idx in all_i:
                                f.write(bp.uint16(idx))
                        elif stride32_layout_c:
                            # Layout C
                            remaining = {s: w for s, w in BME.normalize_weights(raw_weights, 12)}
                            all_i = list(orig_all_i)
                            extra = sorted(remaining.keys() - set(all_i),
                                           key=lambda s: remaining[s], reverse=True)
                            for k in range(12):
                                if all_i[k] not in remaining and extra:
                                    all_i[k] = extra.pop(0)
                            # Pre-encode all 12 weights as a batch to fix rounding sum
                            ordered = [(all_i[k], remaining.pop(all_i[k], 0.0)) for k in range(12)]
                            enc = BME.encode_weights_u8(ordered)
                            for _, e in enc:
                                f.write(bp.uint8(e))
                            for bone in all_i:
                                f.write(bp.uint8(bone))
                        else:
                            # Layout A
                            weight_slots = range(8) if sum(orig_all_w) == 32767 else range(4)
                            remaining = {s: w for s, w in BME.normalize_weights(raw_weights, len(weight_slots))}
                            # Inject added-bone weights into zero-weight positions.
                            all_i = list(orig_all_i)
                            empty_slots = [k for k in weight_slots if orig_all_w[k] == 0]
                            extra = sorted(remaining.keys() - set(all_i),
                                           key=lambda s: remaining[s], reverse=True)
                            for k in empty_slots:
                                if not extra:
                                    break
                                all_i[k] = extra.pop(0)
                            # Pre-encode active weight slots as a batch to fix rounding sum
                            ordered = [(all_i[slot], remaining.pop(all_i[slot], 0.0)) for slot in weight_slots]
                            enc = BME.encode_weights_u16(ordered)
                            for _, e in enc:
                                f.write(bp.uint16(e))
                            for slot in range(len(weight_slots), 8):
                                f.write(bp.uint16(orig_all_w[slot]))
                            for bone in all_i:
                                f.write(bp.uint8(bone))
                    else:
                        # Vert count changed: write by weight desc
                        if stride32_use_u16:
                            # Layout B
                            sw = BME.normalize_weights(raw_weights, 6)
                            enc = BME.encode_weights_u8(sw)
                            for _, e in enc:
                                f.write(bp.uint8(e))
                            for _ in range(6 - len(enc)):
                                f.write(bp.uint8(0))
                            f.write(b'\x00\x00')
                            for s, _ in sw[:6]:
                                f.write(bp.uint16(s))
                            for _ in range(6 - min(len(sw), 6)):
                                f.write(bp.uint16(0))
                        elif stride32_layout_c:
                            # Layout C
                            sw = BME.normalize_weights(raw_weights, 12)
                            enc = BME.encode_weights_u8(sw)
                            for _, e in enc:
                                f.write(bp.uint8(e))
                            for _ in range(12 - len(enc)):
                                f.write(bp.uint8(0))
                            for s, _ in sw:
                                f.write(bp.uint8(s))
                            for _ in range(12 - len(sw)):
                                f.write(bp.uint8(0))
                        else:
                            # Layout A
                            sw = BME.normalize_weights(raw_weights, 8)
                            enc = BME.encode_weights_u16(sw)
                            for _, e in enc:
                                f.write(bp.uint16(e))
                            for _ in range(8 - len(enc)):
                                f.write(bp.uint16(0))
                            for s, _ in sw:
                                f.write(bp.uint8(s))
                            for _ in range(8 - len(sw)):
                                f.write(bp.uint8(0))

                elif stride == 36:
                    max_bones = 8
                    sw = BME.normalize_weights(raw_weights, max_bones)
                    enc = BME.encode_weights_u16(sw)
                    for _, e in enc:
                        f.write(bp.uint16(e))
                    for _ in range(max_bones - len(enc)):
                        f.write(bp.uint16(0))
                    for s, _ in sw:
                        f.write(bp.uint8(s))
                    for _ in range(max_bones - len(sw)):
                        f.write(bp.uint8(0))

                elif stride == 40:
                    max_bones = 8
                    sw = BME.normalize_weights(raw_weights, max_bones)
                    enc = BME.encode_weights_u16(sw)
                    for _, e in enc:
                        f.write(bp.uint16(e))
                    for _ in range(max_bones - len(enc)):
                        f.write(bp.uint16(0))
                    for s, _ in sw:
                        f.write(bp.uint16(s))
                    for _ in range(max_bones - len(sw)):
                        f.write(bp.uint16(0))

                elif stride == 44:
                    max_bones = 12
                    sw = BME.normalize_weights(raw_weights, max_bones)
                    enc = BME.encode_weights_u8(sw)
                    for i in range(max_bones):
                        f.write(bp.uint8(enc[i][1] if i < len(enc) else 0))
                    for i in range(max_bones):
                        f.write(bp.uint16(enc[i][0] if i < len(enc) else 0))

                else:
                    wc = int((stride - pos_length) / 2)
                    sw = BME.normalize_weights(raw_weights, wc)
                    enc = BME.encode_weights_u8(sw)
                    for i in range(wc):
                        f.write(bp.uint8(enc[i][1] if i < len(enc) else 0))
                    orig_idx = orig_index_bytes[vi] if (orig_index_bytes is not None and vi < len(orig_index_bytes)) else None
                    for i in range(wc):
                        if i < len(enc):
                            f.write(bp.uint8(enc[i][0]))
                        else:
                            f.write(bp.uint8(orig_idx[i] if orig_idx is not None else 0))

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
            all_uvs = [[(0.0, 0.0)] * len(data.vertices) for _ in range(mesh.uv_count)]

            # Detect UV Convention
            uv_centred_u = []
            uv_centred_v = []
            for ui in range(mesh.uv_count):
                # Prefer attributes
                cu_attr = data.attributes.get(f'mmb_uv{ui}_centred_u')
                cv_attr = data.attributes.get(f'mmb_uv{ui}_centred_v')
                if cu_attr is not None:
                    uv_centred_u.append(bool(cu_attr.data[0].value))
                elif ui < len(data.uv_layers):
                    uv_centred_u.append(False)
                else:
                    uv_centred_u.append(False)
                if cv_attr is not None:
                    uv_centred_v.append(bool(cv_attr.data[0].value))
                elif ui < len(data.uv_layers):
                    v_vals = [data.uv_layers[ui].data[li].uv[1] for li in range(len(data.loops))]
                    v_min, v_max = min(v_vals), max(v_vals)
                    uv_centred_v.append(v_min < -0.05 and abs((v_min + v_max) / 2.0 - 0.5) < 0.15)
                else:
                    uv_centred_v.append(False)

            uv_accum = [[None] * len(data.vertices) for _ in range(mesh.uv_count)]

            # For vertices shared between faces with different UVs (seams), average all loop UV values
            for ui in range(mesh.uv_count):
                if ui >= len(data.uv_layers):
                    continue
                uv_data = data.uv_layers[ui].data
                for li, loop in enumerate(data.loops):
                    vi = loop.vertex_index
                    u_bl, v_bl = uv_data[li].uv
                    u = u_bl
                    v = v_bl - 0.5 if uv_centred_v[ui] else 1 - v_bl
                    if uv_accum[ui][vi] is None:
                        uv_accum[ui][vi] = [u, v, 1]
                    else:
                        uv_accum[ui][vi][0] += u
                        uv_accum[ui][vi][1] += v
                        uv_accum[ui][vi][2] += 1
            for ui in range(mesh.uv_count):
                for vi in range(len(data.vertices)):
                    if uv_accum[ui][vi] is not None:
                        u_sum, v_sum, count = uv_accum[ui][vi]
                        all_uvs[ui][vi] = (u_sum / count, v_sum / count)

            for l in data.loops:
                flip = -1.0 if l.bitangent_sign == -1 else 1.0
                NTB[l.vertex_index] = (l.normal, l.tangent, flip)

            SWOMT = bpy.context.scene.SWOMT
            export_uvs = SWOMT.export_uvs or len(data.vertices) != lod.vertex_count

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
                orig_vob = unpack('<I', orig_file_bytes[lod.start_offset + lod.lod_field_offset + 16:lod.start_offset + lod.lod_field_offset + 20])[0]
                orig_voa = unpack('<I', orig_file_bytes[lod.start_offset + lod.lod_field_offset + 12:lod.start_offset + lod.lod_field_offset + 16])[0]
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

                # color(4*cc) | normal(4) | tangent(4) | UV (per-set, variable width)
                color_count = mesh.color_count if getattr(mesh, 'color_in_normals', True) else 0
                _normal_block = 8  # tangent(4) + normal(4)
                _color_prefix = 4 * color_count

                # Detect per-UV-set encoding from the original file bytes.
                # _uv_is_float32[ui] : 8-byte float pair
                # _uv_wide[ui]       : signed int16/4096, no modulo, native range ~[-8,8]
                # _uv_compact[ui]    : uv_unorm; else int16_norm.
                _uv_is_float32 = []
                _uv_wide = []
                _uv_compact = []
                _uv_u_divisor = []
                _uv_v_divisor = []
                _uv_field_offs = []
                _cur = _color_prefix + _normal_block
                _exp_divs = getattr(mesh, 'uv_divisors', None)
                for _ui in range(mesh.uv_count):
                    _div = _exp_divs[_ui] if (_exp_divs is not None and _ui < len(_exp_divs)) else None
                    _plausible = 0
                    _compact_ok = True
                    if orig_vc_src > 0:
                        for _ni in range(orig_vc_src):
                            _o = abs_vob_src + _ni * ns + _cur
                            _fv = unpack('<f', orig_file_bytes[_o:_o + 4])[0]
                            if _fv == _fv and (_fv == 0.0 or 1e-4 < abs(_fv) < 500):
                                _plausible += 1
                            if _div is None:
                                _rs = unpack('<h', orig_file_bytes[_o:_o + 2])[0]
                                if abs(_rs) > 8191:
                                    _compact_ok = False
                        _plausible_f32 = (_plausible / orig_vc_src) > 0.90
                    else:
                        _plausible_f32 = False
                    _enc = _resolve_uv_encoding(_div, _plausible_f32, _compact_ok)
                    _uv_field_offs.append(_cur)
                    _is_f32 = (_enc == 'float32')
                    _uv_is_float32.append(_is_f32)
                    if _is_f32:
                        _uv_wide.append(False); _uv_compact.append(False)
                        _uv_u_divisor.append(1.0); _uv_v_divisor.append(1.0)
                        _cur += 8
                    elif _enc == 'wide':
                        _uv_wide.append(True); _uv_compact.append(False)
                        _uv_u_divisor.append(4096.0); _uv_v_divisor.append(4096.0)
                        _cur += 4
                    else:
                        _uv_wide.append(False)
                        _uv_u_divisor.append(1.0); _uv_v_divisor.append(1.0)
                        _uv_compact.append(_enc == 'compact')
                        _cur += 4
                _total_uv_bytes = _cur - (_color_prefix + _normal_block)
                uv_off_in_stride = _color_prefix + _normal_block
                written_per_vert = _color_prefix + _normal_block + _total_uv_bytes
                trailing_per_vert = ns - written_per_vert
                for ni in range(orig_vc_src):
                    off = abs_vob_src + ni * ns
                    normal_off = off + _color_prefix
                    tangent_off = normal_off + 4
                    uv_off = tangent_off + 4
                    orig_nw.append(unpack('<b', orig_file_bytes[normal_off + 3:normal_off + 4])[0])
                    orig_tangents.append(orig_file_bytes[tangent_off:tangent_off + 4])
                    if color_count > 0:
                        orig_colors.append(orig_file_bytes[off:off + 4 * color_count])
                    if mesh.uv_count > 0:
                        orig_uvs.append(orig_file_bytes[uv_off:uv_off + _total_uv_bytes])
                    if trailing_per_vert > 0:
                        trail_off = off + written_per_vert
                        orig_trailing.append(orig_file_bytes[trail_off:trail_off + trailing_per_vert])

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
                    if color_count > 0:
                        if SWOMT.export_vertex_colors:
                            for ci in range(color_count):
                                layer = bm.verts.layers.float_color.get(f"Color_{ci}")
                                if layer is not None:
                                    vertex_color = bm.verts[v.index][layer]
                                    for c in vertex_color:
                                        f.write(bp.uint8_norm(c))
                                elif orig_colors and src_vi < len(orig_colors):
                                    # Preserve original bytes verbatim
                                    f.write(orig_colors[src_vi][ci * 4:ci * 4 + 4])
                                else:
                                    f.write(b'\x00' * 4)
                        elif orig_colors and src_vi < len(orig_colors):
                            # Preserve original bytes verbatim
                            f.write(orig_colors[src_vi])
                        else:
                            # No source - write zeros
                            f.write(b'\x00' * (4 * color_count))

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
                        if export_uvs:
                            for ui in range(mesh.uv_count):
                                u, v_uv = all_uvs[ui][v.index]
                                if _uv_is_float32[ui]:
                                    f.write(bp.float(u))
                                    f.write(bp.float(v_uv))
                                elif _uv_wide[ui]:
                                    # signed int16 = round(u * 4096), no folding/clamping
                                    # to [0,1] since wide encoding represents the full
                                    # tiled value directly.
                                    _u_raw = max(-32768, min(32767, int(round(u * _uv_u_divisor[ui]))))
                                    _v_raw = max(-32768, min(32767, int(round(v_uv * _uv_v_divisor[ui]))))
                                    f.write(pack('<h', _u_raw))
                                    f.write(pack('<h', _v_raw))
                                elif _uv_compact[ui]:
                                    f.write(bp.uv_unorm_u(max(0.0, min(1.0, u))))
                                    f.write(bp.uv_unorm_v(max(0.0, min(1.0, v_uv))))
                                else:
                                    f.write(bp.int16_norm(max(-1.0, min(1.0, u))))
                                    f.write(bp.int16_norm(max(-1.0, min(1.0, v_uv))))
                        elif orig_uvs and src_vi < len(orig_uvs):
                            f.write(orig_uvs[src_vi])
                        else:
                            f.write(b'\x00' * _total_uv_bytes)

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
                orig_vob = unpack('<I', orig_file_bytes[lod.start_offset+lod.lod_field_offset+16:lod.start_offset+lod.lod_field_offset+20])[0]
                abs_vob_src = orig_do + (orig_vob - higher_size)
                orig_vc_src = unpack('<I', orig_file_bytes[lod.start_offset:lod.start_offset+4])[0]
                ns = mesh.normals_stride

                orig_colors = []
                orig_uvs    = []
                orig_trailing = []
                # color(4*cc) | normal(12) | tangent(12) | sign(4) | UV
                color_count = mesh.color_count if getattr(mesh, 'color_in_normals', True) else 0
                color_off_in_stride = 0
                uv_off_in_stride    = 4 * color_count + 12 + 12 + 4
                # Per-set encoding from the binary divisor table (this float-normal
                # path stores every UV set as a 4-byte int form).
                _uv_wide = []
                _uv_compact = []
                _uv_u_divisor = []
                _uv_v_divisor = []
                _exp_divs = getattr(mesh, 'uv_divisors', None)
                for _ui in range(mesh.uv_count):
                    _off0 = uv_off_in_stride + _ui * 4
                    _div = _exp_divs[_ui] if (_exp_divs is not None and _ui < len(_exp_divs)) else None
                    _compact_ok = True
                    if _div is None:
                        for _ni in range(orig_vc_src):
                            _off = abs_vob_src + _ni * ns + _off0
                            _rs = unpack('<h', orig_file_bytes[_off:_off + 2])[0]
                            if abs(_rs) > 8191:
                                _compact_ok = False
                                break
                    # 4-byte fixed stride here, so a float32 divisor cannot widen the
                    # layout; fall back to int16_norm in that (unseen) case.
                    _enc = _resolve_uv_encoding(_div, False, _compact_ok)
                    if _enc == 'wide':
                        _uv_wide.append(True); _uv_compact.append(False)
                        _uv_u_divisor.append(4096.0); _uv_v_divisor.append(4096.0)
                    else:
                        _uv_wide.append(False)
                        _uv_u_divisor.append(1.0); _uv_v_divisor.append(1.0)
                        _uv_compact.append(_enc == 'compact')
                written_per_vert = uv_off_in_stride + 4 * mesh.uv_count
                trailing_per_vert = ns - written_per_vert
                for ni in range(orig_vc_src):
                    off = abs_vob_src + ni * ns
                    if color_count > 0:
                        orig_colors.append(orig_file_bytes[off + color_off_in_stride:off + color_off_in_stride + 4 * color_count])
                    if mesh.uv_count > 0:
                        orig_uvs.append(orig_file_bytes[off + uv_off_in_stride:off + uv_off_in_stride + 4 * mesh.uv_count])
                    if trailing_per_vert > 0:
                        trail_off = off + written_per_vert
                        orig_trailing.append(orig_file_bytes[trail_off:trail_off + trailing_per_vert])

                orig_idx_attr = data.attributes.get("mmb_vertex_order")
                orig_idx_for_vi = ({vi: orig_idx_attr.data[vi].value for vi in range(len(data.vertices))}
                                   if orig_idx_attr is not None else None)

                for vi, v in enumerate(data.vertices):
                    normal  = NTB[v.index][0]
                    tangent = NTB[v.index][1]
                    v_flip  = NTB[v.index][2]
                    src_vi  = min(orig_idx_for_vi.get(vi, vi) if orig_idx_for_vi else vi, orig_vc_src - 1)

                    # Colors first
                    if color_count > 0:
                        if SWOMT.export_vertex_colors:
                            for ci in range(color_count):
                                layer = bm.verts.layers.float_color.get(f"Color_{ci}")
                                if layer is not None:
                                    vertex_color = bm.verts[v.index][layer]
                                    for c in vertex_color:
                                        f.write(bp.uint8_norm(c))
                                elif orig_colors and src_vi < len(orig_colors):
                                    f.write(orig_colors[src_vi][ci * 4:ci * 4 + 4])
                                else:
                                    f.write(b'\x00' * 4)
                        elif orig_colors and src_vi < len(orig_colors):
                            f.write(orig_colors[src_vi])
                        else:
                            f.write(b'\x00' * (4 * color_count))

                    f.write(bp.float(normal[0] * -1))
                    f.write(bp.float(normal[1]))
                    f.write(bp.float(normal[2]))
                    f.write(bp.float(tangent[0] * -1))
                    f.write(bp.float(tangent[1]))
                    f.write(bp.float(tangent[2]))
                    f.write(bp.float(v_flip))

                    # UVs
                    if mesh.uv_count > 0:
                        if export_uvs:
                            for ui in range(mesh.uv_count):
                                u, v_uv = all_uvs[ui][v.index]
                                if _uv_wide[ui]:
                                    _u_raw = max(-32768, min(32767, int(round(u * _uv_u_divisor[ui]))))
                                    _v_raw = max(-32768, min(32767, int(round(v_uv * _uv_v_divisor[ui]))))
                                    f.write(pack('<h', _u_raw))
                                    f.write(pack('<h', _v_raw))
                                elif _uv_compact[ui]:
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

            bm.free()


    @staticmethod
    def write_triangles(file, mesh:SkeletalMeshAsset.Mesh, lod_index=0, force_uint32=False):
        f = file
        obj = BME.find_object_by_name(mesh.name + f"_LOD{lod_index}")
        lod: SkeletalMeshAsset.Mesh.LOD = mesh.lods[lod_index]
        if obj:
            data = obj.data
            use_uint32 = force_uint32 or len(data.vertices) > 65535
            bm = bmesh.new()
            bm.from_mesh(data)
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            for p in data.polygons:
                if use_uint32:
                    f.write(bp.uint32(p.vertices[0]))
                    f.write(bp.uint32(p.vertices[2]))
                    f.write(bp.uint32(p.vertices[1]))
                else:
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

_MOD_SUFFIX_RE = re.compile(r'^(.*?)(_MOD\d*)$')

def _strip_mod_suffix(stem):
    """'head_MOD' -> 'head', 'head_MOD1' -> 'head', 'head' -> 'head' (no change)."""
    m = _MOD_SUFFIX_RE.match(stem)
    return m.group(1) if m else stem

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
                    # AssetPath may be an exported '_MOD'; the armature in this
                    # .blend was never renamed, so prefer its un-suffixed name if found.
                    full_stem = Path(path).stem
                    bare_stem = _strip_mod_suffix(full_stem)
                    if bare_stem != full_stem and bpy.data.objects.get(bare_stem) is not None:
                        sk_mesh.name = bare_stem
                    else:
                        sk_mesh.name = full_stem
                    asset = sk_mesh
                _check_vert_pos_mmb(sk_mesh, path)
                print(f"[AFoPMT] Loaded '{sk_mesh.name}' from '{path}'")
            except Exception as e:
                print(f"[AFoPMT] Failed to Load '{path}': {e}")
            break
    except Exception as e:
        print(f"[AFoPMT] _on_load_post error: {e}")

def _resolve_asset_name(new_path, old_asset):
    """
    Name a newly-parsed asset for `new_path`.

    If the new filename is just the loaded asset's name plus a '_MOD' suffix, and the armature
    for that asset still exists - keep the loaded asset's name so it still matches the exisitng armature.

    Otherwise, derive the name from the path.
    """
    new_stem = Path(new_path).stem
    if (old_asset is not None and old_asset.name
            and _strip_mod_suffix(new_stem) == old_asset.name
            and bpy.data.objects.get(old_asset.name) is not None):
        return old_asset.name
    return new_stem


def _auto_load_mmb(self, context):
    path = self.AssetPath
    if not path or not os.path.isfile(path):
        return
    try:
        global asset
        new_name = _resolve_asset_name(path, asset)
        with open(path, 'rb') as file:
            sk_mesh = SkeletalMeshAsset()
            sk_mesh.parse(file)
            sk_mesh.name = new_name
            asset = sk_mesh
        _check_vert_pos_mmb(sk_mesh, path)
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
            if obj is not None and len(obj.data.vertices) != 0 and len(obj.data.vertices) != lod.vertex_count:
                return True
    return False

def _check_vert_pos_mmb(sk_mesh, path: str):
    """
    Read LOD0 vertex positions for each mesh directly from the MMB file. (We assume that LODs 1-3 are zeroed)
    If all positions for LOD0 are zero, zeroed_out_in_mmb = True
    """
    try:
        merged = get_merged_mmb(path)
    except Exception as e:
        print(f"[AFoPMT] _check_vert_pos_mmb: could not open '{path}': {e}")
        return

    for mesh in sk_mesh.meshes:
        if not mesh.lods:
            continue
        lod = mesh.lods[0]
        if lod.vertex_count == 0:
            continue
        try:
            raw = mesh.extract_mesh_file(merged)
            positions = lod.get_vertex_positions(raw)
            is_zeroed = all(
                abs(x) < 1e-6 and abs(y) < 1e-6 and abs(z) < 1e-6
                for x, y, z in positions
            )
        except Exception as e:
            print(f"[AFoPMT] _check_vert_pos_mmb: error reading '{mesh.name}': {e}")
            continue

        mesh.zeroed_out_in_mmb = is_zeroed

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

def _get_export_normals(self):
    if _vert_count_changed():
        return True
    return self.get("export_normals", False)

def _set_export_normals(self, value):
    if _vert_count_changed():
        return
    old = self.get("export_normals", False)
    self["export_normals"] = value
    if old != value:
        _on_export_normals_update(self, None)

def _get_export_weights(self):
    if _vert_count_changed():
        return True
    return self.get("export_weights", False)

def _set_export_weights(self, value):
    if not _vert_count_changed():
        self["export_weights"] = value

def _get_export_uvs(self):
    if _vert_count_changed():
        return True
    return self.get("export_uvs", False)

def _set_export_uvs(self, value):
    if _vert_count_changed():
        return
    old = self.get("export_uvs", False)
    self["export_uvs"] = value
    if old != value:
        _on_export_uvs_update(self, None)

class SWOMTSettings(bpy.types.PropertyGroup):
    AssetPath: bpy.props.StringProperty(name="Asset Path", update=_auto_load_mmb)
    overwrite_existing: bpy.props.BoolProperty(
        name="Overwrite existing file",
        default=False,
        description="When checked, export overwrites the loaded file instead of creating a new _MOD file",
    )
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
        description="Write normals into the exported file. When unchecked, the original normals from the .mmb are preserved. Automatically forced on when vert count has changed.",
        get=_get_export_normals,
        set=_set_export_normals,
    )
    export_weights: bpy.props.BoolProperty(
        name="Export Weights",
        description="Write bone weights into the exported file. When unchecked, the original weights from the .mmb are preserved. Automatically forced on when vert count has changed.",
        get=_get_export_weights,
        set=_set_export_weights,
    )
    export_vertex_colors: bpy.props.BoolProperty(
        name="Export Vertex Colors",
        default=False,
        description="Write vertex colors from Blender into the exported file. When unchecked, the original vertex colors from the .mmb are preserved.",
        update=_on_export_vertex_colors_update,
    )
    export_uvs: bpy.props.BoolProperty(
        name="Export UVs",
        description="Write UV coordinates from Blender into the exported file. When unchecked, the original UVs from the .mmb are preserved. Automatically forced on when vert count has changed.",
        get=_get_export_uvs,
        set=_set_export_uvs,
    )
    export_options_expanded: bpy.props.BoolProperty(
        name="Export Options",
        default=True,
    )
    force_lod0_mmb_override: bpy.props.StringProperty(
        name="MMB Filename",
        description="Original .mmb filename to look up in mmb_lod_presets.json",
        default="",
    )
    force_lod0_output_path: bpy.props.StringProperty(
        name="Force LOD0 Output Path",
        default="",
    )
    force_lod0_cfg_path: bpy.props.StringProperty(
        name="LOD Presets CFG",
        description="Path to an existing lod_presets.cfg to update. Leave empty to generate a new one alongside the asset.",
        default="",
        subtype="FILE_PATH",
    )

def _force_lod0_generate_cfg(mmb_name: str, asset_dir: str, cfg_override: str = "", operator=None) -> str:
    """
    Look up mmb_name in mmb_lod_presets.json, find its preset names,
    then write a modified lod_presets.cfg into asset_dir (or update cfg_override)
    with those presets having manualLodPixelSteps = {1, 1, 1}.
    Returns a status message string.
    """
    import re as _re
    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(plugin_dir, "mmb_lod_presets.json")
    cfg_path = cfg_override if cfg_override else os.path.join(plugin_dir, "lod_presets.cfg")
    out_path = cfg_override if cfg_override else os.path.join(asset_dir, "lod_presets.cfg")

    if not os.path.isfile(json_path):
        return f"mmb_lod_presets.json not found in plugin folder."
    if not os.path.isfile(cfg_path):
        return f"lod_presets.cfg not found in plugin folder."

    with open(json_path, 'r', encoding='utf-8') as f:
        db = json.load(f)

    # Match any JSON entry whose mmb name is contained within the asset filename.
    mmb_stem = os.path.splitext(mmb_name.strip())[0].lower()
    matches = [e for e in db if os.path.splitext(e['mmb'])[0].lower() in mmb_stem]
    if not matches:
        return f"'{mmb_name}' not found in mmb_lod_presets.json."
    if len(matches) > 1:
        return f"Multiple entries matched '{mmb_name}' - please specify the exact filename."

    target_presets = set(p.strip().lower() for p in matches[0].get('lod_presets', []))
    if not target_presets:
        return f"No LOD presets listed for '{mmb_name}'."

    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg_text = f.read()

    def replace_steps(block: str) -> str:
        """Replace manualLodPixelSteps values with 1, 1, 1 in a preset block."""
        return _re.sub(
            r'(manualLodPixelSteps\s*=\s*\{)[^}]*(\})',
            r'\g<1>\n\t\t\t1,\n\t\t\t1,\n\t\t\t1,\n\t\t\2',
            block,
        )

    # Match each preset block - one level of nesting for manualLodPixelSteps {}
    block_pattern = _re.compile(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', _re.DOTALL)

    patched = 0
    def patch_block(m):
        nonlocal patched
        block = m.group(0)
        name_match = _re.search(r'name\s*=\s*"([^"]*)"', block)
        if name_match and name_match.group(1).strip().lower() in target_presets:
            patched += 1
            return replace_steps(block)
        return block

    new_cfg = block_pattern.sub(patch_block, cfg_text)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(new_cfg)

    return f"Patched {patched} preset(s) for '{mmb_name}' -> lod_presets.cfg updated."

class ForceLOD0(bpy.types.Operator):
    """Generate a lod_presets.cfg that forces LOD0 for this mesh's presets"""
    bl_idname  = 'object.force_lod0'
    bl_label   = 'Force LOD0'
    bl_options = {'REGISTER'}

    needs_override: bpy.props.BoolProperty(default=False, options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return asset is not None

    def invoke(self, context, event):
        SWOMT = context.scene.SWOMT
        mmb_name = os.path.basename(SWOMT.AssetPath) if SWOMT.AssetPath else ''
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        json_path  = os.path.join(plugin_dir, "mmb_lod_presets.json")
        found = False
        if mmb_name and os.path.isfile(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                db = json.load(f)
            mmb_stem = os.path.splitext(mmb_name)[0].lower()
            matches = [e for e in db if os.path.splitext(e['mmb'])[0].lower() in mmb_stem]
            found = len(matches) == 1

        if found:
            self.needs_override = False
            return self.execute(context)
        else:
            self.needs_override = True
            SWOMT.force_lod0_mmb_override = mmb_name # fill with current mmb filename
            return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        layout = self.layout
        SWOMT = context.scene.SWOMT
        mmb_name = os.path.basename(SWOMT.AssetPath) if SWOMT.AssetPath else ''
        if mmb_name:
            layout.label(text=f"'{mmb_name}' was not found in mmb_lod_presets.json.")
        else:
            layout.label(text="No .mmb file loaded.")
        layout.label(text="Enter the original .mmb filename to look up:")
        layout.prop(SWOMT, "force_lod0_mmb_override", text="")

    def execute(self, context):
        SWOMT = context.scene.SWOMT
        if self.needs_override:
            mmb_name = SWOMT.force_lod0_mmb_override.strip()
        else:
            mmb_name = os.path.basename(SWOMT.AssetPath) if SWOMT.AssetPath else ''

        if not mmb_name:
            self.report({'ERROR'}, "No .mmb filename provided.")
            return {'CANCELLED'}

        asset_dir = os.path.dirname(SWOMT.AssetPath) if SWOMT.AssetPath else ''
        if not asset_dir:
            self.report({'ERROR'}, "No asset path loaded.")
            return {'CANCELLED'}

        cfg_override = SWOMT.force_lod0_cfg_path.strip()
        msg = _force_lod0_generate_cfg(mmb_name, asset_dir, cfg_override=cfg_override, operator=self)
        if msg.startswith("Patched"):
            SWOMT.force_lod0_output_path = cfg_override if cfg_override else os.path.join(asset_dir, "lod_presets.cfg")
            self.report({'INFO'}, msg)
        else:
            SWOMT.force_lod0_output_path = ""
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}
        return {'FINISHED'}


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

class BrowseLodPresetsCfg(bpy.types.Operator):
    """Select an existing lod_presets.cfg to update"""
    bl_idname = "object.browse_lod_presets_cfg"
    bl_label  = "Select lod_presets.cfg"

    filepath:    bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.cfg", options={'HIDDEN'})

    def invoke(self, context, event):
        current = context.scene.SWOMT.force_lod0_cfg_path
        if current:
            self.filepath = current
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        context.scene.SWOMT.force_lod0_cfg_path = self.filepath
        return {'FINISHED'}

class ClearLodPresetsCfg(bpy.types.Operator):
    """Clear the selected lod_presets.cfg"""
    bl_idname = "object.clear_lod_presets_cfg"
    bl_label  = "Clear lod_presets.cfg selection"

    def execute(self, context):
        context.scene.SWOMT.force_lod0_cfg_path = ""
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
        mod_file = _mod_file_output(src_path, overwrite=SWOMT.overwrite_existing)

        # Triangulate the Blender mesh before export so all faces are tris
        mesh = asset.meshes[self.mesh_index]
        lod  = mesh.lods[self.lod_index]
        lod_obj_name = lod.blender_obj_name or f"{mesh.name}_LOD{self.lod_index}"
        tri_obj = BME.find_object_by_name(lod_obj_name)
        original_data = {}

        if tri_obj:
            original_data = tri_obj.data
            BME._triangulate_object(tri_obj,
                                    compute_normals=context.scene.SWOMT.compute_normals_on_export)

        try:
            BME._write_mod_file(
                edited_lod_index_per_mesh={self.mesh_index: self.lod_index},
                out_path=mod_file,
            )
        except Exception as e:
            if tri_obj:
                tri_obj.data = original_data
            self.report({'ERROR'}, f"Export failed: {e}")
            return {'CANCELLED'}
        finally:
            # Always restore the original mesh data so the user's mesh is untouched
            if tri_obj:
                export_data = tri_obj.data
                tri_obj.data = original_data
                if export_data != original_data:
                    bpy.data.meshes.remove(export_data)

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
            SWOMT.AssetPath = new_file
            self.report({'INFO'}, f"Exported -> {os.path.basename(new_file)}")
        else:
            SWOMT.AssetPath = mod_file

        return {'FINISHED'}

class RemoveMesh(bpy.types.Operator):
    """Zero out all vertex positions for all LODs of this mesh."""
    bl_idname = "object.remove_mesh"
    bl_label = "Remove Mesh"

    mesh_index: bpy.props.IntProperty()

    @classmethod
    def poll(cls, context):
        return asset is not None

    def execute(self, context):
        mesh = asset.meshes[self.mesh_index]
        mesh.zeroed_out_in_session = True

        # Remove any already-imported Blender objects in the viewport
        for li, lod in enumerate(mesh.lods):
            obj_name = lod.blender_obj_name if lod.blender_obj_name else f"{mesh.name}_LOD{li}"
            obj = bpy.data.objects.get(obj_name)
            if obj is not None:
                bpy.data.objects.remove(obj, do_unlink=True)

        self.report({'INFO'}, f"'{mesh.name}' ({len(mesh.lods)} LOD(s)) will have their vertex positions zeroed out on Export.")
        return {'FINISHED'}

class RevertMesh(bpy.types.Operator):
    """Revert vertex positions for all LODs of this mesh."""
    bl_idname = "object.revert_mesh"
    bl_label = "Revert Mesh"

    mesh_index: bpy.props.IntProperty()

    @classmethod
    def poll(cls, context):
        return asset is not None

    def execute(self, context):
        mesh = asset.meshes[self.mesh_index]

        if not mesh.zeroed_out_in_session:
            self.report({'WARNING'}, "Mesh was not zeroed out in this session.")
            return {'CANCELLED'}

        # Remove any already-imported Blender objects in the viewport
        for li, lod in enumerate(mesh.lods):
            obj_name = lod.blender_obj_name if lod.blender_obj_name else f"{mesh.name}_LOD{li}"
            obj = bpy.data.objects.get(obj_name)
            if obj is not None:
                bpy.data.objects.remove(obj, do_unlink=True)

        mesh.zeroed_out_in_session = False
        self.report({'INFO'}, f"Reverted '{mesh.name}' ({len(mesh.lods)} LOD(s)) to original positions.")
        return {'FINISHED'}

def _scale_uvs_uv_items(self, context):
    """Populate UV set enum from the mesh's uv_count."""
    if asset is None:
        return []
    try:
        mesh = asset.meshes[self.mesh_index]
    except (IndexError, AttributeError):
        return []
    return [(str(i), f"UVMap_{i}", f"UV set {i}") for i in range(mesh.uv_count)]

def _scale_u_update(self, context):
    if self.link:
        self["scale_v"] = self.scale_u

def _scale_v_update(self, context):
    if self.link:
        self["scale_u"] = self.scale_v

class ScaleUVs(bpy.types.Operator):
    """Scale a UV layer on all imported LODs of this mesh"""
    bl_idname = "object.scale_uvs"
    bl_label = "Scale UVs"

    mesh_index: bpy.props.IntProperty()
    uv_set: bpy.props.EnumProperty(
        name="UV Set",
        description="Which UV map to scale",
        items=_scale_uvs_uv_items,
    )
    scale_u: bpy.props.FloatProperty(name="Scale U", default=1.0, min=0.001, max=1000.0, step=100, update=_scale_u_update)
    scale_v: bpy.props.FloatProperty(name="Scale V", default=1.0, min=0.001, max=1000.0, step=100, update=_scale_v_update)
    link: bpy.props.BoolProperty(name="Link", default=True, description="Link Scale U and Scale V together")
    pivot_u: bpy.props.FloatProperty(name="Pivot U", default=0.0, step=100, description="U coordinate to scale from")
    pivot_v: bpy.props.FloatProperty(name="Pivot V", default=1.0, step=100, description="V coordinate to scale from")

    @classmethod
    def poll(cls, context):
        return asset is not None

    def invoke(self, context, event):
        self.scale_u = 1.0
        self.scale_v = 1.0
        self.link = True
        self.pivot_u = 0.0
        self.pivot_v = 1.0
        context.window.cursor_warp(context.window.width // 2, context.window.height // 2)
        return context.window_manager.invoke_props_dialog(self, width=260)

    def draw(self, context):
        mesh = asset.meshes[self.mesh_index]
        layout = self.layout
        layout.label(text=f"Mesh: {mesh.name}")
        if mesh.uv_count > 1:
            layout.prop(self, "uv_set")
        row = layout.row(align=True)
        row.prop(self, "scale_u")
        row.prop(self, "link", text="", icon='LINKED' if self.link else 'UNLINKED')
        row.prop(self, "scale_v")
        pivot_row = layout.row(align=True)
        pivot_row.prop(self, "pivot_u")
        pivot_row.prop(self, "pivot_v")

    def execute(self, context):
        # Exit edit mode first - UV data changes on obj.data are not reflected while the object is being edited.
        if context.active_object and context.active_object.mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')

        mesh = asset.meshes[self.mesh_index]
        uv_index = int(self.uv_set) if self.uv_set else 0
        layer_name = f"UVMap_{uv_index}"
        affected = 0
        seen_data = set() # guard against LODs that share the same mesh data block
        for li, lod in enumerate(mesh.lods):
            obj_name = lod.blender_obj_name if lod.blender_obj_name else f"{mesh.name}_LOD{li}"
            obj = bpy.data.objects.get(obj_name)
            if obj is None or obj.type != 'MESH':
                continue
            if obj.data.name in seen_data:
                continue
            seen_data.add(obj.data.name)
            uv_layer = obj.data.uv_layers.get(layer_name)
            if uv_layer is None:
                continue
            # Scale from the chosen pivot: new = pivot + (uv - pivot) * scale
            pu, pv = self.pivot_u, self.pivot_v
            for loop_uv in uv_layer.data:
                loop_uv.uv = (pu + (loop_uv.uv[0] - pu) * self.scale_u,
                              pv + (loop_uv.uv[1] - pv) * self.scale_v)
            obj.data.update()
            affected += 1
        if affected == 0:
            self.report({'WARNING'}, f"No imported LODs found for '{mesh.name}' with UV layer '{layer_name}'")
        else:
            self.report({'INFO'}, f"Scaled {layer_name} by ({self.scale_u}, {self.scale_v}) from pivot ({self.pivot_u}, {self.pivot_v}) on {affected} LOD(s) of '{mesh.name}'")
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


def _compute_inv_bind_from_skeleton(bone_name):
    """
    Compute the inverse bind matrix for a bone from the currently loaded skeleton.
    """
    if asset is None:
        return None
    bone_idx = next((i for i, b in enumerate(asset.bones) if b.name == bone_name), None)
    if bone_idx is None:
        return None

    SWOMT = bpy.context.scene.SWOMT
    src_path = SWOMT.AssetPath
    if not os.path.isfile(src_path):
        return None

    try:
        with open(src_path, 'rb') as f:
            data = f.read()

        pos = 0
        version = data[3]
        pos += 8
        if version >= 15:
            pos += 4
        bone_count = unpack('<I', data[pos:pos+4])[0]
        pos += 4

        # Read all file_local matrices and parent indices
        file_locals = []
        parents = []
        for i in range(bone_count):
            nlen = unpack('<H', data[pos:pos+2])[0]; pos += 2
            pos += nlen  # skip name
            raw = unpack('<16f', data[pos:pos+64]); pos += 64
            parent_idx = unpack('<H', data[pos:pos+2])[0]; pos += 2
            m = Matrix([
                [raw[0], raw[4], raw[8],  raw[12]],
                [raw[1], raw[5], raw[9],  raw[13]],
                [raw[2], raw[6], raw[10], raw[14]],
                [raw[3], raw[7], raw[11], raw[15]],
            ])
            file_locals.append(m)
            parents.append(parent_idx)

        # Get file_world for target bone
        file_world_cache = [None] * bone_count
        def get_fw(i):
            if file_world_cache[i] is not None:
                return file_world_cache[i]
            if parents[i] == 65535:
                file_world_cache[i] = file_locals[i]
            else:
                file_world_cache[i] = get_fw(parents[i]) @ file_locals[i]
            return file_world_cache[i]

        fw = get_fw(bone_idx)
        try:
            inv_bind = fw.inverted()
        except ValueError:
            return None

        return tuple(inv_bind[r][c] for c in range(4) for r in range(4))

    except Exception:
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
            if version not in (11, 12, 13, 14, 15, 16, 17):
                break
            if u_count > 0 and version != 12:
                f.seek(1 if version in (13, 14) else 2, 1)
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


# Vertex strides whose bone-slot index is written as a single byte on export,
# with no fallback - can only address 256 mesh-bone-table slots.
#
# *** If these sets aren't updated for new or changes in strides, the "ubyte format requires 0 <= number <= 255" error
# will be thrown. ***
#
# Stride 32 is included here even though one of its three layouts (Layout C) is uint8-indexed.
# (write_vertices already self-detects and switches stride 32 to a uint16-index layout)
_UINT8_INDEX_LIMITED_STRIDES = {12, 16, 20, 36}
_UINT16_NON_LIMITED_STRIDES = {32, 40, 44}

def _mesh_is_uint8_index_limited(mesh):
    """True if this mesh's vertex format caps bone-slot indices at 255."""
    stride = getattr(mesh, "vertex_stride", None)
    if not stride or stride <= 0:
        return False
    if stride in _UINT8_INDEX_LIMITED_STRIDES:
        return True
    if stride in _UINT16_NON_LIMITED_STRIDES:
        return False
    # Unrecognized strides fall into 'write_vertices' 'else' uint8 branch.
    return True

def _scan_mesh_used_bone_slots(mesh):
    """
    Scan the `mesh` in the currently loaded asset and return the set of
    mesh-bone-table slot indices that have non-zero weight on any vertex, across
    all LODs.

    Callers adding several bones to the same mesh should call this once and reuse the result via
    _find_unused_mesh_bone_slot's `used` parameter, rather than re-scanning per bone.

    Returns None on any read error.
    """
    try:
        SWOMT = bpy.context.scene.SWOMT
        src_path = SWOMT.AssetPath
        used = set()
        with open(src_path, 'rb') as f:
            raw_mesh_file = mesh.extract_mesh_file(f)
        for lod in mesh.lods:
            if lod.vertex_count == 0:
                continue
            for iw in lod.get_bone_weights(raw_mesh_file):
                used.update(s for s, w in iw.items() if w > 0.0)
        return used
    except Exception as e:
        print(f"[AFoPMT] _scan_mesh_used_bone_slots error: {e}")
        return None


def _find_unused_mesh_bone_slot(mesh, used=None):
    """
    Return the lowest mesh-bone-table slot with zero weight on every vertex - i.e. one
    that can be reused for a new bone without touching existing weights. Slots already pending a remap
    in this session (mesh.pending_bone_remaps) are excluded, even if they have zero weights.

    `used` is the result of _scan_mesh_used_bone_slots(mesh). It is passed when adding
    several bones to the same mesh at the same time, so the file scan only happens once.

    Returns None if no free slot exists, or the scan failed.
    """
    n_slots = len(mesh.mesh_bones)
    if n_slots == 0:
        return None
    if used is None:
        used = _scan_mesh_used_bone_slots(mesh)
        if used is None:
            return None
    else:
        used = set(used)
    used.update(mesh.pending_bone_remaps.keys())
    for slot in range(n_slots):
        if slot not in used:
            return slot
    return None


def _add_or_reuse_mesh_bone_slot(mesh, new_skel_idx, new_matrix, used_slots_cache=None):
    """
    Add a new bone to `mesh`'s bone table, reusing an unused slot instead of
    appending past the uint8 limit when the mesh's vertex stride is limited.

    `used_slots_cache`: pass a dict when adding several bones to the same mesh at once -
    keyed by `id(mesh)`, so the weight scan only happens once per mesh instead of per bone.

    Returns (status, info):
      'appended' - added as a new slot at the end; info=None
      'reused'   - remapped an unused slot; info=slot_index
      'full'     - no free slot available (256-slot limit)
    """
    n_slots = len(mesh.mesh_bones)
    if _mesh_is_uint8_index_limited(mesh) and n_slots >= 256:
        used = None
        if used_slots_cache is not None:
            cache_key = id(mesh)
            if cache_key in used_slots_cache:
                used = used_slots_cache[cache_key]
                if used is None:
                    return 'full', None # If a bone failed to scan - treat as "full"
            else:
                used = _scan_mesh_used_bone_slots(mesh)
                used_slots_cache[cache_key] = used # Failed (None) are also cached
                if used is None:
                    return 'full', None
        free_slot = _find_unused_mesh_bone_slot(mesh, used=used)
        if free_slot is None:
            return 'full', None
        mesh.pending_bone_remaps[free_slot] = (new_skel_idx, new_matrix)
        new_mesh_bones = {}
        for slot_i, (skel_idx, matrix) in enumerate(mesh.mesh_bones.items()):
            if slot_i == free_slot:
                new_mesh_bones[new_skel_idx] = new_matrix
            else:
                new_mesh_bones[skel_idx] = matrix
        mesh.mesh_bones = new_mesh_bones
        return 'reused', free_slot

    mesh.pending_bone_additions.append((new_skel_idx, new_matrix))
    mesh.mesh_bones[new_skel_idx] = new_matrix
    return 'appended', None


class RemapMeshBone(bpy.types.Operator):
    """Remap a mesh bone slot to a different bone"""
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
        description="Derive the inverse bind matrix from the loaded skeleton (Recommended). Uncheck to select a donor MMB file instead.",
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
        if not self.use_auto:
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

        # Get the inverse bind matrix
        # Auto: derive directly from the loaded skeleton (Recommended).
        # Manual: read from a donor MMB file.
        if self.use_auto:
            new_matrix = _compute_inv_bind_from_skeleton(new_name)
            if new_matrix is None:
                self.report({'ERROR'},
                    f"Could not derive inv_bind for '{new_name}' from the loaded skeleton. "
                    f"Uncheck Auto and supply a donor MMB instead.")
                return {'CANCELLED'}
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

        source = "skeleton" if self.use_auto else "donor MMB"
        self.report({'INFO'}, f"Slot {self.slot_index}: '{old_name}' to '{new_name}' via {source} (will patch on export)")
        return {'FINISHED'}

class AddMeshBone(bpy.types.Operator):
    """Add a new bone slot to this mesh's bone table"""
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
        description="Derive the inverse bind matrix from the loaded skeleton (Recommended). Uncheck to select a donor MMB file instead.",
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
        if not self.use_auto:
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

        # Get the inverse bind matrix.
        # Auto: derive directly from the loaded skeleton (Recommended).
        # Manual: read from a donor MMB file.
        if self.use_auto:
            new_matrix = _compute_inv_bind_from_skeleton(new_name)
            if new_matrix is None:
                self.report({'ERROR'},
                    f"Could not derive inv_bind for '{new_name}' from the loaded skeleton. "
                    f"Uncheck Auto and supply a donor MMB instead.")
                return {'CANCELLED'}
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

        # Stage the addition - reuses an unused slot in place if this mesh's vertex is uint8 limited.
        status, info = _add_or_reuse_mesh_bone_slot(mesh, new_skel_idx, new_matrix)
        if status == 'full':
            self.report({'ERROR'},
                f"'{mesh.name}' uses a vertex stride limited to 256 bone slots (uint8) and has "
                f"no un-weighted slot to reuse.")
            return {'CANCELLED'}

        # Create the vertex group on any already-imported Blender objects
        for li, lod in enumerate(mesh.lods):
            obj_name = lod.blender_obj_name if lod.blender_obj_name else f"{mesh.name}_LOD{li}"
            obj = bpy.data.objects.get(obj_name)
            if obj is not None and obj.vertex_groups.get(new_name) is None:
                obj.vertex_groups.new(name=new_name)

        source = "skeleton" if self.use_auto else "donor MMB"
        if status == 'reused':
            self.report({'INFO'},
                f"'{new_name}' staged via {source}, reusing unused slot {info} (will patch on export)")
        else:
            self.report({'INFO'}, f"'{new_name}' staged for addition via {source} (will patch on export)")
        return {'FINISHED'}

def _read_donor_skeleton(donor_path: str):
    """
    Parse a donor .mmb file and return a list of (name, mat_raw, matrix, parent_idx).

    mat_raw - the original 16 floats exactly as stored in the file.
              Used directly for the bone_blob write so no re-encoding is needed.
    matrix  - a Blender Matrix built via the same br.matrix_4x4 convention
              (m[r][c] = mat_raw[c*4+r]) for use in world-matrix accumulation.
    """
    try:
        with open(donor_path, 'rb') as f:
            data = f.read()
        pos = 0
        version = data[3]
        pos += 8
        if version >= 15:
            pos += 4
        bone_count = unpack('<I', data[pos:pos+4])[0]
        pos += 4
        bones = []
        for i in range(bone_count):
            nlen = unpack('<H', data[pos:pos+2])[0]; pos += 2
            name = data[pos:pos+nlen].decode('ascii', errors='replace'); pos += nlen
            mat_raw = unpack('<16f', data[pos:pos+64]); pos += 64
            parent_idx = unpack('<H', data[pos:pos+2])[0]; pos += 2
            # Build Matrix using br.matrix_4x4 convention: m[r][c] = mat_raw[c*4+r].
            # br.matrix_4x4 reads floats and groups them so that the i-th outer loop
            # feeds each row-list in column order - i.e. the file is stored column-first.
            m = Matrix([
                [mat_raw[0], mat_raw[4], mat_raw[8],  mat_raw[12]],
                [mat_raw[1], mat_raw[5], mat_raw[9],  mat_raw[13]],
                [mat_raw[2], mat_raw[6], mat_raw[10], mat_raw[14]],
                [mat_raw[3], mat_raw[7], mat_raw[11], mat_raw[15]],
            ])
            bones.append((name, mat_raw, m, parent_idx))
        return bones
    except Exception as e:
        print(f"[AFoPMT] _read_donor_skeleton error: {e}")
        return None

def _resolve_selected_donor_bones(donor_bones, selected_names):
    """
    From the full donor bone list, keep only `selected_names` plus whatever parents
    each one needs to stay connected.

    Returns (orig_idx, name, mat_raw, matrix, parent_idx) elements, in order from the donor.
    orig_idx is the bone's index in the original unfiltered list - callers must use it
    (not the filtered list's position) when matching against parent_idx, since both
    are original-list indices and filtering changes a bone's position but not its index.
    """
    name_set = set(selected_names)
    keep = set()
    for d_idx, (name, mat_raw, matrix, pidx) in enumerate(donor_bones):
        if name not in name_set:
            continue
        # Walk up the parent chain, adding every parent along the way.
        cur = d_idx
        while cur is not None and cur not in keep:
            keep.add(cur)
            _, _, _, cur_pidx = donor_bones[cur]
            if cur_pidx == 65535 or cur_pidx >= len(donor_bones):
                cur = None
            else:
                cur = cur_pidx
    return [(i,) + donor_bones[i] for i in sorted(keep)]

# Cache at module-level (fixes search issue)
_donor_bone_names_cache = []

# Avoids re-parsing the donor file in execute() right after invoke() already did.
_donor_bones_cache = (None, None)

def _get_cached_donor_bones(donor_path):
    """Return donor_bones for `donor_path`, parsing new, if not already cached."""
    global _donor_bones_cache
    cached_path, cached_bones = _donor_bones_cache
    if cached_path == donor_path and cached_bones is not None:
        return cached_bones
    donor_bones = _read_donor_skeleton(donor_path)
    _donor_bones_cache = (donor_path, donor_bones)
    return donor_bones

def _donor_bone_search_cb(self, context, edit_text):
    """
    Search for the comma-separated bone_names field.

    Filters on the text after the last comma, returning a selected bone name
    and leaving the field empty for the next one.
    """
    if not _donor_bone_names_cache:
        return []
    prefix = ""
    tail = edit_text
    if "," in edit_text:
        prefix, tail = edit_text.rsplit(",", 1)
        prefix = prefix.strip(" ,") + ", "
    tail_lower = tail.strip().lower()
    already = {n.strip().lower() for n in edit_text.split(",") if n.strip()}
    results = []
    for name in _donor_bone_names_cache:
        if tail_lower in name.lower() and name.lower() not in already:
            results.append(f"{prefix}{name}, ")
    return results

def _do_merge_skeletons(context, operator, src_filepath, donor_bones, mode_label):
    """
    Called by MergeSkeletonsPickBones.

    donor_bones is a list of (orig_idx, name, mat_raw, matrix, parent_idx) elements.
    orig_idx must be used for all donor-indexing here (not the list's own position as it may be filtered).
    """
    SWOMT = context.scene.SWOMT
    src_path = SWOMT.AssetPath

    # Index map for the host skeleton
    host_names = {b.name: i for i, b in enumerate(asset.bones)}

    # Collect only bones that are new (not already in the host)
    new_bones = [(orig_idx, name, mat_raw, matrix, pidx)
                 for orig_idx, name, mat_raw, matrix, pidx in donor_bones
                 if name not in host_names]

    if not new_bones:
        operator.report({'INFO'}, f"No new bones to merge ({mode_label}) - already in the loaded skeleton.")
        return {'FINISHED'}

    # Data to insert into the file:
    # - Each bone: uint16 name_len | name bytes | 16 floats (64 bytes) | uint16 parent_idx
    # - Parent indices must be remapped to the combined skeleton's indices.
    # - donor_index -> combined_index, keyed by each bone's ORIGINAL donor index so it
    #   lines up with parent_idx regardless of any filtering.

    # Map: donor bone index -> combined index (host bones first, then appended from donor)
    donor_to_combined = {}
    for orig_idx, name, mat_raw, matrix, pidx in donor_bones:
        if name in host_names:
            donor_to_combined[orig_idx] = host_names[name]

    new_start = len(asset.bones)
    for ni, (orig_idx, name, mat_raw, matrix, pidx) in enumerate(new_bones):
        donor_to_combined[orig_idx] = new_start + ni

    # Build bone data
    bone_blob = bytearray()
    for orig_idx, name, mat_raw, matrix, pidx in new_bones:
        # Name
        name_bytes = name.encode('ascii', errors='replace')
        bone_blob += pack('<H', len(name_bytes))
        bone_blob += name_bytes
        # Matrix - write the original raw bytes verbatim.
        # mat_raw is the 16 floats exactly as they appear in the donor file,
        # which is already in the correct column-first file format that br.matrix_4x4 reads.
        bone_blob += pack('<16f', *mat_raw)
        # Parent index - remap donor index to combined skeleton index
        if pidx == 65535:
            combined_pidx = 65535
        else:
            combined_pidx = donor_to_combined.get(pidx, 65535)
        bone_blob += pack('<H', combined_pidx)

    # Patch the .mmb file:
    # - Increment bone_count (uint32 at a known offset)
    # - Insert bone_blob immediately after the last existing bone

    mod_file = _mod_file_output(src_path, overwrite=SWOMT.overwrite_existing)

    with open(src_path, 'rb') as f:
        file_data = bytearray(f.read())

    version = file_data[3]
    skel_count_offset = 8
    if version >= 15:
        skel_count_offset += 4

    old_bone_count = unpack('<I', file_data[skel_count_offset:skel_count_offset+4])[0]
    new_bone_count = old_bone_count + len(new_bones)
    file_data[skel_count_offset:skel_count_offset+4] = pack('<I', new_bone_count)

    # Walk past existing bones to find the insertion point (start of mesh section)
    pos = skel_count_offset + 4
    for _ in range(old_bone_count):
        nlen = unpack('<H', file_data[pos:pos+2])[0]; pos += 2
        pos += nlen + 64 + 2 # name + matrix + parent_idx

    # pos is now at uint32 mesh_count - insert bone_blob here
    insert_at = pos
    file_data[insert_at:insert_at] = bone_blob

    inserted = len(bone_blob)

    # Update asset.size header field (bytes 4..8) - it covers the header section.
    # Skeleton is always in the header, so bump it.
    old_asset_size = unpack('<I', file_data[4:8])[0]
    file_data[4:8] = pack('<I', old_asset_size + inserted)

    # All data_offset fields in all mesh LODs must be incremented by `inserted`
    # because the skeleton has grown (it lives before the mesh data in the header).
    # We need to patch these fields in the file_data we are building.
    # We do NOT have the in-memory asset offsets updated yet, so we must re-walk
    # the mesh section in the modified file_data to find and patch them.
    #
    # Patch: scan mesh table starting after the new bone data.
    mesh_pos = insert_at + inserted # points to uint32 mesh_count
    mesh_count_val = unpack('<I', file_data[mesh_pos:mesh_pos+4])[0]
    mp = mesh_pos + 4

    for mi_scan in range(mesh_count_val):
        nlen = unpack('<H', file_data[mp:mp+2])[0]; mp += 2
        mp += nlen # mesh name
        mp += 48 + 1 # matrix + flag
        # version-specific x_count skip
        if version == 11:
            mp += 1 # skip x_count (uint8 in v11 too but parsed differently)
            x_count = unpack('<H', file_data[mp:mp+2])[0]; mp += 2
            mp += 4 * x_count
        else:
            x_count = file_data[mp]; mp += 1
            mp += 1 + 4 * x_count
        u_count = unpack('<H', file_data[mp:mp+2])[0]; mp += 2
        for _ in range(u_count):
            mp += 64 # matrix
            mp += 2 # skeleton index
        # Pre-LOD bytes
        if u_count > 0 and version not in (11, 12):
            mp += 1 if version in (13, 14) else 2 # root_bone_index
            lod_info_type = file_data[mp]; mp += 1
        else:
            if version in (11, 12, 13):
                lod_info_type = 0
            else:
                lod_info_type = file_data[mp]; mp += 1
        lod_count_scan = file_data[mp]; mp += 1
        mp += 4 # unknown 4 bytes
        lod_field_size = 40 if version == 11 else 36
        for li_scan in range(lod_count_scan):
            lod_start = mp
            # data_offset field is at byte offset 24 within the 36-byte LOD header
            # (or 28 in v11 due to the extra uint32)
            lod_fo = 4 if version == 11 else 0 # lod_field_offset
            do_field = lod_start + 4 + lod_fo + 20 # start+vc(4)+lod_fo+ic(4)+sa(4)+voa(4)+vob(4)
            old_do = unpack('<I', file_data[do_field:do_field+4])[0]
            file_data[do_field:do_field+4] = pack('<I', old_do + inserted)
            # Shift the second-section absolute offset (field [5], data_offset field + 32;
            # see Mesh.parse) too.
            if lod_info_type == 2:
                sp = do_field + 32
                sec2_val = unpack('<I', file_data[sp:sp+4])[0]
                if sec2_val > 0:
                    file_data[sp:sp+4] = pack('<I', sec2_val + inserted)
            # Also update the in-memory lod.data_offset_file_pos value - we patch the file
            # positions here, and will sync asset afterwards.
            mp += lod_field_size
            if lod_info_type == 2:
                mp += 28
        # Tail section - skip to next mesh
        uv_count_scan = file_data[mp]; mp += 1
        mp += 4 * uv_count_scan
        if version == 11:
            pass # no color_count in v11
        elif version in (16, 17):
            cc = file_data[mp]; mp += 1
            mp += 4 * cc + 4
            count_c = file_data[mp]; mp += 1
            mp += 4 * count_c
        else:
            mp += 4 # unk
            cc = file_data[mp]; mp += 1
            mp += 4 * cc
        mp += 4 # vs + ns (uint16 each)
        mp += 20 if version == 17 else 16 # post-stride skip

    # Write the patched file
    with open(mod_file, 'wb') as f:
        f.write(file_data)

    # Update in-memory asset.bones so the rest of the session works correctly
    for orig_idx, name, mat_raw, matrix, pidx in new_bones:
        if pidx == 65535:
            combined_pidx = 65535
        else:
            combined_pidx = donor_to_combined.get(pidx, 65535)
        new_b = SkeletalMeshAsset.Bone.__new__(SkeletalMeshAsset.Bone)
        new_b.name = name
        new_b.matrix = matrix
        new_b.parent_index = combined_pidx
        asset.bones.append(new_b)
    asset.bone_count = len(asset.bones)

    # Sync lod.data_offset values in memory (they were bumped in the file)
    for m_mem in asset.meshes:
        for lod_mem in m_mem.lods:
            lod_mem.data_offset += inserted
            lod_mem.data_offset_file_pos += inserted
            lod_mem.start_offset += inserted

    # Rebuild the Blender armature from the merged skeleton
    arm_obj = bpy.data.objects.get(asset.name)
    if arm_obj is not None and arm_obj.type == 'ARMATURE':
        # Collect names of every mesh object currently parented to the armature so we can re-parent them afterwards.
        child_meshes = [obj for obj in arm_obj.children if obj.type == 'MESH']

        # Remove the old armature object and data entirely.
        old_arm_data = arm_obj.data
        bpy.data.objects.remove(arm_obj, do_unlink=True)
        bpy.data.armatures.remove(old_arm_data)

        # Re-import the skeleton. import_skeleton reads asset.bones (already updated
        # to include the new bones) and builds a new correctly-transformed
        # armature via the exact same path as a normal LOD import.
        new_arm_obj = BMI.import_skeleton(asset)

        # import_skeleton only applies the X-flip. rotate_model applies the 90deg X
        # rotation that brings the armature into the correct viewport orientation.
        # It is normally called from ImportLOD, so we call it explicitly here.
        dummy = child_meshes[0] if child_meshes else new_arm_obj
        BMI.rotate_model(dummy, new_arm_obj)

        # Restore armature modifiers and parenting on all child meshes.
        # Reset matrix_parent_inverse to 'identity' so the mesh sits correctly
        # relative to the rebuilt armature (same as the original import).
        for mesh_obj in child_meshes:
            mesh_obj.parent = new_arm_obj
            mesh_obj.matrix_parent_inverse = Matrix.Identity(4)
            arm_mod = mesh_obj.modifiers.get('Armature')
            if arm_mod is not None:
                arm_mod.object = new_arm_obj

    SWOMT.AssetPath = mod_file
    operator.report({'INFO'},
        f"Merged {len(new_bones)} new bone(s) ({mode_label}) from '{os.path.basename(src_filepath)}' "
        f"into '{os.path.basename(mod_file)}'. "
        f"Skeleton now has {len(asset.bones)} bones.")
    return {'FINISHED'}

def _all_donor_bones_field_text():
    """The bone_names field text representing 'every donor bone selected'."""
    if not _donor_bone_names_cache:
        return ""
    return ", ".join(_donor_bone_names_cache) + ", "

def _select_all_update_cb(self, context):
    """Fills or clears bone_names when the 'Select All' checkbox changes."""
    self.bone_names = _all_donor_bones_field_text() if self.select_all else ""

class MergeSkeletonsPickBones(bpy.types.Operator):
    """Choose which donor bones to merge (shown after selecting the donor .mmb)"""
    bl_idname = "object.merge_skeletons_pick_bones"
    bl_label = "Choose Bones to Merge"
    bl_options = {'REGISTER'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    select_all: bpy.props.BoolProperty(
        name="Select All",
        description="Merge every bone in the donor skeleton. Un-check to select specific bone names.",
        default=True,
        update=_select_all_update_cb,
    )
    bone_names: bpy.props.StringProperty(
        name="Bones",
        description="Comma-separated list of donor bone names to merge.",
        search=_donor_bone_search_cb,
    )

    @classmethod
    def poll(cls, context):
        return asset is not None

    def invoke(self, context, event):
        global _donor_bone_names_cache
        donor_bones = _get_cached_donor_bones(self.filepath)
        if donor_bones is None:
            self.report({'ERROR'}, "Failed to read donor .mmb skeleton.")
            return {'CANCELLED'}
        _donor_bone_names_cache = [name for name, mat_raw, matrix, pidx in donor_bones]
        # Default to 'Select All'.
        self.select_all = True
        self.bone_names = _all_donor_bones_field_text()
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        layout = self.layout
        layout.label(text=f"Donor: {os.path.basename(self.filepath)}", icon="FILE")
        layout.label(text=f"{len(_donor_bone_names_cache)} bone(s) available in donor skeleton.")
        layout.separator()
        layout.prop(self, "select_all")
        col = layout.column()
        col.enabled = not self.select_all
        col.prop(self, "bone_names", text="")
        layout.label(text="Parents required to keep the skeleton connected are added automatically.")

    def execute(self, context):
        donor_bones = _get_cached_donor_bones(self.filepath)
        if donor_bones is None:
            self.report({'ERROR'}, "Failed to read donor .mmb skeleton.")
            return {'CANCELLED'}

        requested = [n.strip() for n in self.bone_names.split(",") if n.strip()]
        if not requested:
            self.report({'ERROR'}, "Enter at least one bone name to merge, or check 'Select All'.")
            return {'CANCELLED'}

        donor_names = {name for name, mat_raw, matrix, pidx in donor_bones}
        unknown = [n for n in requested if n not in donor_names]
        if unknown:
            self.report({'ERROR'}, f"Bone(s) not found in donor skeleton: {', '.join(unknown)}")
            return {'CANCELLED'}

        filtered_bones = _resolve_selected_donor_bones(donor_bones, requested)
        mode_label = "all bones" if self.select_all else "selected bones"
        return _do_merge_skeletons(context, self, self.filepath, filtered_bones, mode_label)

class MergeSkeletons(bpy.types.Operator):
    """Merge bones from a donor .mmb skeleton into the currently-loaded asset skeleton."""

    # New donor bones are appended to the skeleton (via a _MOD copy) and to asset.bones
    # in memory. The armature is rebuilt so Add/Remap Bone can reference them immediately.

    bl_idname = "object.merge_skeletons"
    bl_label = "Merge Skeleton"
    bl_options = {'REGISTER'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.mmb", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return asset is not None

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        if not self.filepath or not os.path.isfile(self.filepath):
            self.report({'ERROR'}, "Please select a valid .mmb file.")
            return {'CANCELLED'}

        # Hand off to the bone-picker dialog; it reads the donor file itself and
        # performs the merge once the user confirms which bones to include.
        bpy.ops.object.merge_skeletons_pick_bones('INVOKE_DEFAULT', filepath=self.filepath)
        return {'FINISHED'}

class AddBonesFromVertexGroups(bpy.types.Operator):
    """Add a bone slot for every vertex group to this mesh's bone table"""

    # Scan the imported mesh objects for this asset entry and add a bone slot for every
    # vertex group whose name matches a skeleton bone but is not yet in the bone table.
    # Inverse bind matrices are derived automatically from the loaded skeleton.

    bl_idname = "object.add_bones_from_vertex_groups"
    bl_label = "Add Bone Slots from Vertex Groups"
    bl_options = {'REGISTER'}

    mesh_index: bpy.props.IntProperty()

    @classmethod
    def poll(cls, context):
        return asset is not None

    def execute(self, context):
        mesh = asset.meshes[self.mesh_index]

        # Build a set of skeleton bone names for lookup
        skel_name_to_idx = {b.name: i for i, b in enumerate(asset.bones)}

        # Build the set of skeleton indices already in this mesh's bone table
        # (including any pending additions not yet written to file)
        existing_skel_indices = set(mesh.mesh_bones.keys())
        existing_skel_indices.update(idx for idx, _ in mesh.pending_bone_additions)

        # Collect vertex group names from all imported LOD objects for this mesh
        vg_names = set()
        for li, lod in enumerate(mesh.lods):
            obj_name = lod.blender_obj_name if lod.blender_obj_name else f"{mesh.name}_LOD{li}"
            obj = bpy.data.objects.get(obj_name)
            if obj is not None and obj.type == 'MESH':
                for vg in obj.vertex_groups:
                    vg_names.add(vg.name)

        if not vg_names:
            self.report({'WARNING'}, "No vertex groups found on any imported LOD for this mesh.")
            return {'CANCELLED'}

        added = []
        reused_slots = []
        skipped_no_bone = []
        skipped_already = []
        skipped_full = []
        # All bones here target the same mesh, so the used-slots scan only runs once.
        used_slots_cache = {}

        for vg_name in sorted(vg_names):
            skel_idx = skel_name_to_idx.get(vg_name)
            if skel_idx is None:
                skipped_no_bone.append(vg_name)
                continue
            if skel_idx in existing_skel_indices:
                skipped_already.append(vg_name)
                continue

            inv_bind = _compute_inv_bind_from_skeleton(vg_name)
            if inv_bind is None:
                self.report({'WARNING'}, f"Could not compute inv_bind for '{vg_name}' - skipping.")
                continue

            status, info = _add_or_reuse_mesh_bone_slot(mesh, skel_idx, inv_bind, used_slots_cache=used_slots_cache)
            if status == 'full':
                skipped_full.append(vg_name)
                continue
            existing_skel_indices.add(skel_idx)
            added.append(vg_name)
            if status == 'reused':
                reused_slots.append((vg_name, info))

        if added:
            msg = f"Added {len(added)} bone slot(s): {', '.join(added)}"
            if reused_slots:
                msg += f". {len(reused_slots)} reused an existing unused slot"
            self.report({'INFO'}, msg)
        elif skipped_already and not skipped_no_bone and not skipped_full:
            self.report({'INFO'}, "All vertex groups are already in the bone table.")
        else:
            missing = [n for n in sorted(vg_names) if n not in skel_name_to_idx]
            self.report({'WARNING'},
                f"No new slots added. "
                f"{len(skipped_already)} already present, "
                f"{len(skipped_full)} blocked (mesh's bone slots are full and have no "
                f"unused slot to reuse), "
                f"{len(skipped_no_bone)} not in skeleton: {', '.join(skipped_no_bone[:5])}"
                + (" ..." if len(skipped_no_bone) > 5 else ""))

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

    @staticmethod
    def _base_stem(path):
        """Return the stem with any _MOD or _MOD<n> suffix stripped."""
        stem = Path(path).stem
        return re.sub(r'_MOD\d*$', '', stem)

    def invoke(self, context, event):
        old_stem = self._base_stem(context.scene.SWOMT.AssetPath)
        bpy.types.Scene.mmb_file_rename_input = bpy.props.StringProperty(
            name="New Filename",
            maxlen=len(old_stem),
        )
        context.scene.mmb_file_rename_input = old_stem
        context.window.cursor_warp(context.window.width // 2, context.window.height // 2)
        return context.window_manager.invoke_props_dialog(self, width=380)

    def draw(self, context):
        old_stem = self._base_stem(context.scene.SWOMT.AssetPath)
        layout = self.layout
        layout.label(text=f"Original: {old_stem}")
        layout.label(text=f"Must be exactly {len(old_stem)} characters")
        layout.prop(context.scene, "mmb_file_rename_input", text="New Filename")

    def execute(self, context):
        SWOMT = context.scene.SWOMT
        old_path = Path(SWOMT.AssetPath)
        old_stem = self._base_stem(SWOMT.AssetPath)
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
        mod_file = _mod_file_output(src_path, overwrite=SWOMT.overwrite_existing)

        # Bake parent inverse and triangulate every LOD object that exists in the scene.
        # Baking must happen before triangulation so vertex positions are correct.
        original_data = {}  # obj_name -> (obj, original_data)
        for m in asset.meshes:
            for li, lod in enumerate(m.lods):
                lod_obj_name = lod.blender_obj_name or f"{m.name}_LOD{li}"
                tri_obj = BME.find_object_by_name(lod_obj_name)
                if tri_obj:
                    original_data[lod_obj_name] = (tri_obj, tri_obj.data)
                    BME._triangulate_object(tri_obj,
                                            compute_normals=context.scene.SWOMT.compute_normals_on_export)

        # Export each LOD level in order (0 -> 3). After the first pass has
        # written mod_file, subsequent passes read from it so each LOD level
        # accumulates on top of the previous one.
        exported_any = False
        current_src = None  # None -> _write_mod_file reads SWOMT.AssetPath
        try:
            for lod_n in range(4):
                edited = {}
                for m in asset.meshes:
                    if len(m.lods) <= lod_n:
                        continue
                    lod = m.lods[lod_n]
                    obj_name = lod.blender_obj_name or f"{m.name}_LOD{lod_n}"
                    obj = BME.find_object_by_name(obj_name)
                    if obj is None:
                        continue
                    edited[m.index] = lod_n
                if not edited:
                    continue
                try:
                    BME._write_mod_file(edited_lod_index_per_mesh=edited, out_path=mod_file,
                                        src_path=current_src)
                    exported_any = True
                    current_src = mod_file
                except Exception as e:
                    self.report({'ERROR'}, f"Export LOD{lod_n} failed: {e}")
                    return {'CANCELLED'}
        finally:
            # Always restore every object's original mesh data so the user's mesh is untouched
            for obj_name, (obj, orig_data) in original_data.items():
                export_data = obj.data
                obj.data = orig_data
                if export_data != orig_data:
                    bpy.data.meshes.remove(export_data)

        if not exported_any:
            self.report({'WARNING'}, "No LOD objects found in scene to export.")
            return {'CANCELLED'}

        # Apply header-level patches once after all LODs are written
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
            SWOMT.AssetPath = new_file
            self.report({'INFO'}, f"Exported -> {os.path.basename(new_file)}")
        else:
            SWOMT.AssetPath = mod_file

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
        layout.prop(SWOMT, "overwrite_existing")

        layout.separator()
        row = layout.row()
        row.operator("object.compute_normals", text="Compute Normals", icon="NORMALS_FACE")
        row.operator("object.clear_custom_normals", text="Clear Normals", icon="REMOVE")

        layout.separator()
        row = layout.row()
        if asset:
            row.label(text=asset.name)
            row.operator("object.rename_mmb_file", text="", icon="GREASEPENCIL")
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
            exp_row.enabled = any(
                len(m.lods) > lod_n and bpy.data.objects.get(
                    m.lods[lod_n].blender_obj_name if m.lods[lod_n].blender_obj_name else f"{m.name}_LOD{lod_n}"
                ) is not None
                for m in asset.meshes if m.lods
                for lod_n in range(len(m.lods))
            )
            exp_row.operator("object.export_all_lods", text="Export All LODs")
            pose_row = layout.row()
            pose_row.scale_y = 1.2
            arm_obj = bpy.data.objects.get(asset.name) if asset else None
            pose_row.enabled = arm_obj is not None and arm_obj.type == 'ARMATURE'
            pose_row.operator("object.export_posed_bone_matrices",
                              text="Export Pose as New Rest Pose", icon="ARMATURE_DATA")
            force_row = layout.row(align=True)
            cfg_selected = bool(SWOMT.force_lod0_cfg_path.strip())
            btn_text = "Update 'lod_presets.cfg' file (Force LOD0)" if cfg_selected else "Generate 'lod_presets.cfg' file (Force LOD0)"
            force_row.operator("object.force_lod0", text=btn_text, icon='FILE_NEW')
            force_row.operator("object.browse_lod_presets_cfg", text="", icon='FILEBROWSER')
            if cfg_selected:
                force_row.operator("object.clear_lod_presets_cfg", text="", icon='REMOVE')
            if SWOMT.force_lod0_output_path:
                info_box = layout.box()
                if cfg_selected:
                    info_box.label(text="Updated file:", icon='INFO')
                    info_box.label(text=SWOMT.force_lod0_output_path)
                else:
                    info_box.label(text="Place generated file at:", icon='INFO')
                    info_box.label(text=r"...\AFOP\rogue\modules\core\graphobject\lod_presets.cfg")

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
                for prop_name in ("export_normals", "export_weights", "export_uvs"):
                    prop_row = box.row()
                    prop_row.enabled = not forced
                    prop_row.prop(SWOMT, prop_name)
                    if forced:
                        prop_row.label(text="", icon='LOCKED')
                box.prop(SWOMT, "export_vertex_colors")

            if forced:
                tip_row = layout.row()
                tip_row.label(text="Tip: Transfer Weights from original mesh", icon='INFO')
            for mi, m in enumerate(asset.meshes):
                expanded = SWOMT.mesh_expanded[mi] if mi < 32 else True
                mesh_row = layout.row()
                mesh_box = mesh_row.box()
                name_row = mesh_box.row()
                name_row.prop(SWOMT, "mesh_expanded", index=mi, text="",
                              icon='TRIA_DOWN' if expanded else 'TRIA_RIGHT', emboss=False)
                name_row.label(text=m.name, icon="MESH_ICOSPHERE")
                scale_uv_op = name_row.operator("object.scale_uvs", text="", icon="UV")
                scale_uv_op.mesh_index = mi
                rename_op = name_row.operator("object.rename_mesh", text="", icon="GREASEPENCIL")
                rename_op.mesh_index = mi
                remove_row = name_row.row()
                remove_row.enabled = not (m.zeroed_out_in_session or m.zeroed_out_in_mmb)
                remove_op = remove_row.operator("object.remove_mesh", text="", icon="X")
                remove_op.mesh_index = mi
                revert_row = name_row.row()
                revert_row.enabled = m.zeroed_out_in_session
                revert_op = revert_row.operator("object.revert_mesh", text="", icon="LOOP_BACK")
                revert_op.mesh_index = mi
                if expanded:
                    for li,l in enumerate(m.lods):
                        row = mesh_box.row()
                        icon = "CON_SIZELIKE"
                        if m.zeroed_out_in_session or m.zeroed_out_in_mmb:
                            icon = "STRIP_COLOR_01"
                        row.label(text=f"LOD{li} - {l.vertex_count}", icon=icon)
                        lod_import_button = row.operator("object.import_lod")
                        lod_import_button.lod_index = li
                        lod_import_button.mesh_index = mi
                        obj_name = l.blender_obj_name if l.blender_obj_name else f"{m.name}_LOD{li}"
                        lod_export_row = row.row()
                        lod_export_row.enabled = bpy.data.objects.get(obj_name) is not None
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
                        vg_op = bone_header.operator("object.add_bones_from_vertex_groups", text="", icon="GROUP_VERTEX")
                        vg_op.mesh_index = mi
                        bone_header.operator("object.merge_skeletons", text="", icon="ARMATURE_DATA")
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

        # Update data files
        warnings = []
        for url, filename in [
            (_LOD_CFG_URL, _LOD_CFG_FILENAME),
            (_MMB_JSON_URL, _MMB_JSON_FILENAME),
        ]:
            ok, err = _download_data_file(url, filename)
            if not ok:
                warnings.append(f"{filename}: {err}")

        global _update_status
        _update_status = None
        if warnings:
            self.report({'WARNING'}, f"Updated! Restart Blender to apply. Failed to download: {', '.join(warnings)}")
        else:
            self.report({'INFO'}, "Updated! Restart Blender to apply.")
        return {'FINISHED'}


class ExportPosedBoneMatrices(bpy.types.Operator):
    """Export the armature's current pose. Use this after posing the armature in 'Pose Mode'."""
    bl_idname = "object.export_posed_bone_matrices"
    bl_label = "Export Pose as New Rest Pose"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        if asset is None:
            return False
        arm_obj = bpy.data.objects.get(asset.name)
        return arm_obj is not None and arm_obj.type == 'ARMATURE'

    def execute(self, context):
        SWOMT = context.scene.SWOMT
        src_path = SWOMT.AssetPath
        arm_obj = bpy.data.objects.get(asset.name)

        if arm_obj is None or arm_obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Armature object not found in scene.")
            return {'CANCELLED'}

        # Build pose_bone armature-local matrices for every bone.
        arm_pose_local = {} # bone_name -> Matrix (armature-local)
        for pb in arm_obj.pose.bones:
            arm_pose_local[pb.name] = pb.matrix.copy()

        S = Matrix.Scale(-1.0, 4, Vector((1.0, 0.0, 0.0))) # diag(-1,1,1,1)

        def pose_to_file_world(pose_mat):
            """Convert pose_bone.matrix (Blender armature-local) to MMB file-space world matrix."""
            return S @ pose_mat

        # Determine which bones are actually posed (differ from rest pose).
        POSE_EPSILON = 1e-4

        def _is_posed(pb):
            """Return True if this bone's pose differs from its rest."""
            rest_al = pb.bone.matrix_local
            pose_al = pb.matrix
            for r in range(4):
                for c in range(4):
                    if abs(pose_al[r][c] - rest_al[r][c]) > POSE_EPSILON:
                        return True
            return False

        posed_bones = {pb.name for pb in arm_obj.pose.bones if _is_posed(pb)}

        # Patch skeleton section in a copy of the file bytes.
        mod_file = _mod_file_output(src_path, overwrite=SWOMT.overwrite_existing)

        with open(src_path, 'rb') as f:
            file_data = bytearray(f.read())

        patched_skel = 0
        # Re-walk skeleton section to get file offsets (not stored on asset.bones)
        pos = 0
        version = file_data[3]
        pos += 8
        if version >= 15:
            pos += 4

        bone_count = unpack('<I', bytes(file_data[pos:pos+4]))[0]
        pos += 4

        bone_mat_offsets = [] # file offset of each bone's 64-byte matrix
        bone_parents = []
        for i in range(bone_count):
            nlen = unpack('<H', bytes(file_data[pos:pos+2]))[0]
            pos += 2
            pos += nlen  # skip name
            bone_mat_offsets.append(pos)
            pos += 64  # matrix
            parent_idx = unpack('<H', bytes(file_data[pos:pos+2]))[0]
            bone_parents.append(parent_idx)
            pos += 2

        # Build bone name -> index map
        bone_name_to_idx = {b.name: i for i, b in enumerate(asset.bones)}

        # Build file_world for every bone.
        orig_file_local = {}
        _pos2 = 0
        _pos2 += 8
        if file_data[3] >= 15:
            _pos2 += 4
        _bc2 = unpack('<I', bytes(file_data[_pos2:_pos2+4]))[0]
        _pos2 += 4
        for _i in range(_bc2):
            _nlen = unpack('<H', bytes(file_data[_pos2:_pos2+2]))[0]
            _pos2 += 2
            _bname = bytes(file_data[_pos2:_pos2+_nlen]).decode('ascii', errors='replace')
            _pos2 += _nlen
            _raw = unpack('<16f', bytes(file_data[_pos2:_pos2+64]))
            _pos2 += 64
            _pos2 += 2
            orig_file_local[_bname] = Matrix([
                [_raw[0], _raw[4], _raw[8],  _raw[12]],
                [_raw[1], _raw[5], _raw[9],  _raw[13]],
                [_raw[2], _raw[6], _raw[10], _raw[14]],
                [_raw[3], _raw[7], _raw[11], _raw[15]],
            ])

        orig_file_world = {}
        for _i, _b in enumerate(asset.bones):
            _pidx = bone_parents[_i]
            if _pidx == 65535:
                orig_file_world[_b.name] = orig_file_local.get(_b.name, Matrix.Identity(4))
            else:
                _pname = asset.bones[_pidx].name
                orig_file_world[_b.name] = orig_file_world.get(_pname, Matrix.Identity(4)) @ orig_file_local.get(_b.name, Matrix.Identity(4))

        file_world = {}
        for pb in arm_obj.pose.bones:
            if pb.name in posed_bones:
                file_world[pb.name] = pose_to_file_world(pb.matrix)
            else:
                file_world[pb.name] = orig_file_world.get(pb.name, pose_to_file_world(pb.matrix))

        for bone_name, fw in file_world.items():
            # Only write bones that were actually posed
            if bone_name not in posed_bones:
                continue
            bi = bone_name_to_idx.get(bone_name)
            if bi is None:
                continue
            parent_idx = bone_parents[bi]
            if parent_idx == 65535:
                parent_fw = Matrix.Identity(4)
            else:
                parent_bone_name = asset.bones[parent_idx].name
                parent_fw = file_world.get(parent_bone_name, Matrix.Identity(4))

            try:
                new_local = parent_fw.inverted() @ fw
            except ValueError:
                continue

            # The translation delta in file_local must be negated relative to the original.
            orig_local = orig_file_local.get(bone_name)
            if orig_local is not None:
                orig_trans = orig_local.col[3].copy()
                new_trans  = new_local.col[3].copy()
                delta = new_trans - orig_trans
                new_local.col[3] = orig_trans - delta

            # Write new local matrix (row-major 4x4 floats) to file_data
            flat = [new_local[r][c] for c in range(4) for r in range(4)]
            offset = bone_mat_offsets[bi]
            file_data[offset:offset+64] = pack('<16f', *flat)
            patched_skel += 1

        # Patch mesh bone slot (inv_bind) matrices.
        # inv_bind = file_world_matrix.inverted()
        patched_slots = 0
        for mi, mesh in enumerate(asset.meshes):
            mesh_bones_list = list(mesh.mesh_bones.keys())
            for slot_idx, skel_idx in enumerate(mesh_bones_list):
                if skel_idx >= len(asset.bones):
                    continue
                bone_name = asset.bones[skel_idx].name
                # Only update inv_bind for bones that were actually posed
                if bone_name not in posed_bones:
                    continue
                fw = file_world.get(bone_name)
                if fw is None:
                    continue
                try:
                    new_inv_bind = fw.inverted()
                except ValueError:
                    self.report({'WARNING'}, f"Could not invert matrix for bone '{bone_name}' - skipping.")
                    continue
                flat = tuple(new_inv_bind[r][c] for c in range(4) for r in range(4))
                mesh.pending_bone_remaps[slot_idx] = (skel_idx, flat)
                patched_slots += 1

        if patched_skel == 0 and patched_slots == 0:
            self.report({'WARNING'}, "No bones were updated. Was the armature posed?")
            return {'CANCELLED'}

        # Write the skeleton-patched file_data to disk first
        with open(mod_file, 'wb') as f:
            f.write(file_data)

        # Then apply mesh inv_bind patches via the existing header-patch path.
        # _apply_header_patches writes bone remaps directly into the file on disk.
        for mesh in asset.meshes:
            BME._apply_header_patches(mod_file, mesh, asset, operator=self)

        SWOMT.AssetPath = mod_file
        self.report({'INFO'},
            f"Pose exported: {patched_skel} skeleton bone(s) ({len(posed_bones)} posed), "
            f"{patched_slots} inv_bind slot(s) -> {os.path.basename(mod_file)}")
        return {'FINISHED'}


classes=[SWOMTSettings,
         BrowseMMBFile,
         BrowseLodPresetsCfg,
         ClearLodPresetsCfg,
         ComputeNormals,
         ClearNormals,
         ImportAllLOD0s,
         ImportAllLOD1s,
         ImportAllLOD2s,
         ImportAllLOD3s,
         ExportAllLODs,
         ForceLOD0,
         RemapMeshBone,
         AddMeshBone,
         MergeSkeletons,
         MergeSkeletonsPickBones,
         AddBonesFromVertexGroups,
         CheckForUpdates,
         ApplyUpdate,
         SWOMTPanel,
         LoadMMB,
         ImportLOD,
         ExportLOD,
         ScaleUVs,
         RenameMesh,
         SelectMGraphObject,
         RenameMMBFile,
         SelectMGraphObjectFilePatch,
         ExportPosedBoneMatrices,
         RemoveMesh,
         RevertMesh]

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.SWOMT = bpy.props.PointerProperty(type=SWOMTSettings)
    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)
    # Kick off background version check on startup
    threading.Thread(target=_check_update_thread, daemon=True).start()
    # Download data files if they are missing
    _check_data_files()

def unregister():
    for c in classes:
        bpy.utils.unregister_class(c)
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)

if __name__ == "__main__":
    register()

# Original author: AlexPo
# Modified by: JasperZebra — Avatar: Frontiers of Pandora (.mmb version 13) support
# Further modified — multi-version support: v12, v13, v15, v16, v17
#   - Added v12/v15/v16 parsing (multi-version support)
#   - Fixed pre-LOD section: root_bone_index and lod_info_type are version/u_count conditional
#   - Fixed tail section: UV hashes are 4 bytes on all versions; unk field order differs v12-15 vs v16-17
#   - Formula-based position type detection: normals_base = ns - 4*uv - 4*col; 28→float, 12→int16
#   - All fixes are version-conditional (additive only — existing versions unchanged)

bl_info = {
    "name": "AFoP Mesh Tool",
    "author": "JasperZebra, J-Lyt",
    "location": "Scene Properties > AFoP Mesh Tool Panel",
    "version": (0, 1, 22),
    "blender": (5, 0, 0),
    "description": "Imports skeletal meshes from AFoP .mmb files. Supports versions 12, 13, 15, 16, 17.",
    "category": "Import-Export"
    }

import bpy
import bmesh
from struct import unpack, pack
import numpy as np
import math
from mathutils import Matrix, Euler, Vector
from pathlib import Path
import os
import io
import urllib.request
import threading
import re

# Auto-update
_RAW_URL = "https://raw.githubusercontent.com/J-Lyt/AFoPMeshTool/master/__init__.py"
_update_status = None   # None = not checked, "up_to_date", or "vX.X.X available"
_update_error  = None   # set if network fetch failed

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
            i = int(v * ((2 ** 8)-1))
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
        if -1.0 < v < 1.0:
            # if v >= 0:
            #     v = int(abs(v) * (2 ** 15))
            # else:
            #     v = 2 ** 16 - int(abs(v) * (2 ** 15))
            v = int(v * (2 ** 15))
        else:
            raise Exception("Couldn't normalize value as int16Norm, it wasn't between -1.0 and 1.0. Unknown max value.")
        return pack('<h', v)
    @staticmethod
    def uint16_norm(v):
        if 0.0 < v < 1.0:
            i = v * (2 ** 16) - 1
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
                self.parent_mesh:SkeletalMeshAsset.Mesh = parent_mesh
                self.index = index
                self.vertex_count = 0
                self.index_count = 0
                self.vertex_data_offset_a = 0
                self.vertex_data_offset_b = 0
                self.face_block_offset = 0
                self.data_offset = 0
                self.data_size = 0
                self.blender_obj_name = ""  # set at import time; used by exporter to find the object
            def parse(self, f):
                self.vertex_count = br.uint32(f)
                self.index_count = br.uint32(f)
                unknown_size = br.uint32(f)
                self.vertex_data_offset_a = br.uint32(f)
                self.vertex_data_offset_b = br.uint32(f)
                self.face_block_offset = br.uint32(f)
                self.data_offset = br.uint32(f)
                self.data_size = br.uint32(f)
                lod_screen_size = br.float(f)

            def get_vertex_positions(self,raw_mesh_file):
                vertices = []
                stride = self.parent_mesh.vertex_stride
                f = raw_mesh_file
                f.seek(self.vertex_data_offset_a)
                pos = (0.0,0.0,0.0)
                for v in range(self.vertex_count):
                    stride_start = f.tell()
                    if self.parent_mesh.position_type == 0:
                        x = br.int16_norm(f)
                        y = br.int16_norm(f)
                        z = br.int16_norm(f)
                        w = br.int16(f)
                        pos = (x*w,y*w,z*w)
                    elif self.parent_mesh.position_type == 1:
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
                        f.seek(8, 1)
                        weights = [br.uint16(f) / 32767.0 for _ in range(8)]
                        indices = [br.uint8(f) for _ in range(8)]
                        for i in range(8):
                            if weights[i] > 0.0:
                                iw[indices[i]] = weights[i]
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
                        remaining = stride - pos_skip - 24
                        f.seek(remaining, 1)
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
                f = raw_mesh_file
                f.seek(self.vertex_data_offset_b)
                v = Vector((0.0,0.0,1.0))
                for i in range(self.vertex_count):
                    stride_start = f.tell()
                    if self.parent_mesh.normal_type == 0:
                        x = br.int8_norm(f) *-1
                        y = br.int8_norm(f)
                        z = br.int8_norm(f)
                        w = br.int8(f)
                        v = Vector((x*w, y*w, z*w)).normalized()
                        v.negate()  # TODO not sure about this
                    elif self.parent_mesh.normal_type == 1:
                        x = br.float(f) *-1
                        y = br.float(f)
                        z = br.float(f)
                        v = Vector((x,y,z)).normalized()
                    f.seek(stride_start + stride)
                    normals.append(v)
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
                f = raw_mesh_file
                f.seek(self.vertex_data_offset_b)
                for i in range(self.vertex_count):
                    stride_start = f.tell()
                    f.seek(self.get_normals_size(), 1) #skip normals
                    f.seek(4 * color_count, 1) #skip color
                    f.seek(index * 4, 1)  # skip previous uv
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
                f = raw_mesh_file
                f.seek(self.vertex_data_offset_b)
                for i in range(self.vertex_count):
                    stride_start = f.tell()
                    f.seek(self.get_normals_size(), 1) #skip normals
                    f.seek(index * 4, 1) #skip previous color
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
                :param pos: (x,y,z)
                :param scale: the per-vertex w (scale) value read from the original file.
                              If None or 0, the vertex is a zero-displacement override vertex
                              and is skipped (the original bytes are left intact).
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
                    f.write(bp.int16_norm(x / scale))
                    f.write(bp.int16_norm(y / scale))
                    f.write(bp.int16_norm(z / scale))
                    f.write(bp.int16(scale))
                elif self.parent_mesh.position_type == 1:
                    f.write(bp.float(x))
                    f.write(bp.float(y))
                    f.write(bp.float(z))
                f.seek(stride_start + stride)

        def __init__(self,parent_sk_mesh):
            self.parent_sk_mesh:SkeletalMeshAsset = parent_sk_mesh
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
            self.color_count = 0
            self.uv_count = 0
            self.normal_type = 0 # 0:int8_norm 1:floats
            self.position_type = 0 # 0:int16_norm 1:floats
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
            u_count = br.uint16(f)
            for b in range(u_count):
                matrix = br.matrix_4x4(f)
                bone_index = br.uint16(f)
                self.mesh_bones[bone_index] = matrix

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
            # Each LOD is 36 bytes. When lod_info_type == 2, an extra 28 bytes follow each LOD.
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
            # normals_base > 8  → normal_type 1 (float normals, 28-byte block)
            # normals_base <= 8 → normal_type 0 (int8_norm, 4-byte block)
            _normals_base = self.normals_stride - 4 * self.uv_count - 4 * self.color_count
            self.normal_type = 1 if _normals_base > 8 else 0

            # --- Position type detection ---
            # Uses the same normals_base formula:
            #   normals_base == 28 → float positions (3 × float32, 12 bytes)
            #   normals_base == 12 → int16 positions (4 × int16 x/y/z/scale, 8 bytes)
            #   other              → default to int16
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
                  f'\nColor Count: {self.color_count}')

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
            mesh = self.Mesh(self)
            mesh.parse(f)
            self.meshes.append(mesh)

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

        # Store original MMB vertex index on each vertex so the order survives
        # operations like 'Mesh > Separate > By Loose Parts'.
        attr = obj_data.attributes.new(name='mmb_vertex_order', type='INT', domain='POINT')
        for i in range(lod.vertex_count):
            attr.data[i].value = i

        bm = bmesh.new()
        bm.from_mesh(obj_data)
        bm.faces.ensure_lookup_table()
        # Import UVs
        for uv_index in range(mesh.uv_count):
            uvs = lod.get_uvs(raw_mesh_file,uv_index)
            uv_layer = bm.loops.layers.uv.new(f'UVMap_{uv_index}')
            for finder, face in enumerate(bm.faces):
                for lindex, loop in enumerate(face.loops):
                    v_index = loop.vert.index
                    v_uv = (uvs[v_index][0],uvs[v_index][1]*-1+1)
                    loop[uv_layer].uv = v_uv

        # Import Colors
        for color_index in range(mesh.color_count):
            colors = lod.get_color(raw_mesh_file, color_index)
            color_layer = bm.verts.layers.float_color.new(f"Color_{color_index}")
            for v in bm.verts:
                v[color_layer] = colors[v.index]
        bm.to_mesh(obj_data)
        bm.free()
        obj_data.update()

        # Import Normals
        # Some VAT meshes store animation-texture lookup data in the normals stream
        # rather than actual surface normals (e.g. geckofish: all ny=0, nz=0).
        # Detect degenerate normals (all pointing the same direction) and skip
        # custom normal import in that case — Blender will auto-calculate them.
        computed_normals = lod.get_normals(raw_mesh_file)
        if computed_normals:
            first = computed_normals[0]
            degenerate = all(
                abs(n.dot(first)) > 0.999
                for n in computed_normals[1:min(len(computed_normals), 64)]
            )
            if not degenerate:
                obj_data.normals_split_custom_set_from_vertices(computed_normals)

        # Import Bone Weights
        weights = lod.get_bone_weights(raw_mesh_file)
        mesh_bones = list(mesh.mesh_bones.keys())
        for bone in skeletal_mesh.bones:
            obj.vertex_groups.new(name=bone.name)
        for v_index in range(lod.vertex_count):
            v_bone_weights = weights[v_index]
            for bone_index in v_bone_weights.keys():
                if bone_index < len(mesh_bones):
                    real_bone_index = mesh_bones[bone_index] # Convert mesh bone index to skeleton bone index
                    bone_name = skeletal_mesh.bones[real_bone_index].name
                    obj.vertex_groups[bone_name].add([v_index],v_bone_weights[bone_index], "ADD")
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
            bone.parent = _armature.edit_bones[parent_index]
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
    def parent_obj_to_armature(obj,armature):
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
    def find_object_by_name(name=""):
        return bpy.data.objects.get(name, None)
    @staticmethod
    def copy_mmb_file():
        """
        Takes the merged mmb file and creates a copy of it.
        :return: Path to the created file.
        """
        SWOMT = bpy.context.scene.SWOMT
        file = SWOMT.AssetPath
        print(f"file = {file}")
        merged_file = get_merged_mmb(file)
        print(f'merged file size = {merged_file.getbuffer().nbytes}')
        mod_file = os.path.splitext(file)[0] + "_MOD.mmb"
        print(f'mod file = {mod_file}')
        if os.path.exists(mod_file):
            return mod_file
        else:
            with open(mod_file, 'wb') as w:
                CopyFile(merged_file, w, 0, merged_file.getbuffer().nbytes)
        return mod_file
    @staticmethod
    def overwrite_vertex_positions(file, skeletal_mesh:SkeletalMeshAsset, mesh:SkeletalMeshAsset.Mesh, lod_index = 0):
        lod_tracked = mesh.lods[lod_index].blender_obj_name
        obj_lookup = lod_tracked if lod_tracked else mesh.name + f"_LOD{lod_index}"
        obj = BME.find_object_by_name(obj_lookup)
        lod:SkeletalMeshAsset.Mesh.LOD = mesh.lods[lod_index]
        if obj:
            data = obj.data
            with open(file,'rb+') as f:
                if mesh.position_type == 0:
                    original_w = []
                    f.seek(lod.data_offset)
                    for v in range(lod.vertex_count):
                        f.seek(6, 1)
                        w = unpack('<h', f.read(2))[0]
                        original_w.append(w)
                        f.seek(mesh.vertex_stride - 8, 1)
                else:
                    original_w = [None] * lod.vertex_count

                mmb_attr = data.attributes.get('mmb_vertex_order')
                if mmb_attr is not None and len(mmb_attr.data) == len(data.vertices):
                    blender_to_mmb = {bv_idx: mmb_attr.data[bv_idx].value for bv_idx in range(len(data.vertices))}
                    is_valid = (len(blender_to_mmb) == lod.vertex_count and
                                all(0 <= idx < lod.vertex_count for idx in blender_to_mmb.values()))
                    use_mapping = is_valid
                else:
                    use_mapping = False

                for bv_idx, bv in enumerate(data.vertices):
                    mmb_idx = blender_to_mmb[bv_idx] if use_mapping else bv_idx
                    f.seek(lod.data_offset + mmb_idx * mesh.vertex_stride)
                    lod.write_vertex_position(f, pos=bv.co * Vector((-1.0, 1.0, 1.0)), scale=original_w[mmb_idx])
    @staticmethod
    def write_vertices(file, mesh:SkeletalMeshAsset.Mesh, lod_index = 0):
        obj = BME.find_object_by_name(mesh.name+f"_LOD{lod_index}")
        lod:SkeletalMeshAsset.Mesh.LOD = mesh.lods[lod_index]
        if obj:
            data = obj.data

asset : SkeletalMeshAsset = None
BMI = BlenderMeshImporter
BME = BlenderMeshExporter

@bpy.app.handlers.persistent
def _on_load_post(filepath, *args, **kwargs):
    """Resets the asset when a .blend file is loaded."""
    global asset
    asset = None

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

class SWOMTSettings(bpy.types.PropertyGroup):
    AssetPath: bpy.props.StringProperty(name="Asset Path", subtype="FILE_PATH", update=_auto_load_mmb)
    mesh_expanded: bpy.props.BoolVectorProperty(size=32, default=tuple([False]*32))

# OPERATORS #
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
        BMI.parent_obj_to_armature(obj,armature)
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
        mod_file = BME.copy_mmb_file()
        mesh = asset.meshes[self.mesh_index]
        BME.overwrite_vertex_positions(file=mod_file,
                                       skeletal_mesh=asset,
                                       mesh=mesh,
                                       lod_index=self.lod_index)

        # Apply staged Hide Mesh — zero all LODs of this mesh in the _MOD file.
        if mesh.zeroed_out_in_session:
            with open(mod_file, 'rb+') as f:
                for lod in mesh.lods:
                    if mesh.position_type == 0:
                        original_w = []
                        f.seek(lod.data_offset)
                        for v in range(lod.vertex_count):
                            f.seek(6, 1)
                            w = unpack('<h', f.read(2))[0]
                            original_w.append(w)
                            f.seek(mesh.vertex_stride - 8, 1)
                    else:
                        original_w = [1] * lod.vertex_count
                    for v in range(lod.vertex_count):
                        f.seek(lod.data_offset + v * mesh.vertex_stride)
                        lod.write_vertex_position(f, pos=(0.0, 0.0, 0.0), scale=original_w[v] if mesh.position_type == 0 else None)

        # Apply staged mesh name rename to the _MOD copy.
        # Nulls go BEFORE the new name; length prefix stays unchanged.
        if mesh.pending_rename_new:
            padded = b'\x00' * (mesh.name_length - len(mesh.pending_rename_new)) + mesh.pending_rename_new.encode('utf-8')
            try:
                with open(mod_file, 'rb+') as f:
                    f.seek(mesh.name_offset + 2)
                    f.write(padded)
            except Exception as e:
                self.report({'ERROR'}, f"Failed to patch mesh name in mod file: {e}")
                return {'FINISHED'}

        # Apply staged file rename — move _MOD.mmb → new_stem.mmb in the same folder.
        if asset.pending_file_rename_new:
            new_file = str(Path(mod_file).parent / (asset.pending_file_rename_new + '.mmb'))
            try:
                os.replace(mod_file, new_file)
            except Exception as e:
                self.report({'ERROR'}, f"Failed to rename mod file: {e}")
                return {'FINISHED'}
            self.report({'INFO'}, f"Exported → {os.path.basename(new_file)}")

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
    bl_label = "Select MGraphObject File to Patch"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*", options={'HIDDEN'})

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

        self.report({'INFO'}, f"Patched {count} MeshName occurrence(s): '{old_name}' → '{new_name}' in {os.path.basename(self.filepath)} (backup: {os.path.basename(backup_path)})")
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
    bl_label = "Select MGraphObject to Update File References"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*", options={'HIDDEN'})

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

        self.report({'INFO'}, f"Updated {count} .mmb reference(s): '{old_stem}.mmb' → '{new_stem}.mmb' in {os.path.basename(self.filepath)} (backup: {os.path.basename(backup_path)})")
        return {'FINISHED'}


class ImportAllLOD0s(bpy.types.Operator):
    """Imports LOD0 for every mesh in the asset"""
    bl_idname = 'object.import_all_lod0s'
    bl_label = "Import All LOD0's"

    @classmethod
    def poll(cls, context):
        return asset is not None

    def execute(self, context):
        sk_mesh = asset
        SWOMT = context.scene.SWOMT
        merged_mmb = get_merged_mmb(SWOMT["AssetPath"])
        is_new_armature = bpy.data.objects.find(sk_mesh.name) == -1
        armature = BMI.find_or_create_skeleton(sk_mesh)
        for mi, mesh in enumerate(sk_mesh.meshes):
            if not mesh.lods:
                continue
            lod = mesh.lods[0]
            obj = BMI.import_mesh(merged_mmb,
                                  skeletal_mesh=sk_mesh,
                                  mesh=mesh,
                                  lod_index=lod.index)
            BMI.parent_obj_to_armature(obj, armature)
        if is_new_armature:
            BMI.rotate_model(obj, armature)
        return {'FINISHED'}


class ExportAllLOD0s(bpy.types.Operator):
    """Exports LOD0 for every mesh in the asset"""
    bl_idname = 'object.export_all_lod0s'
    bl_label = "Export All LOD0's"

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
        mod_file = BME.copy_mmb_file()
        for mesh in asset.meshes:
            if not mesh.lods:
                continue
            BME.overwrite_vertex_positions(file=mod_file,
                                           skeletal_mesh=asset,
                                           mesh=mesh,
                                           lod_index=0)
            if mesh.zeroed_out_in_session:
                with open(mod_file, 'rb+') as f:
                    for lod in mesh.lods:
                        if mesh.position_type == 0:
                            original_w = []
                            f.seek(lod.data_offset)
                            for v in range(lod.vertex_count):
                                f.seek(6, 1)
                                w = unpack('<h', f.read(2))[0]
                                original_w.append(w)
                                f.seek(mesh.vertex_stride - 8, 1)
                        else:
                            original_w = [1] * lod.vertex_count
                        for v in range(lod.vertex_count):
                            f.seek(lod.data_offset + v * mesh.vertex_stride)
                            lod.write_vertex_position(f, pos=(0.0, 0.0, 0.0), scale=original_w[v] if mesh.position_type == 0 else None)
            if mesh.pending_rename_new:
                padded = b'\x00' * (mesh.name_length - len(mesh.pending_rename_new)) + mesh.pending_rename_new.encode('utf-8')
                try:
                    with open(mod_file, 'rb+') as f:
                        f.seek(mesh.name_offset + 2)
                        f.write(padded)
                except Exception as e:
                    self.report({'ERROR'}, f"Failed to patch mesh name: {e}")
                    return {'FINISHED'}
        if asset.pending_file_rename_new:
            new_file = str(Path(mod_file).parent / (asset.pending_file_rename_new + '.mmb'))
            try:
                os.replace(mod_file, new_file)
            except Exception as e:
                self.report({'ERROR'}, f"Failed to rename mod file: {e}")
                return {'FINISHED'}
            self.report({'INFO'}, f"Exported → {os.path.basename(new_file)}")
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
        row = layout.row()
        row.prop(SWOMT, "AssetPath")

class MeshPanel(bpy.types.Panel):
    bl_label = "Mesh"
    bl_idname = "OBJECT_PT_meshpanel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "scene"
    bl_parent_id = "OBJECT_PT_swomtpanel"

    def draw(self,context):
        SWOMT = context.scene.SWOMT

        layout = self.layout
        row = layout.row()
        if asset:
            row.label(text=asset.name)
            row.operator("object.rename_mmb_file", text="Rename File")
            all_row = layout.row()
            all_row.operator("object.import_all_lod0s", text="Import All LOD0's")
            all_row.operator("object.export_all_lod0s", text="Export All LOD0's")
            for mi, m in enumerate(asset.meshes):
                expanded = SWOMT.mesh_expanded[mi] if mi < 32 else True
                mesh_row = layout.row()
                mesh_box = mesh_row.box()
                name_row = mesh_box.row()
                name_row.prop(SWOMT, "mesh_expanded", index=mi, text="",
                              icon='TRIA_DOWN' if expanded else 'TRIA_RIGHT', emboss=False)
                name_row.label(text=m.name, icon="MESH_ICOSPHERE")
                if expanded:
                    rename_op = name_row.operator("object.rename_mesh", text="Rename Mesh")
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
                        lod_export_button = row.operator("object.export_lod")
                        lod_export_button.lod_index = li
                        lod_export_button.mesh_index = mi

class CheckForUpdates(bpy.types.Operator):
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

        global _update_status
        _update_status = None
        self.report({'INFO'}, f"Updated! Restart Blender to apply.")
        return {'FINISHED'}


classes=[SWOMTSettings,
         ImportAllLOD0s,
         ExportAllLOD0s,
         ZeroOutMesh,
         RevertMesh,
         CheckForUpdates,
         ApplyUpdate,
         SWOMTPanel,
         MeshPanel,
         LoadMMB,
         ImportLOD,
         ExportLOD,
         RenameMesh,
         SelectMGraphObject,
         RenameMMBFile,
         SelectMGraphObjectFilePatch]

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.SWOMT = bpy.props.PointerProperty(type=SWOMTSettings)
    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)
    # Kick off background version check on startup
    threading.Thread(target=_check_update_thread, daemon=True).start()

def unregister():
    for c in classes:
        bpy.utils.unregister_class(c)
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)

if __name__ == "__main__":
    register()

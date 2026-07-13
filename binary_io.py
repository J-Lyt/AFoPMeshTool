"""Low-level Snowdrop binary readers and packers."""

import math
from struct import pack, unpack

import numpy as np
from mathutils import Matrix

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

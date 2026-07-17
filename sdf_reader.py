"""
sdf_reader.py - Standalone reader for AFOP (Snowdrop) SDF v41 archives.

Decrypts + parses sdf.sdftoc and extracts individual assets from the
sdf-A-NNNN.sdfdata blobs, byte-identical to the original game files. Pure Python
apart from the Oodle decompression runtime.

Pipeline (reverse-engineered from the Snowdrop engine):
  file table = XTEA(first 8 bytes) -> DES-CFB(custom S-boxes) -> zlib -> asset trie
  each asset slice in .sdfdata = Oodle (per 64KB page)
  each ENCRYPTED asset slice (DataSlice.is_encrypted) = [XTEA(first 8) -> DES-CFB(custom
    S-boxes), same key material as the file table cipher, no zlib] -> Oodle (per page)
"""
import ctypes
import os
import re
import struct
import zlib


try:
    from . import sdf_toc
except ImportError:  # Allow direct execution outside the Blender package.
    import sdf_toc

M64 = (1 << 64) - 1
M32 = 0xffffffff

# ---------------------------------------------------------------------------
# v41 DES (custom S-boxes) + XTEA  -- key material recovered from the engine
# ---------------------------------------------------------------------------
SBOX = {
  'c80': bytes.fromhex('0e00040f0d070104020e0f020b0d0801030a0a06060c0c0b0509090500030708040f010c0e0808020d04060902010b070f050c0b0903070e030a0a000506000d'),
  'cc0': bytes.fromhex('0f03010d08040e07060f0b020308040e090c070002010d0a0c060009050b0a05000d0e08070a0b010a03040f0d040102050b08060c07060c09000305020e0f09'),
  'd00': bytes.fromhex('0a0d000709000e09060303040f06050a01020d080c05070e0b0c040b020f08010d01060a040d090008060f09030800070b04010f020e0c03050b0a050e02070c'),
  'd40': bytes.fromhex('070d0d080e0b03050006060f09000a03010402070802050c0b010c0a040e0f090a03060f090000060c0a0b01070d0d080f09010403050e0b050c02070802040e'),
  'd80': bytes.fromhex('020e0c0b0402010c07040a070b0d060108050500030f0f0a0d0300090e080906040b0208010c0b070a010d0e0702080d0f06090f0c000509060a030400050e03'),
  'dc0': bytes.fromhex('0c0a010f0a040f020907020c0609080500060d01030d040e0e00070b05030b0809040e030f02050c020908050c0f030a070b000e04010a0701060d000b08060d'),
  'e00': bytes.fromhex('040d0b00020b0e070f04000908010d0a030e0c030905070c05020a0f060801060106040b0b0d0d080c010304070a0e070a090f050600080f000e05020903020c'),
  'e40': bytes.fromhex('0d01020f080d0408060a0f030b0701040a0c090503060e0b0500000e0c09070207020b01040e010709040c0a0e08020d000f060c0a090d000f0303050506080b'),
}
ROT = bytes.fromhex('01010202020202020102020202020201')  # DES key-schedule rotations
V1 = 0x0b5d2640f2ceb519   # DES key seed (byteswapped into schedule)
V2 = 0xce4a4a8629bad33b   # DES-CFB feedback IV
XKEY = [0x0b, 0x11, 0x17, 0x1f]  # XTEA key, 32 cycles


def _be680(p1, p2):
    p1 &= M64; p2 &= M64
    u2 = p1 & M32
    expand = (((u2>>0xf)&0x10000) | ((u2&0x1f)<<0x11) | ((u2&0x1f8)<<0x13)
              | ((u2&0x1f80)<<0x15) | ((u2&0x1f800)<<0x17) | ((u2&0x1f8000)<<0x19)
              | ((u2&0x1f80000)<<0x1b) | ((u2&0x1f800000)<<0x1d) | ((u2&0xf8000000)<<0x1f)
              | ((p1<<0x3f)&M64)) & M64
    p2 = (expand ^ p2) & M64
    u2 = (p2>>0x20) & M32
    S = SBOX
    v3 = (S['d00'][(u2>>0xe)&0x3f]<<0x34) | (S['cc0'][(u2>>0x14)&0x3f]<<0x38) | (S['c80'][(p2>>0x3a)&0x3f]<<0x3c)
    v5 = (S['dc0'][((p2>>0x1c)&M32)&0x3f]<<0x28) | (S['d80'][(u2>>2)&0x3f]<<0x2c) | (S['d40'][(u2>>8)&0x3f]<<0x30)
    v6 = v5 | v3
    v7 = (S['e00'][((p2>>0x16)&M32)&0x3f]<<0x24) | v6
    v4 = (S['e40'][((p2&M32)>>0x10)&0x3f]<<0x20) | v7
    v1 = v4 >> 0x14
    res = ((((v1 & 0x910040000)*0xc04000020)&M64 & 0x8410010000000000)
           | (((v1 & 0x20280015000)*0x20080800083)&M64 & 0x2000a6400000000)
           | ((v5 & 0x400000000000)<<4) | ((v7 & 0x180000000000)<<0x11)
           | ((v6>>6) & 0x1108000000000) | ((v4 & 0x202012000000000)<<5)
           | ((v3>>8) & 0x88000000000000) | ((v4>>7) & 0x900000000)
           | ((v7 & 0x20000000000)<<0xc)
           | ((((v1<<0x1d | v6>>0x37)&0x1001400000000aa)*0x210210008081)&M64 & 0x902c01200000000)) & M64
    res = (res ^ p1) & M64
    return ((res>>0x20) | ((p1<<0x20)&M64)) & M64


def _be910(seed):
    p2 = seed & M64
    p2 = (((((p2>>2)^p2)&0x3333000033330000)*5)&M64 ^ p2) & M64
    u6 = ((p2>>4)^p2)&0xf0f0f0f00000000
    p2 = ((u6<<4 | u6)^p2)&M64
    u6 = ((p2>>8)^p2)&0x9a000a00a200a8
    p2 = ((u6<<8 | u6)^p2)&M64
    u6 = ((p2>>0x10)^p2)&0x6c6c0000cccc
    p2 = ((u6<<0x10 | u6)^p2)&M64
    p2 = (((((p2>>1)^p2)&0x1045500500550550)*3)&M64 ^ p2)&M64
    u3 = ((p2>>0x20)^p2)&M32
    p2 = ((((u3<<32)|u3)&0xf0f0f5faf0f0f5fa)^p2)&M64
    u6 = ((p2>>8)^p2)&0x550055006a00aa
    p2 = ((u6<<8 | u6)^p2)&M64
    u6 = ((p2 & 0xffffffffffffff00) ^ ((((p2>>2)^p2)&0x333330000300)*5)) & M64
    c = u6>>0x24; d = (u6>>8)&0xfffffff
    keys = []
    for i in range(16):
        b1 = ROT[i]
        c = (((c<<(b1&0x3f))&0xfffffff) | (c>>((0x1c-b1)&0x3f)))
        d8 = d>>((0x1c-b1)&0x3f)
        d3 = (d<<(b1&0x3f))&0xfffffff
        d = d3|d8
        u2 = (d<<8 | (c<<0x24))&M64
        u5 = u2>>3
        k = (((((((d3|(d8&M32))>>10)&0x24084)*0x2040005)&M64)&0xa030000)
             | ((((((d&M32)*2)&0x820280)*0x89001)&M64)&0x110880000)
             | (((d3|(d8&M32))&0x8001)<<0x18) | (u5 & 0x2200000000000)
             | ((u2>>2)&0x10040020100000) | ((u2>>10)&0x420000040000)
             | (((u5&0x1000004c0011100)*0x4284)&M64 & 0x400082244400000)
             | ((((u2>>0xd)&0x5312400000011)*0x94200201)&M64 & 0xea40100880000000)
             | ((((u5<<6 | u2>>0x3d)&0x520040200002)*0x80000000c1)&M64 & 0x28811000200000)
             | ((((u5<<7 | u2>>0x3c)&0x22110000012001)*0x1000000610006)&M64 & 0x1185004400000000)) & M64
        keys.append(k)
    return keys


def _delta(x, mask, sh):
    dd = ((x>>sh)^x)&mask
    return (((dd<<sh)|dd)^x)&M64


def _bswap64(x):
    return int.from_bytes(x.to_bytes(8, 'little'), 'big')


_IP = ((0x55005500550055,9),(0x333300003333,0x12),(0x0f0f0f0f,0x24),(0xff00ff00,0x18),(0xff000000ff,0x18))
_FP = tuple(reversed(_IP))


def _des_keystream(block, keys):
    x = _bswap64(block)
    for mask, sh in _IP:
        x = _delta(x, mask, sh)
    for k in keys[::-1]:
        x = _be680(x, k)
    x = ((x<<0x20)|(x>>0x20))&M64
    for mask, sh in _FP:
        x = _delta(x, mask, sh)
    return x


def _xtea_dec_block(v0, v1, key, cycles=32):
    delta = 0x9e3779b9; s = (delta*cycles)&M32
    for _ in range(cycles):
        v1 = (v1 - ((((v0<<4)&M32 ^ (v0>>5)) + v0 & M32) ^ (s + key[(s>>11)&3] & M32)))&M32
        s = (s-delta)&M32
        v0 = (v0 - ((((v1<<4)&M32 ^ (v1>>5)) + v1 & M32) ^ (s + key[s&3] & M32)))&M32
    return v0, v1


# ---------------------------------------------------------------------------
# Vectorized DES (numpy) — same cipher as the scalar path above, but the per-block
# keystream is computed for a whole chunk of blocks at once (numpy uint64 ops), and
# the CFB feedback is a cumulative XOR (prefix-xor). ~100x faster than the loop.
# CHUNKED so peak memory stays small regardless of archive size (safe on low RAM).
# ---------------------------------------------------------------------------
try:
    import numpy as _np
except Exception:
    _np = None

_SB_NP = None


def _be680_vec(p1, key):
    U = _np.uint64; S = _SB_NP
    u2 = p1 & U(0xffffffff)
    expand = (((u2 >> U(0xf)) & U(0x10000)) | ((u2 & U(0x1f)) << U(0x11)) | ((u2 & U(0x1f8)) << U(0x13))
              | ((u2 & U(0x1f80)) << U(0x15)) | ((u2 & U(0x1f800)) << U(0x17)) | ((u2 & U(0x1f8000)) << U(0x19))
              | ((u2 & U(0x1f80000)) << U(0x1b)) | ((u2 & U(0x1f800000)) << U(0x1d)) | ((u2 & U(0xf8000000)) << U(0x1f))
              | (p1 << U(0x3f)))
    p2 = expand ^ key
    u2 = p2 >> U(0x20)
    v3 = (S['d00'][(u2 >> U(0xe)) & U(0x3f)] << U(0x34)) | (S['cc0'][(u2 >> U(0x14)) & U(0x3f)] << U(0x38)) | (S['c80'][(p2 >> U(0x3a)) & U(0x3f)] << U(0x3c))
    v5 = (S['dc0'][(p2 >> U(0x1c)) & U(0x3f)] << U(0x28)) | (S['d80'][(u2 >> U(0x2)) & U(0x3f)] << U(0x2c)) | (S['d40'][(u2 >> U(0x8)) & U(0x3f)] << U(0x30))
    v6 = v5 | v3
    v7 = (S['e00'][(p2 >> U(0x16)) & U(0x3f)] << U(0x24)) | v6
    v4 = (S['e40'][((p2 & U(0xffffffff)) >> U(0x10)) & U(0x3f)] << U(0x20)) | v7
    v1 = v4 >> U(0x14)
    res = (((v1 & U(0x910040000)) * U(0xc04000020) & U(0x8410010000000000))
           | ((v1 & U(0x20280015000)) * U(0x20080800083) & U(0x2000a6400000000))
           | ((v5 & U(0x400000000000)) << U(0x4))
           | ((v7 & U(0x180000000000)) << U(0x11))
           | ((v6 >> U(0x6)) & U(0x1108000000000))
           | ((v4 & U(0x202012000000000)) << U(0x5))
           | ((v3 >> U(0x8)) & U(0x88000000000000))
           | ((v4 >> U(0x7)) & U(0x900000000))
           | ((v7 & U(0x20000000000)) << U(0xc))
           | ((((v1 << U(0x1d)) | (v6 >> U(0x37))) & U(0x1001400000000aa)) * U(0x210210008081) & U(0x902c01200000000)))
    res = res ^ p1
    return (res >> U(0x20)) | (p1 << U(0x20))


def _delta_vec(x, mask, sh):
    U = _np.uint64
    d = ((x >> U(sh)) ^ x) & U(mask)
    return ((d << U(sh)) | d) ^ x


def _des_cfb_decrypt_vec(ct, keys, iv, progress=None, chunk=200_000):
    # Smaller than the original 1,000,000: with the on-disk TOC cache removed,
    # this decrypt now runs on EVERY launch of EVERY archive (not just the
    # first), so smaller DLC-sized file tables (well under 1M blocks) used to
    # get exactly ONE progress update (0% to 100% in a single jump) -- this
    # gives multiple real updates across any archive size. Numpy's per-call
    # dispatch overhead is negligible at this chunk size, unlike the pickle-
    # batch-size sensitivity found elsewhere in this file.
    global _SB_NP
    if _SB_NP is None:
        _SB_NP = {k: _np.frombuffer(v, dtype=_np.uint8).astype(_np.uint64) for k, v in SBOX.items()}
    U = _np.uint64
    rk = [U(keys[i]) for i in (15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)]
    out = bytearray(ct)
    n = len(ct) & ~7
    nblk = n // 8
    f = _np.frombuffer(iv.to_bytes(8, 'little'), dtype=_np.uint8).astype(_np.uint8).copy()
    bi = 0
    while bi < nblk:
        m = min(chunk, nblk - bi)
        seg = _np.frombuffer(ct, dtype=_np.uint64, count=m, offset=bi * 8).copy()
        in_b = _np.frombuffer(ct, dtype=_np.uint8, count=m * 8, offset=bi * 8).reshape(m, 8)
        x = seg.byteswap()                              # bswap(input block)
        for mask, sh in _IP:
            x = _delta_vec(x, mask, sh)
        for k in rk:
            x = _be680_vec(x, k)
        x = (x << U(0x20)) | (x >> U(0x20))
        for mask, sh in _FP:
            x = _delta_vec(x, mask, sh)
        ks_b = x.view(_np.uint8).reshape(m, 8)          # keystream bytes (LE)
        ksrev = ks_b[:, ::-1]                            # ks[7-i]
        term = (in_b ^ ksrev).astype(_np.uint8)
        cum = _np.bitwise_xor.accumulate(term, axis=0)   # inclusive prefix XOR
        fblk = _np.empty((m, 8), dtype=_np.uint8)
        fblk[0] = f
        if m > 1:
            fblk[1:] = f ^ cum[:-1]                      # exclusive prefix, seeded by carry
        o = ksrev ^ fblk
        out[bi * 8: bi * 8 + m * 8] = o.tobytes()
        f = (f ^ cum[-1]).astype(_np.uint8)              # carry feedback to next chunk
        bi += m
        if progress:
            progress(bi, nblk)
    return out


def decrypt_file_table(ct, progress=None):
    """XTEA(first 8) -> DES-CFB(whole) -> zlib. Returns the inflated asset trie."""
    ct = bytearray(ct)
    v0 = int.from_bytes(ct[0:4], 'little'); v1 = int.from_bytes(ct[4:8], 'little')
    v0, v1 = _xtea_dec_block(v0, v1, XKEY, 32)
    ct[0:4] = v0.to_bytes(4, 'little'); ct[4:8] = v1.to_bytes(4, 'little')
    keys = _be910(_bswap64(V1))

    if _np is not None:
        out = _des_cfb_decrypt_vec(bytes(ct), keys, V2, progress=progress)
        return zlib.decompress(bytes(out))

    # fallback: scalar loop (no numpy)
    f = list(V2.to_bytes(8, 'little'))
    out = bytearray(ct)
    n = len(ct) & ~7
    nblk = n // 8
    for bi in range(nblk):
        off = bi * 8
        IN = struct.unpack_from("<Q", ct, off)[0]
        ks = _des_keystream(IN, keys).to_bytes(8, 'little')
        inb = IN.to_bytes(8, 'little')
        o = bytes(ks[7 - i] ^ f[i] for i in range(8))
        f = [inb[i] ^ o[i] for i in range(8)]
        out[off:off + 8] = o
        if progress and (bi & 0x3fff) == 0:
            progress(bi, nblk)
    if progress:
        progress(nblk, nblk)
    return zlib.decompress(bytes(out))


# ---------------------------------------------------------------------------
# Per-asset (per-slice) encryption -- SOLVED. Same cipher/key material as the file
# table above (XTEA(first 8 bytes) -> custom-S-box DES-CFB, using the same XKEY/
# V1/V2/SBOX/ROT), just WITHOUT the trailing zlib step, because slice data is Oodle-
# compressed rather than zlib-compressed. Confirmed by decrypting real encrypted
# slices from the live "rogue" archive across every affected extension (.juice,
# .blueitemtype, .mreward, .mtalent, .mrumblelibrary, etc.) and Oodle-decompressing
# the result: sizes match exactly and the output is legible "binjuice"-format text
# (asset names, GUIDs, include paths, human-readable strings) -- not noise.
#
# Each 64KB Oodle PAGE within a multi-page encrypted slice is its own independent
# XTEA+DES-CFB stream (its own first-8-bytes XTEA, its own CFB feedback starting
# from V2) -- verified empirically: running one continuous CFB stream across the
# concatenation of all pages in a slice decrypts the first page correctly but
# corrupts every page after it (Oodle fails to decompress); decrypting each page
# independently fixes this and every page decompresses cleanly.
# ---------------------------------------------------------------------------
def _des_cfb_decrypt_page(ct):
    """XTEA(first 8 bytes) -> custom-S-box DES-CFB(remainder), same key material as
    decrypt_file_table's cipher but with no zlib step. `ct` is one independent
    encryption unit (either a whole unpaged slice, or a single Oodle page within a
    paged slice) -- see module note above."""
    ct = bytearray(ct)
    if len(ct) < 8:
        return bytes(ct)
    v0 = int.from_bytes(ct[0:4], 'little'); v1 = int.from_bytes(ct[4:8], 'little')
    v0, v1 = _xtea_dec_block(v0, v1, XKEY, 32)
    ct[0:4] = v0.to_bytes(4, 'little'); ct[4:8] = v1.to_bytes(4, 'little')
    keys = _be910(_bswap64(V1))
    n = len(ct) & ~7

    if _np is not None:
        out = bytearray(_des_cfb_decrypt_vec(bytes(ct), keys, V2))
        out[n:] = ct[n:]
        return bytes(out)

    # fallback: scalar loop (no numpy)
    f = list(V2.to_bytes(8, 'little'))
    out = bytearray(ct)
    nblk = n // 8
    for bi in range(nblk):
        off = bi * 8
        IN = struct.unpack_from("<Q", ct, off)[0]
        ks = _des_keystream(IN, keys).to_bytes(8, 'little')
        inb = IN.to_bytes(8, 'little')
        o = bytes(ks[7 - i] ^ f[i] for i in range(8))
        f = [inb[i] ^ o[i] for i in range(8)]
        out[off:off + 8] = o
    return bytes(out)


def decrypt_asset_slice(comp, page_sizes):
    """Decrypt an is_encrypted DataSlice's still-Oodle-compressed bytes (`comp`),
    ready to hand to Oodle.decompress (whole-buffer or per-page, exactly like an
    unencrypted slice). `page_sizes` is the DataSlice's own page_sizes list (may be
    None/empty/single-element for an unpaged slice)."""
    if not page_sizes or len(page_sizes) <= 1:
        return _des_cfb_decrypt_page(comp)
    out = bytearray()
    pos = 0
    for psz in page_sizes:
        if psz == 0:
            psz = 0x10000   # u16 overflow: a full 64KB page stored uncompressed
        out += _des_cfb_decrypt_page(comp[pos:pos + psz])
        pos += psz
    return bytes(out)


# ---------------------------------------------------------------------------
# Oodle
# ---------------------------------------------------------------------------
class Oodle:
    def __init__(self, dll_path):
        self.lib = ctypes.WinDLL(dll_path)
        fn = self.lib.OodleLZ_Decompress
        fn.restype = ctypes.c_int64
        fn.argtypes = [ctypes.c_char_p, ctypes.c_int64, ctypes.c_char_p, ctypes.c_int64,
                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                       ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int64, ctypes.c_int]
        self.fn = fn

    def decompress(self, src, out_size):
        ob = ctypes.create_string_buffer(out_size)
        r = self.fn(src, len(src), ob, out_size, 1, 0, 0, None, 0, None, None, None, 0, 3)
        if r != out_size:
            raise ValueError(f"Oodle returned {r}, expected {out_size}")
        return ob.raw


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------
# Full archive parsing remains cache-free here. The Blender integration owns the
# separate targeted cache containing only MMB and mcloth index records.
# The TOC's texture-header table: `dds_slot_count` (TocHeader u32 @ 0x18) slots
# of 152 bytes each, laid out as [u32 header_len <= 148][header bytes][pad].
# An asset's dds_index indexes this table DIRECTLY (slot = dds_index * 152) --
# confirmed against the Hunter v2.2.4 decompile (src/sdftoc.rs bounds-checks
# dds_index < slot_count and reads `table + index * 0x98`) and validated on all
# 4 live archives (24,000 sampled textures, 100% slice<->mip-table agreement).
# Headers are DEDUPLICATED: many textures with the same format/dims/mip layout
# share one slot. The old formula here (anchor on the first 'STF\x02' magic and
# assume it was slot index 3) happened to equal direct indexing on the DLC
# archives (their slots 0-2 are DDS headers) but was off by 3 slots on rogue
# (its slot 0 is already STF), silently giving thousands of textures a
# NEIGHBOR's header -- wrong dims/format = sheared/wrong-color previews.
STF_STRIDE = 152
STF_HDR_LEN = 76


class SdfArchive:
    def __init__(self, toc_path, oodle_dll):
        self.toc_path = toc_path
        self.data_dir = os.path.dirname(toc_path)
        self.oodle = Oodle(oodle_dll)
        self.assets = []          # list of sdf_toc.Asset
        self.header = None
        self.dds_block = b""      # raw TOC DDS block (STF texture headers)
        self._blob_locale_map = None   # lazy: (letter, idx) -> locale-suffixed path

    def load(self, progress=None, parse_progress=None):
        """Decrypt + parse the TOC. Always fresh -- no on-disk cache (see the
        module comment above `class SdfArchive` for why). progress(done, total,
        sample=None) is called during the slow DES pass; `sample` is always None
        for this phase (per-asset names aren't known until parsing completes).

        `parse_progress(done, total, sample=None)` -- same shape, a SEPARATE
        callback -- is called during the asset-trie parse (`sdf_toc.
        parse_asset_table`) that follows the DES pass. This used to be a
        completely silent, unreported phase even though it's the single
        biggest real cost in the whole TOC load (measured ~16.5s of a ~31.8s
        decrypt+zlib+parse+index total on the real "rogue" archive, bigger
        than the DES-CFB decrypt itself) -- the decrypt progress bar would
        hit 100% of its own sub-range and then the status bar would sit
        frozen for that entire stretch before indexing's own progress took
        over. `sample` here IS available (the most recently parsed asset's
        full name), unlike the DES-pass callback. Kept as a separate
        parameter (rather than reusing `progress` for both phases) so a
        caller mapping progress onto a single overall bar can give each
        phase its own weighted sub-range instead of the two phases'
        different `(done, total)` units (DES blocks vs. trie bytes) colliding
        in one linear 0-1 interpolation -- see `sdf_extract_gui.py`'s
        `LoadAllWorker.run()` for the real caller."""
        buf = open(self.toc_path, "rb").read()
        tf = sdf_toc.parse_toc(buf)
        table = decrypt_file_table(tf.file_table_compressed, progress=progress)
        self.assets = sdf_toc.parse_asset_table(table, tf.has_sign, tf.header.version,
                                                 progress=parse_progress)
        self.header = tf.header
        self.dds_block = buf[tf.preamble_end:tf.file_table_offset]

    def _texture_header(self, dds_index):
        """Header bytes the engine prepends to a texture. The TOC texture-header
        table holds 152-byte slots, each [u32 len][header bytes][pad]; the header
        is EITHER an STF descriptor (76B, 'STF\\x02') OR a standard DDS header
        (128B, or 148B with a DX10 ext, 'DDS '). dds_index selects the slot
        DIRECTLY (see the layout comment above STF_STRIDE). Returns b'' if the
        slot isn't a recognised header."""
        if dds_index is None or dds_index < 0:
            return b""
        off = dds_index * STF_STRIDE
        if off + 8 > len(self.dds_block):
            return b""
        ln = struct.unpack_from("<I", self.dds_block, off)[0]
        if not 0 < ln <= STF_STRIDE - 4:
            return b""
        hdr = self.dds_block[off + 4:off + 4 + ln]
        if hdr[:4] in (b"STF\x02", b"DDS "):
            return hdr
        return b""

    def _locale_blob_map(self):
        """Lazy one-time scan of data_dir for LOCALE-SUFFIXED data files.

        The C series (index 2000+) holds localized content (voice audio): on
        disk those files are named `sdf-C-2006-en-US.sdfdata`, not the plain
        `sdf-C-2006.sdfdata` the index alone would suggest. Maps
        (letter, idx) -> full path, preferring 'en-US' when several locales
        of the same index are installed, else the alphabetically first.
        Building the dict twice under a rare thread race is harmless (same
        result either way), so no lock is needed."""
        m = self._blob_locale_map
        if m is None:
            m = {}
            try:
                names = os.listdir(self.data_dir)
            except OSError:
                names = []
            rx = re.compile(r"^sdf-([A-Z])-(\d{4})-(.+)\.sdfdata$", re.IGNORECASE)
            for n in sorted(names):
                g = rx.match(n)
                if not g:
                    continue
                key = (g.group(1).upper(), int(g.group(2)))
                if key not in m or g.group(3).lower() == "en-us":
                    m[key] = os.path.join(self.data_dir, n)
            self._blob_locale_map = m
        return m

    def _blob(self, idx):
        # data slices are partitioned into series by index range:
        # 0-999 -> A, 1000-1999 -> B, 2000-2999 -> C, ... (matches TOC PartA/PartB/... ranges)
        letter = chr(ord('A') + idx // 1000)
        p = os.path.join(self.data_dir, f"sdf-{letter}-{idx:04d}.sdfdata")
        if os.path.isfile(p):
            return p
        # Localized series (C = voice audio): the real on-disk name carries a
        # locale suffix (sdf-C-2006-en-US.sdfdata). Resolve via the dir scan.
        alt = self._locale_blob_map().get((letter, idx))
        if alt is not None:
            return alt
        raise FileNotFoundError(
            f"data file sdf-{letter}-{idx:04d}.sdfdata is not in "
            f"{self.data_dir} (no locale-suffixed variant either) - that "
            f"part of the game data is not installed, so this asset's "
            f"content is not available")

    def extract(self, asset):
        """Return the decompressed bytes for an asset (STF header + all slices).
        Thread-safe: per-call file handles + Oodle buffers, immutable dds_block."""
        parts = []
        hdr = self._texture_header(getattr(asset, "dds_index", -1))
        if hdr:
            parts.append(hdr)
        for s in asset.data_slices:
            with open(self._blob(s.index), "rb") as f:
                f.seek(s.offset)
                comp = f.read(s.compressed_size)
            if s.is_encrypted:
                comp = decrypt_asset_slice(comp, s.page_sizes)
            if not s.is_compressed:
                parts.append(comp[:s.decompressed_size]); continue
            dec = s.decompressed_size
            if dec <= 0x10000 or not s.page_sizes or len(s.page_sizes) <= 1:
                parts.append(self.oodle.decompress(comp, dec))
            else:
                out = bytearray(); pos = 0; rem = dec
                for psz in s.page_sizes:
                    if psz == 0:
                        psz = 0x10000   # u16 overflow: a full 64KB page stored uncompressed
                    dsz = min(0x10000, rem)
                    page = comp[pos:pos+psz]
                    out += page if psz == dsz else self.oodle.decompress(page, dsz)
                    pos += psz; rem -= dsz
                parts.append(bytes(out))
        return b"".join(parts)

    def extract_to(self, asset, out_root):
        data = self.extract(asset)
        dest = os.path.join(out_root, *asset.name.split("/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)
        return dest

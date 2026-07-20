#!/usr/bin/env python3
"""
sdf/toc.py - Reader for AFOP (Avatar: Frontiers of Pandora) Snowdrop SDF
table-of-contents files (sdf.sdftoc), magic 'WEST', VERSION 41 (0x29).

Ported from the open-source Galanthus C# unpacker (SdfToc.cs / GameManager.cs /
Structs/*) and adjusted for the v41 header layout, which was reverse engineered
by matching the known anchor values:
    FileTableDecompressedSize = 15,555,624
    FileTableCompressedSize   =  7,120,055   (header field at 0x0C; the task brief
                                              calls this "TocDataSize")
    DataFileCount             =        112   (physical .sdfdata blobs referenced)
    DdsCount                  =         15
plus the 'massive\\0' .. 'ubisoft\\0' tag at offset 0x30 and the per-game
DataSliceIndexSettings hash table.

What this tool does
-------------------
1. Parses the full v41 header (every field + offset, see HEADER_LAYOUT below).
2. Parses the preamble: start tag, sign flag, signature, index-range settings,
   locales, per-slice data-file sizes, the 5000 data-file tags, the per-slice
   data-file hashes and the DDS header block.
3. Locates the compressed file table (anchored on the trailing 'massive' tag).
4. Detects the file-table codec. zlib / zstd / lz4-block / lz4-frame are all
   tried; if none match it falls back to Oodle (oo2core.dll via ctypes, if you
   drop one next to this script or pass --oodle).
5. If the file table can be decompressed it parses the asset trie (a port of
   Galanthus ParseEntry) and lists every asset: full path, data slices
   (sdfdata index / offset / compressed+decompressed size), dds index and the
   per-slice compression/encryption/oodle flags.

Findings for this specific archive are printed by the CLI; see the module
docstring at the bottom and the accompanying report for the headline result
(the v41 file table is Oodle-compressed, so a full listing needs oo2core.dll).
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import time
import zlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
MAGIC_WEST = 0x54534557                  # 'WEST' little-endian
TAG_MASSIVE = 0x006576697373616D         # 'massive\0'
TAG_UBISOFT = 0x0074666F73696275         # 'ubisoft\0'
TAG_SIZE = 0x30                          # 8 + 32 + 8
SIGNATURE_SIZE = 0x140                   # 320-byte Snowdrop signature
PAGE = 0x10000

# DataSliceIndexSettings hashes -> name (only the ones AFOP actually emits)
SETTING_HASHES = {
    0x353CFA07: "IndexRangeSizeForPartCLocalizedAudio",
    0x7E54C40B: "IndexRangeSizeForDlc",
    0x66D1031A: "StartIndexAlwaysResident",
    0xDA9609CF: "EndIndexAlwaysResident",
    0x07CDE36F: "StartIndexPartA",
    0xDC42F4E5: "EndIndexPartA",
    0x9243CB90: "StartIndexPartB",
    0xE2CF6B76: "EndIndexPartB",
    0xF8C55AE0: "StartIndexPartCLocalizedAudio",
    0x14C9D726: "EndIndexPartCLocalizedAudio",
    0xF319E8CE: "StartIndexDlc",
    0xE01C7E59: "EndIndexDlc",
    0x8253DE68: "StartIndexUnk4000",
    0x926D8FB8: "EndIndexUnk4999",
    0x986CE4C8: "MaxIndex",
}


# ---------------------------------------------------------------------------
# little binary cursor (mirrors Galanthus DataStream read path)
# ---------------------------------------------------------------------------
class Cursor:
    def __init__(self, buf: bytes, pos: int = 0):
        self.buf = buf
        self.pos = pos

    def u8(self) -> int:
        v = self.buf[self.pos]; self.pos += 1; return v

    def u16(self) -> int:
        v = struct.unpack_from("<H", self.buf, self.pos)[0]; self.pos += 2; return v

    def i32(self) -> int:
        v = struct.unpack_from("<i", self.buf, self.pos)[0]; self.pos += 4; return v

    def u32(self) -> int:
        v = struct.unpack_from("<I", self.buf, self.pos)[0]; self.pos += 4; return v

    def u64(self) -> int:
        v = struct.unpack_from("<Q", self.buf, self.pos)[0]; self.pos += 8; return v

    def bytes(self, n: int) -> bytes:
        v = self.buf[self.pos:self.pos + n]; self.pos += n; return v

    def sized_int(self, size: int, fallback: int = 0) -> int:
        """Galanthus ReadSizedInt: little-endian int of <size> bytes."""
        if size <= 0:
            return fallback
        v = 0
        for i in range(size):
            v |= self.buf[self.pos + i] << (i * 8)
        self.pos += size
        return v


# ---------------------------------------------------------------------------
# data structures
# ---------------------------------------------------------------------------
@dataclass
class DataSlice:
    decompressed_size: int
    compressed_size: int
    is_compressed: bool
    is_oodle: bool
    is_encrypted: bool
    offset: int
    index: int                       # sdfdata slice index
    page_sizes: Optional[List[int]] = None


@dataclass
class Asset:
    name: str
    hash: int
    dds_index: int
    unk: int
    data_slices: List[DataSlice] = field(default_factory=list)


@dataclass
class TocHeader:
    magic: int
    version: int
    file_table_decompressed_size: int     # 0x08
    file_table_compressed_size: int       # 0x0C  (brief: "TocDataSize")
    first_data_slice_index: int           # 0x10
    data_slice_count: int                 # 0x14  (== MaxIndex+1, drives tag loops)
    data_file_count: int                  # 0x18  -- MISNOMER, kept for compat: this
                                          # is actually the texture-header SLOT COUNT
                                          # (the dds block = this many 152-byte slots;
                                          # verified block_len == count*152 on all 4
                                          # archives, and Hunter's decompile bounds-
                                          # checks dds_index against this field).
    dds_count: int                        # 0x1C  (NOT the header-slot count; 15 on
                                          # every archive, meaning unknown)
    unk1: int                             # 0x20
    unk2: int                             # 0x24
    unk3: int                             # 0x28  (two int16)
    unk4: int                             # 0x2C  (two int16)


HEADER_LAYOUT = [
    ("magic",                        0x00, "u32",  "'WEST'"),
    ("version",                      0x04, "u32",  "41 (0x29)"),
    ("file_table_decompressed_size", 0x08, "i32",  "decompressed size of the asset file table"),
    ("file_table_compressed_size",   0x0C, "i32",  "compressed size of the asset file table (brief: TocDataSize)"),
    ("first_data_slice_index",       0x10, "i32",  "0"),
    ("data_slice_count",             0x14, "i32",  "MaxIndex+1 = number of data-slice index slots / tags"),
    ("data_file_count",              0x18, "i32",  "texture-header slot count (misnomer, see TocHeader)"),
    ("dds_count",                    0x1C, "i32",  "unknown, 15 on every archive (NOT the header count)"),
    ("unk1",                         0x20, "u32",  "unknown (maybe a decompressed size)"),
    ("unk2",                         0x24, "u32",  "unknown (maybe a compressed size)"),
    ("unk3",                         0x28, "u32",  "unknown (two int16)"),
    ("unk4",                         0x2C, "u32",  "unknown (two int16)"),
    # 0x30: start tag 'massive'..hash..'ubisoft' (0x30 bytes)
    # 0x60: hasSign(u8), flag2/isEncrypted(u8), then 0x140 signature
]


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def _is_tag(buf: bytes, off: int) -> bool:
    if off + TAG_SIZE > len(buf):
        return False
    m = struct.unpack_from("<Q", buf, off)[0]
    u = struct.unpack_from("<Q", buf, off + 40)[0]
    return m == TAG_MASSIVE and u == TAG_UBISOFT


def find_end_tag(buf: bytes) -> int:
    """Return offset of the trailing 'massive' tag (end of file table)."""
    i = buf.rfind(b"massive\x00")
    while i >= 0:
        if _is_tag(buf, i):
            return i
        i = buf.rfind(b"massive\x00", 0, i)
    raise ValueError("no end tag found")


@dataclass
class TocFile:
    header: TocHeader
    settings: dict
    locales: List[str]
    data_file_sizes: List[int]
    data_file_hashes: List[int]
    file_table_offset: int
    file_table_compressed: bytes
    has_sign: bool
    flag2_isencrypted: int
    preamble_end: int                 # where dds block ends / file table begins


def parse_toc(buf: bytes) -> TocFile:
    c = Cursor(buf)
    magic = c.u32()
    if magic != MAGIC_WEST:
        raise ValueError(f"bad magic 0x{magic:08x}, expected 'WEST'")
    version = c.u32()
    if version != 41:
        print(f"[warn] version {version} (0x{version:x}); this reader targets v41", file=sys.stderr)

    h = TocHeader(
        magic=magic, version=version,
        file_table_decompressed_size=c.i32(),
        file_table_compressed_size=c.i32(),
        first_data_slice_index=c.i32(),
        data_slice_count=c.i32(),
        data_file_count=c.i32(),
        dds_count=c.i32(),
        unk1=c.u32(), unk2=c.u32(), unk3=c.u32(), unk4=c.u32(),
    )

    # start tag @ 0x30
    if not _is_tag(buf, c.pos):
        raise ValueError(f"start tag not found at 0x{c.pos:x}")
    c.pos += TAG_SIZE

    has_sign = c.u8() != 0
    flag2 = c.u8()                  # Galanthus reads this as isEncrypted (v>=0x25)
    if has_sign:
        c.pos += SIGNATURE_SIZE

    # index-range settings: read (hash, value) pairs while the hash is recognised
    settings = {}
    while True:
        save = c.pos
        hsh = c.u32()
        val = c.i32()
        name = SETTING_HASHES.get(hsh)
        if name is None:
            c.pos = save
            break
        settings[name] = val
        if name == "MaxIndex":
            break

    n_slices = h.data_slice_count   # 5000 for this archive

    # locales: 10 x 6-byte fixed strings (empty in AFOP)
    locales = []
    for _ in range(10):
        s = c.bytes(6).split(b"\x00", 1)[0].decode("ascii", "replace")
        locales.append(s)

    # per-slice data file sizes (one uint32 per slice index)
    data_file_sizes = list(struct.unpack_from("<%dI" % n_slices, buf, c.pos))
    c.pos += 4 * n_slices

    # per-slice data file tags (validate they are all 'massive' tags)
    tags_ok = all(_is_tag(buf, c.pos + i * TAG_SIZE) for i in range(min(n_slices, 8)))
    c.pos += TAG_SIZE * n_slices

    # per-slice data file hashes (uint64 each, v>=0x25)
    data_file_hashes = list(struct.unpack_from("<%dQ" % n_slices, buf, c.pos))
    c.pos += 8 * n_slices

    # DDS header block runs from here up to the file table; we locate the file
    # table by anchoring on the trailing tag (robust), then sanity-check.
    end_tag = find_end_tag(buf)
    ft_off = end_tag - h.file_table_compressed_size
    preamble_end = c.pos            # = start of dds block

    if ft_off < preamble_end:
        raise ValueError("file table overlaps preamble; layout mismatch")

    file_table_compressed = buf[ft_off:end_tag]

    return TocFile(
        header=h, settings=settings, locales=locales,
        data_file_sizes=data_file_sizes, data_file_hashes=data_file_hashes,
        file_table_offset=ft_off, file_table_compressed=file_table_compressed,
        has_sign=has_sign, flag2_isencrypted=flag2, preamble_end=preamble_end,
    )


# ---------------------------------------------------------------------------
# file-table decompression (codec detection + optional Oodle)
# ---------------------------------------------------------------------------
def _try_oodle(comp: bytes, out_size: int, dll_path: Optional[str]) -> Optional[bytes]:
    import ctypes
    candidates = []
    if dll_path:
        candidates.append(dll_path)
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("oo2core_9_win64.dll", "oo2core_8_win64.dll", "oo2core_7_win64.dll", "oo2core_win64.dll"):
        candidates.append(os.path.join(here, name))
        candidates.append(name)
    for cand in candidates:
        try:
            lib = ctypes.WinDLL(cand)
        except OSError:
            continue
        fn = lib.OodleLZ_Decompress
        fn.restype = ctypes.c_int64
        fn.argtypes = [ctypes.c_char_p, ctypes.c_int64, ctypes.c_char_p, ctypes.c_int64,
                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
                       ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int64, ctypes.c_int]
        out = ctypes.create_string_buffer(out_size)
        r = fn(comp, len(comp), out, out_size, 0, 0, 0, None, 0, None, None, None, 0, 3)
        if r == out_size:
            print(f"[oodle] decompressed via {cand}")
            return out.raw
    return None


def decompress_file_table(tf: TocFile, oodle_dll: Optional[str] = None) -> Tuple[Optional[bytes], str]:
    comp = tf.file_table_compressed
    out_size = tf.header.file_table_decompressed_size
    # 1) standard codecs Galanthus uses for the file table
    try:
        d = zlib.decompress(comp)
        if len(d) == out_size:
            return d, "zlib"
    except Exception:
        pass
    try:
        import lz4.block
        d = lz4.block.decompress(comp, uncompressed_size=out_size)
        return d, "lz4-block"
    except Exception:
        pass
    try:
        import lz4.frame
        d = lz4.frame.decompress(comp)
        if len(d) == out_size:
            return d, "lz4-frame"
    except Exception:
        pass
    try:
        import zstandard
        d = zstandard.ZstdDecompressor().decompress(comp, max_output_size=out_size)
        return d, "zstd"
    except Exception:
        pass
    # 2) Oodle (what v41 actually uses)
    d = _try_oodle(comp, out_size, oodle_dll)
    if d is not None:
        return d, "oodle"
    return None, "oodle (no oo2core.dll available)"


def looks_like_oodle(comp: bytes) -> bool:
    # Oodle Kraken/Mermaid chunks start with a byte in the 0x8c/0xcc/0x2c family;
    # AFOP asset (BERG) data starts 0x8c, the file table starts 0x8b.
    return len(comp) > 1 and (comp[0] & 0x7f) in (0x0b, 0x0c, 0x2c & 0x7f)


# ---------------------------------------------------------------------------
# asset trie parser (port of Galanthus ParseEntry, v>=0x29 branch)
# ---------------------------------------------------------------------------
def parse_asset_table(table: bytes, is_signed: bool, version: int, progress=None) -> List[Asset]:
    """Walk the asset trie and return every :class:`Asset`.

    ``progress(done, total, sample=None)`` -- same shape/convention as
    ``sdf.reader.decrypt_file_table``'s callback -- is invoked periodically
    WHILE walking the trie (not just at start/end). This closes a real,
    measured silent gap: on the real "rogue" archive, this function is the
    single biggest cost in the whole TOC-load pipeline (~16.5s of a ~31.8s
    decrypt+zlib+parse+index total -- bigger than the DES-CFB decrypt itself,
    ~10.6s), and before this, nothing reported progress during it at all --
    the decrypt phase's own progress bar would hit 100% of its sub-range, then
    the status bar would sit frozen for that entire ~16.5s before the indexing
    phase's progress started ticking. `done`/`total` are byte offsets into
    `table` (the running high-water-mark trie-cursor position vs. the table's
    total length) -- a natural, always-available, monotonically-non-decreasing
    proxy for "how much of the trie has been consumed" (the DFS walk doesn't
    visit strictly in file order because branch nodes jump to arbitrary
    offsets, so raw `pos` isn't monotonic node-to-node, but the running max
    is). `sample` is the most recently completed asset's full name, mirroring
    `sdf_extract_gui.py`'s `_index()` showing the current asset name. Throttled
    by node-count (cheap bitmask check every node) THEN wall-clock (~30ms,
    matching `_index()`'s own throttle) so the callback overhead is
    negligible even across an 8.4M-node trie -- see AGENTS.md for the measured
    before/after timing that justified this interval.

    Perf note: a real archive's file table can hold millions of trie nodes (a
    "rogue" AFOP archive profiled at 2.3M assets / ~8.4M trie nodes -- this
    function was, before an earlier rewrite, ~24s of a ~36s total TOC-load time, i.e.
    the single largest cost in the whole load, bigger than the DES-CFB decrypt
    of the file table itself). The trie is NOT numpy-vectorizable: each node's
    byte width depends on the id byte just read (variable-length name chunks,
    variable-length sized-ints keyed off flag bits, forward jump pointers to
    arbitrary offsets for branch nodes) -- there is no fixed stride to reshape
    into an array. So this is a scalar-Python speedup, not a numpy one:
      - the original recursive ``parse_entry(name)`` closure is replaced with
        an explicit LIFO stack of ``(pos, name)`` pairs. For a branch node the
        original recursed into the "local" continuation first and the
        jump-address ("p_next") continuation second, fully depth-first before
        moving to the second branch; pushing p_next THEN pos (so pos pops
        first, LIFO) reproduces that exact traversal order, so `assets` comes
        out in the identical order.
      - the ``Cursor`` object's per-field method-call indirection (attribute
        lookups on ``self.buf``/``self.pos`` plus a full function call per
        u8/u16/u32/sized_int) is replaced with direct local-variable buffer
        indexing / ``struct.unpack_from`` / ``int.from_bytes``.
      - ``sized_int``'s manual byte-shift-OR loop is replaced with
        ``int.from_bytes(..., 'little')``, which for size 0 naturally returns
        0 (matching ``sized_int``'s default `fallback=0`) without a branch;
        only the one call site that passes a non-zero fallback (``dds_index``,
        fallback=-1) needs an explicit size==0 check.
      - ``ord('A')``/``ord('Z')`` (recomputed on every single node in the
        original) are replaced with the literal constants 65/90.
    Field-for-field decode logic (which bytes mean what, in what order) is
    UNCHANGED -- verified byte-identical against the original recursive
    implementation on the full 2,322,420-asset "rogue" archive file table
    (see AGENTS.md).
    """
    assets: List[Asset] = []
    buf = table
    unpack_from = struct.unpack_from
    count_mask = 15 if version >= 0x29 else 7
    flag_shift = 4 if version >= 0x29 else 3
    enc_shift = 7 if version >= 0x29 else 6
    v29 = version >= 0x29

    total_bytes = len(buf) or 1
    hi_pos = 0
    node_i = 0
    last_emit = 0.0
    last_name = None

    stack = [(0, "")]
    push = stack.append
    pop = stack.pop
    while stack:
        pos, name = pop()
        if progress is not None:
            # Throttled two ways: a cheap bitmask check on every node (no
            # time.time() call at all most iterations), THEN a wall-clock
            # gate once that fires -- same two-stage throttle style as
            # sdf_extract_gui.py's _index(). `hi_pos` is a running max so the
            # reported percentage never jumps backward even though the DFS
            # walk itself isn't strictly sequential through `buf` (branch
            # nodes jump to arbitrary offsets).
            node_i += 1
            if pos > hi_pos:
                hi_pos = pos
            if (node_i & 0xFFF) == 0:
                now = time.time()
                if now - last_emit >= 0.03:
                    progress(hi_pos, total_bytes, last_name)
                    last_emit = now
                    time.sleep(0)   # explicit GIL-release point, same reasoning as SdfArchive.load()/_index()
        idc = buf[pos]
        pos += 1
        if idc == 0:
            raise ValueError("null id in asset trie")
        if 1 <= idc <= 0x1F:
            seg = buf[pos:pos + idc]
            pos += idc
            nul = seg.find(b"\x00")
            if nul != -1:
                seg = seg[:nul]
            push((pos, name + seg.decode("latin1")))
            continue
        if idc < 65 or idc > 90:          # not in 'A'..'Z'
            p_next = unpack_from("<I", buf, pos)[0]
            pos += 4
            push((p_next, name))          # jump-address branch, processed 2nd
            push((pos, name))             # local branch, processed 1st (LIFO)
            continue
        var = idc - 65
        count = var & count_mask
        flags = var >> flag_shift
        if count <= 0:
            continue
        hsh = unpack_from("<I", buf, pos)[0]
        pos += 4
        packed = buf[pos]
        pos += 1
        unk = packed >> 2
        dds_size = packed & 3
        if dds_size == 0:
            dds_index = -1
        else:
            dds_index = int.from_bytes(buf[pos:pos + dds_size], 'little')
            pos += dds_size
        asset = Asset(name=name, hash=hsh, dds_index=dds_index, unk=unk)
        slices = asset.data_slices
        for _ in range(count):
            saf = buf[pos]
            pos += 1
            is_compressed = (saf >> 5 & 1) != 0
            is_encrypted = (saf >> enc_shift & 1) != 0
            no_page_size = v29 and (saf >> 6 & 1) != 0
            dsz_size = (saf & 3) + 1
            decompressed_size = int.from_bytes(buf[pos:pos + dsz_size], 'little')
            pos += dsz_size
            if is_compressed:
                csize = (saf & 3) + 1
                if v29:
                    pos += 1             # unk1 (always 2)
                    csize = buf[pos]
                    pos += 1
                compressed_size = int.from_bytes(buf[pos:pos + csize], 'little')
                pos += csize
            else:
                compressed_size = decompressed_size
            off_size = (saf >> 2) & 7
            offset = int.from_bytes(buf[pos:pos + off_size], 'little')
            pos += off_size
            index = unpack_from("<H", buf, pos)[0]
            pos += 2
            page_sizes = None
            if is_compressed:
                page_count = (decompressed_size + 0xFFFF) >> 16
                if page_count > 1 and not no_page_size:
                    page_sizes = list(unpack_from("<%dH" % page_count, buf, pos))
                    pos += 2 * page_count
                else:
                    page_sizes = [compressed_size]
            if is_signed:
                pos += 4                 # sign (discarded, like the original)
            slices.append(DataSlice(
                decompressed_size=decompressed_size, compressed_size=compressed_size,
                is_compressed=is_compressed,
                is_oodle=v29 or (saf >> 7 & 1) != 0,
                is_encrypted=is_encrypted, offset=offset, index=index,
                page_sizes=page_sizes,
            ))
        assets.append(asset)
        if progress is not None:
            last_name = asset.name
        if flags != 0:
            n2 = buf[pos]
            pos += 1
            pos += 2 * n2
        # leaf node: nothing pushed, this path of the trie ends here

    if progress is not None:
        progress(total_bytes, total_bytes, last_name)
    return assets


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="AFOP Snowdrop SDF v41 TOC reader")
    ap.add_argument("toc", help="path to sdf.sdftoc")
    ap.add_argument("--oodle", help="path to oo2core_*.dll (optional)")
    ap.add_argument("--sample", type=int, default=20, help="number of sample assets to print")
    ap.add_argument("--list", help="write the full asset path list to this file")
    args = ap.parse_args(argv)

    buf = open(args.toc, "rb").read()
    tf = parse_toc(buf)
    h = tf.header

    print("=" * 72)
    print(f"  {os.path.basename(args.toc)}  ({len(buf):,} bytes)")
    print("=" * 72)
    print("v41 HEADER LAYOUT")
    for name, off, kind, desc in HEADER_LAYOUT:
        val = getattr(h, name)
        sval = f"'{val.to_bytes(4,'little').decode('latin1')}'" if name == "magic" else f"{val:,}"
        print(f"  0x{off:02x}  {name:<30} = {sval:>14}   {desc}")
    print(f"  0x30  start tag 'massive'..'ubisoft'")
    print(f"  0x60  hasSign={int(tf.has_sign)}  flag2/isEncrypted={tf.flag2_isencrypted}"
          f"  + 0x140 signature")

    print("\nINDEX-RANGE SETTINGS")
    for k, v in tf.settings.items():
        print(f"  {k:<38} = {v}")

    n_present = sum(1 for s in tf.data_file_sizes if s)
    print("\nPREAMBLE")
    print(f"  data-slice slots                 = {h.data_slice_count:,} (sizes + tags + hashes)")
    print(f"  slices with non-zero size        = {n_present}")
    print(f"  texture-header slots (152B each) = {h.data_file_count}")
    print(f"  dds headers                      = {h.dds_count}")
    print(f"  dds/preamble block               = [0x{tf.preamble_end:x} .. 0x{tf.file_table_offset:x}]")

    print("\nFILE TABLE")
    print(f"  compressed   @ 0x{tf.file_table_offset:x}  size {h.file_table_compressed_size:,}")
    print(f"  decompressed   size {h.file_table_decompressed_size:,}")
    print(f"  first bytes  {tf.file_table_compressed[:8].hex(' ')}"
          f"  ({'Oodle-like header' if looks_like_oodle(tf.file_table_compressed) else 'unknown'})")

    table, codec = decompress_file_table(tf, args.oodle)
    print(f"  codec        {codec}")

    if table is None:
        print("\n[blocked] could not decompress the file table.")
        print("          The v41 file table is Oodle (Kraken) compressed - the same")
        print("          codec as the BERG asset data - so listing the individual")
        print("          assets requires oo2core_9_win64.dll (pass it with --oodle).")
        print("          No standard zlib/zstd/lz4 stream and no AES/DES key was needed")
        print("          to reach this point; the TOC structure itself is fully parsed.")
        return 2

    print(f"\n  decompressed {len(table):,} bytes; parsing asset trie...")
    assets = parse_asset_table(table, tf.has_sign, h.version)
    print(f"  ASSET COUNT: {len(assets):,}")
    print(f"\nSAMPLE ({min(args.sample, len(assets))} of {len(assets):,}):")
    n_oodle = 0
    for a in assets:
        if any(s.is_oodle and s.is_compressed for s in a.data_slices):
            n_oodle += 1
    for a in assets[:args.sample]:
        s0 = a.data_slices[0] if a.data_slices else None
        loc = (f"slice {s0.index} @ {s0.offset} csz {s0.compressed_size} dsz {s0.decompressed_size}"
               f" {'oodle' if s0.is_oodle and s0.is_compressed else 'raw'}") if s0 else "-"
        print(f"  {a.name}   [{loc}]")
    print(f"\n  assets with Oodle-compressed slices: {n_oodle:,} / {len(assets):,}")
    if args.list:
        with open(args.list, "w", encoding="utf-8") as fh:
            for a in assets:
                fh.write(a.name + "\n")
        print(f"  wrote full listing -> {args.list}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# .mcloth reader/writer for the AFoP Mesh Tool.
#
# Pure module: no bpy imports, usable standalone (tests, CLI tooling).
# The Blender-side layer (_export_mcloth_for_asset) lives in __init__.py.
#
# ---------------- Format ----------------
# .mcloth files drive cloth: a low-res <name>_CLOTH_SIM mesh is simulated and
# each cloth-driven vertex of the <name>_CLOTH_RENDER mesh is skinned onto a
# sim-mesh triangle. Format: 20-byte file header (u32[5]: 205, 1,
# stream_size+8, 1, stream_size), then a flat stream of chunks
# [u32 tag 0xECD7xxxx][u32 size incl. this 8-byte header][payload], then a
# small footer (LOD id list + hashes, independent of vertex data).
#
# Per render LOD the stream contains, in order:
#   ecd71116 header : u32 render_vc, u32 A, u8 1, u32 B, u8 1, 5 zero bytes.
#                     B = number of cloth-driven render vertices. A is an
#                     unknown count (~1.13*B, no table is sized by it and
#                     restoring the vanilla value changes nothing in-game);
#                     preserved by ratio on rewrite.
#   ecd71129        : u16[B] sim-triangle index per driven vertex (unpadded)
#   ecd71118/19/1b/1c/33/35/37/39 : 12-byte float32 (scale, min, max)
#                     dequantization params for the quantized tables.
#                     scale = (max - min) / 254.5, bytes 0..254.
#   ecd71125/27/34/38 : 2 lanes x u8 per driven vertex, 4-ROW SIMD BLOCKS
#                     [lane0 x4][lane1 x4] repeating; partial final block and
#                     the tail up to 16-alignment are padding.
#   ecd71126/28/36/3a : 1 x u8 per driven vertex (linear), padded to 16
#   ecd71122 name   : "<mesh>_CLOTH_RENDER" (LOD0) or "..._LODn"
#   ecd7113e        : u32[B] ascending render-vertex indices (the driven set)
#
# Full per-row semantics (verified against vanilla to quantization accuracy;
# a,b,c = stored sim tri corners, e1=b-a, e2=c-a, n = normalized cross(e1,e2)):
#   0x1129 : sim triangle index
#   0x1125 : (w, u) barycentric coords of the perpendicular foot point of the
#            rest position on the tri plane (v = 1 - w - u)
#   0x1126 : signed perpendicular height h of the rest position
#   0x1127 : (w, u) corner-affine coefficients of the rest NORMAL's in-plane
#            part (N_ip = u*e1 + v*e2, w = -(u + v));  0x1128 : dot(N, n)
#   0x1134 : same for the rest TANGENT;                0x1136 : dot(T, n)
#   0x1138/0x113a : byte-for-byte duplicates of 0x1134/0x1136
# Runtime reconstruction on the deformed tri (a',b',c',n'):
#   P' = w*a' + u*b' + v*c' + h*n';  N' = u_N*e1' + v_N*e2' + dot(N,n)*n'; etc.
# Rows for NEW vertices are therefore computable exactly: compute_row_values().
# The exporter still uses a slot-preserving layout for cloth render meshes
# (see _write_mod_file in __init__.py) as the safest, in-game-verified path.

import math
import os
import re
from struct import unpack, pack

T_HEADER  = 0x1116
T_TRI     = 0x1129
T_NAME    = 0x1122
T_INDICES = 0x113e
TABLES_2B = (0x1125, 0x1127, 0x1134, 0x1138)
TABLES_1B = (0x1126, 0x1128, 0x1136, 0x113a)
SCALES    = (0x1118, 0x1119, 0x111b, 0x111c, 0x1133, 0x1135, 0x1137, 0x1139)
# (scale chunk, table chunk, lanes) for the six independent tables
VALUE_TABLES = ((0x1118, 0x1125, 2), (0x1119, 0x1126, 1), (0x111b, 0x1127, 2),
                (0x111c, 0x1128, 1), (0x1133, 0x1134, 2), (0x1135, 0x1136, 1))
# duplicate chunks: written as copies of their originals
DUP_TABLES = {0x1138: 0x1134, 0x113a: 0x1136}
DUP_SCALES = {0x1137: 0x1133, 0x1139: 0x1135}
# kept for compatibility with older callers/tests
SCALE_TABLE_PAIRS = VALUE_TABLES + ((0x1137, 0x1138, 2), (0x1139, 0x113a, 1))


def _pad16(b):
    return b + b'\x00' * ((-len(b)) % 16)


def read_row_bytes(data, table_off, r, lanes):
    """Raw byte tuple of row r. 2-lane tables use the 4-row SIMD block layout
    [lane0 x4][lane1 x4]; 1-lane tables are linear."""
    if lanes == 1:
        return (data[table_off + r],)
    base = table_off + (r // 4) * 8 + (r % 4)
    return (data[base], data[base + 4])


def write_table(rows, lanes):
    """Build a table payload from per-row byte tuples (blocked for 2 lanes)."""
    if lanes == 1:
        return _pad16(bytes(rw[0] for rw in rows))
    nblocks = (len(rows) + 3) // 4
    out = bytearray(nblocks * 8)
    for r, rw in enumerate(rows):
        base = (r // 4) * 8 + (r % 4)
        out[base] = rw[0]
        out[base + 4] = rw[1]
    return _pad16(bytes(out))


def _closest_point_tri_dist2(a, b, c, p):
    """Squared distance from point p to triangle abc (Christer Ericson's method)."""
    ab = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
    ac = (c[0]-a[0], c[1]-a[1], c[2]-a[2])
    ap = (p[0]-a[0], p[1]-a[1], p[2]-a[2])
    def dot(u, v): return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
    d1 = dot(ab, ap); d2 = dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        q = a
    else:
        bp = (p[0]-b[0], p[1]-b[1], p[2]-b[2])
        d3 = dot(ab, bp); d4 = dot(ac, bp)
        if d3 >= 0.0 and d4 <= d3:
            q = b
        else:
            vc = d1*d4 - d3*d2
            if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                t = d1 / (d1 - d3)
                q = (a[0]+t*ab[0], a[1]+t*ab[1], a[2]+t*ab[2])
            else:
                cp = (p[0]-c[0], p[1]-c[1], p[2]-c[2])
                d5 = dot(ab, cp); d6 = dot(ac, cp)
                if d6 >= 0.0 and d5 <= d6:
                    q = c
                else:
                    vb = d5*d2 - d1*d6
                    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                        t = d2 / (d2 - d6)
                        q = (a[0]+t*ac[0], a[1]+t*ac[1], a[2]+t*ac[2])
                    else:
                        va = d3*d6 - d5*d4
                        if va <= 0.0 and (d4-d3) >= 0.0 and (d5-d6) >= 0.0:
                            t = (d4-d3) / ((d4-d3) + (d5-d6))
                            q = (b[0]+t*(c[0]-b[0]), b[1]+t*(c[1]-b[1]), b[2]+t*(c[2]-b[2]))
                        else:
                            denom = 1.0 / (va + vb + vc)
                            v = vb * denom; w = vc * denom
                            q = (a[0]+ab[0]*v+ac[0]*w, a[1]+ab[1]*v+ac[1]*w, a[2]+ab[2]*v+ac[2]*w)
    dx, dy, dz = p[0]-q[0], p[1]-q[1], p[2]-q[2]
    return dx*dx + dy*dy + dz*dz


def nearest_tri(sim_verts, sim_tris, p):
    """Index of the sim triangle closest to point p (true point-to-triangle
    distance), or None for an empty list."""
    best, bd = None, None
    for ti, tri in enumerate(sim_tris):
        d2 = _closest_point_tri_dist2(sim_verts[tri[0]], sim_verts[tri[1]],
                                      sim_verts[tri[2]], p)
        if bd is None or d2 < bd:
            bd, best = d2, ti
    return best


def bary_within(vals, tol=0.5):
    """True when a computed row's foot barycentrics lie reasonably inside its
    triangle - used to detect appended verts that should re-attach to the
    nearest triangle instead of an inherited one."""
    w, u = vals[0x1125]
    v = 1.0 - w - u
    return all(-tol <= x <= 1.0 + tol for x in (w, u, v))


def _perp_height(sim_verts, tri, p):
    """Signed perpendicular distance from point p to the triangle's plane,
    or None for a degenerate triangle."""
    i0, i1, i2 = tri
    a = sim_verts[i0]; b = sim_verts[i1]; c = sim_verts[i2]
    e1 = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    e2 = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    nx = e1[1] * e2[2] - e1[2] * e2[1]
    ny = e1[2] * e2[0] - e1[0] * e2[2]
    nz = e1[0] * e2[1] - e1[1] * e2[0]
    nl = math.sqrt(nx * nx + ny * ny + nz * nz)
    if nl == 0.0:
        return None
    return ((p[0] - a[0]) * nx + (p[1] - a[1]) * ny + (p[2] - a[2]) * nz) / nl


def compute_row_values(sim_verts, sim_tris, ti, P, N, T):
    """
    Compute the full driven-row values for a render vertex with rest position
    P, normal N and tangent T, attached to sim triangle ti (see the format
    notes above). Returns {'tri': ti, table_tag: value tuple, ...} or None for
    degenerate geometry. Verified to reproduce vanilla rows to quantization
    accuracy.
    """
    i0, i1, i2 = sim_tris[ti]
    a = sim_verts[i0]; b = sim_verts[i1]; c = sim_verts[i2]
    e1 = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    e2 = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    nx = e1[1] * e2[2] - e1[2] * e2[1]
    ny = e1[2] * e2[0] - e1[0] * e2[2]
    nz = e1[0] * e2[1] - e1[1] * e2[0]
    nl = math.sqrt(nx * nx + ny * ny + nz * nz)
    if nl == 0.0:
        return None
    nh = (nx / nl, ny / nl, nz / nl)

    g11 = e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]
    g12 = e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2]
    g22 = e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]
    det = g11 * g22 - g12 * g12
    if det == 0.0:
        return None

    def in_plane_coeffs(vx, vy, vz):
        """Solve v_in_plane = u*e1 + v*e2 (the n-component is removed first)."""
        zc = vx * nh[0] + vy * nh[1] + vz * nh[2]
        px, py, pz = vx - zc * nh[0], vy - zc * nh[1], vz - zc * nh[2]
        r1 = px * e1[0] + py * e1[1] + pz * e1[2]
        r2 = px * e2[0] + py * e2[1] + pz * e2[2]
        return (r1 * g22 - r2 * g12) / det, (r2 * g11 - r1 * g12) / det, zc

    u, v, h = in_plane_coeffs(P[0] - a[0], P[1] - a[1], P[2] - a[2])
    nu, nv, nzc = in_plane_coeffs(N[0], N[1], N[2])
    tu, tv, tzc = in_plane_coeffs(T[0], T[1], T[2])
    return {
        'tri': ti,
        0x1125: (1.0 - u - v, u),      # (w, u) bary of the foot point
        0x1126: (h,),                  # perpendicular height
        0x1127: (-(nu + nv), nu),      # (w, u) of the normal's in-plane part
        0x1128: (nzc,),                # dot(N, n)
        0x1134: (-(tu + tv), tu),      # (w, u) of the tangent's in-plane part
        0x1136: (tzc,),                # dot(T, n)
    }


def source_path(asset_path):
    """The .mcloth paired with the loaded mmb: same stem, or the stem with the
    _MOD suffix stripped (when a _MOD.mmb was loaded but its mcloth was never
    exported)."""
    stem, _ = os.path.splitext(asset_path)
    cand = stem + '.mcloth'
    if os.path.isfile(cand):
        return cand
    base = re.sub(r'_MOD\d*$', '', os.path.basename(stem))
    cand = os.path.join(os.path.dirname(stem), base + '.mcloth')
    return cand if os.path.isfile(cand) else None


def parse_blocks(data):
    """Parse the chunk stream. Returns (stream_end, blocks) where blocks maps
    the LOD name from the ecd71122 chunk -> {'order': [(tag, off, size)...],
    'chunks': {low16 tag: (off, size)}} covering ecd71116 .. ecd7113e."""
    hdr = unpack('<5I', data[0:20])
    stream_end = 20 + hdr[4]
    if stream_end > len(data):
        raise ValueError("mcloth stream size exceeds file size")
    blocks = {}
    cur = None
    off = 20
    while off < stream_end:
        tag, size = unpack('<II', data[off:off + 8])
        if size < 8 or off + size > stream_end:
            raise ValueError(f"Bad mcloth chunk at offset {off}")
        t = tag & 0xFFFF
        if t == T_HEADER:
            cur = {'order': [], 'chunks': {}}
        if cur is not None:
            cur['order'].append((tag, off, size))
            cur['chunks'][t] = (off, size)
            if t == T_INDICES:
                noff, nsize = cur['chunks'][T_NAME]
                name = data[noff + 8:noff + nsize].split(b'\0')[0].decode('latin-1')
                blocks[name] = cur
                cur = None
        off += size
    return stream_end, blocks


def rebind_heights(data, srcb, rows, sim_verts, sim_tris, new_pos):
    """
    Recompute the per-driven-vertex height byte (table 0x1126) from the NEW
    render vertex positions: h = signed perpendicular distance to the stored
    sim triangle's plane. Returns (new_scale_floats, height_bytes) or None
    when the geometry is unusable.
    """
    toff, tsize = srcb['chunks'][T_TRI]
    hoff, hsize = srcb['chunks'][0x1126]
    s2off, _s2s = srcb['chunks'][0x1119]
    old_s, old_mn, old_mx = unpack('<fff', data[s2off + 8:s2off + 20])
    heights = []
    for nv, r in rows:
        ti = unpack('<H', data[toff + 8 + r * 2:toff + 10 + r * 2])[0]
        if ti >= len(sim_tris) or nv >= len(new_pos):
            return None
        i0, i1, i2 = sim_tris[ti]
        a = sim_verts[i0]; b = sim_verts[i1]; c = sim_verts[i2]
        e1 = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        e2 = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        nx = e1[1] * e2[2] - e1[2] * e2[1]
        ny = e1[2] * e2[0] - e1[0] * e2[2]
        nz = e1[0] * e2[1] - e1[1] * e2[0]
        nl = math.sqrt(nx * nx + ny * ny + nz * nz)
        if nl == 0.0:
            # degenerate triangle: keep the old stored height
            old_b = data[hoff + 8 + r]
            heights.append(old_mn + old_b * old_s)
            continue
        p = new_pos[nv]
        h = ((p[0] - a[0]) * nx + (p[1] - a[1]) * ny + (p[2] - a[2]) * nz) / nl
        heights.append(h)
    if not heights:
        return None
    # Requantize with a fresh range so edits can never clip (vanilla scale
    # convention, see the format notes above)
    mn = min(heights)
    mx = max(heights)
    rng = mx - mn
    if rng <= 0.0:
        rng = max(abs(mn), 1e-6)
        mx = mn + rng
    s = rng / 254.5
    hbytes = bytes(max(0, min(254, int(round((h - mn) / s)))) for h in heights)
    return pack('<fff', s, mn, mx), hbytes


def _patched_header(data, tgt, new_vc, new_A, new_B):
    thoff, thsize = tgt['chunks'][T_HEADER]
    tp = bytearray(data[thoff + 8:thoff + thsize])
    tp[0:4] = pack('<I', new_vc)
    tp[4:8] = pack('<I', new_A)
    tp[9:13] = pack('<I', new_B)
    return bytes(tp)


def _build_single(data, blocks, tgt, block_name, new_vc, source, rebind):
    """Single-source block rebuild: rows are kept at byte level (blocked
    layout aware). A full-identity mapping copies table payloads verbatim."""
    src_name, old_to_new = source
    srcb = blocks[src_name]

    soff, ssize = srcb['chunks'][T_INDICES]
    src_B = (ssize - 8) // 4
    src_idx = unpack(f'<{src_B}I', data[soff + 8:soff + ssize])

    # rows: (new render-vertex index, row index in the source tables)
    rows = []
    for r, ov in enumerate(src_idx):
        nv = old_to_new.get(ov)
        if nv is not None and nv < new_vc:
            rows.append((nv, r))
    rows.sort()
    new_B = len(rows)
    # identity: same block, every row kept with its original index and order
    identity = (src_name == block_name and new_B == src_B
                and all(nv == src_idx[r] for nv, r in rows))

    # Recompute heights from new positions when geometry is available
    rebind_result = None
    if rebind and block_name in rebind and new_B:
        sim_verts, sim_tris, new_pos = rebind[block_name]
        rebind_result = rebind_heights(data, srcb, rows,
                                       sim_verts, sim_tris, new_pos)

    # preserve A by ratio (semantics unknown, see format notes above)
    hoff, hsize = srcb['chunks'][T_HEADER]
    src_A = unpack('<I', data[hoff + 8 + 4:hoff + 8 + 8])[0]
    src_Bh = unpack('<I', data[hoff + 8 + 9:hoff + 8 + 13])[0]
    new_A = max(new_B, int(round(new_B * (src_A / src_Bh)))) if src_Bh else new_B

    header = _patched_header(data, tgt, new_vc, new_A, new_B)
    pieces = []
    for tag, off, size in tgt['order']:
        t = tag & 0xFFFF
        if t == T_HEADER:
            payload = header
        elif t == T_INDICES:
            payload = pack(f'<{new_B}I', *[nv for nv, _r in rows]) if new_B else b''
        elif t == 0x1126 and rebind_result is not None:
            # height table recomputed from the exported vertex positions
            payload = _pad16(rebind_result[1])
        elif t == 0x1119 and rebind_result is not None:
            # matching requantization range for the recomputed heights
            payload = rebind_result[0]
        elif t == T_TRI or t in TABLES_2B or t in TABLES_1B:
            so2, ss2 = srcb['chunks'][t]
            if identity:
                payload = data[so2 + 8:so2 + ss2]  # verbatim, incl. block padding
            elif t == T_TRI:
                tbl = data[so2 + 8:so2 + ss2]
                payload = b''.join(tbl[r * 2:r * 2 + 2] for _nv, r in rows)
            else:
                lanes = 2 if t in TABLES_2B else 1
                payload = write_table(
                    [read_row_bytes(data, so2 + 8, r, lanes) for _nv, r in rows],
                    lanes)
        elif t in SCALES and src_name != block_name:
            # dequant params belong with the rows they decode
            so2, ss2 = srcb['chunks'][t]
            payload = data[so2 + 8:so2 + ss2]
        else:
            payload = data[off + 8:off + size]
        pieces.append(pack('<II', tag, 8 + len(payload)) + payload)
    return b''.join(pieces), new_B


def _build_values(data, blocks, tgt, block_name, new_vc, source, computed, rebind):
    """
    Value-pipeline block rebuild: kept rows are dequantized from the source
    block and merged with computed rows (from compute_row_values), then
    everything is requantized under fresh per-table ranges and written in the
    blocked layout. Used when new (e.g. appended) vertices need rows.
    """
    src_name, old_to_new = source
    srcb = blocks[src_name]

    soff, ssize = srcb['chunks'][T_INDICES]
    src_B = (ssize - 8) // 4
    src_idx = unpack(f'<{src_B}I', data[soff + 8:soff + ssize])

    # scales + table offsets of the source block
    scales = {}
    offs = {}
    for sc_tag, tb_tag, lanes in VALUE_TABLES:
        so, _ss = srcb['chunks'][sc_tag]
        s, mn, _mx = unpack('<fff', data[so + 8:so + 20])
        scales[tb_tag] = (s, mn)
        offs[tb_tag] = srcb['chunks'][tb_tag][0] + 8
    tri_off = srcb['chunks'][T_TRI][0] + 8

    # kept rows, dequantized into value space
    rowvals = []  # (new index, values dict)
    for r, ov in enumerate(src_idx):
        nv = old_to_new.get(ov)
        if nv is None or nv >= new_vc:
            continue
        vals = {'tri': unpack('<H', data[tri_off + r * 2:tri_off + r * 2 + 2])[0]}
        for sc_tag, tb_tag, lanes in VALUE_TABLES:
            s, mn = scales[tb_tag]
            vals[tb_tag] = tuple(mn + x * s
                                 for x in read_row_bytes(data, offs[tb_tag], r, lanes))
        rowvals.append((nv, vals))
    rowvals.extend(computed)
    rowvals.sort(key=lambda rv: rv[0])
    new_B = len(rowvals)

    # fresh heights from the exported positions when geometry is available
    if rebind and block_name in rebind:
        sim_verts, sim_tris, new_pos = rebind[block_name]
        for nv, vals in rowvals:
            if vals['tri'] < len(sim_tris) and nv < len(new_pos):
                h = _perp_height(sim_verts, sim_tris[vals['tri']], new_pos[nv])
                if h is not None:
                    vals[0x1126] = (h,)

    # requantize under fresh per-table ranges (scale = range/254.5)
    table_payloads = {}
    scale_payloads = {}
    for sc_tag, tb_tag, lanes in VALUE_TABLES:
        flat = [x for _nv, vals in rowvals for x in vals[tb_tag]]
        mn = min(flat) if flat else 0.0
        mx = max(flat) if flat else 0.0
        rng = mx - mn
        if rng <= 0.0:
            rng = max(abs(mn), 1e-6)
            mx = mn + rng
        s = rng / 254.5
        rows_b = [tuple(max(0, min(254, int(round((x - mn) / s)))) for x in vals[tb_tag])
                  for _nv, vals in rowvals]
        table_payloads[tb_tag] = write_table(rows_b, lanes)
        scale_payloads[sc_tag] = pack('<fff', s, mn, mx)

    # A preserved by the TARGET block's ratio
    thoff, _ths = tgt['chunks'][T_HEADER]
    t_A = unpack('<I', data[thoff + 8 + 4:thoff + 8 + 8])[0]
    t_B = unpack('<I', data[thoff + 8 + 9:thoff + 8 + 13])[0]
    new_A = max(new_B, int(round(new_B * (t_A / t_B)))) if t_B else new_B

    header = _patched_header(data, tgt, new_vc, new_A, new_B)
    pieces = []
    for tag, off, size in tgt['order']:
        t = tag & 0xFFFF
        if t == T_HEADER:
            payload = header
        elif t == T_INDICES:
            payload = pack(f'<{new_B}I', *[nv for nv, _v in rowvals]) if new_B else b''
        elif t == T_TRI:
            payload = pack(f'<{new_B}H', *[v['tri'] for _nv, v in rowvals]) if new_B else b''
        elif t in table_payloads:
            payload = table_payloads[t]
        elif t in scale_payloads:
            payload = scale_payloads[t]
        elif t in DUP_TABLES:
            payload = table_payloads[DUP_TABLES[t]]
        elif t in DUP_SCALES:
            payload = scale_payloads[DUP_SCALES[t]]
        else:
            payload = data[off + 8:off + size]
        pieces.append(pack('<II', tag, 8 + len(payload)) + payload)
    return b''.join(pieces), new_B


def rewrite(data, remaps, rebind=None, computed=None):
    """
    Rewrite the mcloth chunk stream, remapping the per-LOD driven-vertex blocks.

    :param remaps: dict target_block_name -> (new_vc, source_block_name, old_to_new)
           where old_to_new maps a source-LOD original vertex index to the new
           vertex index in the edited/generated Blender mesh.
    :param rebind: optional dict target_block_name -> (sim_verts, sim_tris, new_pos)
           used to recompute the height table from the exported positions.
    :param computed: optional dict target_block_name -> [(new_index, values), ...]
           rows synthesized by compute_row_values() for NEW vertices (e.g. the
           appended vertices of a generated LOD). Blocks with computed rows go
           through the dequantize/requantize value pipeline; others keep their
           rows at byte level.
    :return: (new file bytes, {block_name: (old_B, new_B)})
    """
    stream_end, blocks = parse_blocks(data)
    stats = {}
    spans = []
    for block_name, (new_vc, src_name, old_to_new) in remaps.items():
        tgt = blocks[block_name]
        extra = (computed or {}).get(block_name) or []
        if extra:
            pieces, new_B = _build_values(data, blocks, tgt, block_name, new_vc,
                                          (src_name, old_to_new), extra, rebind)
        else:
            pieces, new_B = _build_single(data, blocks, tgt, block_name, new_vc,
                                          (src_name, old_to_new), rebind)

        start = tgt['order'][0][1]
        last_tag, last_off, last_size = tgt['order'][-1]
        spans.append((start, last_off + last_size, pieces))
        # old B of the TARGET block for reporting
        toff2, tsize2 = tgt['chunks'][T_INDICES]
        stats[block_name] = ((tsize2 - 8) // 4, new_B)

    spans.sort()
    out = bytearray(data[0:20])
    pos = 20
    for start, end, rep in spans:
        out += data[pos:start]
        out += rep
        pos = end
    out += data[pos:stream_end]
    new_stream_size = len(out) - 20
    out[16:20] = pack('<I', new_stream_size)
    out[8:12] = pack('<I', new_stream_size + 8)
    out += data[stream_end:]  # footer unchanged
    return bytes(out), stats


# ---- mmb geometry readers used to prepare height rebinding ----

def mmb_lod_float_positions(file_bytes, mesh, li):
    """Float32 vertex positions of one LOD, read directly from mmb bytes.
    `mesh` is a parsed SkeletalMeshAsset.Mesh (duck-typed: vertex_stride,
    position_type and lods[] with data_offset/vertex_data_offset_a/data_size)."""
    lod = mesh.lods[li]
    if lod.vertex_count == 0 or mesh.position_type != 1:
        return None
    higher = sum(mesh.lods[k].data_size for k in range(li + 1, len(mesh.lods)))
    base = lod.data_offset + (lod.vertex_data_offset_a - higher)
    vs = mesh.vertex_stride
    return [unpack('<fff', file_bytes[base + i * vs: base + i * vs + 12])
            for i in range(lod.vertex_count)]


def mmb_lod_u16_tris(file_bytes, mesh, li):
    """uint16 triangles of one LOD, or None when the LOD uses uint32 indices."""
    lod = mesh.lods[li]
    if lod.index_count == 0 or lod.size_a != lod.face_block_offset // 2:
        return None
    higher = sum(mesh.lods[k].data_size for k in range(li + 1, len(mesh.lods)))
    base = lod.data_offset + (lod.face_block_offset - higher)
    return [unpack('<HHH', file_bytes[base + i * 6: base + i * 6 + 6])
            for i in range(lod.index_count // 3)]


def mmb_lod_color_bytes(file_bytes, mesh, li):
    """Per-vertex color bytes (all color sets, 4*color_count each) of one LOD,
    or None when the mesh stores no colors in the normals stride. Color set 0's
    R channel is the cloth driven mask; set 1's G channel the blend weight."""
    cc = mesh.color_count if getattr(mesh, 'color_in_normals', True) else 0
    if cc <= 0:
        return None
    lod = mesh.lods[li]
    if lod.vertex_count == 0:
        return None
    higher = sum(mesh.lods[k].data_size for k in range(li + 1, len(mesh.lods)))
    base = lod.data_offset + (lod.vertex_data_offset_b - higher)
    ns = mesh.normals_stride
    n = 4 * cc
    return [file_bytes[base + i * ns: base + i * ns + n]
            for i in range(lod.vertex_count)]


def mmb_lod_normals_tangents(file_bytes, mesh, li):
    """Per-vertex (normal, tangent) float3 pairs of one LOD, read directly
    from mmb bytes. Only for normal_type 1 meshes (float normal layout:
    color(4*cc) | normal(12) | tangent(12) | sign(4) | ...); returns None
    otherwise."""
    if getattr(mesh, 'normal_type', None) != 1:
        return None
    lod = mesh.lods[li]
    if lod.vertex_count == 0:
        return None
    higher = sum(mesh.lods[k].data_size for k in range(li + 1, len(mesh.lods)))
    base = lod.data_offset + (lod.vertex_data_offset_b - higher)
    ns = mesh.normals_stride
    cc = mesh.color_count if getattr(mesh, 'color_in_normals', True) else 0
    off0 = 4 * cc
    out = []
    for i in range(lod.vertex_count):
        o = base + i * ns + off0
        out.append((unpack('<fff', file_bytes[o:o + 12]),
                    unpack('<fff', file_bytes[o + 12:o + 24])))
    return out

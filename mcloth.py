# .mcloth reader/writer for the AFoP Mesh Tool.
#
# Pure module: no bpy imports, usable standalone (tests, CLI tooling).
# The Blender-side layer (_export_mcloth_for_asset) lives in __init__.py.
#
# ---------------- Format ----------------
# .mcloth files drive cloth: a low-res <name>_CLOTH_SIM mesh is simulated and
# each cloth-driven vertex of the <name>_CLOTH_RENDER mesh is skinned onto a
# sim-mesh triangle. Container: u32 magic 205, u32 stream_count, then that many
# records [u32 stream_size+8, u32 1, u32 stream_size, flat ECD7 chunk stream].
# A small file-level footer follows the last record. The rewriter currently
# accepts one stream only; multi-stream files are detected and rejected rather
# than partially rewriting stream 0 and misclassifying the others as a footer.
#
# Per render LOD the stream contains, in order:
#   ecd71116 header : u32 render_vc, u32 A, u8 1, u32 B, u8 1, 5 zero bytes.
#                     B = number of cloth-driven render vertices. A is the sum
#                     of the 16-byte allocation counts of every per-row table:
#                     ceil(1129/16) + ceil(113e/16) + each present 1125..113a
#                     table. This is NOT simply a fixed ratio of B.
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

# ---------------- Sim section ----------------
# Outside the render LOD blocks the stream carries one _CLOTH_SIM description
# (header chunks before the first 0x1116, and a tail after the last block).
# Decoded so far (corpus-verified) and rewritten to keep the sim mesh count
# fields consistent after a sim-mesh edit:
#   0x1102 u32 = sim vertex count           0x1112 u32 = sim triangle count
#   0x11d1 u32[2], [0] = sim triangle count 0x111d u32 = ceil(V,16)
#   0x110b = 16B float4 per sim vert, ceil(V,16) rows: rest (x,y,z) matching the
#            mmb sim positions + a w flag (0.0 free / -2.0 pinned/kinematic)
#   0x1113 = byte-identical copy of the sim triangle index buffer (6B per tri)
#   per sim VERT (padded to ceil(V,16)): 0x11b3 (1B), 0x112a (2B)
#   per sim TRI: 0x11d2 count + 0x11d3 offset into 0x11d5's packed
#                barycentric virtual-particle/collision samples; one 0x111e
#                occurrence is the triangle permutation (identified by content)
# Constraint fabric (serialized NvCloth cooked fabric, decoded 2026-07-09):
#   0x1123 u32[3] = constraint counts of the 3 sections/phases:
#            [stretch (mesh edges), shearing (quad diagonals + shear-dominant
#            edges), bending (2-apart pairs)]. Every constraint has >=1
#            movable endpoint (0x110b flag == 0.0).
#   0x1124 = u16 vertex-index PAIRS, section by section (mIndices)
#   0x110d = u8 rest length per constraint, per section quantized to that
#            section's own range, EACH SECTION padded to 16B, then a trailing
#            32-byte record (kept verbatim; meaning unknown)
#   0x110e = 6 x (scale float, min float, max float) quantization triplets,
#            scale = (max-min)/254.5; [0:3] = the 3 rest-length sections,
#            [3:6] = other quantized data (kept verbatim)
#   0x110c u32[6]: [0:3] = ceil(section/16) SIMD-16 group counts. Each of
#            [3:6] is either zero (disabled for that section) or ceil(V/16).
#            Preserve that zero/nonzero mask and refresh every nonzero particle
#            block count when V crosses a 16-vertex boundary.
#   0x1111 u32[3] = per-section count (multiple of 16) of leading constraints
#            arranged as conflict-free SIMD-16 blocks.  Within each complete
#            block a movable vertex occurs at most once; the unscheduled tail
#            is solved conservatively.
#   another 0x111e, when present = free (movable) vertex list, u16 each
# Confirmed ADD strategy - BUDGET REUSE: new sim verts REUSE orphaned (deleted)
# vert slots and new tris reuse phantom tri slots; every chunk keeps its
# vanilla size and count, only VALUES change:
#   0x110b positions refreshed; reused slots keep their slot's free/pinned
#   flag (free list must not change); 0x112a/0x11b3 rows -> donor values;
#   0x1113 mirrors the new tri buffer (same size); constraint rows in
#   0x1124/0x110d whose pairs reference a reused slot are REWRITTEN in place
#   (cooked pairs + rest lengths quantized under the section's EXISTING
#   0x110e range, clamped) - spare rows are repointed to a duplicate of a
#   cooked pair (a doubled distance constraint is valid), excess cooked
#   constraints are dropped. 0x11dX/0x12eX etc. stay verbatim (per-tri
#   records go stale in VALUE for reused tris but stay structurally valid).
# Vertex/triangle append modes (pinned or cooked-free) are experimental.  Their
# fabric rows must be re-scheduled after rows are removed or added; preserving
# a stale 0x1111 prefix while compacting 0x1124 creates SIMD write conflicts.
SIM_VC_FIELDS = (0x1102,)              # u32 at payload +0 = sim vertex count
SIM_TC_FIELDS = (0x1112, 0x11d1)       # u32 at payload +0 = sim triangle count
SIM_PADV_FIELDS = (0x111d,)            # u32 = ceil(sim vertex count, 16)
# Optional per-vert tables. Non-empty 0x112a = TETHER ANCHOR: u16 index of a
# PINNED sim vert per vertex (corpus: every value is a pinned vert). Non-empty
# 0x11b3 = 1B/vert quantized data, described by 0x11b2/0x11b4 but semantically
# undecoded. Either payload may intentionally be empty and must remain empty.
# Appended verts get their nearest-original DONOR's row (a zero here would
# anchor new verts to vertex 0 - usually a FREE vert = invalid solver input).
SIM_VERT_TABLES = {0x11b3: 1, 0x112a: 2}  # bytes per sim vert, padded to 16
T_SIM_AUX_HEADER = 0x11b2              # u32[2]: [11b3 SIMD block count, 8]
T_SIM_AUX_VALUES = 0x11b3              # optional u8[ceil16(V)], may be empty
T_SIM_AUX_DESCS = 0x11b4               # 8 x 16B; one descriptor repeats blocks
SIM_TRI_TABLES = {0x11d2: 1, 0x11d3: 2}  # bytes per sim tri (unpadded)
T_SIM_VP_META = 0x11d1                 # [triangle count, 0x11d5 u16 count]
T_SIM_VP_COUNTS = 0x11d2               # u8 count per triangle
T_SIM_VP_OFFSETS = 0x11d3              # u16 CSR offset per triangle
T_SIM_VP_SCALE = 0x11d4                # float3 global sampling parameters
T_SIM_VP_VALUES = 0x11d5               # packed u8x2 barycentric sample weights
T_SIM_REST = 0x110b                    # float4 rest position + flag per vert
T_SIM_TRIS = 0x1113                    # sim triangle index buffer copy
T_SIM_PERM = 0x111e                    # free-vert list (optional) and triangle
                                       # permutation; do not assume occurrence order
SIM_PIN_FLAG = -2.0                    # 0x110b w flag: pinned/kinematic vertex
T_SIM_SECTS = 0x1123                   # u32[3] constraint section counts
T_SIM_PAIRS = 0x1124                   # u16 constraint vertex pairs
T_SIM_RESTL = 0x110d                   # u8 quantized rest lengths per section
T_SIM_QUANT = 0x110e                   # 6 x (scale,min,max) float triplets
T_SIM_GROUPS = 0x110c                  # u32[6] constraint + particle block counts
T_SIM_PREFIX16 = 0x1111                # u32[3] conflict-free SIMD-16 prefixes


def _pad16(b):
    return b + b'\x00' * ((-len(b)) % 16)


def _mapping_allocation_count(block, row_count):
    """Header 0x1116 field A for ``row_count`` driven render vertices.

    The loader allocates every row table independently on a 16-byte boundary.
    A is the sum of those allocation sizes in 16-byte units.  This rule matches
    every vanilla block in the scanned corpus; proportional A/B scaling can
    under-allocate after render-row growth and crash during mapping cook.
    """
    lanes_by_tag = {T_TRI: 2, T_INDICES: 4}
    lanes_by_tag.update({t: 2 for t in TABLES_2B})
    lanes_by_tag.update({t: 1 for t in TABLES_1B})
    return sum((row_count * lanes + 15) // 16
               for tag, lanes in lanes_by_tag.items()
               if tag in block['chunks'])


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


def nearest_tri(sim_verts, sim_tris, p, valid=None):
    """Index of the sim triangle closest to point p (true point-to-triangle
    distance), or None for an empty list. `valid` optionally restricts the
    candidate triangle indices (e.g. excluding phantom tri slots)."""
    return nearest_tri_dist(sim_verts, sim_tris, p, valid)[0]


def nearest_tri_dist(sim_verts, sim_tris, p, valid=None):
    """(nearest valid triangle index, distance) - or (None, None)."""
    best, bd = None, None
    for ti, tri in enumerate(sim_tris):
        if valid is not None and ti not in valid:
            continue
        try:
            d2 = _closest_point_tri_dist2(sim_verts[tri[0]], sim_verts[tri[1]],
                                          sim_verts[tri[2]], p)
        except ZeroDivisionError:
            continue
        if bd is None or d2 < bd:
            bd, best = d2, ti
    return best, (math.sqrt(bd) if bd is not None else None)


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


def parse_streams(data):
    """Return ``(streams, footer_offset)`` for the outer .mcloth container.

    Each stream dict has ``header``, ``start``, ``end``, and ``blocks`` keys.
    Offsets are absolute file offsets. Legacy/raw non-ECD7 .mcloth variants
    deliberately raise ValueError here.
    """
    if len(data) < 20 or unpack('<I', data[0:4])[0] != 205:
        raise ValueError("Unsupported legacy/raw mcloth container")
    stream_count = unpack('<I', data[4:8])[0]
    if stream_count == 0:
        raise ValueError("mcloth container has no streams")

    streams = []
    pos = 8
    for stream_index in range(stream_count):
        if pos + 12 > len(data):
            raise ValueError(f"Truncated mcloth stream header {stream_index}")
        outer_size, marker, stream_size = unpack('<3I', data[pos:pos + 12])
        if marker != 1 or outer_size != stream_size + 8:
            raise ValueError(f"Bad mcloth stream header {stream_index}")
        stream_start = pos + 12
        stream_end = stream_start + stream_size
        if stream_end > len(data):
            raise ValueError("mcloth stream size exceeds file size")

        blocks = _parse_blocks_range(data, stream_start, stream_end)
        streams.append({'header': pos, 'start': stream_start,
                        'end': stream_end, 'blocks': blocks})
        pos = stream_end
    return streams, pos


def _parse_blocks_range(data, stream_start, stream_end):
    """Render mapping blocks within one already-validated chunk stream."""
    blocks = {}
    cur = None
    off = stream_start
    while off < stream_end:
        if off + 8 > stream_end:
            raise ValueError(f"Truncated mcloth chunk header at offset {off}")
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
    return blocks


def parse_blocks(data):
    """Parse the render blocks of a single-stream .mcloth.

    Returns ``(stream_end, blocks)`` for compatibility with existing callers.
    Multi-stream files require a stream-name-aware rewrite and are rejected so
    that later streams can never be mistaken for an inert footer.
    """
    streams, _footer_offset = parse_streams(data)
    if len(streams) != 1:
        raise ValueError(
            f"Multi-stream mcloth ({len(streams)} streams) is unsupported")
    stream = streams[0]
    blocks = stream['blocks']
    stream_end = stream['end']
    return stream_end, blocks


def rebind_heights(data, srcb, rows, sim_verts, sim_tris, new_pos,
                   tri_remap=None):
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
        if tri_remap:
            ti = tri_remap.get(ti, ti)
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


def _build_single(data, blocks, tgt, block_name, new_vc, source, rebind,
                  tri_remap=None):
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
    identity = (not tri_remap and src_name == block_name and new_B == src_B
                and all(nv == src_idx[r] for nv, r in rows))

    # Recompute heights from new positions when geometry is available
    rebind_result = None
    if rebind and block_name in rebind and new_B:
        sim_verts, sim_tris, new_pos = rebind[block_name]
        rebind_result = rebind_heights(data, srcb, rows,
                                       sim_verts, sim_tris, new_pos,
                                       tri_remap=tri_remap)

    new_A = _mapping_allocation_count(tgt, new_B)

    header = _patched_header(data, tgt, new_vc, new_A, new_B)

    def tri_row(tbl, row):
        raw = unpack('<H', tbl[row * 2:row * 2 + 2])[0]
        return pack('<H', tri_remap.get(raw, raw))

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
                payload = b''.join(tri_row(tbl, r) if tri_remap
                                   else tbl[r * 2:r * 2 + 2]
                                   for _nv, r in rows)
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


def _build_values(data, blocks, tgt, block_name, new_vc, source, computed,
                  rebind, tri_remap=None):
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
        ti = unpack('<H', data[tri_off + r * 2:tri_off + r * 2 + 2])[0]
        if tri_remap:
            ti = tri_remap.get(ti, ti)
        vals = {'tri': ti}
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

    new_A = _mapping_allocation_count(tgt, new_B)

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


def _sim_rest_table(old_pay, orig_vc, new_pos, free_set=None):
    """Rebuild 0x110b: float4 (x,y,z, flag) per sim vert, padded to ceil(V,16)
    rows. Kept verts keep their flag byte-for-byte (their position refreshes
    from new_pos - relevant when verts were moved). Appended verts are marked
    FREE (0.0) when in free_set, otherwise PINNED. Growth derives that class
    from the nearest original donor and guarantees every new disconnected
    topology component has a pinned particle."""
    new_vc = len(new_pos)
    pin = pack('<f', SIM_PIN_FLAG)
    free = pack('<f', 0.0)
    rows = bytearray()
    for i in range(new_vc):
        x, y, z = new_pos[i]
        if i < orig_vc and (i + 1) * 16 <= len(old_pay):
            flag = old_pay[i * 16 + 12:i * 16 + 16]
        else:
            flag = free if (free_set and i in free_set) else pin
        rows += pack('<fff', x, y, z) + flag
    rows += b'\x00' * (((new_vc + 15) // 16 * 16 - new_vc) * 16)
    return bytes(rows)


def _resize_padded(old_pay, orig_n, new_n, per, blk):
    """Resize a fixed per-element array (per bytes each, padded to blk elements).
    Unchanged or shrunk counts preserve the original bytes verbatim (incl. the
    original padding, so a null edit is byte-identical); a grown count keeps the
    real elements and zero-fills the appended ones and the new padding."""
    total = per * ((new_n + blk - 1) // blk * blk)
    out = bytearray(total)
    if new_n <= orig_n:
        out[:] = old_pay[:total]
    else:
        out[:orig_n * per] = old_pay[:orig_n * per]
    return bytes(out)


def cook_appended_constraints(pos, mov, tris, orig_vc, new_set=None):
    """
    NvCloth-cooker reimplementation, restricted to constraints that involve a
    NEW vertex (index >= orig_vc, or membership in new_set when given - used
    by the budget-reuse path where new verts occupy reused slots). Returns
    three [(pair, rest_length)] lists - [stretch, shearing, bending] -
    matching the .mcloth sections.
      stretch  : mesh edges (unless shear accumulation dominates)
      shearing : shear-dominant edges + quadifier quad diagonals
      bending  : non-edge 2-ring pairs with dominant nonzero bend accumulation
    Every constraint needs >=1 movable endpoint. `mov` covers all verts
    (new verts should be passed as movable).
    """
    def sub(a, b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
    def dt(a, b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    def mg(a): return math.sqrt(dt(a, a))
    def crs(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
                           a[0]*b[1]-a[1]*b[0])
    SV = len(pos)
    adj = {}
    for a, b, c in tris:
        for x, y in ((a, b), (b, c), (c, a)):
            adj.setdefault(x, set()).add(y)
            adj.setdefault(y, set()).add(x)
    # classification accumulators over 1-ring and 2-ring paths (NvCloth Edge)
    acc = {}
    for i in range(SV):
        wi = 1.0 if mov[i] else 0.0
        for m in adj.get(i, ()):
            if wi + (1.0 if mov[m] else 0.0) > 0:
                e = acc.setdefault((min(i, m), max(i, m)), [0.0, 0.0, 0.0])
                e[0] += 0.1
            for n2 in adj.get(m, ()):
                if n2 != i and wi + (1.0 if mov[n2] else 0.0) > 0:
                    p0, p1, p2 = pos[i], pos[m], pos[n2]
                    ar = mg(crs(sub(p1, p0), sub(p2, p1)))
                    d2 = dt(sub(p2, p0), sub(p2, p0)) or 1e-12
                    r = ar / d2
                    e = acc.setdefault((min(i, n2), max(i, n2)), [0.0, 0.0, 0.0])
                    e[2] += max(0.0, 0.15 - abs(0.45 - r))
                    e[1] += max(0.0, 0.1 - r) * 3
    edge_tris = {}
    for ti, (a, b, c) in enumerate(tris):
        for x, y in ((a, b), (b, c), (c, a)):
            edge_tris.setdefault((min(x, y), max(x, y)), []).append(ti)
    # quadifier: greedy squarest-first triangle matching -> shear diagonals
    def cosc(o, p, q):
        u = sub(pos[p], pos[o]); v = sub(pos[q], pos[o])
        dn = mg(u) * mg(v) or 1e-12
        return abs(dt(u, v)) / dn
    sin60 = math.sin(math.radians(60))
    cands = []
    for e, tl in edge_tris.items():
        if len(tl) != 2:
            continue
        a, b = e; t1, t2 = tl
        c = [v for v in tris[t1] if v not in e][0]
        d_ = [v for v in tris[t2] if v not in e][0]
        cs = [cosc(c, b, a), cosc(a, c, d_), cosc(d_, a, b), cosc(b, d_, c)]
        cands.append((max(cs), (min(c, d_), max(c, d_)), (t1, t2)))
    cands.sort()
    used = set(); diagonals = set()
    for mx, dg, (t1, t2) in cands:
        if mx > sin60 or t1 in used or t2 in used:
            continue
        if not (mov[dg[0]] or mov[dg[1]]):
            continue
        used.add(t1); used.add(t2); diagonals.add(dg)

    if new_set is None:
        def is_new(p): return p[0] >= orig_vc or p[1] >= orig_vc
    else:
        def is_new(p): return p[0] in new_set or p[1] in new_set
    def length(p): return mg(sub(pos[p[0]], pos[p[1]]))
    stretch = []; shear = []; bend = []
    for e in edge_tris:
        if not is_new(e) or not (mov[e[0]] or mov[e[1]]):
            continue
        st, bd, sh = acc.get(e, (0.0, 0.0, 0.0))
        (shear if sh > max(st, bd) else stretch).append((e, length(e)))
    for dg in diagonals:
        if is_new(dg) and dg not in edge_tris:
            shear.append((dg, length(dg)))
    for pr, (st, bd, sh) in acc.items():
        if (is_new(pr) and pr not in edge_tris and pr not in diagonals
                and bd > 0 and bd > max(st, sh)):
            bend.append((pr, length(pr)))
    return [sorted(stretch), sorted(shear), sorted(bend)]


def _schedule_constraint_blocks(rows, movable, width=16):
    """Order ``[(pair, rest_length), ...]`` into conflict-free SIMD blocks.

    Snowdrop's 0x1111 value is the length of the leading scheduled prefix, not
    a generic performance hint: in every vanilla reference block before that
    boundary, a movable particle appears at most once.  Constraints after the
    boundary may conflict and are handled by the conservative tail path.

    Build as many complete ``width``-row blocks as possible.  If the remaining
    graph cannot fill another independent block, leave all remaining rows in
    the scalar tail.  Pinned endpoints do not conflict, matching NvCloth's
    cooker (which skips inverse-mass-zero particles while colouring).
    """
    remaining = list(rows)
    scheduled = []
    while len(remaining) >= width:
        block = []
        used = set()
        deferred = []
        for row in remaining:
            pair = row[0]
            active = {v for v in pair if movable[v]}
            if len(block) < width and not (active & used):
                block.append(row)
                used.update(active)
            else:
                deferred.append(row)
        if len(block) != width:
            break
        scheduled.extend(block)
        remaining = deferred
    return scheduled + remaining, len(scheduled)


def _grow_tether_trailer(trailing, orig_vc, new_vc, new_pos, donors, a1_pay,
                         trips, free_set=None, moved=None):
    """Rebuild the 0x110d per-vert tether TRAILER for GROWN counts. Returns
    (new_trailing_bytes, {0x110e trip index: (sc, mn, mx)}).

    Windows resize from ceil16(orig_vc) to ceil16(new_vc) with REAL rows for
    the appended verts: donor anchor in the u16 anchor-dup table (consistent
    with the 0x112a donor copy), recomputed 1.25*euclid radii in the window
    that encodes them, donor+slack transfer in the other length windows;
    appended verts NOT in free_set are pinned (self anchor, zero lengths).
    Original verts moved during the same growth export also get their decoded
    1.25*euclid tether radius refreshed (as in the count-preserving MOVE path).
    The u16 table spans TWO windows and must be re-laid out as one buffer -
    naive per-window padding inserts bytes into the table's MIDDLE and
    corrupts its second half. u8 windows get a fresh quant range over
    kept+new values (existing rows requantized)."""
    cvo = (orig_vc + 15) // 16 * 16
    cvn = (new_vc + 15) // 16 * 16
    if (not trailing or cvo == 0 or len(trailing) % cvo
            or new_vc <= orig_vc):
        return trailing, {}
    k = len(trailing) // cvo
    a1 = ([unpack('<H', a1_pay[i*2:i*2+2])[0] for i in range(orig_vc)]
          if a1_pay is not None and len(a1_pay) >= orig_vc * 2 else None)
    free_set = free_set or set()
    moved = moved or set()

    def _d(p, q):
        return ((p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2) ** 0.5

    out = []
    newtrips = {}
    wi = 0
    u8i = 0
    while wi < k:
        w = trailing[wi*cvo:(wi+1)*cvo]
        if wi + 1 < k:
            pair = trailing[wi*cvo:(wi+2)*cvo]
            vals = [unpack('<H', pair[i*2:i*2+2])[0] for i in range(orig_vc)]
            pad_ok = (orig_vc >= cvo or unpack(
                '<H', pair[orig_vc*2:orig_vc*2+2])[0] == orig_vc)
            if pad_ok and all(x < orig_vc for x in vals):
                ents = []
                for v in range(cvn):
                    if v < orig_vc:
                        ents.append(vals[v])
                    elif v < new_vc:
                        s = donors[v - orig_vc] \
                            if v - orig_vc < len(donors) else 0
                        ents.append(a1[s] if (a1 and v in free_set) else v)
                    else:
                        ents.append(v)  # self-index padding
                out.append(b''.join(pack('<H', e) for e in ents))
                wi += 2
                continue
        ti = 3 + u8i
        sc, mn = (trips[ti][0], trips[ti][1]) if ti < len(trips) \
            else (0.0, 0.0)
        rows = bytearray(b'\xff' * cvn)
        rows[:min(orig_vc, len(w))] = w[:orig_vc]
        if sc > 0 and a1:
            tmoved = {v for v in range(orig_vc) if v in moved
                      or (a1[v] < orig_vc and a1[v] in moved)}
            errs = sorted(abs(w[v]*sc+mn - 1.25*_d(new_pos[v],
                                                   new_pos[a1[v]]))
                          for v in range(orig_vc)
                          if v not in tmoved and a1[v] < orig_vc
                          and w[v] != 255)
            e125 = bool(errs) and errs[len(errs)//2] < 3.0 * sc
            newvals = {}
            if e125:
                for v in tmoved:
                    if a1[v] < orig_vc and w[v] != 255:
                        newvals[v] = 1.25 * _d(new_pos[v], new_pos[a1[v]])
            for v in range(orig_vc, new_vc):
                if v not in free_set:
                    newvals[v] = 0.0  # pinned: zero-length self tether
                    continue
                s = donors[v - orig_vc] if v - orig_vc < len(donors) else 0
                if e125:
                    newvals[v] = 1.25 * _d(new_pos[v], new_pos[a1[s]])
                elif s < len(w) and w[s] == 255:
                    newvals[v] = None  # donor row is filler: inherit it
                elif mn < -sc:
                    newvals[v] = w[s]*sc + mn - _d(new_pos[v], new_pos[s])
                else:
                    newvals[v] = w[s]*sc + mn + _d(new_pos[v], new_pos[s])
            deq = [w[v] * sc + mn for v in range(orig_vc)
                   if v not in newvals and w[v] != 255]
            allv = deq + [x for x in newvals.values() if x is not None]
            if allv:
                nmn, nmx = min(allv), max(allv)
                if nmx - nmn <= 0.0:
                    nmx = nmn + max(abs(nmn), 1e-6)
                nsc = (nmx - nmn) / 254.5
                for v in range(orig_vc):
                    if w[v] != 255:
                        rows[v] = max(0, min(254, int(round(
                            (w[v]*sc + mn - nmn) / nsc))))
                for v, val in newvals.items():
                    rows[v] = 255 if val is None else max(0, min(254, int(
                        round((val - nmn) / nsc))))
                newtrips[ti] = (nsc, nmn, nmx)
        else:
            for v in range(orig_vc, new_vc):
                s = donors[v - orig_vc] if v - orig_vc < len(donors) else 0
                rows[v] = w[s] if s < len(w) else 255
        out.append(bytes(rows))
        u8i += 1
        wi += 1
    return b''.join(out), newtrips


def _grow_triangle_samples(orig, orig_tc, tris, pos, valid_tris=None):
    """Grow the 0x11d1..0x11d5 per-triangle virtual-particle sample table.

    0x11d2/0x11d3 form a CSR into 0x11d5.  Every 0x11d5 u16 is two quantized
    triangle-local barycentric coordinates.  Snowdrop varies the number of
    samples with triangle size/shape; the weights themselves are portable to
    another triangle because they are barycentric.  For each appended triangle
    copy the complete record from the closest-shaped LIVE original triangle.
    Existing records and the terminal 0xFFFF words stay byte-identical.

    Returns replacement payloads keyed by chunk tag, or None when this optional
    table is absent/inconsistent or the source asset has no sampled triangle.
    """
    required = (T_SIM_VP_META, T_SIM_VP_COUNTS,
                T_SIM_VP_OFFSETS, T_SIM_VP_VALUES)
    if len(tris) <= orig_tc or not all(t in orig for t in required):
        return None
    meta = orig[T_SIM_VP_META]
    counts_pay = orig[T_SIM_VP_COUNTS]
    offsets_pay = orig[T_SIM_VP_OFFSETS]
    values_pay = orig[T_SIM_VP_VALUES]
    if (len(meta) < 8 or len(counts_pay) != orig_tc
            or len(offsets_pay) != orig_tc * 2 or len(values_pay) % 2):
        return None
    mt, word_count = unpack('<II', meta[:8])
    if mt != orig_tc or word_count * 2 != len(values_pay):
        return None
    counts = list(counts_pay)
    offsets = list(unpack(f'<{orig_tc}H', offsets_pay)) if orig_tc else []
    if orig_tc:
        if offsets[0] != 0 or any(offsets[i] + counts[i] != offsets[i + 1]
                                  for i in range(orig_tc - 1)):
            return None
        closing = offsets[-1] + counts[-1]
    else:
        closing = 0
    if closing > word_count:
        return None

    live = ({i for i in valid_tris if 0 <= i < orig_tc}
            if valid_tris is not None else set(range(orig_tc)))
    donors = [i for i in sorted(live) if counts[i] > 0]
    if not donors:
        donors = [i for i in range(orig_tc) if counts[i] > 0]
    if not donors:
        return None

    def feature(t):
        try:
            a, b, c = (pos[t[0]], pos[t[1]], pos[t[2]])
        except (IndexError, TypeError):
            return None
        def d2(p, q):
            return ((p[0]-q[0])**2 + (p[1]-q[1])**2
                    + (p[2]-q[2])**2)
        # Sorted squared edge lengths are orientation-independent and capture
        # both physical sampling density and triangle aspect ratio.
        return tuple(sorted((d2(a, b), d2(b, c), d2(c, a))))

    donor_features = [(i, feature(tris[i])) for i in donors]
    donor_features = [(i, f) for i, f in donor_features if f is not None]
    if not donor_features:
        return None

    body = bytearray(values_pay[:closing * 2])
    tail = values_pay[closing * 2:]
    new_counts = bytearray(counts_pay)
    new_offsets = list(offsets)
    cursor = closing
    for ti in range(orig_tc, len(tris)):
        f = feature(tris[ti])
        if f is None:
            donor = donor_features[0][0]
        else:
            donor = min(donor_features,
                        key=lambda it: sum((it[1][j] - f[j]) ** 2
                                          for j in range(3)))[0]
        count = counts[donor]
        if cursor > 0xFFFF or cursor + count > 0xFFFF:
            raise ValueError("0x11d5 virtual-particle offsets exceed uint16")
        new_offsets.append(cursor)
        new_counts.append(count)
        start = offsets[donor] * 2
        record = values_pay[start:start + count * 2]
        body += record
        cursor += count

    new_values = bytes(body) + tail
    return {
        T_SIM_VP_META: pack('<II', len(tris), len(new_values) // 2) + meta[8:],
        T_SIM_VP_COUNTS: bytes(new_counts),
        T_SIM_VP_OFFSETS: pack(f'<{len(new_offsets)}H', *new_offsets),
        T_SIM_VP_VALUES: new_values,
    }


def _sim_overrides(data, stream_end, new_pos, new_tri_bytes, valid_tris=None,
                   free_append=False):
    """Map {chunk_file_offset: full replacement chunk bytes} for the sim-section
    chunks whose contents depend on the sim vertex/triangle counts. Chunks not
    in the map are copied verbatim (including every undecoded chunk). With
    free_append, appended verts get cooked constraints and inherit the nearest
    original donor's free/pinned class; otherwise they are pinned (kinematic).
    ``valid_tris`` is the set of triangle slots present in the edited Blender
    SIM mesh; preserved phantom slots remain serialized but are excluded from
    constraint cooking."""
    new_vc = len(new_pos)
    new_tc = len(new_tri_bytes) // 6
    ov = {}
    # pre-scan: original counts + fabric chunk payloads
    orig_vc = orig_tc = None
    orig = {}
    scan = 20
    while scan + 8 <= stream_end:
        tag, size = unpack('<II', data[scan:scan + 8])
        t = tag & 0xFFFF
        if t in SIM_VC_FIELDS and orig_vc is None:
            orig_vc = unpack('<I', data[scan + 8:scan + 12])[0]
        elif t in SIM_TC_FIELDS and orig_tc is None:
            orig_tc = unpack('<I', data[scan + 8:scan + 12])[0]
        if t in (T_SIM_SECTS, T_SIM_PAIRS, T_SIM_RESTL, T_SIM_QUANT, T_SIM_REST,
                 T_SIM_VP_META, T_SIM_VP_COUNTS, T_SIM_VP_OFFSETS,
                 T_SIM_VP_SCALE, T_SIM_VP_VALUES, 0x112a,
                 T_SIM_AUX_HEADER, T_SIM_AUX_VALUES, T_SIM_AUX_DESCS):
            orig.setdefault(t, data[scan + 8:scan + size])
        scan += size
    if orig_vc is None:
        orig_vc = new_vc
    if orig_tc is None:
        orig_tc = new_tc

    tris_all = [unpack('<HHH', new_tri_bytes[i * 6:i * 6 + 6])
                for i in range(new_tc)]
    tri_samples = _grow_triangle_samples(
        orig, orig_tc, tris_all, new_pos, valid_tris=valid_tris)

    # Growth and movement can occur in the same Blender edit.  A moved
    # original slot needs the MOVE refresh even though growth takes precedence
    # at the rewrite entry point.  Detect against the source 0x110b while its
    # bytes are still available.
    moved = set()
    old_rest = orig.get(T_SIM_REST)
    if old_rest is not None and len(old_rest) >= orig_vc * 16:
        for v in range(min(orig_vc, new_vc)):
            op = unpack('<fff', old_rest[v*16:v*16+12])
            np = new_pos[v]
            if ((np[0]-op[0])**2 + (np[1]-op[1])**2
                    + (np[2]-op[2])**2) > 2.5e-9:  # 0.05mm
                moved.add(v)

    # Optional 0x11b3 is a quantized u8-per-vertex SIMD table. 0x11b2[0]
    # stores its 16-lane block count and exactly one 16-byte descriptor in
    # 0x11b4 repeats that count. All three must move together when ceil16(V)
    # crosses a boundary. An empty 0x11b3 is a distinct vanilla mode and must
    # remain empty.
    aux_meta = None
    aux_values = orig.get(T_SIM_AUX_VALUES, b'')
    if aux_values:
        aux_header = orig.get(T_SIM_AUX_HEADER, b'')
        aux_descs = orig.get(T_SIM_AUX_DESCS, b'')
        if len(aux_header) < 8:
            raise ValueError("0x11b3 is present without a valid 0x11b2 header")
        old_groups, desc_count = unpack('<II', aux_header[:8])
        if (old_groups <= 0 or len(aux_values) != old_groups * 16
                or len(aux_descs) != desc_count * 16):
            raise ValueError("Inconsistent 0x11b2/0x11b3/0x11b4 metadata")
        active = [i for i in range(desc_count)
                  if unpack('<I', aux_descs[i*16:i*16+4])[0] == old_groups]
        if len(active) != 1:
            raise ValueError("Could not identify the active 0x11b4 descriptor")
        aux_meta = (old_groups, desc_count, active[0])

    # Donor map for appended verts: nearest ORIGINAL sim vert by rest position.
    # Unknown per-vert tables get the donor's row - in-domain values everywhere
    # (e.g. 0x112a tether anchors stay valid pinned verts).
    donors = []
    if new_vc > orig_vc and orig_vc > 0:
        for v in range(orig_vc, new_vc):
            p = new_pos[v]
            best, bd = 0, float('inf')
            for u in range(orig_vc):
                q = new_pos[u]
                d2 = ((p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2)
                if d2 < bd:
                    bd, best = d2, u
            donors.append(best)

    # --- free-append mode: cook constraints for the appended verts ---
    fab = None
    if (free_append and new_vc > orig_vc
            and all(t in orig for t in (T_SIM_SECTS, T_SIM_PAIRS,
                                        T_SIM_RESTL, T_SIM_QUANT))):
        if valid_tris is None:
            tris = tris_all
        else:
            tris = [tris_all[i] for i in sorted(valid_tris)
                    if 0 <= i < len(tris_all)]
        rest = orig[T_SIM_REST]
        mov = [unpack('<f', rest[i * 16 + 12:i * 16 + 16])[0] == 0.0
               for i in range(orig_vc)]
        for k, _v in enumerate(range(orig_vc, new_vc)):
            donor = donors[k] if k < len(donors) else 0
            donor_free = (donor < orig_vc and
                          unpack('<f', rest[donor * 16 + 12:
                                            donor * 16 + 16])[0] == 0.0)
            mov.append(donor_free)

        # Every vanilla topology component in the paired corpus has at least
        # one pinned particle (373/373 components across 136 assets). A fully
        # movable disconnected island is not a valid cooked Snowdrop cloth:
        # the 202V/244T test was exactly such an island and crashed in-game.
        # Donor inheritance normally supplies its boundary pins. If spatial
        # transfer finds only free donors, pin the appended vertex closest to
        # any original pinned particle as a deterministic safety anchor.
        adjacency = [set() for _ in range(new_vc)]
        for a, b, c in tris:
            if a >= new_vc or b >= new_vc or c >= new_vc:
                continue
            for x, y in ((a, b), (b, c), (c, a)):
                adjacency[x].add(y)
                adjacency[y].add(x)
        original_pins = [v for v in range(orig_vc) if not mov[v]]
        seen = set()
        for start in range(new_vc):
            if start in seen:
                continue
            stack = [start]
            seen.add(start)
            component = []
            while stack:
                v = stack.pop()
                component.append(v)
                for other in adjacency[v]:
                    if other not in seen:
                        seen.add(other)
                        stack.append(other)
            appended = [v for v in component if v >= orig_vc]
            if not appended or any(not mov[v] for v in component):
                continue
            if original_pins:
                def pin_distance(v):
                    p = new_pos[v]
                    return min(sum((p[j] - new_pos[u][j]) ** 2
                                   for j in range(3))
                               for u in original_pins)
                anchor = min(appended, key=pin_distance)
            else:
                anchor = min(appended)
            mov[anchor] = False

        changed = set(range(orig_vc, new_vc)) | moved
        sections = cook_appended_constraints(new_pos, mov, tris, orig_vc,
                                             new_set=changed)
        free_set = {v for v in range(orig_vc, new_vc) if mov[v]}
        s3 = list(unpack('<3I', orig[T_SIM_SECTS][:12]))
        pairs_pay = orig[T_SIM_PAIRS]
        old_pairs = [[unpack('<HH', pairs_pay[(o + k) * 4:(o + k) * 4 + 4])
                      for k in range(c)]
                     for o, c in zip((0, s3[0], s3[0] + s3[1]), s3)]
        # rest lengths: dequant kept, merge appended, requant fresh per section
        trip = [unpack('<fff', orig[T_SIM_QUANT][i * 12:i * 12 + 12])
                for i in range(len(orig[T_SIM_QUANT]) // 12)]
        dpay = orig[T_SIM_RESTL]
        new_sects = []   # per section: (pairs list, value list)
        boff = 0
        for si in range(3):
            sc, mn, _mx = trip[si]
            vals = [mn + dpay[boff + k] * sc for k in range(s3[si])]
            boff += (s3[si] + 15) // 16 * 16
            # Keep byte-proven vanilla constraints that do not touch a moved
            # slot. Constraints touching moved originals are topology-sensitive
            # (the old faces may now be phantoms), so replace them with the
            # freshly cooked current-topology rows together with appended rows.
            kept = [(p, val) for p, val in zip(old_pairs[si], vals)
                    if p[0] not in moved and p[1] not in moved]
            prs = [p for p, _val in kept] + [p for p, _l in sections[si]]
            vals = [val for _p, val in kept] + [l for _p, l in sections[si]]
            new_sects.append((prs, vals))
        # Removing topology-sensitive vanilla rows closes gaps in the original
        # SIMD schedule; merely appending the replacement rows then makes the
        # old 0x1111 prefix lie. Rebuild conflict-free 16-lane blocks and carry
        # their exact prefix lengths into 0x1111.
        scheduled_sects = []
        safe_prefix = []
        for prs, vals in new_sects:
            rows, prefix = _schedule_constraint_blocks(
                list(zip(prs, vals)), mov, 16)
            scheduled_sects.append(
                ([p for p, _v in rows], [v for _p, v in rows]))
            safe_prefix.append(prefix)
        new_sects = scheduled_sects

        # requantize each section under a fresh range (render-table convention)
        newtrip = {}
        restl_out = bytearray()
        for si, (_prs, vals) in enumerate(new_sects):
            mn = min(vals) if vals else 0.0
            mx = max(vals) if vals else 0.0
            rng = mx - mn
            if rng <= 0.0:
                rng = max(abs(mn), 1e-6)
                mx = mn + rng
            s = rng / 254.5
            newtrip[si] = (s, mn, mx)
            restl_out += _pad16(bytes(max(0, min(254, int(round((v - mn) / s))))
                                      for v in vals))
        fab = dict(sections=new_sects, free_set=free_set,
                   s3new=[len(p) for p, _v in new_sects],
                   restl_sections=bytes(restl_out), newtrip=newtrip,
                   safe_prefix=safe_prefix)

    # --- 0x110d tether TRAILER rebuild for grown counts (both modes):
    #     appended free verts get real donor-anchor/radius rows, pinned ones
    #     get self/zero rows; the u16 anchor-dup table is re-laid out as one
    #     buffer (per-window padding would corrupt its second half). ---
    trail_new = None
    trail_trips = {}
    trail_b0 = 0
    if (new_vc > orig_vc and T_SIM_RESTL in orig and T_SIM_SECTS in orig
            and T_SIM_QUANT in orig):
        s3p = unpack('<3I', orig[T_SIM_SECTS][:12])
        trail_b0 = sum((s + 15) // 16 * 16 for s in s3p)
        trips_all = [unpack('<fff', orig[T_SIM_QUANT][i*12:i*12+12])
                     for i in range(len(orig[T_SIM_QUANT]) // 12)]
        trail_new, trail_trips = _grow_tether_trailer(
            orig[T_SIM_RESTL][trail_b0:], orig_vc, new_vc, new_pos, donors,
            orig.get(0x112a), trips_all,
            free_set=fab['free_set'] if fab else set(), moved=moved)

    off = 20
    while off + 8 <= stream_end:
        tag, size = unpack('<II', data[off:off + 8])
        t = tag & 0xFFFF
        pay = data[off + 8:off + size]
        new_pay = None
        if t in SIM_VC_FIELDS:
            new_pay = pack('<I', new_vc) + pay[4:]
        elif t in SIM_TC_FIELDS:
            if tri_samples is not None and t == T_SIM_VP_META:
                new_pay = tri_samples[t]
            else:
                new_pay = pack('<I', new_tc) + pay[4:]
        elif t in SIM_PADV_FIELDS:
            new_pay = pack('<I', (new_vc + 15) // 16 * 16) + pay[4:]
        elif t == T_SIM_AUX_HEADER and aux_meta is not None:
            new_groups = (new_vc + 15) // 16
            new_pay = pack('<I', new_groups) + pay[4:]
        elif t == T_SIM_AUX_DESCS and aux_meta is not None:
            new_groups = (new_vc + 15) // 16
            active = aux_meta[2]
            new_pay = bytearray(pay)
            new_pay[active*16:active*16+4] = pack('<I', new_groups)
            new_pay = bytes(new_pay)
        elif t == T_SIM_REST:
            new_pay = _sim_rest_table(pay, orig_vc, new_pos,
                                      fab['free_set'] if fab else None)
        elif t == T_SIM_TRIS:
            new_pay = new_tri_bytes
        elif t in SIM_VERT_TABLES:
            # SIMD-16 tables: the engine processes the PADDING lanes too, and
            # vanilla keeps them inert (0x112a: self-index anchor = no-op;
            # 0x11b3: 0xFF). Zeroing them turns phantom lanes into live work
            # against vertex 0 -> intermittent corruption. Appended REAL verts
            # get their donor's row; padding rows get the table's inert filler.
            per = SIM_VERT_TABLES[t]
            if not pay:
                # Empty 0x112a/0x11b3 payloads are intentional optional-table
                # modes. Materializing them as zero-filled arrays introduces
                # live vertex-0 anchors and is not equivalent to vanilla.
                new_pay = pay
            elif new_vc <= orig_vc:
                new_pay = _resize_padded(pay, orig_vc, new_vc, per, 16)
            else:
                cvn = (new_vc + 15) // 16 * 16
                buf = bytearray(per * cvn)
                keep = min(len(pay), per * cvn)
                buf[:keep] = pay[:keep]
                for k, v in enumerate(range(orig_vc, new_vc)):
                    s = donors[k] if k < len(donors) else 0
                    free_slots = fab['free_set'] if fab else set()
                    if t == 0x112a and v not in free_slots:
                        # Pinned appended rows use the same self anchor in both
                        # 0x112a and its duplicate in the 0x110d trailer.
                        buf[v * 2:(v + 1) * 2] = pack('<H', v)
                    elif (s + 1) * per <= len(pay):
                        buf[v * per:(v + 1) * per] = pay[s * per:(s + 1) * per]
                for v in range(new_vc, cvn):
                    if t == 0x112a:
                        buf[v * 2:(v + 1) * 2] = pack('<H', v)  # self-index
                    else:
                        buf[v * per:(v + 1) * per] = b'\xff' * per
                new_pay = bytes(buf)
        elif tri_samples is not None and t in (
                T_SIM_VP_COUNTS, T_SIM_VP_OFFSETS, T_SIM_VP_VALUES):
            new_pay = tri_samples[t]
        elif t in SIM_TRI_TABLES:
            if (t == 0x11d3 and new_tc > orig_tc and orig_tc > 0
                    and 0x11d2 in orig and len(pay) >= orig_tc * 2):
                # 0x11d2/0x11d3 = CSR (count, offset) per tri into the 0x11d5
                # records (d3[t]+d2[t] == d3[t+1], corpus 99/99). Appended tris
                # must carry the CLOSING offset - not 0 - so both d2-based and
                # offset-difference reads see an empty record range. A zero here
                # makes the last original tri's range negative: heap overread,
                # intermittent crash on equip.
                last_off = unpack('<H', pay[(orig_tc - 1) * 2:orig_tc * 2])[0]
                closing = last_off + orig[0x11d2][orig_tc - 1]
                new_pay = (pay[:orig_tc * 2]
                           + pack('<H', closing) * (new_tc - orig_tc))
            else:
                new_pay = _resize_padded(pay, orig_tc, new_tc, SIM_TRI_TABLES[t], 1)
        elif fab and t == T_SIM_SECTS:
            new_pay = pack('<3I', *fab['s3new'])
        elif fab and t == T_SIM_PAIRS:
            new_pay = b''.join(pack('<HH', *p)
                               for prs, _v in fab['sections'] for p in prs)
        elif t == T_SIM_RESTL and fab:
            new_pay = fab['restl_sections'] + (
                trail_new if trail_new is not None else pay[trail_b0:])
        elif t == T_SIM_RESTL and trail_new is not None:
            # pinned mode: sections verbatim, trailer rebuilt (self/zero rows
            # for the appended kinematic verts, windows resized correctly)
            new_pay = pay[:trail_b0] + trail_new
        elif t == T_SIM_QUANT and (fab or trail_trips):
            out = bytearray(pay)
            merged = dict(fab['newtrip']) if fab else {}
            merged.update(trail_trips)
            for si, tr in merged.items():
                if si * 12 + 12 <= len(out):
                    out[si * 12:si * 12 + 12] = pack('<fff', *tr)
            new_pay = bytes(out)
        elif t == T_SIM_GROUPS and (fab or new_vc != orig_vc):
            g = list(unpack(f'<{len(pay) // 4}I', pay))
            if fab:
                for si in range(3):
                    g[si] = (fab['s3new'][si] + 15) // 16
            particle_groups = (new_vc + 15) // 16
            for si in range(3, min(6, len(g))):
                if g[si] != 0:
                    g[si] = particle_groups
            new_pay = pack(f'<{len(g)}I', *g)
        elif fab and t == T_SIM_PREFIX16:
            new_pay = pack('<3I', *fab['safe_prefix']) + pay[12:]
        elif t == T_SIM_PERM:
            # Usually 0x111e appears twice (free-vertex list, then triangle
            # permutation), but some vanilla streams omit the free list and
            # carry only the permutation. Identify it by CONTENT rather than
            # occurrence order: exactly 2*T bytes containing 0..T-1 once.
            is_tri_perm = False
            if orig_tc >= 0 and len(pay) == orig_tc * 2:
                vals = unpack(f'<{orig_tc}H', pay) if orig_tc else ()
                is_tri_perm = sorted(vals) == list(range(orig_tc))
            if is_tri_perm:
                new_pay = pay + b''.join(pack('<H', i)
                                         for i in range(orig_tc, new_tc))
            elif fab and fab['free_set']:
                new_pay = pay + b''.join(pack('<H', v)
                                         for v in sorted(fab['free_set']))
        if new_pay is not None:
            ov[off] = pack('<II', tag, 8 + len(new_pay)) + new_pay
        off += size
    return ov


def _sim_reuse_overrides(data, stream_end, new_pos, new_tri_bytes, reused):
    """
    Budget-reuse rewrite: counts and chunk sizes stay EXACTLY vanilla; only
    values change. `reused` = set of vert slots now occupied by new geometry.
    Rewrites: 0x110b positions (flags preserved per slot), 0x112a/0x11b3 donor
    rows for reused slots, 0x1113 tri buffer mirror, the constraint rows
    referencing reused slots in 0x1124/0x110d (cooked pairs + rest lengths),
    the per-vert TETHER trailer of 0x110d (stale rows there clamp reused
    slots back to the DELETED vert's position), and 0x110e quant triplets,
    which WIDEN when new lengths/radii fall outside the vanilla range (the
    old clamping actively contracted long new edges); existing rows are
    requantized under any widened range.
    """
    V = len(new_pos)
    ov = {}
    orig = {}
    scan = 20
    while scan + 8 <= stream_end:
        tag, size = unpack('<II', data[scan:scan + 8])
        t = tag & 0xFFFF
        if t in (T_SIM_SECTS, T_SIM_PAIRS, T_SIM_RESTL, T_SIM_QUANT,
                 T_SIM_REST, 0x1102, 0x112a):
            orig.setdefault(t, data[scan + 8:scan + size])
        scan += size
    if (0x1102 not in orig or unpack('<I', orig[0x1102][:4])[0] != V
            or not reused):
        return {}
    rest = orig[T_SIM_REST]
    flags = [unpack('<f', rest[i * 16 + 12:i * 16 + 16])[0] for i in range(V)]
    mov = [f == 0.0 for f in flags]

    # donors for reused slots: nearest NON-reused vert
    donor = {}
    keep = [u for u in range(V) if u not in reused]
    for v in reused:
        p = new_pos[v]
        best, bd = None, float('inf')
        for u in keep:
            q = new_pos[u]
            d2 = (p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2
            if d2 < bd:
                bd, best = d2, u
        donor[v] = best if best is not None else 0

    def _dist(p, q):
        return ((p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2) ** 0.5

    # Quantization triplets (0x110e) are WIDENED when new values fall outside
    # a section's/window's vanilla range - the old clamp-to-range behavior
    # actively CONTRACTED long new edges to vanilla lengths (the 'clumped
    # together' bug). Widening requantizes the existing rows under the new
    # range: value-only, sizes identical, precision degrades gracefully.
    trips_pay = orig.get(T_SIM_QUANT)
    trips = ([list(unpack('<fff', trips_pay[i*12:i*12+12]))
              for i in range(len(trips_pay) // 12)] if trips_pay else [])
    quant_changed = False
    restl_pay = (bytearray(orig[T_SIM_RESTL])
                 if T_SIM_RESTL in orig else None)
    s3 = (list(unpack('<3I', orig[T_SIM_SECTS][:12]))
          if T_SIM_SECTS in orig else None)

    def _q(val, sc, mn):
        if sc <= 0:
            return 0
        return max(0, min(254, int(round((val - mn) / sc))))

    def _widen_requant(ti, needed, offsets):
        """Grow quant triplet `ti` to cover every value in `needed`,
        requantizing the existing restl_pay bytes at `offsets` under the new
        range. Returns the (scale, min) to quantize new values with."""
        nonlocal quant_changed
        sc, mn, mx = trips[ti]
        if sc <= 0 or not needed:
            return sc, mn
        lo, hi = min(needed), max(needed)
        if lo >= mn - 1e-9 and hi <= mx + 1e-9:
            return sc, mn
        nmn, nmx = min(mn, lo), max(mx, hi)
        nsc = (nmx - nmn) / 254.5
        for o in offsets:
            b = restl_pay[o]
            if b == 255:  # window filler, not a quantized value
                continue
            restl_pay[o] = _q(mn + b * sc, nsc, nmn)
        trips[ti] = [nsc, nmn, nmx]
        quant_changed = True
        return nsc, nmn

    # cook replacement constraints for the reused slots
    new_pairs = None
    if restl_pay is not None and s3 is not None and len(trips) >= 3 \
            and T_SIM_PAIRS in orig:
        tris = [unpack('<HHH', new_tri_bytes[i*6:i*6+6])
                for i in range(len(new_tri_bytes) // 6)]
        cooked = cook_appended_constraints(new_pos, mov, tris, V,
                                           new_set=reused)
        pairs_pay = bytearray(orig[T_SIM_PAIRS])
        base_pair = 0
        base_rest = 0
        for si in range(3):
            cnt = s3[si]
            rows = []      # row indices (within section) referencing reused
            keep_rows = [] # rows with no reused endpoint (dup donors)
            for r in range(cnt):
                o = (base_pair + r) * 4
                a, b = unpack('<HH', pairs_pay[o:o + 4])
                (rows if (a in reused or b in reused) else keep_rows).append(r)
            ck = cooked[si]
            if rows and ck:
                sc, mn = _widen_requant(
                    si, [L for _pr, L in ck],
                    [base_rest + r for r in range(cnt)])
            else:
                sc, mn = trips[si][0], trips[si][1]
            nwr = min(len(rows), len(ck))
            for k in range(nwr):
                r = rows[k]
                (a, b), L = ck[k]
                pairs_pay[(base_pair + r)*4:(base_pair + r)*4 + 4] = pack('<HH', a, b)
                restl_pay[base_rest + r] = _q(L, sc, mn)
            # spare old rows (deleted verts had more constraints than the new
            # geometry needs): repoint to a duplicate of a valid row - a
            # doubled distance constraint is a valid, slightly stiffer no-op
            for k in range(nwr, len(rows)):
                r = rows[k]
                if ck:
                    (a, b), L = ck[k % len(ck)]
                    pairs_pay[(base_pair + r)*4:(base_pair + r)*4 + 4] = pack('<HH', a, b)
                    restl_pay[base_rest + r] = _q(L, sc, mn)
                elif keep_rows:
                    src = keep_rows[k % len(keep_rows)]
                    pairs_pay[(base_pair + r)*4:(base_pair + r)*4 + 4] = \
                        pairs_pay[(base_pair + src)*4:(base_pair + src)*4 + 4]
                    restl_pay[base_rest + r] = restl_pay[base_rest + src]
            base_pair += cnt
            base_rest += (cnt + 15) // 16 * 16
        new_pairs = bytes(pairs_pay)

    # --- 0x110d TETHER TRAILER: per-vert tables in ceil16(V)-byte windows
    #     after the padded constraint sections - u8 quantized tether lengths
    #     (quant params = 0x110e triplets [3:6], in window order; the window
    #     matching 1.25*|pos[v]-pos[anchor]| is recomputed exactly, others
    #     are donor-transferred with slack) and a u16 anchor table (a
    #     DUPLICATE of 0x112a spanning two windows, self-index padding).
    #     Left stale, these tether a reused slot to its DELETED position. ---
    if restl_pay is not None and s3 is not None and 0x112a in orig:
        c16 = (V + 15) // 16 * 16
        base = sum((c + 15) // 16 * 16 for c in s3)
        if base < len(restl_pay) and (len(restl_pay) - base) % c16 == 0:
            k = (len(restl_pay) - base) // c16
            a1p = orig[0x112a]
            a1 = [unpack('<H', a1p[i*2:i*2+2])[0] for i in range(V)]
            wi = 0
            u8i = 0
            while wi < k:
                w0 = base + wi * c16
                if wi + 1 < k:
                    # u16 anchor-dup detection: all real entries in vert
                    # range, first padding row self-indexes (== V)
                    vals = [unpack('<H', restl_pay[w0+i*2:w0+i*2+2])[0]
                            for i in range(V)]
                    pad_ok = (V >= c16 or unpack(
                        '<H', restl_pay[w0+V*2:w0+V*2+2])[0] == V)
                    if pad_ok and all(x < V for x in vals):
                        for v in reused:
                            restl_pay[w0+v*2:w0+v*2+2] = \
                                pack('<H', a1[donor[v]])
                        wi += 2
                        continue
                ti = 3 + u8i
                sc, mn = (trips[ti][0], trips[ti][1]) if ti < len(trips) \
                    else (0.0, 0.0)
                if sc > 0:
                    # is this window the 1.25*euclid(v, anchor) table?
                    errs = []
                    for v in range(V):
                        if v in reused or a1[v] >= V:
                            continue
                        got = restl_pay[w0 + v] * sc + mn
                        errs.append(abs(got - 1.25 * _dist(new_pos[v],
                                                           new_pos[a1[v]])))
                        if len(errs) >= 48:
                            break
                    errs.sort()
                    euclid125 = bool(errs) and errs[len(errs)//2] < 3.0 * sc
                    newvals = {}
                    for v in reused:
                        s = donor[v]
                        if euclid125:
                            newvals[v] = 1.25 * _dist(new_pos[v],
                                                      new_pos[a1[s]])
                        elif restl_pay[w0 + s] == 255:
                            newvals[v] = None  # donor has filler: copy it
                        elif mn < -sc:  # negative range: widen downward
                            newvals[v] = restl_pay[w0 + s] * sc + mn \
                                - _dist(new_pos[v], new_pos[s])
                        else:           # positive range: widen upward
                            newvals[v] = restl_pay[w0 + s] * sc + mn \
                                + _dist(new_pos[v], new_pos[s])
                    if ti < len(trips):
                        sc, mn = _widen_requant(
                            ti, [x for x in newvals.values()
                                 if x is not None],
                            [w0 + u for u in range(V) if u not in reused])
                    for v, val in newvals.items():
                        restl_pay[w0 + v] = 255 if val is None \
                            else _q(val, sc, mn)
                else:
                    for v in reused:
                        restl_pay[w0 + v] = restl_pay[w0 + donor[v]]
                u8i += 1
                wi += 1

    off = 20
    while off + 8 <= stream_end:
        tag, size = unpack('<II', data[off:off + 8])
        t = tag & 0xFFFF
        pay = data[off + 8:off + size]
        new_pay = None
        if t == T_SIM_REST:
            rows = bytearray(pay)
            for i in range(V):
                x, y, z = new_pos[i]
                rows[i * 16:i * 16 + 12] = pack('<fff', x, y, z)
            new_pay = bytes(rows)
        elif t == T_SIM_TRIS and len(pay) == len(new_tri_bytes):
            new_pay = new_tri_bytes
        elif t in SIM_VERT_TABLES:
            per = SIM_VERT_TABLES[t]
            buf = bytearray(pay)
            for v in reused:
                s = donor[v]
                if (max(v, s) + 1) * per <= len(buf):
                    buf[v * per:(v + 1) * per] = pay[s * per:(s + 1) * per]
            new_pay = bytes(buf)
        elif new_pairs is not None and t == T_SIM_PAIRS:
            new_pay = new_pairs
        elif restl_pay is not None and t == T_SIM_RESTL:
            new_pay = bytes(restl_pay)
        elif quant_changed and t == T_SIM_QUANT:
            body = b''.join(pack('<fff', *tr) for tr in trips)
            new_pay = body + pay[len(body):]
        if new_pay is not None and len(new_pay) == len(pay):
            ov[off] = pack('<II', tag, 8 + len(new_pay)) + new_pay
        off += size
    return ov


def _sim_move_overrides(data, stream_end, new_pos, moved):
    """Value-refresh for a sim MOVE (topology UNCHANGED): refresh 0x110b
    positions to the moved geometry, recompute rest lengths (0x110d) for
    constraint pairs touching a moved vert (widening 0x110e ranges as needed),
    and recompute the decoded per-vert tether radii for moved verts. Constraint
    PAIRS, the tri buffer, tether anchors and all counts stay vanilla.

    Two purposes: (1) the sim rest state in the mcloth matches the moved mmb
    geometry, and (2) changing the sim-section bytes makes the engine re-cook
    the render->sim mapping, so render-vertex edits bundled into the same
    export take effect (they are otherwise cached against the old fabric)."""
    V = len(new_pos)
    ov = {}
    orig = {}
    scan = 20
    while scan + 8 <= stream_end:
        tag, size = unpack('<II', data[scan:scan + 8])
        t = tag & 0xFFFF
        if t in (T_SIM_SECTS, T_SIM_PAIRS, T_SIM_RESTL, T_SIM_QUANT,
                 T_SIM_REST, 0x1102, 0x112a):
            orig.setdefault(t, data[scan + 8:scan + size])
        scan += size
    if (0x1102 not in orig or unpack('<I', orig[0x1102][:4])[0] != V
            or not moved or T_SIM_REST not in orig):
        return {}

    def _dist(a, b):
        p, q = new_pos[a], new_pos[b]
        return ((p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2) ** 0.5

    trips_pay = orig.get(T_SIM_QUANT)
    trips = ([list(unpack('<fff', trips_pay[i*12:i*12+12]))
              for i in range(len(trips_pay) // 12)] if trips_pay else [])
    quant_changed = False
    restl_pay = (bytearray(orig[T_SIM_RESTL])
                 if T_SIM_RESTL in orig else None)
    s3 = (list(unpack('<3I', orig[T_SIM_SECTS][:12]))
          if T_SIM_SECTS in orig else None)
    pairs_pay = orig.get(T_SIM_PAIRS)

    def _q(val, sc, mn):
        if sc <= 0:
            return 0
        return max(0, min(254, int(round((val - mn) / sc))))

    def _widen_requant(ti, needed, offsets):
        """Grow triplet `ti` to cover `needed`, requantizing existing bytes at
        `offsets` under the new range. Returns the (scale, min) to use."""
        nonlocal quant_changed
        sc, mn, mx = trips[ti]
        if sc <= 0 or not needed:
            return sc, mn
        lo, hi = min(needed), max(needed)
        if lo >= mn - 1e-9 and hi <= mx + 1e-9:
            return sc, mn
        nmn, nmx = min(mn, lo), max(mx, hi)
        nsc = (nmx - nmn) / 254.5
        for o in offsets:
            b = restl_pay[o]
            if b == 255:
                continue
            restl_pay[o] = _q(mn + b * sc, nsc, nmn)
        trips[ti] = [nsc, nmn, nmx]
        quant_changed = True
        return nsc, nmn

    if restl_pay is not None and s3 is not None and len(trips) >= 3 \
            and pairs_pay:
        # rest lengths for pairs touching a moved vert
        base_pair = 0
        base_rest = 0
        for si in range(3):
            cnt = s3[si]
            rows, lens = [], []
            for r in range(cnt):
                o = (base_pair + r) * 4
                a, b = unpack('<HH', pairs_pay[o:o + 4])
                if a in moved or b in moved:
                    rows.append(r)
                    lens.append(_dist(a, b))
            if rows:
                sc, mn = _widen_requant(si, lens,
                                        [base_rest + r for r in range(cnt)])
                for r, L in zip(rows, lens):
                    restl_pay[base_rest + r] = _q(L, sc, mn)
            base_pair += cnt
            base_rest += (cnt + 15) // 16 * 16

        # tether trailer: recompute the decoded 1.25*euclid radius window for
        # verts that moved (or whose anchor moved); anchors themselves stay.
        if 0x112a in orig:
            c16 = (V + 15) // 16 * 16
            base = sum((c + 15) // 16 * 16 for c in s3)
            if base < len(restl_pay) and (len(restl_pay) - base) % c16 == 0:
                k = (len(restl_pay) - base) // c16
                a1 = [unpack('<H', orig[0x112a][i*2:i*2+2])[0]
                      for i in range(V)]
                tmoved = {v for v in range(V) if v in moved
                          or (a1[v] < V and a1[v] in moved)}
                wi = 0
                u8i = 0
                while wi < k:
                    w0 = base + wi * c16
                    if wi + 1 < k:
                        vals = [unpack('<H', restl_pay[w0+i*2:w0+i*2+2])[0]
                                for i in range(V)]
                        pad_ok = (V >= c16 or unpack(
                            '<H', restl_pay[w0+V*2:w0+V*2+2])[0] == V)
                        if pad_ok and all(x < V for x in vals):
                            wi += 2   # u16 anchor dup: unchanged on a move
                            continue
                    ti = 3 + u8i
                    sc, mn = (trips[ti][0], trips[ti][1]) if ti < len(trips) \
                        else (0.0, 0.0)
                    if sc > 0:
                        errs = []
                        for v in range(V):
                            if v in tmoved or a1[v] >= V:
                                continue
                            got = restl_pay[w0 + v] * sc + mn
                            errs.append(abs(got - 1.25 * _dist(v, a1[v])))
                            if len(errs) >= 48:
                                break
                        errs.sort()
                        if errs and errs[len(errs)//2] < 3.0 * sc:
                            nv = {v: 1.25 * _dist(v, a1[v]) for v in tmoved
                                  if a1[v] < V and restl_pay[w0 + v] != 255}
                            sc, mn = _widen_requant(
                                ti, list(nv.values()),
                                [w0 + u for u in range(V) if u not in tmoved])
                            for v, val in nv.items():
                                restl_pay[w0 + v] = _q(val, sc, mn)
                    u8i += 1
                    wi += 1

    off = 20
    while off + 8 <= stream_end:
        tag, size = unpack('<II', data[off:off + 8])
        t = tag & 0xFFFF
        pay = data[off + 8:off + size]
        new_pay = None
        if t == T_SIM_REST:
            rows = bytearray(pay)
            for i in range(V):
                x, y, z = new_pos[i]
                rows[i * 16:i * 16 + 12] = pack('<fff', x, y, z)
            new_pay = bytes(rows)
        elif restl_pay is not None and t == T_SIM_RESTL:
            new_pay = bytes(restl_pay)
        elif quant_changed and t == T_SIM_QUANT:
            body = b''.join(pack('<fff', *tr) for tr in trips)
            new_pay = body + pay[len(body):]
        if new_pay is not None and len(new_pay) == len(pay):
            ov[off] = pack('<II', tag, 8 + len(new_pay)) + new_pay
        off += size
    return ov


def sim_counts(data):
    """(sim vertex count, sim triangle count) read from the mcloth sim count
    fields, or (None, None) when absent. Lets callers detect a sim topology
    change (vs the exported mmb) before deciding to rewrite the sim section."""
    try:
        streams, _footer_offset = parse_streams(data)
    except ValueError:
        return None, None
    if len(streams) != 1:
        return None, None
    stream_start = streams[0]['start']
    stream_end = streams[0]['end']
    vc = tc = None
    off = stream_start
    while off + 8 <= stream_end:
        tag, size = unpack('<II', data[off:off + 8])
        t = tag & 0xFFFF
        if t in SIM_VC_FIELDS and vc is None:
            vc = unpack('<I', data[off + 8:off + 12])[0]
        elif t in SIM_TC_FIELDS and tc is None:
            tc = unpack('<I', data[off + 8:off + 12])[0]
        off += size
    return vc, tc


def _emit_region(data, lo, hi, overrides):
    """Copy [lo, hi) chunk by chunk, substituting any overridden chunk."""
    if not overrides:
        return data[lo:hi]
    out = bytearray()
    off = lo
    while off < hi:
        _tag, size = unpack('<II', data[off:off + 8])
        out += overrides.get(off, data[off:off + size])
        off += size
    return bytes(out)


def rewrite(data, remaps, rebind=None, computed=None, sim=None, sim_free=False,
            sim_reuse=None, sim_move=None, sim_tri_remap=None):
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
    :param sim: optional (new_sim_positions, new_sim_tri_bytes) to refresh the
           sim-section count fields and decoded tables after a _CLOTH_SIM edit.
           new_sim_tri_bytes is the sim triangle index buffer (u16[3] per tri).
    :param sim_free: with sim, cook constraints for appended sim verts and
           transfer each one's free/pinned class from its nearest original
           donor (also enforcing a pin per new disconnected component). When
           False, every appended vertex is pinned/kinematic.
    :param sim_reuse: (new_sim_positions, new_sim_tri_bytes, reused_slot_set)
           BUDGET-REUSE mode: counts unchanged, new verts occupy the reused
           (orphaned) slots; values rewritten in place. Takes precedence over
           `sim` and is the engine-stable path for adding sim geometry.
    :param sim_move: (new_sim_positions, moved_slot_set) for a MOVE (topology
           unchanged): refresh 0x110b positions + rest lengths/tethers for the
           moved verts. Also forces the engine to re-cook the render mapping.
    :param sim_tri_remap: optional old->new sim triangle-ID map applied only to
           rows retained from the source render blocks. Synthesized rows already
           address the exported topology and are deliberately not remapped.
    :return: (new file bytes, {block_name: (old_B, new_B)})
    """
    stream_end, blocks = parse_blocks(data)
    if sim_reuse:
        sim_ov = _sim_reuse_overrides(data, stream_end, *sim_reuse)
    elif sim:
        sim_ov = _sim_overrides(data, stream_end, *sim, free_append=sim_free)
    elif sim_move:
        sim_ov = _sim_move_overrides(data, stream_end, *sim_move)
    else:
        sim_ov = {}
    stats = {}
    spans = []
    for block_name, (new_vc, src_name, old_to_new) in remaps.items():
        tgt = blocks[block_name]
        extra = (computed or {}).get(block_name) or []
        if extra:
            pieces, new_B = _build_values(data, blocks, tgt, block_name, new_vc,
                                          (src_name, old_to_new), extra, rebind,
                                          tri_remap=sim_tri_remap)
        else:
            pieces, new_B = _build_single(data, blocks, tgt, block_name, new_vc,
                                          (src_name, old_to_new), rebind,
                                          tri_remap=sim_tri_remap)

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
        out += _emit_region(data, pos, start, sim_ov)
        out += rep
        pos = end
    out += _emit_region(data, pos, stream_end, sim_ov)
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


def mmb_lod_swap_components(file_bytes, mesh, li, vertex_pairs, triangle_pairs):
    """Swap two equal-sized SIM components in an MMB LOD without changing
    counts or layout.

    ``vertex_pairs`` and ``triangle_pairs`` are ``(old_id, appended_id)``
    pairs. Complete position and normal-stream records are exchanged. Each
    exchanged triangle is also rewritten through the bidirectional vertex-ID
    map, so the geometry stays intact while the two components trade IDs.
    """
    lod = mesh.lods[li]
    if mesh.position_type != 1:
        raise ValueError("SIM component swapping requires float positions")
    if lod.size_a != lod.face_block_offset // 2:
        raise ValueError("SIM component swapping requires uint16 indices")
    vp = [(int(a), int(b)) for a, b in vertex_pairs]
    tp = [(int(a), int(b)) for a, b in triangle_pairs]
    if not vp or not tp:
        raise ValueError("SIM component swap cannot be empty")
    if len({a for p in vp for a in p}) != len(vp) * 2:
        raise ValueError("SIM vertex swap IDs overlap")
    if len({a for p in tp for a in p}) != len(tp) * 2:
        raise ValueError("SIM triangle swap IDs overlap")
    if any(a < 0 or b < 0 or a >= lod.vertex_count or b >= lod.vertex_count
           for a, b in vp):
        raise ValueError("SIM vertex swap ID is out of range")
    tc = lod.index_count // 3
    if any(a < 0 or b < 0 or a >= tc or b >= tc for a, b in tp):
        raise ValueError("SIM triangle swap ID is out of range")

    higher = sum(mesh.lods[k].data_size
                 for k in range(li + 1, len(mesh.lods)))
    pos_base = lod.data_offset + (lod.vertex_data_offset_a - higher)
    nor_base = lod.data_offset + (lod.vertex_data_offset_b - higher)
    tri_base = lod.data_offset + (lod.face_block_offset - higher)
    out = bytearray(file_bytes)

    def swap_records(base, stride, pairs):
        if stride <= 0:
            return
        for a, b in pairs:
            ao, bo = base + a * stride, base + b * stride
            ar, br = bytes(out[ao:ao + stride]), bytes(out[bo:bo + stride])
            if len(ar) != stride or len(br) != stride:
                raise ValueError("SIM component record extends past the MMB")
            out[ao:ao + stride], out[bo:bo + stride] = br, ar

    swap_records(pos_base, mesh.vertex_stride, vp)
    swap_records(nor_base, mesh.normals_stride, vp)

    vmap = {a: b for a, b in vp}
    vmap.update({b: a for a, b in vp})
    for a, b in tp:
        ao, bo = tri_base + a * 6, tri_base + b * 6
        at = unpack('<HHH', bytes(out[ao:ao + 6]))
        bt = unpack('<HHH', bytes(out[bo:bo + 6]))
        if not all(v in vmap for v in at + bt):
            raise ValueError("SIM component triangle crosses the swap boundary")
        out[ao:ao + 6] = pack('<HHH', *(vmap[v] for v in bt))
        out[bo:bo + 6] = pack('<HHH', *(vmap[v] for v in at))
    return bytes(out)


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

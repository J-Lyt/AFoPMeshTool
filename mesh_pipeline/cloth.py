"""Blender orchestration for paired .mcloth exports."""

import importlib.util
import io
import operator
import os
from struct import pack, unpack

import bpy

from .. import addon_state
from ..log import logger
from ..mmb import SkeletalMeshAsset

try:
    from .. import mcloth
except ImportError:
    try:
        _path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'mcloth.py',
        )
        _spec = importlib.util.spec_from_file_location('afop_mcloth', _path)
        mcloth = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(mcloth)
    except Exception as error:
        mcloth = None
        logger.warning("mcloth.py is unavailable; cloth export is disabled: %s", error)

def _sim_free_slot_flags(orig_vc):
    """[bool]*orig_vc: True where the source .mcloth marks the sim vert FREE
    (simulating). Used to hand reused slots to new verts free-first. None when
    the mcloth is unavailable."""
    try:
        src = mcloth.source_path(bpy.context.scene.SWOMT.AssetPath) if mcloth else None
        if not src:
            return None
        with open(src, 'rb') as f:
            d = f.read()
        streams, _footer_offset = mcloth.parse_streams(d)
        if len(streams) != 1:
            return None
        off = streams[0]['start']
        stream_end = streams[0]['end']
        while off + 8 <= stream_end:
            tag, size = unpack('<II', d[off:off + 8])
            if (tag & 0xFFFF) == mcloth.T_SIM_REST:
                pay = d[off + 8:off + size]
                if len(pay) >= orig_vc * 16:
                    return [unpack('<f', pay[i*16+12:i*16+16])[0] == 0.0
                            for i in range(orig_vc)]
                return None
            off += size
    except Exception:
        logger.debug("Could not read cloth free-slot flags", exc_info=True)
    return None


def _export_mcloth_for_asset(out_mmb_path, operator=None):
    """
    After the mmb has been exported, remap the paired .mcloth for every cloth
    render mesh LOD object present in the scene and write it next to the
    exported mmb. No-op when the asset has no _CLOTH_RENDER meshes.
    """
    cloth_meshes = [m for m in addon_state.asset.meshes if m.name.endswith('_CLOTH_RENDER')]
    if not cloth_meshes:
        return
    if mcloth is None:
        if operator:
            operator.report({'WARNING'},
                "mcloth.py is missing - cloth vertex mapping was NOT updated. "
                "Restart Blender to let the plugin download it.")
        return
    SWOMT = bpy.context.scene.SWOMT
    src_path = mcloth.source_path(SWOMT.AssetPath)
    if src_path is None:
        if operator:
            operator.report({'WARNING'},
                "Asset has cloth meshes but no paired .mcloth file was found - "
                "cloth vertex mapping was NOT updated.")
        return
    with open(src_path, 'rb') as f:
        data = f.read()
    try:
        _stream_end, blocks = mcloth.parse_blocks(data)
    except ValueError as e:
        if operator:
            operator.report({'WARNING'}, f"Could not parse '{os.path.basename(src_path)}': {e}")
        return

    remaps = {}
    gen_info = {} # block_name -> (obj, lod, li) for slot exports with appended verts
    for mesh in cloth_meshes:
        for li, lod in enumerate(mesh.lods):
            obj_name = lod.blender_obj_name or f"{mesh.name}_LOD{li}"
            obj = bpy.data.objects.get(obj_name)
            if obj is None or len(obj.data.vertices) == 0:
                continue
            block_name = mesh.name if li == 0 else f"{mesh.name}_LOD{li}"
            if block_name not in blocks:
                continue
            slot_vc = getattr(lod, 'exported_slot_identity', 0)
            if slot_vc:
                # Slot-preserving export: original indices stay at their
                # original slots and every source row remains eligible.
                remaps[block_name] = (slot_vc, block_name,
                                      {i: i for i in range(slot_vc)})
                # Register for row synthesis whenever appended slots exist:
                # generated LODs, added verts, seam splits, custom meshes.
                if slot_vc > getattr(lod, 'exported_append_base', slot_vc):
                    gen_info[block_name] = (obj, lod, li)
                continue
            # Renumbered fallback (non-slot exports).
            # Generated LODs carry LOD0-based mmb_vertex_order values, so their rows must be pulled
            # from LOD0's block (see GenerateLODs).
            src_li = obj.get('mmb_lod_source', li)
            src_block = mesh.name if src_li == 0 else f"{mesh.name}_LOD{src_li}"
            if src_block not in blocks:
                continue
            new_vc = len(obj.data.vertices)
            attr = obj.data.attributes.get('mmb_vertex_order')
            old_to_new = {}
            if attr is not None:
                for vi in range(new_vc):
                    ov = attr.data[vi].value
                    if ov not in old_to_new:
                        old_to_new[ov] = vi
            else:
                # No mapping attribute (fully replaced mesh): fall back to identity
                old_to_new = {i: i for i in range(new_vc)}
            remaps[block_name] = (new_vc, src_block, old_to_new)

    # Sim-section sync: BUDGET REUSE remains the preferred path. New sim verts
    # occupy reused (deleted) slots while the vanilla budget is sufficient,
    # producing a value-only rewrite. If the vertex budget is exceeded, the
    # confirmed growth path appends complete vertex/triangle topology and
    # rebuilds all count-dependent tables plus the SIMD constraint schedule.
    # Move and inert-delete stay on the passthrough.
    # Source-asset baseline: a driven RENDER vert's in-game position is
    # reconstructed from its ROW (relative to the sim triangle), NOT the mmb
    # rest position - so moving a render vert without re-encoding its row
    # snaps it back in-game. Used below to find moved render verts (their rows
    # are recomputed). Moved SIM verts are handled separately - the render
    # follows the cage there, so their rows stay vanilla.
    _src_bytes = _src_asset = None
    try:
        with open(SWOMT.AssetPath, 'rb') as _sf:
            _src_bytes = _sf.read()
        _src_asset = SkeletalMeshAsset()
        _src_asset.parse(io.BytesIO(_src_bytes))
    except Exception as _se:
        logger.warning("Moved-vertex baseline is unavailable: %s", _se)
        _src_asset = None

    sim_arg = None
    sim_reuse_arg = None
    sim_move_arg = None  # (new_sim_positions, moved_slot_set) for a sim move
    sim_fix = None # (valid_tri_set_or_None, reused_vert_set) for render-row fix
    try:
        with open(out_mmb_path, 'rb') as f:
            _nb = f.read()
        _na = SkeletalMeshAsset()
        _na.parse(io.BytesIO(_nb))
        _sims = [m for m in _na.meshes if m.name.endswith('_CLOTH_SIM')]
        if len(_sims) == 1:
            _sm = _sims[0]
            _sp = mcloth.mmb_lod_float_positions(_nb, _sm, 0)
            _st = mcloth.mmb_lod_u16_tris(_nb, _sm, 0)
            _mem = next((m for m in addon_state.asset.meshes
                         if m.name.endswith('_CLOTH_SIM') and m.lods), None)
            _reused = (getattr(_mem.lods[0], 'exported_sim_reused', set())
                       if _mem else set())
            _valid = (getattr(_mem.lods[0], 'exported_sim_valid_tris', None)
                      if _mem else None)
            # Grown counts (mismatched .mmb/.mcloth pair fallback): appended
            # vert slots join the exclusion set so unzoned vanilla rows never
            # silently re-attach onto appended sim (same policy as reuse
            # mode; riding new sim is a zone opt-in).
            _ovc0, _otc0 = mcloth.sim_counts(data)
            _appended_v = (set(range(_ovc0, len(_sp)))
                           if _sp is not None and _ovc0 is not None
                           and len(_sp) > _ovc0 else set())
            # A sim MOVE (positions changed, topology same) refreshes the
            # sim-section values (0x110b positions + rest lengths/tethers of
            # the moved verts). This keeps the mcloth sim rest state matching
            # the moved mmb geometry AND - because it changes the sim-section
            # bytes - forces the engine to re-cook the render->sim mapping, so
            # render-vertex edits in the same export take effect (otherwise
            # they are cached against the old fabric and ignored in-game).
            # The render rows themselves stay VANILLA: the render follows the
            # reshaped cage via its stored coefficients. The moved-slot set is
            # recorded during _write_mod_file (the source bytes are intact
            # there - the exported mmb may overwrite the source in a chained
            # _MOD export, so a post-hoc position compare would see nothing).
            _moved = (getattr(_mem.lods[0], 'exported_sim_moved', set())
                      if _mem else set())
            # Render rows riding deleted/rebuilt sim triangles stretch: when the
            # sim was exported this session, re-attach or release them below.
            if _st is not None and _valid is not None and (
                    _reused or _appended_v or len(_valid) < len(_st)):
                sim_fix = (set(_valid), set(_reused) | _appended_v)
            _counts_changed = (_sp is not None and _st is not None
                               and (_ovc0 is None or len(_sp) != _ovc0
                                    or len(_st) != _otc0))
            if _counts_changed:
                _tb = b''.join(pack('<HHH', *t) for t in _st)
                # Growth can include moved original slots as well as appended
                # ones.  Pass the current mesh's live triangle slots so the
                # fabric cooker excludes preserved phantom faces.
                sim_arg = (_sp, _tb,
                           set(_valid) if _valid is not None else None)
                logger.info(
                    "SIM topology-growth path: %s->%d vertices, "
                    "%s->%d triangles",
                    _ovc0, len(_sp), _otc0, len(_st))
                if operator:
                    operator.report({'INFO'},
                        f"_CLOTH_SIM topology grew to {len(_sp)} vertices and "
                        f"{len(_st)} triangles.")
            elif _sp is not None and _st is not None and _reused:
                _tb = b''.join(pack('<HHH', *t) for t in _st)
                sim_reuse_arg = (_sp, _tb, set(_reused))
                logger.debug(
                    "Sim budget reuse: %d slot(s) rewritten in place",
                    len(_reused))
                if operator:
                    operator.report({'INFO'},
                        f"New _CLOTH_SIM vertices reuse {len(_reused)} deleted "
                        f"slot(s); constraints rewritten in place.")
            elif _sp is not None and _st is not None:
                if _moved:
                    sim_move_arg = (_sp, _moved)
                    logger.debug(
                        "Sim move: refreshing rest state for %d moved vertex(s)",
                        len(_moved))
        elif len(_sims) > 1:
            logger.warning(
                "Multiple _CLOTH_SIM meshes found; sim rewrite is unsupported")
    except Exception as e:
        logger.exception("Sim-section pass was skipped: %s", e)
        sim_arg = None
        sim_reuse_arg = None
        sim_move_arg = None

    # ZONE_* vertex groups on any imported cloth object mean rows may need
    # re-assignment even without a sim/render edit.
    _zones_present = False
    for _zm in addon_state.asset.meshes:
        if not (_zm.name.endswith('_CLOTH_RENDER')
                or _zm.name.endswith('_CLOTH_SIM')):
            continue
        for _zl in _zm.lods:
            _zo = bpy.data.objects.get(_zl.blender_obj_name or "")
            if _zo is not None and any(g.name.startswith('ZONE_')
                                       for g in _zo.vertex_groups):
                _zones_present = True
                break
        if _zones_present:
            break

    if not remaps and sim_arg is None and sim_reuse_arg is None \
            and sim_move_arg is None and sim_fix is None \
            and not _zones_present:
        # Nothing to remap (render unedited), no sim edit and no zone
        # assignments. Emit a verbatim copy so the exported mmb still ships
        # with a matched .mcloth pair.
        with open(os.path.splitext(out_mmb_path)[0] + '.mcloth', 'wb') as f:
            f.write(data)
        return

    # Read sim geometry + new render geometry from the exported mmb: used to
    # rebind heights (moved vertices) and to build full rows for appended
    # vertices (generated LODs, added verts, seam splits, custom meshes).
    rebind = {}
    computed = {}
    mmb_color_patches = []
    try:
        with open(out_mmb_path, 'rb') as f:
            new_bytes = f.read()
        new_asset = SkeletalMeshAsset()
        new_asset.parse(io.BytesIO(new_bytes))
        for mesh in cloth_meshes:
            sim_name = mesh.name[:-len('_RENDER')] + '_SIM'
            new_render = next((m for m in new_asset.meshes if m.name == mesh.name), None)
            new_sim = next((m for m in new_asset.meshes if m.name == sim_name), None)
            if new_render is None or new_sim is None or not new_sim.lods:
                continue
            sim_verts = mcloth.mmb_lod_float_positions(new_bytes, new_sim, 0)
            sim_tris = mcloth.mmb_lod_u16_tris(new_bytes, new_sim, 0)
            if sim_verts is None or sim_tris is None:
                continue

            # ZONE ASSIGNMENTS: vertex groups named 'ZONE_*' painted on
            # BOTH the sim object and render objects pair regions - a
            # zoned render vert only binds to sim tris whose 3 verts all
            # carry a matching zone; a zoned vert with no candidate tris
            # releases to skinning. Unzoned verts keep default behavior.
            def _grid_of(points, cell):
                g = {}
                for i, p in enumerate(points):
                    g.setdefault((int(p[0] / cell), int(p[1] / cell),
                                  int(p[2] / cell)), []).append(i)
                def nearest(p, maxd):
                    c = (int(p[0] / cell), int(p[1] / cell),
                         int(p[2] / cell))
                    rng = int(maxd / cell) + 1
                    best, bd = None, maxd * maxd
                    for dx in range(-rng, rng + 1):
                        for dy in range(-rng, rng + 1):
                            for dz in range(-rng, rng + 1):
                                for i in g.get((c[0]+dx, c[1]+dy, c[2]+dz),
                                               ()):
                                    q = points[i]
                                    d = ((p[0]-q[0])**2 + (p[1]-q[1])**2
                                         + (p[2]-q[2])**2)
                                    if d < bd:
                                        bd, best = d, i
                    return best
                return nearest

            def _obj_zone_verts(o):
                zg = {g.index: g.name for g in o.vertex_groups
                      if g.name.startswith('ZONE_')}
                if not zg:
                    return None
                out = []
                for v in o.data.vertices:
                    zs = frozenset(zg[ge.group] for ge in v.groups
                                   if ge.group in zg and ge.weight > 0)
                    # FILE space: Blender x is negated on import/export
                    out.append(((-v.co[0], v.co[1], v.co[2]), zs))
                return out

            zone_tris = {}
            _zone_lookups = {} # li -> callable(pos)->frozenset
            _lod0_zone_fn = None
            _smesh_mem = next((m for m in addon_state.asset.meshes
                               if m.name == sim_name and m.lods), None)
            sim_obj = None
            if _smesh_mem:
                _sn = (_smesh_mem.lods[0].blender_obj_name
                       or f"{sim_name}_LOD0")
                sim_obj = bpy.data.objects.get(_sn)
            if sim_obj is not None:
                _szv = _obj_zone_verts(sim_obj)
                if _szv:
                    _sfind = _grid_of(sim_verts, 0.002)
                    _slot_zones = {}
                    for p, zs in _szv:
                        if not zs:
                            continue
                        s = _sfind(p, 0.002)
                        if s is not None:
                            _slot_zones[s] = _slot_zones.get(
                                s, frozenset()) | zs
                    for t, tri in enumerate(sim_tris):
                        zs = None
                        for v in tri:
                            vz = _slot_zones.get(v, frozenset())
                            zs = vz if zs is None else (zs & vz)
                        for z in (zs or ()):
                            zone_tris.setdefault(z, set()).add(t)
                    if zone_tris:
                        _zmsg = ", ".join(f"{z} ({len(ts)} sim tris)"
                                          for z, ts in zone_tris.items())
                        logger.debug("%s: cloth zones %s", sim_name, _zmsg)
                        if operator:
                            operator.report({'INFO'},
                                            f"Cloth zones active: {_zmsg}")
                    elif operator:
                        operator.report({'WARNING'},
                            "ZONE_ groups found on the sim mesh but no sim "
                            "triangle has ALL 3 vertices zoned - zones are "
                            "inactive. Paint whole triangles (including "
                            "anchor verts).")

            def _make_zone_fn(zone_pts):
                pts = [p for p, _zs in zone_pts]
                zss = [zs for _p, zs in zone_pts]
                rad = max(SWOMT.cloth_donor_radius, 1e-4)
                find = _grid_of(pts, rad)
                def fn(p):
                    i = find(p, rad)
                    return zss[i] if i is not None else frozenset()
                return fn

            if not zone_tris and operator:
                for _lz in range(len(mesh.lods)):
                    _ozn = (mesh.lods[_lz].blender_obj_name
                            or f"{mesh.name}_LOD{_lz}")
                    _oz = bpy.data.objects.get(_ozn)
                    if _oz is not None and any(
                            g.name.startswith('ZONE_')
                            for g in _oz.vertex_groups):
                        operator.report({'WARNING'},
                            "ZONE_ groups found on the render mesh but the "
                            "sim mesh has no matching zones (not imported, "
                            "or no ZONE_ groups) - zones are inactive.")
                        break

            if zone_tris:
                for li in range(len(new_render.lods)):
                    _on = (mesh.lods[li].blender_obj_name
                           or f"{mesh.name}_LOD{li}") \
                        if li < len(mesh.lods) else None
                    _ro = bpy.data.objects.get(_on) if _on else None
                    _zv = _obj_zone_verts(_ro) if _ro is not None else None
                    if _zv is not None:
                        _zone_lookups[li] = _make_zone_fn(_zv)
                        if li == 0:
                            _lod0_zone_fn = _zone_lookups[0]
                for li in range(len(new_render.lods)):
                    if li not in _zone_lookups and _lod0_zone_fn is not None:
                        _zone_lookups[li] = _lod0_zone_fn

            # Original render positions per LOD (moved-driven-vert baseline).
            _orig_rpos = {}
            if _src_asset is not None:
                try:
                    _orm = next((m for m in _src_asset.meshes
                                 if m.name == mesh.name), None)
                    if _orm is not None:
                        for _li3 in range(len(_orm.lods)):
                            _op3 = mcloth.mmb_lod_float_positions(
                                _src_bytes, _orm, _li3)
                            if _op3:
                                _orig_rpos[_li3] = _op3
                except Exception as _oe:
                    logger.warning("Moved-vertex baseline was skipped: %s", _oe)
                    _orig_rpos = {}

            _zone_moved_total = 0
            for li in range(len(new_render.lods)):
                block_name = mesh.name if li == 0 else f"{mesh.name}_LOD{li}"
                if block_name not in remaps:
                    if (sim_fix is None and _zone_lookups.get(li) is None) \
                            or block_name not in blocks:
                        continue
                    # Sim was edited (or zones re-assign rows) but this render
                    # LOD wasn't exported: give it an identity remap so its
                    # stale rows can be fixed too.
                    _bb0 = blocks[block_name]
                    _h0 = _bb0['chunks'][mcloth.T_HEADER][0]
                    _vc0 = unpack('<I', data[_h0 + 8:_h0 + 12])[0]
                    _i0, _s0 = _bb0['chunks'][mcloth.T_INDICES]
                    _n0 = (_s0 - 8) // 4
                    _six0 = unpack(f'<{_n0}I', data[_i0 + 8:_i0 + _s0])
                    remaps[block_name] = (_vc0, block_name,
                                          {ov: ov for ov in _six0})
                pos = mcloth.mmb_lod_float_positions(new_bytes, new_render, li)
                if pos is None:
                    continue
                rebind[block_name] = (sim_verts, sim_tris, pos)

                # Sim-edit row fix: rows whose sim triangle was deleted
                # (phantom) or rebuilt (touches a reused slot) would
                # STRETCH. Re-attach them to the nearest surviving tri
                # when close enough, otherwise release them to skinning
                # (the mapping shader treats absent rows as skinned).
                _fix_rows = {}
                _zone_of = _zone_lookups.get(li)

                def _zone_cand(p, default):
                    """Candidate sim tris for a render vert at p: the union
                    of its zones' tris when zoned, else `default` (None =
                    unrestricted). Caller intersects with its valid set."""
                    if _zone_of is None or not zone_tris:
                        return default
                    zs = _zone_of(p)
                    if not zs:
                        return default
                    c = set()
                    for z in zs:
                        c |= zone_tris.get(z, set())
                    return c

                _zone_gate = bool(zone_tris) and _zone_of is not None
                if sim_fix is not None or _zone_gate:
                    if sim_fix is not None:
                        _valid_t, _reused_v = sim_fix
                    else:
                        _valid_t = set(range(len(sim_tris)))
                        _reused_v = set()
                    # Unzoned vanilla rows only re-attach to SURVIVING
                    # original tris - never silently onto new/rebuilt sim
                    # (riding a new sim region is a zone opt-in).
                    _surv_t = ({t for t in _valid_t
                                if t < len(sim_tris) and all(
                                    v not in _reused_v for v in sim_tris[t])}
                               if _reused_v else _valid_t)
                    _bb = blocks[block_name]
                    _i3, _s3 = _bb['chunks'][mcloth.T_INDICES]
                    _nB = (_s3 - 8) // 4
                    _six = unpack(f'<{_nB}I', data[_i3 + 8:_i3 + _s3])
                    _t3 = _bb['chunks'][mcloth.T_TRI][0] + 8
                    _o2n = remaps[block_name][2]
                    _ntb = mcloth.mmb_lod_normals_tangents(new_bytes, new_render, li)
                    _rad = SWOMT.cloth_donor_radius
                    _re = _rel = _zmoved = 0
                    for _r, _ov in enumerate(_six):
                        _ti = unpack('<H', data[_t3 + _r*2:_t3 + _r*2 + 2])[0]
                        _nv = _o2n.get(_ov)
                        _zs = (_zone_of(pos[_nv]) if _zone_gate
                               and _nv is not None and _nv < len(pos)
                               else frozenset())
                        _struct_bad = (
                            _ti >= len(sim_tris) or _ti not in _valid_t
                            or any(v in _reused_v for v in sim_tris[_ti]))
                        # zones are AUTHORITATIVE assignments: a zoned vert
                        # riding an out-of-zone tri is re-attached into its
                        # zone (no radius cap - painting it means binding it)
                        _zone_bad = bool(_zs) and not any(
                            _ti in zone_tris.get(z, ()) for z in _zs)
                        if not (_struct_bad or _zone_bad):
                            continue
                        if _nv is None:
                            continue
                        del _o2n[_ov]
                        if _ntb is None or _nv >= len(pos):
                            _rel += 1
                            continue
                        if _zs:
                            _cand = set()
                            for z in _zs:
                                _cand |= zone_tris.get(z, set())
                            _cand &= _valid_t
                        else:
                            _cand = _surv_t
                        if not _cand:
                            _rel += 1
                            continue
                        _ti2, _dist = mcloth.nearest_tri_dist(
                            sim_verts, sim_tris, pos[_nv], valid=_cand)
                        if _ti2 is None or (not _zs and _dist > _rad):
                            _rel += 1
                            continue
                        _vals = mcloth.compute_row_values(
                            sim_verts, sim_tris, _ti2,
                            pos[_nv], _ntb[_nv][0], _ntb[_nv][1])
                        if _vals is None:
                            _rel += 1
                            continue
                        _fix_rows[_nv] = _vals
                        _re += 1
                        _zmoved += int(_zone_bad and not _struct_bad)
                    _zone_moved_total += _zmoved
                    if _re or _rel:
                        logger.debug(
                            "%s: sim-edit row fix re-attached %d (%d by zone) "
                            "and released %d to skinning",
                            block_name, _re, _zmoved, _rel)
                    if _fix_rows:
                        computed[block_name] = sorted(_fix_rows.items())

                # MOVED RENDER VERTS: a driven vert's in-game position is
                # reconstructed from its ROW (sim tri + bary foot +
                # height + normal/tangent), NOT from the mmb rest
                # position. The height is rebound on every export, but an
                # IN-PLANE move kept the vanilla bary foot and snapped
                # back in-game. Recompute the full row of every claimed
                # RENDER vert whose exported position moved. (Moved SIM
                # verts are deliberately NOT handled here - the render is
                # meant to follow the reshaped sim cage.)
                _opos = _orig_rpos.get(li)
                _ntb2 = mcloth.mmb_lod_normals_tangents(new_bytes,
                                                        new_render, li)
                if _opos is not None and _ntb2 is not None:
                    _bb2 = blocks[block_name]
                    _i4, _s4 = _bb2['chunks'][mcloth.T_INDICES]
                    _n4 = (_s4 - 8) // 4
                    _six2 = unpack(f'<{_n4}I', data[_i4 + 8:_i4 + _s4])
                    _t4 = _bb2['chunks'][mcloth.T_TRI][0] + 8
                    _o2n2 = remaps[block_name][2]
                    _done = dict(computed.get(block_name, []))
                    _valid2 = sim_fix[0] if sim_fix else None
                    _mvd = {}
                    for _r2, _ov2 in enumerate(_six2):
                        _nv2 = _o2n2.get(_ov2)
                        if (_nv2 is None or _nv2 in _done
                                or _ov2 >= len(_opos) or _nv2 >= len(pos)
                                or _nv2 >= len(_ntb2)):
                            continue
                        _q0 = _opos[_ov2]
                        _p0 = pos[_nv2]
                        if ((_p0[0]-_q0[0])**2 + (_p0[1]-_q0[1])**2
                                + (_p0[2]-_q0[2])**2) <= 2.5e-9:  # 0.05mm
                            continue
                        _ti4 = unpack('<H', data[_t4+_r2*2:_t4+_r2*2+2])[0]
                        if _ti4 >= len(sim_tris):
                            continue
                        _vals2 = mcloth.compute_row_values(
                            sim_verts, sim_tris, _ti4,
                            _p0, _ntb2[_nv2][0], _ntb2[_nv2][1])
                        # moved far off its tri: re-attach (zone-aware)
                        if _vals2 is not None \
                                and not mcloth.bary_within(_vals2):
                            _cand2 = _zone_cand(_p0, _valid2)
                            _ti5 = mcloth.nearest_tri(
                                sim_verts, sim_tris, _p0, valid=_cand2)
                            if _ti5 is not None and _ti5 != _ti4:
                                _v5 = mcloth.compute_row_values(
                                    sim_verts, sim_tris, _ti5,
                                    _p0, _ntb2[_nv2][0], _ntb2[_nv2][1])
                                if _v5 is not None:
                                    _vals2 = _v5
                        if _vals2 is not None:
                            # replace the vanilla row: drop it from the remap
                            # so the computed row wins (same as the fix pass)
                            del _o2n2[_ov2]
                            _mvd[_nv2] = _vals2
                    if _mvd:
                        _done.update(_mvd)
                        computed[block_name] = sorted(_done.items())
                        logger.debug(
                            "%s: recomputed rows for %d moved render vertex(s)",
                            block_name, len(_mvd))

                # Computed rows for appended slots. Three sources of binding:
                #   1. a driven mmb_vertex_order source vertex (edited/generated
                #      meshes) - inherit its sim triangle;
                #   2. no usable source (custom from-scratch meshes, attribute
                #      interpolation noise) - inherit driven status, triangle
                #      AND mask/blend colors from the NEAREST ORIGINAL vertex
                #      (the orphan slots 0..append_base carry the originals);
                #   3. vanilla rules: overlapping twins of driven verts and
                #      mask-flagged verts get rows too.
                # Row values are always computed from the vertex's own exported
                # geometry; colors are patched into the mmb where computed.
                gi = gen_info.get(block_name)
                if gi is None:
                    continue
                obj, lod_mem, li_ = gi
                append_sources = getattr(lod_mem, 'exported_append_sources', {})
                append_base = getattr(lod_mem, 'exported_append_base', 0)
                slot_total = getattr(lod_mem, 'exported_slot_identity', 0)
                src_li = obj.get('mmb_lod_source', li_)
                src_block = mesh.name if src_li == 0 else f"{mesh.name}_LOD{src_li}"
                nt = mcloth.mmb_lod_normals_tangents(new_bytes, new_render, li_)
                if nt is None or slot_total <= append_base:
                    continue

                def _tri_map(bname):
                    bb = blocks[bname]
                    i3, s3 = bb['chunks'][mcloth.T_INDICES]
                    nB = (s3 - 8) // 4
                    six = unpack(f'<{nB}I', data[i3 + 8:i3 + s3])
                    t3 = bb['chunks'][mcloth.T_TRI][0] + 8
                    return {ov: unpack('<H', data[t3 + r * 2:t3 + r * 2 + 2])[0]
                            for r, ov in enumerate(six)}

                src_tri = _tri_map(src_block) if src_block in blocks else {}
                own_tri = (src_tri if src_block == block_name
                           else _tri_map(block_name))
                colors = mcloth.mmb_lod_color_bytes(new_bytes, new_render, li_)
                attr_present = obj.data.attributes.get('mmb_vertex_order') is not None
                # Donor color patching is the fallback for meshes without
                # authored cloth colors. With 'Export Vertex Colors' checked the
                # user's own colors are already in the file - leave them alone,
                # and let their mask also decide driven status: verts painted
                # black get no row at all (vanilla non-driven verts have no
                # rows - the engine honors driven status, not just blend 0).
                # Vanilla rule: non-driven flag is EXACTLY 0; driven runs
                # 18..255 - so only flag 0 opts out.
                donor_colors = None if SWOMT.export_vertex_colors else colors
                user_mask = SWOMT.export_vertex_colors and colors is not None

                # Spatial grid over the original (orphan) vertices for
                # nearest-original lookups.
                _MAXD = SWOMT.cloth_donor_radius # inherit nothing beyond this
                _CELL = max(0.02, _MAXD / 2.0)   # keep the probe loop bounded
                grid = {}
                for v in range(min(append_base, len(pos))):
                    p = pos[v]
                    grid.setdefault((int(p[0] / _CELL), int(p[1] / _CELL),
                                     int(p[2] / _CELL)), []).append(v)

                def _nearest_original(p):
                    cx, cy, cz = int(p[0] / _CELL), int(p[1] / _CELL), int(p[2] / _CELL)
                    rng = int(_MAXD / _CELL) + 1
                    best, bd = None, _MAXD * _MAXD
                    for dx in range(-rng, rng + 1):
                        for dy in range(-rng, rng + 1):
                            for dz in range(-rng, rng + 1):
                                for v in grid.get((cx + dx, cy + dy, cz + dz), ()):
                                    q = pos[v]
                                    d = ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
                                         + (p[2] - q[2]) ** 2)
                                    if d < bd:
                                        bd, best = d, v
                    return best

                _valid_tris = sim_fix[0] if sim_fix else None
                _reused_vv = sim_fix[1] if sim_fix else ()

                def _syn_cand(slot):
                    """Candidate tris for a synthesized row: zone tris when
                    the vert is zoned (new geometry may ride new sim), else
                    the full valid set (None = all)."""
                    c = _zone_cand(pos[slot], None)
                    if c is None:
                        return _valid_tris
                    return (c & _valid_tris) if _valid_tris is not None else c

                def _computed_row(slot, ti):
                    cand = _syn_cand(slot)
                    # A source tri that was deleted, rebuilt, or outside the
                    # vert's zone can't be inherited: swap to the nearest
                    # candidate tri up front.
                    _bad_ti = (ti >= len(sim_tris)
                               or (cand is not None and ti not in cand)
                               or (sim_fix and (ti not in _valid_tris
                                   or any(v in _reused_vv
                                          for v in sim_tris[ti]))))
                    if _bad_ti:
                        if cand is not None and not cand:
                            return None, False
                        ti = mcloth.nearest_tri(sim_verts, sim_tris, pos[slot],
                                                valid=cand)
                        if ti is None:
                            return None, False
                    vals = mcloth.compute_row_values(
                        sim_verts, sim_tris, ti,
                        pos[slot], nt[slot][0], nt[slot][1])
                    # Re-attach to the truly nearest triangle when the foot
                    # lands far outside the given one - keeps the quantization
                    # ranges tight and the binding local.
                    if vals is not None and not mcloth.bary_within(vals):
                        ti2 = mcloth.nearest_tri(sim_verts, sim_tris, pos[slot],
                                                 valid=cand)
                        if ti2 is not None and ti2 != ti:
                            v2 = mcloth.compute_row_values(
                                sim_verts, sim_tris, ti2,
                                pos[slot], nt[slot][0], nt[slot][1])
                            if v2 is not None:
                                return v2, True
                    return vals, False

                rows = {}
                pos_twin = {} # position -> a driven appended slot there
                color_patches = [] # (slot, donor vertex index)
                retargeted = twins = flagged = transferred = 0

                # Pass 1: driven source vertex, else nearest-original transfer
                for slot in range(append_base, min(slot_total, len(pos))):
                    if user_mask and colors[slot][0] == 0:
                        continue
                    ov = append_sources.get(slot)
                    if ov is not None and ov in src_tri:
                        vals, retg = _computed_row(slot, src_tri[ov])
                        if vals is not None:
                            rows[slot] = vals
                            retargeted += int(retg)
                            pos_twin.setdefault(pos[slot], slot)
                        continue
                    o = _nearest_original(pos[slot])
                    if o is None:
                        continue
                    if o in own_tri:
                        vals, _ = _computed_row(slot, own_tri[o])
                        if vals is not None:
                            rows[slot] = vals
                            transferred += 1
                            pos_twin.setdefault(pos[slot], slot)
                            if donor_colors is not None:
                                color_patches.append((slot, o)) # driven needs the donor's mask/blend colors
                    elif not attr_present and donor_colors is not None:
                        # non-driven region: still give custom meshes proper (non-driven) mask colors
                        color_patches.append((slot, o))

                # Pass 2: vanilla rules for the remaining slots -
                # overlapping twins of driven verts get rows + the twin's colors
                # (blend weight); mask-flagged verts get nearest-tri rows.
                for slot in range(append_base, min(slot_total, len(pos))):
                    if slot in rows or (user_mask and colors[slot][0] == 0):
                        continue
                    twin = pos_twin.get(pos[slot])
                    if twin is not None:
                        vals, _ = _computed_row(slot, rows[twin]['tri'])
                        if vals is not None:
                            rows[slot] = vals
                            twins += 1
                            if donor_colors is not None:
                                color_patches.append((slot, twin))
                        continue
                    # File colors are reliable when they came from the
                    # original (attr present) or from the user's own layers
                    # (Export Vertex Colors) - never for attr-less appends.
                    # User colors follow the vanilla >0 rule; preserved colors
                    # keep the conservative 128 cut (interp noise).
                    if (colors is not None
                            and ((user_mask and colors[slot][0] > 0)
                                 or (attr_present and colors[slot][0] >= 128))):
                        ti = mcloth.nearest_tri(sim_verts, sim_tris, pos[slot],
                                                valid=_syn_cand(slot))
                        if ti is not None:
                            vals, _ = _computed_row(slot, ti)
                            if vals is not None:
                                rows[slot] = vals
                                flagged += 1

                if rows:
                    _merged = dict(computed.get(block_name, []))
                    _merged.update(rows)
                    computed[block_name] = sorted(_merged.items())
                    logger.debug(
                        "%s: computed rows for %d/%d appended vertices "
                        "(%d re-attached, %d nearest-transferred, %d twins, "
                        "%d mask-flagged)",
                        block_name, len(rows), slot_total - append_base,
                        retargeted, transferred, twins, flagged)
                if color_patches:
                    lodn = new_render.lods[li_]
                    hi = sum(new_render.lods[k].data_size
                             for k in range(li_ + 1, len(new_render.lods)))
                    vob_abs = lodn.data_offset + (lodn.vertex_data_offset_b - hi)
                    ns_ = new_render.normals_stride
                    for slot, donor in color_patches:
                        mmb_color_patches.append(
                            (vob_abs + slot * ns_, bytes(colors[donor])))
            if _zone_moved_total and operator:
                operator.report({'INFO'},
                    f"Cloth zones re-assigned {_zone_moved_total} render "
                    f"row(s) across LODs.")
    except Exception as e:
        logger.exception("mcloth geometry pass was skipped: %s", e)
        rebind = {}
        computed = {}
        mmb_color_patches = []

    # Computed verts need their donor's cloth mask/blend colors in the
    # exported mmb as well - a row alone blends to fully skinned at blend 0.
    if mmb_color_patches:
        try:
            with open(out_mmb_path, 'r+b') as f:
                for _off, _byts in mmb_color_patches:
                    f.seek(_off)
                    f.write(_byts)
            logger.debug(
                "Patched cloth mask colors for %d appended vertex(s)",
                len(mmb_color_patches))
        except OSError as e:
            logger.warning("Cloth color patch failed: %s", e)

    out_bytes, stats = mcloth.rewrite(data, remaps, rebind=rebind,
                                      computed=computed, sim=sim_arg,
                                      sim_free=True,
                                      sim_reuse=sim_reuse_arg,
                                      sim_move=sim_move_arg)
    out_path = os.path.splitext(out_mmb_path)[0] + '.mcloth'
    with open(out_path, 'wb') as f:
        f.write(out_bytes)
    if operator:
        parts = ', '.join(f"{name.rsplit('_CLOTH_RENDER', 1)[-1] or 'LOD0'} {ob}->{nb}"
                          for name, (ob, nb) in stats.items())
        detail = f"driven verts: {parts}" if parts else "sim section updated"
        operator.report({'INFO'},
            f"Cloth mapping updated -> {os.path.basename(out_path)} ({detail})")

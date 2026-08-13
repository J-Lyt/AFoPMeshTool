"""Blender orchestration for paired .mcloth exports."""

import importlib.util
import io
import operator
import os
from struct import pack, unpack

import bpy

from .. import addon_state
from ..log import logger
from ..formats.mmb import SkeletalMeshAsset

try:
    from ..formats import mcloth
except ImportError:
    try:
        _path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'formats',
            'mcloth.py',
        )
        _spec = importlib.util.spec_from_file_location('afop_mcloth', _path)
        mcloth = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(mcloth)
    except Exception as error:
        mcloth = None
        logger.warning("mcloth.py is unavailable; cloth export is disabled: %s", error)


def _source_mcloth_path(settings):
    """Selected source mcloth, falling back to the MMB's paired sidecar."""
    selected = settings.MClothPath.strip()
    if selected:
        return bpy.path.abspath(selected)
    return mcloth.source_path(bpy.path.abspath(settings.AssetPath)) if mcloth else None


def _sim_free_slot_flags(orig_vc, sim_name=None):
    """[bool]*orig_vc: True where the source .mcloth marks the sim vert FREE
    (simulating). Used to hand reused slots to new verts free-first. None when
    the mcloth is unavailable."""
    try:
        src = _source_mcloth_path(bpy.context.scene.SWOMT)
        if not src or not os.path.isfile(src):
            return None
        with open(src, 'rb') as f:
            d = f.read()
        return mcloth.sim_free_flags(d, orig_vc, sim_name)
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
    force_recook = bool(SWOMT.force_cloth_recook)
    src_path = _source_mcloth_path(SWOMT)
    if not src_path or not os.path.isfile(src_path):
        if operator:
            if SWOMT.MClothPath:
                detail = f"Selected .mcloth file was not found: {src_path}"
            else:
                detail = "Asset has cloth meshes but no paired .mcloth file was found"
            operator.report(
                {'WARNING'}, f"{detail} - cloth vertex mapping was NOT updated.")
        return
    with open(src_path, 'rb') as f:
        data = f.read()
    try:
        streams, _footer_offset = mcloth.parse_streams(data)
        _stream_end, blocks = mcloth.parse_blocks(data)
    except ValueError as e:
        if operator:
            operator.report({'WARNING'}, f"Could not parse '{os.path.basename(src_path)}': {e}")
        return

    remaps = {}
    gen_info = {} # block_name -> (obj, lod, li) for slot exports with appended verts
    dense_unsupported = []
    dense_unsupported_blocks = set()
    for mesh in cloth_meshes:
        for li, lod in enumerate(mesh.lods):
            obj_name = lod.blender_obj_name or f"{mesh.name}_LOD{li}"
            obj = bpy.data.objects.get(obj_name)
            if obj is None or len(obj.data.vertices) == 0:
                continue
            block_name = mesh.name if li == 0 else f"{mesh.name}_LOD{li}"
            if block_name not in blocks:
                continue
            block = blocks[block_name]
            original_vc = mcloth.block_vertex_count(data, block)
            slot_vc = getattr(lod, 'exported_slot_identity', 0)
            if slot_vc:
                append_base = getattr(lod, 'exported_append_base', slot_vc)
                if block.get('dense') and (
                        slot_vc < original_vc
                        or (slot_vc > original_vc
                            and append_base != original_vc)):
                    dense_unsupported_blocks.add(block_name)
                    dense_unsupported.append(
                        f"{block_name}: implicit slots cannot be preserved "
                        f"({original_vc}->{slot_vc}, append base {append_base})")
                    continue
                # Slot-preserving export: original indices stay at their
                # original slots and every source row remains eligible.
                remaps[block_name] = (slot_vc, block_name,
                                      {i: i for i in range(slot_vc)})
                # Register for row synthesis whenever appended slots exist:
                # generated LODs, added verts, seam splits, custom meshes.
                if slot_vc > append_base:
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
            if block.get('dense') and (
                    new_vc != original_vc
                    or old_to_new != {i: i for i in range(original_vc)}):
                dense_unsupported_blocks.add(block_name)
                dense_unsupported.append(
                    f"{block_name}: dense implicit vertex IDs changed")
                continue
            remaps[block_name] = (new_vc, src_block, old_to_new)

    if dense_unsupported:
        detail = "; ".join(dense_unsupported)
        logger.warning(
            "Dense cloth topology rewrite skipped (%s). Mapping rows were "
            "preserved and may be stale.", detail)
        if operator:
            operator.report(
                {'WARNING'},
                "Dense cloth destructive renumbering/shrink is unsupported; "
                "affected mapping rows were preserved. See the log for details.")

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

    sim_args = {}
    sim_reuse_args = {}
    sim_move_args = {}  # sim_name -> (new_sim_positions, moved_slot_set)
    sim_moved_slots = {}  # sim_name -> slots whose positions were moved
    sim_fixes = {}  # sim_name -> (valid_tri_set_or_None, invalid_vert_set)
    try:
        with open(out_mmb_path, 'rb') as f:
            _nb = f.read()
        _na = SkeletalMeshAsset()
        _na.parse(io.BytesIO(_nb))
        _sims = [m for m in _na.meshes if m.name.endswith('_CLOTH_SIM')]
        for _sm in _sims:
            _sp = mcloth.mmb_lod_float_positions(_nb, _sm, 0)
            _st = mcloth.mmb_lod_u16_tris(_nb, _sm, 0)
            _mem = next((m for m in addon_state.asset.meshes
                         if m.name == _sm.name and m.lods), None)
            _reused = (getattr(_mem.lods[0], 'exported_sim_reused', set())
                       if _mem else set())
            _valid = (getattr(_mem.lods[0], 'exported_sim_valid_tris', None)
                      if _mem else None)
            _rebuilt = (getattr(_mem.lods[0], 'exported_sim_rebuilt', set())
                        if _mem else set())
            # Grown counts (mismatched .mmb/.mcloth pair fallback): appended
            # vert slots join the exclusion set so unzoned vanilla rows never
            # silently re-attach onto appended sim (same policy as reuse
            # mode; riding new sim is a zone opt-in).
            _ovc0, _otc0 = mcloth.sim_counts(data, _sm.name)
            if _ovc0 is None:
                logger.warning(
                    "%s has no uniquely matched mcloth stream; sim rewrite skipped",
                    _sm.name)
                if operator:
                    operator.report(
                        {'WARNING'},
                        f"{_sm.name} has no uniquely matched mcloth stream; "
                        "its SIM section was not rewritten.")
                continue
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
            sim_moved_slots[_sm.name] = set(_moved)
            # Render rows riding deleted/rebuilt sim triangles stretch: when the
            # sim was exported this session, re-attach or release them below.
            if _st is not None and _valid is not None and (
                    _reused or _appended_v or _rebuilt
                    or len(_valid) < len(_st)):
                sim_fixes[_sm.name] = (
                    set(_valid), set(_reused) | _appended_v)
            _counts_changed = (_sp is not None and _st is not None
                               and (_ovc0 is None or len(_sp) != _ovc0
                                    or len(_st) != _otc0))
            if _counts_changed:
                _tb = b''.join(pack('<HHH', *t) for t in _st)
                # Growth can include moved original slots as well as appended
                # ones.  Pass the current mesh's live triangle slots so the
                # fabric cooker excludes preserved phantom faces.
                sim_args[_sm.name] = (
                    _sp, _tb, set(_valid) if _valid is not None else None,
                    set(_rebuilt))
                logger.info(
                    "SIM topology-growth path: %s->%d vertices, "
                    "%s->%d triangles",
                    _ovc0, len(_sp), _otc0, len(_st))
                if operator:
                    operator.report({'INFO'},
                        f"{_sm.name} topology grew to {len(_sp)} vertices and "
                        f"{len(_st)} triangles.")
            elif _sp is not None and _st is not None and _reused:
                _tb = b''.join(pack('<HHH', *t) for t in _st)
                sim_reuse_args[_sm.name] = (_sp, _tb, set(_reused))
                logger.debug(
                    "Sim budget reuse: %d slot(s) rewritten in place",
                    len(_reused))
                if operator:
                    operator.report({'INFO'},
                        f"New {_sm.name} vertices reuse {len(_reused)} deleted "
                        f"slot(s); constraints rewritten in place.")
            elif _sp is not None and _st is not None:
                if _moved:
                    sim_move_args[_sm.name] = (_sp, _moved)
                    logger.debug(
                        "Sim move: refreshing rest state for %d moved vertex(s)",
                        len(_moved))
    except Exception as e:
        logger.exception("Sim-section pass was skipped: %s", e)
        sim_args = {}
        sim_reuse_args = {}
        sim_move_args = {}
        sim_moved_slots = {}
        sim_fixes = {}

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

    if not remaps and not sim_args and not sim_reuse_args \
            and not sim_move_args and not sim_fixes \
            and not _zones_present and not force_recook:
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
    promoted_dense_attachments = set()
    mmb_color_patches = []
    try:
        with open(out_mmb_path, 'rb') as f:
            new_bytes = f.read()
        new_asset = SkeletalMeshAsset()
        new_asset.parse(io.BytesIO(new_bytes))
        for mesh in cloth_meshes:
            sim_name = mesh.name[:-len('_RENDER')] + '_SIM'
            sim_fix = sim_fixes.get(sim_name)
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
                if block_name in dense_unsupported_blocks:
                    continue
                if block_name not in remaps:
                    if (sim_fix is None and _zone_lookups.get(li) is None) \
                            or block_name not in blocks:
                        continue
                    # Sim was edited (or zones re-assign rows) but this render
                    # LOD wasn't exported: give it an identity remap so its
                    # stale rows can be fixed too.
                    _bb0 = blocks[block_name]
                    _vc0 = mcloth.block_vertex_count(data, _bb0)
                    _six0 = mcloth.block_vertex_indices(data, _bb0)
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
                    _dense_rows = bool(_bb.get('dense'))
                    _six = mcloth.block_vertex_indices(data, _bb)
                    _t3 = _bb['chunks'][mcloth.T_TRI][0] + 8
                    _o2n = remaps[block_name][2]
                    _ntb = mcloth.mmb_lod_normals_tangents(new_bytes, new_render, li)
                    _rad = SWOMT.cloth_donor_radius
                    _re = _rel = _dense_kept = _zmoved = 0
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
                        if _ntb is None or _nv >= len(pos):
                            if _dense_rows:
                                _dense_kept += 1
                            else:
                                _o2n.pop(_ov, None)
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
                            if _dense_rows:
                                _dense_kept += 1
                            else:
                                _o2n.pop(_ov, None)
                                _rel += 1
                            continue
                        _ti2, _dist = mcloth.nearest_tri_dist(
                            sim_verts, sim_tris, pos[_nv], valid=_cand)
                        if _ti2 is None or (not _zs and _dist > _rad):
                            if _dense_rows:
                                _dense_kept += 1
                            else:
                                _o2n.pop(_ov, None)
                                _rel += 1
                            continue
                        _vals = mcloth.compute_row_values(
                            sim_verts, sim_tris, _ti2,
                            pos[_nv], _ntb[_nv][0], _ntb[_nv][1])
                        if _vals is None:
                            if _dense_rows:
                                _dense_kept += 1
                            else:
                                _o2n.pop(_ov, None)
                                _rel += 1
                            continue
                        _o2n.pop(_ov, None)
                        _fix_rows[_nv] = _vals
                        _re += 1
                        _zmoved += int(_zone_bad and not _struct_bad)
                    _zone_moved_total += _zmoved
                    if _re or _rel or _dense_kept:
                        logger.debug(
                            "%s: sim-edit row fix re-attached %d (%d by zone) "
                            "and released %d to skinning; preserved %d dense "
                            "row(s) with no valid replacement",
                            block_name, _re, _zmoved, _rel, _dense_kept)
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
                    _six2 = mcloth.block_vertex_indices(data, _bb2)
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
                    six = mcloth.block_vertex_indices(data, bb)
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

                def _nearest_original_any(p):
                    """Nearest preserved slot without the donor-radius cap.

                    Dense mappings cannot omit an appended implicit row, so
                    they also need a deterministic donor outside the normal
                    custom-mesh transfer radius.
                    """
                    best = None
                    bd = None
                    for v in range(min(append_base, len(pos))):
                        q = pos[v]
                        d = ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
                             + (p[2] - q[2]) ** 2)
                        if bd is None or d < bd:
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

                # Dense blocks have no serialized vertex-index table: row N
                # always belongs to vertex N. Growth therefore requires a row
                # for every appended slot, including vertices outside the
                # ordinary donor radius or painted with a zero cloth mask.
                # Preserve authored colors, but use an unrestricted nearest
                # original as the color donor for exporter-generated colors.
                dense_filled = 0
                if blocks[block_name].get('dense'):
                    for slot in range(append_base,
                                      min(slot_total, len(pos))):
                        if slot in rows:
                            continue
                        cand = _syn_cand(slot)
                        zone_fallback = cand is not None and not cand
                        if zone_fallback:
                            # A dense row cannot be released. If a painted zone
                            # has no live triangle, retain a structurally valid
                            # attachment instead of emitting an incomplete
                            # implicit table.
                            cand = _valid_tris
                        ti = mcloth.nearest_tri(
                            sim_verts, sim_tris, pos[slot], valid=cand)
                        if ti is None:
                            raise ValueError(
                                f"{block_name}: no valid SIM triangle for "
                                f"dense appended slot {slot}")
                        if zone_fallback:
                            vals = mcloth.compute_row_values(
                                sim_verts, sim_tris, ti, pos[slot],
                                nt[slot][0], nt[slot][1])
                        else:
                            vals, _ = _computed_row(slot, ti)
                        if vals is None:
                            raise ValueError(
                                f"{block_name}: could not compute dense row "
                                f"for appended slot {slot}")
                        rows[slot] = vals
                        dense_filled += 1
                    if donor_colors is not None:
                        patched_slots = {slot for slot, _donor
                                         in color_patches}
                        for slot in range(
                                append_base, min(slot_total, len(pos))):
                            if slot in patched_slots:
                                continue
                            donor = append_sources.get(slot)
                            if donor is None or not (0 <= donor < append_base):
                                donor = _nearest_original_any(pos[slot])
                            if donor is not None:
                                color_patches.append((slot, donor))

                if rows:
                    _merged = dict(computed.get(block_name, []))
                    _merged.update(rows)
                    computed[block_name] = sorted(_merged.items())
                    logger.debug(
                        "%s: computed rows for %d/%d appended vertices "
                        "(%d re-attached, %d nearest-transferred, %d twins, "
                        "%d mask-flagged, %d dense-required)",
                        block_name, len(rows), slot_total - append_base,
                        retargeted, transferred, twins, flagged, dense_filled)
                    if blocks[block_name].get('dense'):
                        logger.info(
                            "%s: synthesized all %d required dense appended "
                            "mapping row(s)", block_name,
                            slot_total - append_base)
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

        # CROSS-STREAM SIM ATTACHMENTS: a mapping block named for a different
        # _CLOTH_SIM mesh makes that target SIM ride the current stream's
        # driver fabric. Dense blocks use implicit identity row IDs, so target
        # growth appends complete rows. Reused/moved target slots and rows
        # riding rebuilt driver triangles are recomputed against live geometry.
        # Sparse attachment topology edits remain gated because new row
        # membership cannot be inferred safely.
        _attachment_warnings = []
        _new_meshes = {m.name: m for m in new_asset.meshes}
        for _aname, _ablock in blocks.items():
            if not _aname.endswith('_CLOTH_SIM') \
                    or _aname not in _new_meshes:
                continue
            _asi = _ablock.get('stream_index')
            if _asi is None or _asi >= len(streams):
                continue
            _driver_name = mcloth.stream_sim_name(streams[_asi])
            if not _driver_name or _driver_name == _aname:
                continue

            _dense_attachment = bool(_ablock.get('dense'))
            _topology_edit = (
                _aname in sim_args or _aname in sim_reuse_args
                or _driver_name in sim_args
                or _driver_name in sim_reuse_args)
            if _topology_edit and not _dense_attachment:
                _attachment_warnings.append(
                    f"{_aname} attached to {_driver_name}: sparse "
                    "topology/reuse edit")
                continue

            _target_moved = set(sim_moved_slots.get(_aname, ()))
            _target_reused = set(
                sim_reuse_args.get(_aname, (None, None, set()))[2])
            _driver_fix = sim_fixes.get(_driver_name)
            if not (_target_moved or _target_reused
                    or _aname in sim_args or _driver_fix is not None):
                continue

            _target_mesh = _new_meshes[_aname]
            _driver_mesh = _new_meshes.get(_driver_name)
            if _driver_mesh is None or not _driver_mesh.lods:
                _attachment_warnings.append(
                    f"{_aname}: driver mesh {_driver_name} is unavailable")
                continue
            _target_pos = mcloth.mmb_lod_float_positions(
                new_bytes, _target_mesh, 0)
            _driver_pos = mcloth.mmb_lod_float_positions(
                new_bytes, _driver_mesh, 0)
            _driver_tris = mcloth.mmb_lod_u16_tris(
                new_bytes, _driver_mesh, 0)
            _target_nt = mcloth.mmb_lod_normals_tangents(
                new_bytes, _target_mesh, 0)
            _avc = mcloth.block_vertex_count(data, _ablock)
            if (_target_pos is None or _driver_pos is None
                    or _driver_tris is None or _target_nt is None
                    or len(_target_pos) < _avc
                    or (not _dense_attachment
                        and len(_target_pos) != _avc)):
                _attachment_warnings.append(
                    f"{_aname}: attachment geometry/count mismatch")
                continue

            _aids = mcloth.block_vertex_indices(data, _ablock)
            _a_o2n = {v: v for v in _aids}
            _atri_off = _ablock['chunks'][mcloth.T_TRI][0] + 8
            _arows = {}
            _appended_target = set(range(_avc, len(_target_pos)))
            _target_refresh = (
                _target_moved | _target_reused | _appended_target)
            if _driver_fix is not None:
                _driver_valid, _driver_invalid_vertices = _driver_fix
                _driver_valid = set(_driver_valid)
                _driver_invalid_vertices = set(_driver_invalid_vertices)
            else:
                _driver_valid = set(range(len(_driver_tris)))
                _driver_invalid_vertices = set()

            def _attachment_values(_av, _old_ti=None):
                if _av >= len(_target_pos) or _av >= len(_target_nt):
                    return None
                _ap = _target_pos[_av]
                _an, _at = _target_nt[_av]
                _ati = _old_ti
                if (_ati is None or _ati >= len(_driver_tris)
                        or _ati not in _driver_valid):
                    _ati = mcloth.nearest_tri(
                        _driver_pos, _driver_tris, _ap,
                        valid=_driver_valid)
                if _ati is None:
                    return None
                _avals = mcloth.compute_row_values(
                    _driver_pos, _driver_tris, _ati, _ap, _an, _at)
                if _avals is not None and not mcloth.bary_within(_avals):
                    _ati2 = mcloth.nearest_tri(
                        _driver_pos, _driver_tris, _ap,
                        valid=_driver_valid)
                    if _ati2 is not None and _ati2 != _ati:
                        _avals2 = mcloth.compute_row_values(
                            _driver_pos, _driver_tris, _ati2,
                            _ap, _an, _at)
                        if _avals2 is not None:
                            _avals = _avals2
                return _avals

            for _arow, _av in enumerate(_aids):
                _ati = unpack(
                    '<H', data[_atri_off + _arow * 2:
                               _atri_off + _arow * 2 + 2])[0]
                _driver_bad = (
                    _ati >= len(_driver_tris)
                    or _ati not in _driver_valid
                    or (_ati < len(_driver_tris)
                        and any(v in _driver_invalid_vertices
                                for v in _driver_tris[_ati])))
                if _av not in _target_refresh and not _driver_bad:
                    continue
                _avals = _attachment_values(
                    _av, None if _driver_bad else _ati)
                if _avals is not None:
                    _a_o2n.pop(_av, None)
                    _arows[_av] = _avals

            # Dense growth adds one implicit row per appended target slot.
            for _av in sorted(_appended_target):
                _avals = _attachment_values(_av)
                if _avals is None:
                    _attachment_warnings.append(
                        f"{_aname}: no driver triangle for appended slot {_av}")
                    break
                _arows[_av] = _avals
            else:
                if not _arows:
                    continue
                _new_avc = len(_target_pos)
                remaps[_aname] = (_new_avc, _aname, _a_o2n)
                rebind[_aname] = (
                    _driver_pos, _driver_tris, _target_pos)
                computed[_aname] = sorted(_arows.items())
                if _dense_attachment and _appended_target:
                    promoted_dense_attachments.add(_aname)
                logger.info(
                    "%s: recomputed %d attachment row(s) against %s "
                    "(%d appended, %d reused/moved)",
                    _aname, len(_arows), _driver_name,
                    len(_appended_target),
                    len((_target_moved | _target_reused) & set(_aids)))
                if operator:
                    operator.report(
                        {'INFO'},
                        f"{_aname}: recomputed {len(_arows)} attachment "
                        f"row(s) against {_driver_name}.")

        if _attachment_warnings:
            logger.warning(
                "Cloth attachment rewrite skipped for unsupported edits: %s",
                "; ".join(_attachment_warnings))
            if operator:
                operator.report(
                    {'WARNING'},
                    "A SIM attachment has an unsupported or invalid edit; "
                    "its mapping was preserved and may be stale.")
    except Exception as e:
        logger.exception("mcloth geometry pass was skipped: %s", e)
        rebind = {}
        computed = {}
        promoted_dense_attachments = set()
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
                                      computed=computed, sim=sim_args,
                                      sim_free=True,
                                      sim_reuse=sim_reuse_args,
                                      sim_move=sim_move_args,
                                      force_recook=force_recook,
                                      promote_dense=promoted_dense_attachments)
    out_path = os.path.splitext(out_mmb_path)[0] + '.mcloth'
    with open(out_path, 'wb') as f:
        f.write(out_bytes)
    grown_quad_tables = []
    for sim_name in sim_args:
        old_quads = mcloth.sim_quad_record_count(data, sim_name)
        new_quads = mcloth.sim_quad_record_count(out_bytes, sim_name)
        if (old_quads is not None and new_quads is not None
                and new_quads > old_quads):
            grown_quad_tables.append((sim_name, old_quads, new_quads))
            logger.info(
                "%s: expanded SIM quad table from %d to %d records",
                sim_name, old_quads, new_quads)
            if operator:
                operator.report(
                    {'INFO'},
                    f"{sim_name}: expanded SIM quad table "
                    f"{old_quads}->{new_quads} records.")
    if force_recook:
        logger.info(
            "Force Recook changed a numeric-zero SIM flag in all %d "
            "mcloth stream(s)", len(streams))
        if operator:
            operator.report(
                {'INFO'},
                f"Force Recook applied to {len(streams)} mcloth stream(s).")
    for attachment_name in sorted(promoted_dense_attachments):
        logger.info(
            "%s: promoted grown dense attachment to explicit sparse rows",
            attachment_name)
        if operator:
            operator.report(
                {'INFO'},
                f"{attachment_name}: promoted grown attachment to explicit "
                "rows.")
    if operator:
        def _stat_label(name):
            base, _sep, lod_suffix = name.partition('_CLOTH_RENDER')
            return f"{base} {lod_suffix.lstrip('_') or 'LOD0'}"

        parts = ', '.join(
            f"{_stat_label(name)} {ob}->{nb}"
            for name, (ob, nb) in stats.items())
        detail = f"driven verts: {parts}" if parts else "sim section updated"
        if force_recook:
            detail += f"; force recook: {len(streams)} stream(s)"
        if promoted_dense_attachments:
            detail += (
                f"; explicit attachments: "
                f"{len(promoted_dense_attachments)}")
        if grown_quad_tables:
            detail += f"; quad tables grown: {len(grown_quad_tables)}"
        operator.report({'INFO'},
            f"Cloth mapping updated -> {os.path.basename(out_path)} ({detail})")

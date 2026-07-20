"""Shared Blender object and mesh preparation helpers."""

import bmesh
import bpy
from mathutils import Vector

from ..log import logger

def compute_normals_for_object(obj):
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

    # This should already be one normal per original vertex, but keep the list
    # exact before handing it to Blender's custom-normal C API.
    nverts = len(obj_data.vertices)
    if len(smooth_normals) != nverts:
        logger.warning(
            "%s: computed normal count %d does not match vertex count %d; repairing buffer",
            obj.name, len(smooth_normals), nverts)
        smooth_normals = (
            smooth_normals + [Vector((0.0, 0.0, 1.0))] * nverts
        )[:nverts]
    try:
        obj_data.normals_split_custom_set_from_vertices(smooth_normals)
    except Exception as error:
        logger.warning("%s: could not apply computed custom normals (%s)", obj.name, error)
    obj_data.update()

def find_object_by_name(name=""):
    return bpy.data.objects.get(name, None)

def bake_parent_inverse(obj):
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

def triangulate_object(obj, compute_normals=False, split_seams=False):
    """
    Build a temporary export mesh with seam/UV splits, parent inverse, and
    triangulation applied. The original mesh datablock is left untouched.
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

        # Blender stores UVs per face corner, while MMB stores one UV per
        # vertex. Detect unmarked UV-island boundaries so their corners become
        # separate export vertices instead of being averaged together later.
        # SIM topology and slot identity are load-bearing for cloth, so only
        # explicitly marked seams retain the old behavior on _CLOTH_SIM.
        uv_layers = list(bm.loops.layers.uv)

        def _loop_uvs(loop):
            return tuple(
                (loop[layer].uv.x, loop[layer].uv.y)
                for layer in uv_layers)

        def _uvs_differ(first_uvs, current_uvs):
            return any(
                abs(first[0] - current[0]) > 1e-5
                or abs(first[1] - current[1]) > 1e-5
                for first, current in zip(first_uvs, current_uvs))

        def _ambiguous_uv_vertices():
            ambiguous = []
            for vert in bm.verts:
                first_uvs = None
                for loop in vert.link_loops:
                    loop_uvs = _loop_uvs(loop)
                    if first_uvs is None:
                        first_uvs = loop_uvs
                    elif _uvs_differ(first_uvs, loop_uvs):
                        ambiguous.append(vert)
                        break
            return ambiguous

        def _vertex_face_fans(vert):
            """Return face components connected through edges at this vertex."""
            remaining = set(vert.link_faces)
            fans = []
            while remaining:
                seed = min(remaining, key=lambda face: face.index)
                remaining.remove(seed)
                fan = {seed}
                pending = [seed]
                while pending:
                    face = pending.pop()
                    for edge in sorted(face.edges, key=lambda item: item.index):
                        if vert not in edge.verts:
                            continue
                        for linked_face in sorted(
                                edge.link_faces, key=lambda item: item.index):
                            if linked_face in remaining:
                                remaining.remove(linked_face)
                                fan.add(linked_face)
                                pending.append(linked_face)
                fans.append(fan)
            return fans

        def _fan_uvs(fan, vert):
            face = min(fan, key=lambda item: item.index)
            loop = next(loop for loop in face.loops if loop.vert == vert)
            return _loop_uvs(loop)

        def _separate_point_uv_fans(vert):
            """Separate disconnected face fans until each vertex has one UV."""
            added_vertices = 0
            pending = [vert]
            while pending:
                current = pending.pop()
                if not current.is_valid:
                    continue
                current_uvs = [_loop_uvs(loop) for loop in current.link_loops]
                if not current_uvs or not any(
                        _uvs_differ(current_uvs[0], value)
                        for value in current_uvs[1:]):
                    continue

                fans = _vertex_face_fans(current)
                if len(fans) < 2:
                    continue

                # Prefer detaching a UV group represented by the fewest fans.
                # This keeps same-UV fans sharing their original vertex where
                # possible and minimizes appended export vertices.
                uv_groups = []
                for fan in fans:
                    fan_uvs = _fan_uvs(fan, current)
                    group = next((item for item in uv_groups
                                  if not _uvs_differ(item[0], fan_uvs)), None)
                    if group is None:
                        uv_groups.append([fan_uvs, [fan]])
                    else:
                        group[1].append(fan)
                if len(uv_groups) < 2:
                    continue
                detach_fan = min(uv_groups, key=lambda item: len(item[1]))[1][0]

                fan_edges = sorted({
                    edge
                    for face in detach_fan
                    for edge in face.edges
                    if current in edge.verts
                }, key=lambda item: item.index)
                boundary_edges = [
                    edge for edge in fan_edges
                    if sum(face in detach_fan for face in edge.link_faces) == 1
                ]
                representative = next(iter(boundary_edges or fan_edges), None)
                if representative is None:
                    continue

                separated = _bmesh.utils.vert_separate(
                    current, [representative])
                bm.verts.index_update()
                bm.edges.index_update()
                added_vertices += max(0, len(separated) - 1)
                pending.extend(separated)
            return added_vertices

        uv_split_edges = []
        auto_split_uvs = '_CLOTH_SIM' not in obj.name.upper()
        if uv_layers and auto_split_uvs:
            seam_set = set(seam_edges)
            for edge in bm.edges:
                if edge in seam_set or len(edge.link_loops) < 2:
                    continue
                split_edge = False
                for vert in edge.verts:
                    first_uvs = None
                    for edge_loop in edge.link_loops:
                        loop = (edge_loop if edge_loop.vert == vert
                                else edge_loop.link_loop_next)
                        loop_uvs = _loop_uvs(loop)
                        if first_uvs is None:
                            first_uvs = loop_uvs
                        elif _uvs_differ(first_uvs, loop_uvs):
                            split_edge = True
                            break
                    if split_edge:
                        break
                if split_edge:
                    uv_split_edges.append(edge)
        elif uv_layers:
            logger.debug(
                "%s: automatic UV-boundary splitting skipped for _CLOTH_SIM",
                obj.name)

        split_edges = seam_edges + uv_split_edges
        initial_ambiguous = (_ambiguous_uv_vertices()
                             if uv_layers and auto_split_uvs else [])
        loop_normals = None
        if split_edges or initial_ambiguous:
            # Splitting disconnects the topology, so retain the pre-split loop
            # normals to avoid introducing visible shading seams.
            try:
                loop_normals = [normal.vector.copy()
                                for normal in obj.data.corner_normals]
            except (AttributeError, RuntimeError):
                pass

        if split_edges:
            _bmesh.ops.split_edges(bm, edges=split_edges)
            bm.verts.index_update()
            bm.edges.index_update()

        # UV islands may touch only at a vertex, leaving no shared edge whose
        # adjacent loops expose the discontinuity. After the edge pass, split
        # each remaining disconnected face fan directly at that vertex.
        point_split_count = 0
        if uv_layers and auto_split_uvs:
            for vert in _ambiguous_uv_vertices():
                if vert.is_valid:
                    point_split_count += _separate_point_uv_fans(vert)

        if split_edges or point_split_count:
            logger.debug(
                "%s: splitting %d marked seam edge(s), %d "
                "UV-discontinuous edge(s), and %d point-touching UV fan(s) "
                "before export",
                obj.name, len(seam_edges), len(uv_split_edges),
                point_split_count)

            bm.to_mesh(obj.data)
            obj.data.update()

            if loop_normals is not None and len(loop_normals) == len(obj.data.loops):
                try:
                    obj.data.normals_split_custom_set(loop_normals)
                except Exception as error:
                    logger.warning(
                        "%s: could not restore normals after UV splitting (%s)",
                        obj.name, error)

        # Any remaining ambiguity indicates a topology the automatic edge/fan
        # passes could not safely separate. Keep the existing averaging fallback
        # visible rather than silently changing its UVs.
        if uv_layers and auto_split_uvs:
            ambiguous_vertices = len(_ambiguous_uv_vertices())
            if ambiguous_vertices:
                logger.warning(
                    "%s: automatic splitting could not separate %d export "
                    "vertex/vertices that still use multiple UVs; those UVs "
                    "will use the averaging fallback",
                    obj.name, ambiguous_vertices)
        bm.free()

        bake_parent_inverse(obj)

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
        logger.debug("Computing normals before export")
        compute_normals_for_object(obj)

"""Shared Blender object and mesh preparation helpers."""

import bmesh
import bpy
from mathutils import Vector

from .log import logger

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
            logger.debug("Splitting seam edges before export")
            _bmesh.ops.split_edges(bm, edges=seam_edges)
            bm.to_mesh(obj.data)
            obj.data.update()
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

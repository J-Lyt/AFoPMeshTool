"""Blender-side skeletal mesh importer."""

import math

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

from .blender_utils import compute_normals_for_object
from ..log import logger
from ..formats.mmb import SkeletalMeshAsset

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

        # Get any extra bytes between the end of vertex/normal data and the next block.
        # These are preserved on export to maintain correct vob/fb offsets.
        real_vert_size = lod.vertex_count * mesh.vertex_stride
        real_normal_size = lod.vertex_count * mesh.normals_stride
        vert_extra = (lod.vertex_data_offset_b - lod.vertex_data_offset_a) - real_vert_size
        normal_extra = (lod.face_block_offset - lod.vertex_data_offset_b) - real_normal_size
        if vert_extra > 0:
            raw_mesh_file.seek(lod.vertex_data_offset_a + real_vert_size)
            lod.vertex_end_bytes = raw_mesh_file.read(vert_extra)
        else:
            lod.vertex_end_bytes = b''
        if normal_extra > 0:
            raw_mesh_file.seek(lod.vertex_data_offset_b + real_normal_size)
            lod.normals_end_bytes = raw_mesh_file.read(normal_extra)
        else:
            lod.normals_end_bytes = b''

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
            # degenerate tris (repeated vertex) exist in exported cloth sim
            # meshes: retargeted phantom slots can collapse; skip them.
            if len(set(tris)) < 3:
                continue
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

        # Store original MMB vertex index as a named int attribute.
        # This survives separate/join operations and vertex reordering, allowing
        # write_normals to look up the correct original tangent/nw for each vertex
        # without relying on raw-position matching.
        orig_idx_attr = obj_data.attributes.new(name="mmb_vertex_order", type='INT', domain='POINT')
        for vi in range(len(obj_data.vertices)):
            orig_idx_attr.data[vi].value = vi

        bm = bmesh.new()
        bm.from_mesh(obj_data)
        bm.faces.ensure_lookup_table()
        # Import UVs
        uv_centred_flags = []
        for uv_index in range(mesh.uv_count):
            uvs = lod.get_uvs(raw_mesh_file,uv_index)
            uv_layer = bm.loops.layers.uv.new(f'UVMap_{uv_index}')
            # Detect UV convention
            v_vals = [uvs[i][1] for i in range(len(uvs))]
            u_vals = [uvs[i][0] for i in range(len(uvs))]
            v_min, v_max = min(v_vals), max(v_vals)
            centred_v = v_max > 0 and v_min < 0 and abs(v_min + v_max) < 0.15 and v_max < 0.6
            centred_u = False # Imported raw; game maps [-1,1] directly to texture width
            uv_centred_flags.append((centred_u, centred_v))
            for finder, face in enumerate(bm.faces):
                for lindex, loop in enumerate(face.loops):
                    v_index = loop.vert.index
                    u, v = uvs[v_index][0], uvs[v_index][1]
                    u_out = u
                    v_out = v + 0.5 if centred_v else v * -1 + 1
                    loop[uv_layer].uv = (u_out, v_out)

        bm.to_mesh(obj_data)
        bm.free()
        obj_data.update()

        if mesh.uv_count > 0:
            _enc_map = getattr(lod, 'last_uv_encodings', {})
            _enc_list = '/'.join(_enc_map.get(i, '?') for i in range(mesh.uv_count))
            logger.debug("UV encoding: %d (%s)", mesh.uv_count, _enc_list)

        # Store per-UV-layer centred flags as mesh attributes so export can
        # reverse the correct transform without re-detecting from Blender values.
        for uv_index, (centred_u, centred_v) in enumerate(uv_centred_flags):
            if centred_u:
                cu_attr = obj_data.attributes.get(f'mmb_uv{uv_index}_centred_u') or \
                          obj_data.attributes.new(name=f'mmb_uv{uv_index}_centred_u', type='INT', domain='POINT')
                for vi in range(len(cu_attr.data)):
                    cu_attr.data[vi].value = 1
            if centred_v:
                cv_attr = obj_data.attributes.get(f'mmb_uv{uv_index}_centred_v') or \
                          obj_data.attributes.new(name=f'mmb_uv{uv_index}_centred_v', type='INT', domain='POINT')
                for vi in range(len(cv_attr.data)):
                    cv_attr.data[vi].value = 1

        # Import Colors
        # Written directly to obj_data.attributes (POINT domain, FLOAT_COLOR type) rather
        # than via a BMesh float_color layer, because the BMesh vertex-layer -> to_mesh
        # round-trip corrupts float_color values, causing them to appear white in the viewport.
        for color_index in range(mesh.color_count):
            colors = lod.get_color(raw_mesh_file, color_index)
            attr_name = f"Color_{color_index}"
            attr = obj_data.attributes.new(name=attr_name, type='FLOAT_COLOR', domain='POINT')
            for vi, (r, g, b, a) in enumerate(colors):
                attr.data[vi].color = (r, g, b, a)

        # Import Normals
        # Some VAT meshes store animation-texture lookup data in the normals stream
        # rather than actual surface normals (e.g. geckofish: all ny=0, nz=0).
        # Detect degenerate normals (all pointing the same direction) and fall back to computing normals.
        # Faces must be smooth-shaded for custom split normals to actually render smooth
        # (flat-shaded faces ignore per-vertex normals, giving harsh per-facet lighting).
        for poly in obj_data.polygons:
            poly.use_smooth = True
        computed_normals = lod.get_normals(raw_mesh_file)
        degenerate = False
        if computed_normals:
            first = computed_normals[0]
            sample = computed_normals[1:min(len(computed_normals), 256)]

            all_w_zero = getattr(lod, '_all_w_zero', False)
            tangent_space = getattr(lod, '_tangent_space', False)
            all_same_dir = all(abs(n.dot(first)) > 0.999 for n in sample)
            zero_count = sum(1 for n in computed_normals if n.length < 0.1)
            mostly_zero = zero_count > len(computed_normals) * 0.5

            # Compare normals in file against the meshes own face normals.
            # Create a temporary mesh and measure the average dot between each stored normal and the face it belongs to.
            # Valid normals should match with their faces (avg_dot typically between 0.4 and 0.9).
            import bmesh as _bmesh
            bm_check = _bmesh.new()
            bm_check.from_mesh(obj_data)
            bm_check.faces.ensure_lookup_table()
            dot_sum = 0.0
            dot_count = 0
            for face in bm_check.faces:
                face.normal_update()
                fn = face.normal
                for loop in face.loops:
                    vi = loop.vert.index
                    if vi < len(computed_normals):
                        sn = computed_normals[vi]
                        if sn.length > 0.1:
                            dot_sum += fn.dot(sn)
                            dot_count += 1
                    if dot_count >= 4096:
                        break
                if dot_count >= 4096:
                    break
            bm_check.free()
            avg_dot = (dot_sum / dot_count) if dot_count > 0 else 0.0
            # Stored normals that don't match with face normals (avg_dot < 0.1)
            # or normals that point consistently opposite to faces (avg_dot < -0.1) are degenerate.
            bad_correlation = abs(avg_dot) < 0.1

            degenerate = all_w_zero or all_same_dir or mostly_zero or bad_correlation

            if not degenerate:
                # Blender's custom-normal API expects exactly one usable vector
                # per mesh vertex. Validate the buffer before entering the C API,
                # where a short list can otherwise cause a hard crash.
                nverts = len(obj_data.vertices)
                if len(computed_normals) != nverts:
                    logger.warning(
                        "%s: file normal count %d does not match vertex count %d; "
                        "computing normals instead",
                        obj.name, len(computed_normals), nverts)
                    degenerate = True
                else:
                    safe_normals = []
                    for normal in computed_normals:
                        try:
                            length = normal.length
                            finite = all(math.isfinite(component) for component in normal)
                        except (AttributeError, TypeError, ValueError):
                            length = 0.0
                            finite = False
                        if not finite or not math.isfinite(length) or length < 1e-6:
                            safe_normals.append(Vector((0.0, 0.0, 1.0)))
                        else:
                            safe_normals.append(normal)
                    try:
                        logger.debug("%s: importing normals from file", obj.name)
                        obj_data.normals_split_custom_set_from_vertices(safe_normals)
                    except Exception as error:
                        logger.warning(
                            "%s: applying file normals failed (%s); computing normals instead",
                            obj.name, error)
                        degenerate = True
            else:
                reasons = []
                if all_w_zero: reasons.append("All w=0 (e.g. VAT Data)")
                if all_same_dir: reasons.append("All Same Direction")
                if mostly_zero: reasons.append(f"Mostly Zero ({zero_count}/{len(computed_normals)})")
                if bad_correlation: reasons.append(f"No Face Correlation (avg_dot={avg_dot:.3f})")
                logger.warning(
                    "%s: degenerate normals detected (%s); computing normals",
                    obj.name, ', '.join(reasons))

        if degenerate:
            compute_normals_for_object(obj)

        # Import Bone Weights
        weights = lod.get_bone_weights(raw_mesh_file)
        # The declaration describes storage capacity, not necessarily the
        # runtime influence count used by this asset. Preserve the observed
        # source limit for generated LODs (for example, Banshee declares eight
        # lanes but every stock vertex uses at most six).
        source_influence_limit = max(
            (len(vertex_weights) for vertex_weights in weights), default=0)
        if source_influence_limit:
            obj["mmb_source_influence_limit"] = min(
                mesh.influence_capacity(), source_influence_limit)
        mesh_bones = list(mesh.mesh_bones.keys())
        # Only create vertex groups for bones this mesh references
        for real_bone_index in mesh_bones:
            if real_bone_index < len(skeletal_mesh.bones):
                bone_name = skeletal_mesh.bones[real_bone_index].name
                if obj.vertex_groups.get(bone_name) is None:
                    obj.vertex_groups.new(name=bone_name)

        for v_index in range(lod.vertex_count):
            v_bone_weights = weights[v_index]
            for bone_index in v_bone_weights.keys():
                if bone_index < len(mesh_bones):
                    real_bone_index = mesh_bones[bone_index] # Convert mesh bone index to skeleton bone index
                    bone_name = skeletal_mesh.bones[real_bone_index].name
                    obj.vertex_groups[bone_name].add([v_index], v_bone_weights[bone_index], "ADD")
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
            bone.parent = _armature.edit_bones[parent_index] if parent_index != -1 else None
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
    def parent_obj_to_armature(obj, armature, mesh):
        # Skip the armature modifier for meshes with no bone slots.
        if not mesh.mesh_bones:
            obj.parent = armature
            return
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

BMI = BlenderMeshImporter

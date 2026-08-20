"""Mesh-bone and skeleton-merge operators."""

import operator
import os
from struct import unpack

import bpy
from mathutils import Matrix

from .. import addon_state
from ..formats.binary_io import br
from ..mesh_pipeline.files import (
    get_merged_mmb,
    source_setting_path,
)
from ..mesh_pipeline.importer import BMI
from ..log import logger
from ..formats.mmb import SkeletalMeshAsset
from ..settings import save_staged_state

def _compute_inv_bind_from_skeleton(bone_name):
    """
    Compute inverse bind from the complete runtime skeleton.

    The runtime skeleton includes export-staged Merge Skeleton additions that
    intentionally do not exist in the retained source MMB yet.
    """
    asset = addon_state.asset
    if asset is None:
        return None
    bone_idx = next(
        (index for index, bone in enumerate(asset.bones)
         if bone.name == bone_name),
        None,
    )
    if bone_idx is None:
        return None

    try:
        file_world_cache = {}
        visiting = set()

        def get_file_world(index):
            cached = file_world_cache.get(index)
            if cached is not None:
                return cached
            if not 0 <= index < len(asset.bones) or index in visiting:
                raise ValueError("Invalid skeleton parent hierarchy")
            visiting.add(index)
            bone = asset.bones[index]
            parent_index = int(bone.parent_index)
            if parent_index == 65535:
                file_world = bone.matrix.copy()
            else:
                file_world = get_file_world(parent_index) @ bone.matrix
            visiting.remove(index)
            file_world_cache[index] = file_world
            return file_world

        inv_bind = get_file_world(bone_idx).inverted()
        return tuple(inv_bind[r][c] for c in range(4) for r in range(4))
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def _read_donor_matrix(donor_path, target_bone_name, mesh_name):
    """
    Search a donor MMB file for a mesh bone slot that maps to target_bone_name
    """
    try:
        donor_mmb = get_merged_mmb(donor_path)
        f = donor_mmb
        f.seek(0)
        br.string(f, 3)
        version = br.uint8(f)
        f.seek(4, 1)
        if version >= 15:
            f.seek(4, 1)

        bone_count = br.uint32(f)
        donor_bone_index = None
        for i in range(bone_count):
            nlen = unpack('<H', f.read(2))[0]
            name = br.string(f, nlen)
            f.seek(64, 1)
            f.seek(2, 1)
            if name == target_bone_name:
                donor_bone_index = i

        if donor_bone_index is None:
            return None

        mesh_count = br.uint32(f)
        fallback_matrix = None
        for mi in range(mesh_count):
            nlen = unpack('<H', f.read(2))[0]
            dname = br.string(f, nlen).rstrip('\x00')
            f.seek(48, 1); f.seek(1, 1)
            x_count = br.uint8(f); f.seek(1, 1); f.seek(4 * x_count, 1)
            u_count = br.uint16(f)
            slots = []
            for b in range(u_count):
                mat = unpack('<16f', f.read(64))
                idx = br.uint16(f)
                slots.append((idx, mat))
            for idx, mat in slots:
                if idx == donor_bone_index:
                    if dname == mesh_name:
                        return mat
                    if fallback_matrix is None:
                        fallback_matrix = mat
            if version not in (11, 12, 13, 14, 15, 16, 17):
                break
            if u_count > 0 and version != 12:
                f.seek(1 if version == 13 else 2, 1)
                lod_info_type = br.uint8(f)
            else:
                lod_info_type = 0 if version in (12, 13) else br.uint8(f)
            lod_count = br.uint8(f); f.seek(4, 1)
            for _ in range(lod_count):
                f.seek(36, 1)
                if lod_info_type == 2:
                    f.seek(28, 1)
            uv_count = br.uint8(f); f.seek(4 * uv_count, 1)
            if version in (16, 17):
                color_count = br.uint8(f); f.seek(4 * color_count, 1)
                f.seek(4, 1); count_c = br.uint8(f); f.seek(4 * count_c, 1)
            else:
                f.seek(4, 1); color_count = br.uint8(f); f.seek(4 * color_count, 1)
            f.seek(4, 1)
            f.seek(20 if version == 17 else 16, 1)
        return fallback_matrix
    except Exception as e:
        logger.warning("Could not read donor matrix: %s", e)
        return None


def _bone_search_cb(self, context, edit_text):
    """filters skeleton bone names by typed text"""
    if addon_state.asset is None:
        return []
    edit_lower = edit_text.lower()
    return [
        b.name
        for b in addon_state.asset.bones
        if edit_lower in b.name.lower()
    ]


def _mesh_is_uint8_index_limited(mesh):
    """True when a declared bone-index element stores uint8 components."""
    index_elements = mesh.elements(semantic=3, stream=0)
    if not index_elements:
        return False
    return any(element['format'] == 15 for element in index_elements)

def _scan_mesh_used_bone_slots(mesh):
    """
    Scan the `mesh` in the currently loaded asset and return the set of
    mesh-bone-table slot indices that have non-zero weight on any vertex, across
    all LODs.

    Callers adding several bones to the same mesh should call this once and reuse the result via
    _find_unused_mesh_bone_slot's `used` parameter, rather than re-scanning per bone.

    Returns None on any read error.
    """
    try:
        SWOMT = bpy.context.scene.SWOMT
        src_path = bpy.path.abspath(source_setting_path(
            SWOMT, "AssetPath", "SourceAssetPath"))
        used = set()
        with open(src_path, 'rb') as f:
            raw_mesh_file = mesh.extract_mesh_file(f)
        for lod in mesh.lods:
            if lod.vertex_count == 0:
                continue
            for iw in lod.get_bone_weights(raw_mesh_file):
                used.update(s for s, w in iw.items() if w > 0.0)
        return used
    except Exception as e:
        logger.warning("Could not scan used mesh bone slots: %s", e)
        return None


def _find_unused_mesh_bone_slot(mesh, used=None):
    """
    Return the lowest mesh-bone-table slot with zero weight on every vertex - i.e. one
    that can be reused for a new bone without touching existing weights. Slots already pending a remap
    in this session (mesh.pending_bone_remaps) are excluded, even if they have zero weights.

    `used` is the result of _scan_mesh_used_bone_slots(mesh). It is passed when adding
    several bones to the same mesh at the same time, so the file scan only happens once.

    Returns None if no free slot exists, or the scan failed.
    """
    n_slots = len(mesh.mesh_bones)
    if n_slots == 0:
        return None
    if used is None:
        used = _scan_mesh_used_bone_slots(mesh)
        if used is None:
            return None
    else:
        used = set(used)
    used.update(mesh.pending_bone_remaps.keys())
    for slot in range(n_slots):
        if slot not in used:
            return slot
    return None


def _add_or_reuse_mesh_bone_slot(mesh, new_skel_idx, new_matrix, used_slots_cache=None):
    """
    Add a new bone to `mesh`'s bone table, reusing an unused slot instead of
    appending past the uint8 limit when its declared bone indices are uint8.

    `used_slots_cache`: pass a dict when adding several bones to the same mesh at once -
    keyed by `id(mesh)`, so the weight scan only happens once per mesh instead of per bone.

    Returns (status, info):
      'appended' - added as a new slot at the end; info=None
      'reused'   - remapped an unused slot; info=slot_index
      'full'     - no free slot available (256-slot limit)
    """
    n_slots = len(mesh.mesh_bones)
    if _mesh_is_uint8_index_limited(mesh) and n_slots >= 256:
        used = None
        if used_slots_cache is not None:
            cache_key = id(mesh)
            if cache_key in used_slots_cache:
                used = used_slots_cache[cache_key]
                if used is None:
                    return 'full', None # If a bone failed to scan - treat as "full"
            else:
                used = _scan_mesh_used_bone_slots(mesh)
                used_slots_cache[cache_key] = used # Failed (None) are also cached
                if used is None:
                    return 'full', None
        free_slot = _find_unused_mesh_bone_slot(mesh, used=used)
        if free_slot is None:
            return 'full', None
        mesh.pending_bone_remaps[free_slot] = (new_skel_idx, new_matrix)
        new_mesh_bones = {}
        for slot_i, (skel_idx, matrix) in enumerate(mesh.mesh_bones.items()):
            if slot_i == free_slot:
                new_mesh_bones[new_skel_idx] = new_matrix
            else:
                new_mesh_bones[skel_idx] = matrix
        mesh.mesh_bones = new_mesh_bones
        return 'reused', free_slot

    mesh.pending_bone_additions.append((new_skel_idx, new_matrix))
    mesh.mesh_bones[new_skel_idx] = new_matrix
    return 'appended', None


class RemapMeshBone(bpy.types.Operator):
    """Remap a mesh bone slot to a different bone"""
    bl_idname = "object.remap_mesh_bone"
    bl_label = "Remap Bone Slot"

    mesh_index: bpy.props.IntProperty()
    slot_index: bpy.props.IntProperty()

    new_bone_name: bpy.props.StringProperty(
        name="New Bone",
        description="Skeleton bone to remap this slot to",
        search=_bone_search_cb,
        search_options={'SORT'},
    )
    use_auto: bpy.props.BoolProperty(
        name="Auto",
        description="Derive the inverse bind matrix from the loaded skeleton (Recommended). Uncheck to select a donor MMB file instead.",
        default=True,
    )
    donor_path: bpy.props.StringProperty(
        name="Donor MMB",
        description="An MMB file whose mesh already references the new bone",
        subtype="FILE_PATH",
    )

    @classmethod
    def poll(cls, context):
        return addon_state.asset is not None

    def invoke(self, context, event):
        mesh = addon_state.asset.meshes[self.mesh_index]
        mesh_bones_list = list(mesh.mesh_bones.keys())
        current_skel_idx = mesh_bones_list[self.slot_index]
        self.new_bone_name = addon_state.asset.bones[current_skel_idx].name if current_skel_idx < len(addon_state.asset.bones) else ""
        self.use_auto = True
        self.donor_path = ""
        context.window.cursor_warp(context.window.width // 2, context.window.height // 2)
        return context.window_manager.invoke_props_dialog(self, width=450)

    def draw(self, context):
        mesh = addon_state.asset.meshes[self.mesh_index]
        mesh_bones_list = list(mesh.mesh_bones.keys())
        current_skel_idx = mesh_bones_list[self.slot_index]
        current_name = addon_state.asset.bones[current_skel_idx].name if current_skel_idx < len(addon_state.asset.bones) else str(current_skel_idx)
        layout = self.layout
        layout.label(text=f"Mesh: {mesh.name}   Slot: {self.slot_index}   Current: {current_name}")
        layout.separator()
        layout.prop(self, "new_bone_name", text="New Bone", icon="BONE_DATA")
        layout.separator()
        layout.prop(self, "use_auto")
        if not self.use_auto:
            layout.label(text="Donor MMB - an MMB file whose mesh already uses the new bone:", icon="FILE")
            layout.prop(self, "donor_path", text="")

    def execute(self, context):
        mesh = addon_state.asset.meshes[self.mesh_index]
        new_name = self.new_bone_name.strip()

        if not new_name:
            self.report({'ERROR'}, "Bone name cannot be empty")
            return {'CANCELLED'}

        new_skel_idx = next((i for i, b in enumerate(addon_state.asset.bones) if b.name == new_name), None)
        if new_skel_idx is None:
            self.report({'ERROR'}, f"Bone '{new_name}' not found in skeleton")
            return {'CANCELLED'}

        mesh_bones_list = list(mesh.mesh_bones.keys())
        if self.slot_index >= len(mesh_bones_list):
            self.report({'ERROR'}, "Slot index out of range")
            return {'CANCELLED'}

        old_skel_idx = mesh_bones_list[self.slot_index]
        old_name = addon_state.asset.bones[old_skel_idx].name if old_skel_idx < len(addon_state.asset.bones) else str(old_skel_idx)

        if old_skel_idx == new_skel_idx:
            self.report({'INFO'}, "Slot already maps to that bone")
            return {'FINISHED'}

        if new_skel_idx in mesh_bones_list:
            self.report({'ERROR'}, f"Slot already has '{new_name}' at position {mesh_bones_list.index(new_skel_idx)}")
            return {'CANCELLED'}

        # Get the inverse bind matrix
        # Auto: derive directly from the loaded skeleton (Recommended).
        # Manual: read from a donor MMB file.
        if self.use_auto:
            new_matrix = _compute_inv_bind_from_skeleton(new_name)
            if new_matrix is None:
                self.report({'ERROR'},
                    f"Could not derive inv_bind for '{new_name}' from the loaded skeleton. "
                    f"Uncheck Auto and supply a donor MMB instead.")
                return {'CANCELLED'}
        else:
            donor_path = self.donor_path.strip()
            if not donor_path or not os.path.isfile(donor_path):
                self.report({'ERROR'}, "Please select a valid donor MMB file.")
                return {'CANCELLED'}
            new_matrix = _read_donor_matrix(donor_path, new_name, mesh.name)
            if new_matrix is None:
                self.report({'ERROR'},
                    f"Donor file found but '{new_name}' is not referenced by any mesh in it. "
                    f"Select a donor MMB whose mesh already uses that bone.")
                return {'CANCELLED'}

        # Stage both the skeleton index AND the matrix for export
        mesh.pending_bone_remaps[self.slot_index] = (new_skel_idx, new_matrix)

        # Update mesh.mesh_bones
        new_mesh_bones = {}
        for slot_i, (skel_idx, matrix) in enumerate(mesh.mesh_bones.items()):
            if slot_i == self.slot_index:
                new_mesh_bones[new_skel_idx] = new_matrix
            else:
                new_mesh_bones[skel_idx] = matrix
        mesh.mesh_bones = new_mesh_bones

        # Rename the vertex group on any already-imported objects
        for li, lod in enumerate(mesh.lods):
            obj_name = lod.blender_obj_name if lod.blender_obj_name else f"{mesh.name}_LOD{li}"
            obj = bpy.data.objects.get(obj_name)
            if obj is not None:
                vg = obj.vertex_groups.get(old_name)
                if vg is not None:
                    vg.name = new_name

        source = "skeleton" if self.use_auto else "donor MMB"
        save_staged_state(context.scene.SWOMT)
        self.report({'INFO'}, f"Slot {self.slot_index}: '{old_name}' to '{new_name}' via {source} (will patch on export)")
        return {'FINISHED'}

class AddMeshBone(bpy.types.Operator):
    """Add a new bone slot to this mesh's bone table"""
    bl_idname = "object.add_mesh_bone"
    bl_label = "Add Bone Slot"

    mesh_index: bpy.props.IntProperty()

    new_bone_name: bpy.props.StringProperty(
        name="New Bone",
        description="Bone to add as a new slot",
        search=_bone_search_cb,
        search_options={'SORT'},
    )
    use_auto: bpy.props.BoolProperty(
        name="Auto",
        description="Derive the inverse bind matrix from the loaded skeleton (Recommended). Uncheck to select a donor MMB file instead.",
        default=True,
    )
    donor_path: bpy.props.StringProperty(
        name="Donor MMB",
        description="An MMB file whose mesh already references the new bone",
        subtype="FILE_PATH",
    )

    @classmethod
    def poll(cls, context):
        return addon_state.asset is not None

    def invoke(self, context, event):
        self.new_bone_name = ""
        self.use_auto = True
        self.donor_path = ""
        context.window.cursor_warp(context.window.width // 2, context.window.height // 2)
        return context.window_manager.invoke_props_dialog(self, width=450)

    def draw(self, context):
        mesh = addon_state.asset.meshes[self.mesh_index]
        layout = self.layout
        layout.label(text=f"Mesh: {mesh.name}   Current slots: {len(mesh.mesh_bones)}")
        layout.separator()
        layout.prop(self, "new_bone_name", text="New Bone", icon="BONE_DATA")
        layout.separator()
        layout.prop(self, "use_auto")
        if not self.use_auto:
            layout.label(text="Donor MMB - an MMB file whose mesh already uses the new bone:", icon="FILE")
            layout.prop(self, "donor_path", text="")

    def execute(self, context):
        mesh = addon_state.asset.meshes[self.mesh_index]
        new_name = self.new_bone_name.strip()

        if not new_name:
            self.report({'ERROR'}, "Bone name cannot be empty.")
            return {'CANCELLED'}

        new_skel_idx = next((i for i, b in enumerate(addon_state.asset.bones) if b.name == new_name), None)
        if new_skel_idx is None:
            self.report({'ERROR'}, f"Bone '{new_name}' not found in skeleton.")
            return {'CANCELLED'}

        if new_skel_idx in mesh.mesh_bones:
            self.report({'ERROR'}, f"'{new_name}' is already in this mesh's bone table.")
            return {'CANCELLED'}

        # Check it's not already staged for addition
        if any(idx == new_skel_idx for idx, _ in mesh.pending_bone_additions):
            self.report({'ERROR'}, f"'{new_name}' is already staged for addition.")
            return {'CANCELLED'}

        # Get the inverse bind matrix.
        # Auto: derive directly from the loaded skeleton (Recommended).
        # Manual: read from a donor MMB file.
        if self.use_auto:
            new_matrix = _compute_inv_bind_from_skeleton(new_name)
            if new_matrix is None:
                self.report({'ERROR'},
                    f"Could not derive inv_bind for '{new_name}' from the loaded skeleton. "
                    f"Uncheck Auto and supply a donor MMB instead.")
                return {'CANCELLED'}
        else:
            donor_path = self.donor_path.strip()
            if not donor_path or not os.path.isfile(donor_path):
                self.report({'ERROR'}, "Please select a valid donor MMB file.")
                return {'CANCELLED'}
            new_matrix = _read_donor_matrix(donor_path, new_name, mesh.name)
            if new_matrix is None:
                self.report({'ERROR'},
                    f"Donor file found but '{new_name}' is not referenced by any mesh in it. "
                    f"Select a donor MMB whose mesh already uses that bone.")
                return {'CANCELLED'}

        # Stage the addition; reuse an unused slot when this mesh declares uint8 indices.
        status, info = _add_or_reuse_mesh_bone_slot(mesh, new_skel_idx, new_matrix)
        if status == 'full':
            self.report({'ERROR'},
                f"'{mesh.name}' uses uint8 bone indices (256 slots maximum) and has "
                f"no un-weighted slot to reuse.")
            return {'CANCELLED'}

        # Create the vertex group on any already-imported Blender objects
        for li, lod in enumerate(mesh.lods):
            obj_name = lod.blender_obj_name if lod.blender_obj_name else f"{mesh.name}_LOD{li}"
            obj = bpy.data.objects.get(obj_name)
            if obj is not None and obj.vertex_groups.get(new_name) is None:
                obj.vertex_groups.new(name=new_name)

        source = "skeleton" if self.use_auto else "donor MMB"
        save_staged_state(context.scene.SWOMT)
        if status == 'reused':
            self.report({'INFO'},
                f"'{new_name}' staged via {source}, reusing unused slot {info} (will patch on export)")
        else:
            self.report({'INFO'}, f"'{new_name}' staged for addition via {source} (will patch on export)")
        return {'FINISHED'}

def _read_donor_skeleton(donor_path: str):
    """
    Parse a donor .mmb file and return a list of (name, mat_raw, matrix, parent_idx).

    mat_raw - the original 16 floats exactly as stored in the file.
              Used directly for the bone_blob write so no re-encoding is needed.
    matrix  - a Blender Matrix built via the same br.matrix_4x4 convention
              (m[r][c] = mat_raw[c*4+r]) for use in world-matrix accumulation.
    """
    try:
        with open(donor_path, 'rb') as f:
            data = f.read()
        pos = 0
        version = data[3]
        pos += 8
        if version >= 15:
            pos += 4
        bone_count = unpack('<I', data[pos:pos+4])[0]
        pos += 4
        bones = []
        for i in range(bone_count):
            nlen = unpack('<H', data[pos:pos+2])[0]; pos += 2
            name = data[pos:pos+nlen].decode('ascii', errors='replace'); pos += nlen
            mat_raw = unpack('<16f', data[pos:pos+64]); pos += 64
            parent_idx = unpack('<H', data[pos:pos+2])[0]; pos += 2
            # Build Matrix using br.matrix_4x4 convention: m[r][c] = mat_raw[c*4+r].
            # br.matrix_4x4 reads floats and groups them so that the i-th outer loop
            # feeds each row-list in column order - i.e. the file is stored column-first.
            m = Matrix([
                [mat_raw[0], mat_raw[4], mat_raw[8],  mat_raw[12]],
                [mat_raw[1], mat_raw[5], mat_raw[9],  mat_raw[13]],
                [mat_raw[2], mat_raw[6], mat_raw[10], mat_raw[14]],
                [mat_raw[3], mat_raw[7], mat_raw[11], mat_raw[15]],
            ])
            bones.append((name, mat_raw, m, parent_idx))
        return bones
    except Exception as e:
        logger.warning("Could not read donor skeleton: %s", e)
        return None

def _resolve_selected_donor_bones(donor_bones, selected_names):
    """
    From the full donor bone list, keep only `selected_names` plus whatever parents
    each one needs to stay connected.

    Returns (orig_idx, name, mat_raw, matrix, parent_idx) elements, in order from the donor.
    orig_idx is the bone's index in the original unfiltered list - callers must use it
    (not the filtered list's position) when matching against parent_idx, since both
    are original-list indices and filtering changes a bone's position but not its index.
    """
    name_set = set(selected_names)
    keep = set()
    for d_idx, (name, mat_raw, matrix, pidx) in enumerate(donor_bones):
        if name not in name_set:
            continue
        # Walk up the parent chain, adding every parent along the way.
        cur = d_idx
        while cur is not None and cur not in keep:
            keep.add(cur)
            _, _, _, cur_pidx = donor_bones[cur]
            if cur_pidx == 65535 or cur_pidx >= len(donor_bones):
                cur = None
            else:
                cur = cur_pidx
    return [(i,) + donor_bones[i] for i in sorted(keep)]

# Cache at module-level (fixes search issue)
_donor_bone_names_cache = []

# Avoids re-parsing the donor file in execute() right after invoke() already did.
_donor_bones_cache = (None, None)

def _get_cached_donor_bones(donor_path):
    """Return donor_bones for `donor_path`, parsing new, if not already cached."""
    global _donor_bones_cache
    cached_path, cached_bones = _donor_bones_cache
    if cached_path == donor_path and cached_bones is not None:
        return cached_bones
    donor_bones = _read_donor_skeleton(donor_path)
    _donor_bones_cache = (donor_path, donor_bones)
    return donor_bones

def _donor_bone_search_cb(self, context, edit_text):
    """
    Search for the comma-separated bone_names field.

    Filters on the text after the last comma, returning a selected bone name
    and leaving the field empty for the next one.
    """
    if not _donor_bone_names_cache:
        return []
    prefix = ""
    tail = edit_text
    if "," in edit_text:
        prefix, tail = edit_text.rsplit(",", 1)
        prefix = prefix.strip(" ,") + ", "
    tail_lower = tail.strip().lower()
    already = {n.strip().lower() for n in edit_text.split(",") if n.strip()}
    results = []
    for name in _donor_bone_names_cache:
        if tail_lower in name.lower() and name.lower() not in already:
            results.append(f"{prefix}{name}, ")
    return results

def _do_merge_skeletons(context, operator, src_filepath, donor_bones, mode_label):
    """
    Called by MergeSkeletonsPickBones.

    donor_bones is a list of (orig_idx, name, mat_raw, matrix, parent_idx) elements.
    orig_idx must be used for all donor-indexing here (not the list's own position as it may be filtered).
    """
    SWOMT = context.scene.SWOMT

    # Index map for the host skeleton
    host_names = {b.name: i for i, b in enumerate(addon_state.asset.bones)}

    # Collect only bones that are new (not already in the host)
    new_bones = [(orig_idx, name, mat_raw, matrix, pidx)
                 for orig_idx, name, mat_raw, matrix, pidx in donor_bones
                 if name not in host_names]

    if not new_bones:
        operator.report({'INFO'}, f"No new bones to merge ({mode_label}) - already in the loaded skeleton.")
        return {'FINISHED'}

    # Data to insert into the file:
    # - Each bone: uint16 name_len | name bytes | 16 floats (64 bytes) | uint16 parent_idx
    # - Parent indices must be remapped to the combined skeleton's indices.
    # - donor_index -> combined_index, keyed by each bone's ORIGINAL donor index so it
    #   lines up with parent_idx regardless of any filtering.

    # Map: donor bone index -> combined index (host bones first, then appended from donor)
    donor_to_combined = {}
    for orig_idx, name, mat_raw, matrix, pidx in donor_bones:
        if name in host_names:
            donor_to_combined[orig_idx] = host_names[name]

    new_start = len(addon_state.asset.bones)
    for ni, (orig_idx, name, mat_raw, matrix, pidx) in enumerate(new_bones):
        donor_to_combined[orig_idx] = new_start + ni

    # Store the final combined parent indices and exact donor matrix floats.
    # Export starts from the retained source and inserts these records later.
    for orig_idx, name, mat_raw, _matrix, pidx in new_bones:
        if pidx == 65535:
            combined_pidx = 65535
        else:
            combined_pidx = donor_to_combined.get(pidx, 65535)
        addon_state.asset.stage_skeleton_bone(
            name, mat_raw, combined_pidx)

    save_staged_state(SWOMT)

    # Rebuild the Blender armature from the merged skeleton
    arm_obj = bpy.data.objects.get(addon_state.asset.name)
    if arm_obj is not None and arm_obj.type == 'ARMATURE':
        # Collect names of every mesh object currently parented to the armature so we can re-parent them afterwards.
        child_meshes = [obj for obj in arm_obj.children if obj.type == 'MESH']

        # Remove the old armature object and data entirely.
        old_arm_data = arm_obj.data
        bpy.data.objects.remove(arm_obj, do_unlink=True)
        bpy.data.armatures.remove(old_arm_data)

        # Re-import the skeleton. import_skeleton reads asset.bones (already updated
        # to include the new bones) and builds a new correctly-transformed
        # armature via the exact same path as a normal LOD import.
        new_arm_obj = BMI.import_skeleton(addon_state.asset)

        # import_skeleton only applies the X-flip. rotate_model applies the 90deg X
        # rotation that brings the armature into the correct viewport orientation.
        # It is normally called from ImportLOD, so we call it explicitly here.
        dummy = child_meshes[0] if child_meshes else new_arm_obj
        BMI.rotate_model(dummy, new_arm_obj)

        # Restore armature modifiers and parenting on all child meshes.
        # Reset matrix_parent_inverse to 'identity' so the mesh sits correctly
        # relative to the rebuilt armature (same as the original import).
        for mesh_obj in child_meshes:
            mesh_obj.parent = new_arm_obj
            mesh_obj.matrix_parent_inverse = Matrix.Identity(4)
            arm_mod = mesh_obj.modifiers.get('Armature')
            if arm_mod is not None:
                arm_mod.object = new_arm_obj

    operator.report({'INFO'},
        f"Staged {len(new_bones)} new skeleton bone(s) ({mode_label}) from "
        f"'{os.path.basename(src_filepath)}'. Skeleton now has "
        f"{len(addon_state.asset.bones)} bones; the merge will be applied on export.")
    return {'FINISHED'}

def _all_donor_bones_field_text():
    """The bone_names field text representing 'every donor bone selected'."""
    if not _donor_bone_names_cache:
        return ""
    return ", ".join(_donor_bone_names_cache) + ", "

def _select_all_update_cb(self, context):
    """Fills or clears bone_names when the 'Select All' checkbox changes."""
    self.bone_names = _all_donor_bones_field_text() if self.select_all else ""

class MergeSkeletonsPickBones(bpy.types.Operator):
    """Select which donor bones to merge (shown after selecting the donor .mmb)"""
    bl_idname = "object.merge_skeletons_pick_bones"
    bl_label = "Select Bones to Merge"
    bl_options = {'REGISTER'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    select_all: bpy.props.BoolProperty(
        name="Select All",
        description="Merge every bone in the donor skeleton. Un-check to select specific bone names.",
        default=True,
        update=_select_all_update_cb,
    )
    bone_names: bpy.props.StringProperty(
        name="Bones",
        description="Comma-separated list of donor bone names to merge.",
        search=_donor_bone_search_cb,
    )

    @classmethod
    def poll(cls, context):
        return addon_state.asset is not None

    def invoke(self, context, event):
        global _donor_bone_names_cache
        donor_bones = _get_cached_donor_bones(self.filepath)
        if donor_bones is None:
            self.report({'ERROR'}, "Failed to read donor .mmb skeleton.")
            return {'CANCELLED'}
        _donor_bone_names_cache = [name for name, mat_raw, matrix, pidx in donor_bones]
        # Default to 'Select All'.
        self.select_all = True
        self.bone_names = _all_donor_bones_field_text()
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        layout = self.layout
        layout.label(text=f"Donor: {os.path.basename(self.filepath)}", icon="FILE")
        layout.label(text=f"{len(_donor_bone_names_cache)} bone(s) available in donor skeleton.")
        layout.separator()
        layout.prop(self, "select_all")
        col = layout.column()
        col.enabled = not self.select_all
        col.prop(self, "bone_names", text="")
        layout.label(text="Parents required to keep the skeleton connected are added automatically.")

    def execute(self, context):
        donor_bones = _get_cached_donor_bones(self.filepath)
        if donor_bones is None:
            self.report({'ERROR'}, "Failed to read donor .mmb skeleton.")
            return {'CANCELLED'}

        requested = [n.strip() for n in self.bone_names.split(",") if n.strip()]
        if not requested:
            self.report({'ERROR'}, "Enter at least one bone name to merge, or check 'Select All'.")
            return {'CANCELLED'}

        donor_names = {name for name, mat_raw, matrix, pidx in donor_bones}
        unknown = [n for n in requested if n not in donor_names]
        if unknown:
            self.report({'ERROR'}, f"Bone(s) not found in donor skeleton: {', '.join(unknown)}")
            return {'CANCELLED'}

        filtered_bones = _resolve_selected_donor_bones(donor_bones, requested)
        mode_label = "all bones" if self.select_all else "selected bones"
        return _do_merge_skeletons(context, self, self.filepath, filtered_bones, mode_label)

class MergeSkeletons(bpy.types.Operator):
    """Merge bones from a donor .mmb skeleton into the currently-loaded asset skeleton."""

    # New donor bones are staged in the .blend and appended to asset.bones in
    # memory. The armature is rebuilt so Add/Remap Bone can reference them immediately.

    bl_idname = "object.merge_skeletons"
    bl_label = "Merge Skeleton"
    bl_options = {'REGISTER'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.mmb", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return addon_state.asset is not None

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        if not self.filepath or not os.path.isfile(self.filepath):
            self.report({'ERROR'}, "Please select a valid .mmb file.")
            return {'CANCELLED'}

        # Hand off to the bone-picker dialog; it reads the donor file itself and
        # performs the merge once the user confirms which bones to include.
        bpy.ops.object.merge_skeletons_pick_bones('INVOKE_DEFAULT', filepath=self.filepath)
        return {'FINISHED'}

class AddBonesFromVertexGroups(bpy.types.Operator):
    """Add a bone slot for every vertex group to this mesh's bone table"""

    # Scan the imported mesh objects for this asset entry and add a bone slot for every
    # vertex group whose name matches a skeleton bone but is not yet in the bone table.
    # Inverse bind matrices are derived automatically from the loaded skeleton.

    bl_idname = "object.add_bones_from_vertex_groups"
    bl_label = "Add Bone Slots from Vertex Groups"
    bl_options = {'REGISTER'}

    mesh_index: bpy.props.IntProperty()

    @classmethod
    def poll(cls, context):
        return addon_state.asset is not None

    def execute(self, context):
        mesh = addon_state.asset.meshes[self.mesh_index]

        # Build a set of skeleton bone names for lookup
        skel_name_to_idx = {b.name: i for i, b in enumerate(addon_state.asset.bones)}

        # Build the set of skeleton indices already in this mesh's bone table
        # (including any pending additions not yet written to file)
        existing_skel_indices = set(mesh.mesh_bones.keys())
        existing_skel_indices.update(idx for idx, _ in mesh.pending_bone_additions)

        # Collect vertex group names from all imported LOD objects for this mesh
        vg_names = set()
        for li, lod in enumerate(mesh.lods):
            obj_name = lod.blender_obj_name if lod.blender_obj_name else f"{mesh.name}_LOD{li}"
            obj = bpy.data.objects.get(obj_name)
            if obj is not None and obj.type == 'MESH':
                for vg in obj.vertex_groups:
                    vg_names.add(vg.name)

        if not vg_names:
            self.report({'WARNING'}, "No vertex groups found on any imported LOD for this mesh.")
            return {'CANCELLED'}

        added = []
        reused_slots = []
        skipped_no_bone = []
        skipped_already = []
        skipped_full = []
        skipped_inv_bind = []
        # All bones here target the same mesh, so the used-slots scan only runs once.
        used_slots_cache = {}

        for vg_name in sorted(vg_names):
            skel_idx = skel_name_to_idx.get(vg_name)
            if skel_idx is None:
                skipped_no_bone.append(vg_name)
                continue
            if skel_idx in existing_skel_indices:
                skipped_already.append(vg_name)
                continue

            inv_bind = _compute_inv_bind_from_skeleton(vg_name)
            if inv_bind is None:
                skipped_inv_bind.append(vg_name)
                self.report({'WARNING'}, f"Could not compute inv_bind for '{vg_name}' - skipping.")
                continue

            status, info = _add_or_reuse_mesh_bone_slot(mesh, skel_idx, inv_bind, used_slots_cache=used_slots_cache)
            if status == 'full':
                skipped_full.append(vg_name)
                continue
            existing_skel_indices.add(skel_idx)
            added.append(vg_name)
            if status == 'reused':
                reused_slots.append((vg_name, info))

        if added:
            save_staged_state(context.scene.SWOMT)
            msg = f"Added {len(added)} bone slot(s): {', '.join(added)}"
            if reused_slots:
                msg += f". {len(reused_slots)} reused an existing unused slot"
            if skipped_inv_bind:
                msg += f". {len(skipped_inv_bind)} skipped due to inverse-bind errors"
            self.report({'INFO'}, msg)
        elif len(skipped_already) == len(vg_names):
            self.report({'INFO'}, "All vertex groups are already in the bone table.")
        else:
            self.report({'WARNING'},
                f"No new slots added. "
                f"{len(skipped_already)} already present, "
                f"{len(skipped_full)} blocked (mesh's bone slots are full and have no "
                f"unused slot to reuse), "
                f"{len(skipped_inv_bind)} inverse-bind failures, "
                f"{len(skipped_no_bone)} not in skeleton: {', '.join(skipped_no_bone[:5])}"
                + (" ..." if len(skipped_no_bone) > 5 else ""))

        return {'FINISHED'}

CLASSES = (
    RemapMeshBone, AddMeshBone, MergeSkeletons, MergeSkeletonsPickBones,
    AddBonesFromVertexGroups,
)

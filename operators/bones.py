"""Mesh-bone, skeleton merge, and posed-rest-pose operators."""

import operator
import os
from struct import pack, unpack

import bpy
from mathutils import Matrix, Vector

from .. import addon_state
from ..binary_io import br
from ..mesh_pipeline.exporter import BME
from ..mesh_pipeline.files import _mod_file_output, get_merged_mmb
from ..mesh_pipeline.importer import BMI
from ..log import logger
from ..mmb import SkeletalMeshAsset

def _compute_inv_bind_from_skeleton(bone_name):
    """
    Compute the inverse bind matrix for a bone from the currently loaded skeleton.
    """
    if addon_state.asset is None:
        return None
    bone_idx = next((i for i, b in enumerate(addon_state.asset.bones) if b.name == bone_name), None)
    if bone_idx is None:
        return None

    SWOMT = bpy.context.scene.SWOMT
    src_path = SWOMT.AssetPath
    if not os.path.isfile(src_path):
        return None

    try:
        with open(src_path, 'rb') as f:
            data = f.read()

        pos = 0
        version = data[3]
        pos += 8
        if version >= 15:
            pos += 4
        bone_count = unpack('<I', data[pos:pos+4])[0]
        pos += 4

        # Read all file_local matrices and parent indices
        file_locals = []
        parents = []
        for i in range(bone_count):
            nlen = unpack('<H', data[pos:pos+2])[0]; pos += 2
            pos += nlen  # skip name
            raw = unpack('<16f', data[pos:pos+64]); pos += 64
            parent_idx = unpack('<H', data[pos:pos+2])[0]; pos += 2
            m = Matrix([
                [raw[0], raw[4], raw[8],  raw[12]],
                [raw[1], raw[5], raw[9],  raw[13]],
                [raw[2], raw[6], raw[10], raw[14]],
                [raw[3], raw[7], raw[11], raw[15]],
            ])
            file_locals.append(m)
            parents.append(parent_idx)

        # Get file_world for target bone
        file_world_cache = [None] * bone_count
        def get_fw(i):
            if file_world_cache[i] is not None:
                return file_world_cache[i]
            if parents[i] == 65535:
                file_world_cache[i] = file_locals[i]
            else:
                file_world_cache[i] = get_fw(parents[i]) @ file_locals[i]
            return file_world_cache[i]

        fw = get_fw(bone_idx)
        try:
            inv_bind = fw.inverted()
        except ValueError:
            return None

        return tuple(inv_bind[r][c] for c in range(4) for r in range(4))

    except Exception:
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
        src_path = SWOMT.AssetPath
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
                    f"Choose a donor MMB whose mesh already uses that bone.")
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
                    f"Choose a donor MMB whose mesh already uses that bone.")
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
    src_path = SWOMT.AssetPath

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

    # Build bone data
    bone_blob = bytearray()
    for orig_idx, name, mat_raw, matrix, pidx in new_bones:
        # Name
        name_bytes = name.encode('ascii', errors='replace')
        bone_blob += pack('<H', len(name_bytes))
        bone_blob += name_bytes
        # Matrix - write the original raw bytes verbatim.
        # mat_raw is the 16 floats exactly as they appear in the donor file,
        # which is already in the correct column-first file format that br.matrix_4x4 reads.
        bone_blob += pack('<16f', *mat_raw)
        # Parent index - remap donor index to combined skeleton index
        if pidx == 65535:
            combined_pidx = 65535
        else:
            combined_pidx = donor_to_combined.get(pidx, 65535)
        bone_blob += pack('<H', combined_pidx)

    # Patch the .mmb file:
    # - Increment bone_count (uint32 at a known offset)
    # - Insert bone_blob immediately after the last existing bone

    mod_file = _mod_file_output(src_path, overwrite=SWOMT.overwrite_existing)

    with open(src_path, 'rb') as f:
        file_data = bytearray(f.read())

    version = file_data[3]
    skel_count_offset = 8
    if version >= 15:
        skel_count_offset += 4

    old_bone_count = unpack('<I', file_data[skel_count_offset:skel_count_offset+4])[0]
    new_bone_count = old_bone_count + len(new_bones)
    file_data[skel_count_offset:skel_count_offset+4] = pack('<I', new_bone_count)

    # Walk past existing bones to find the insertion point (start of mesh section)
    pos = skel_count_offset + 4
    for _ in range(old_bone_count):
        nlen = unpack('<H', file_data[pos:pos+2])[0]; pos += 2
        pos += nlen + 64 + 2 # name + matrix + parent_idx

    # pos is now at uint32 mesh_count - insert bone_blob here
    insert_at = pos
    file_data[insert_at:insert_at] = bone_blob

    inserted = len(bone_blob)

    # Update asset.size header field (bytes 4..8) - it covers the header section.
    # Skeleton is always in the header, so bump it.
    old_asset_size = unpack('<I', file_data[4:8])[0]
    file_data[4:8] = pack('<I', old_asset_size + inserted)

    # All data_offset fields in all mesh LODs must be incremented by `inserted`
    # because the skeleton has grown (it lives before the mesh data in the header).
    # We need to patch these fields in the file_data we are building.
    # We do NOT have the in-memory asset offsets updated yet, so we must re-walk
    # the mesh section in the modified file_data to find and patch them.
    #
    # Patch: scan mesh table starting after the new bone data.
    mesh_pos = insert_at + inserted # points to uint32 mesh_count
    mesh_count_val = unpack('<I', file_data[mesh_pos:mesh_pos+4])[0]
    mp = mesh_pos + 4

    for mi_scan in range(mesh_count_val):
        nlen = unpack('<H', file_data[mp:mp+2])[0]; mp += 2
        mp += nlen # mesh name
        mp += 48 + 1 # matrix + flag
        # version-specific x_count skip
        if version == 11:
            mp += 1 # skip x_count (uint8 in v11 too but parsed differently)
            x_count = unpack('<H', file_data[mp:mp+2])[0]; mp += 2
            mp += 4 * x_count
        else:
            x_count = file_data[mp]; mp += 1
            mp += 1 + 4 * x_count
        u_count = unpack('<H', file_data[mp:mp+2])[0]; mp += 2
        for _ in range(u_count):
            mp += 64 # matrix
            mp += 2 # skeleton index
        # Pre-LOD bytes
        if u_count > 0 and version not in (11, 12):
            mp += 1 if version == 13 else 2 # root_bone_index
            lod_info_type = file_data[mp]; mp += 1
        else:
            if version in (11, 12, 13):
                lod_info_type = 0
            else:
                lod_info_type = file_data[mp]; mp += 1
        lod_count_scan = file_data[mp]; mp += 1
        mp += 4 # unknown 4 bytes
        lod_field_size = 40 if version == 11 else 36
        for li_scan in range(lod_count_scan):
            lod_start = mp
            # data_offset field is at byte offset 24 within the 36-byte LOD header
            # (or 28 in v11 due to the extra uint32)
            lod_fo = 4 if version == 11 else 0 # lod_field_offset
            do_field = lod_start + 4 + lod_fo + 20 # start+vc(4)+lod_fo+ic(4)+sa(4)+voa(4)+vob(4)
            old_do = unpack('<I', file_data[do_field:do_field+4])[0]
            file_data[do_field:do_field+4] = pack('<I', old_do + inserted)
            # Shift the second-section absolute offset (field [5], data_offset field + 32;
            # see Mesh.parse) too.
            if lod_info_type == 2:
                sp = do_field + 32
                sec2_val = unpack('<I', file_data[sp:sp+4])[0]
                if sec2_val > 0:
                    file_data[sp:sp+4] = pack('<I', sec2_val + inserted)
            # Also update the in-memory lod.data_offset_file_pos value - we patch the file
            # positions here, and will sync asset afterwards.
            mp += lod_field_size
            if lod_info_type == 2:
                mp += 28
        # Tail section - skip to next mesh
        uv_count_scan = file_data[mp]; mp += 1
        mp += 4 * uv_count_scan
        if version == 11:
            pass # no color_count in v11
        elif version in (16, 17):
            cc = file_data[mp]; mp += 1
            mp += 4 * cc + 4
            count_c = file_data[mp]; mp += 1
            mp += 4 * count_c
        else:
            mp += 4 # unk
            cc = file_data[mp]; mp += 1
            mp += 4 * cc
        mp += 4 # vs + ns (uint16 each)
        mp += 20 if version == 17 else 16 # post-stride skip

    # Write the patched file
    with open(mod_file, 'wb') as f:
        f.write(file_data)

    # Update in-memory asset.bones so the rest of the session works correctly
    for orig_idx, name, mat_raw, matrix, pidx in new_bones:
        if pidx == 65535:
            combined_pidx = 65535
        else:
            combined_pidx = donor_to_combined.get(pidx, 65535)
        new_b = SkeletalMeshAsset.Bone.__new__(SkeletalMeshAsset.Bone)
        new_b.name = name
        new_b.matrix = matrix
        new_b.parent_index = combined_pidx
        addon_state.asset.bones.append(new_b)
    addon_state.asset.bone_count = len(addon_state.asset.bones)

    # Sync lod.data_offset values in memory (they were bumped in the file)
    for m_mem in addon_state.asset.meshes:
        for lod_mem in m_mem.lods:
            lod_mem.data_offset += inserted
            lod_mem.data_offset_file_pos += inserted
            lod_mem.start_offset += inserted

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

    SWOMT.AssetPath = mod_file
    operator.report({'INFO'},
        f"Merged {len(new_bones)} new bone(s) ({mode_label}) from '{os.path.basename(src_filepath)}' "
        f"into '{os.path.basename(mod_file)}'. "
        f"Skeleton now has {len(addon_state.asset.bones)} bones.")
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
    """Choose which donor bones to merge (shown after selecting the donor .mmb)"""
    bl_idname = "object.merge_skeletons_pick_bones"
    bl_label = "Choose Bones to Merge"
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

    # New donor bones are appended to the skeleton (via a _MOD copy) and to asset.bones
    # in memory. The armature is rebuilt so Add/Remap Bone can reference them immediately.

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
            msg = f"Added {len(added)} bone slot(s): {', '.join(added)}"
            if reused_slots:
                msg += f". {len(reused_slots)} reused an existing unused slot"
            self.report({'INFO'}, msg)
        elif skipped_already and not skipped_no_bone and not skipped_full:
            self.report({'INFO'}, "All vertex groups are already in the bone table.")
        else:
            missing = [n for n in sorted(vg_names) if n not in skel_name_to_idx]
            self.report({'WARNING'},
                f"No new slots added. "
                f"{len(skipped_already)} already present, "
                f"{len(skipped_full)} blocked (mesh's bone slots are full and have no "
                f"unused slot to reuse), "
                f"{len(skipped_no_bone)} not in skeleton: {', '.join(skipped_no_bone[:5])}"
                + (" ..." if len(skipped_no_bone) > 5 else ""))

        return {'FINISHED'}

class ExportPosedBoneMatrices(bpy.types.Operator):
    """Export the armature's current pose. Use this after posing the armature in 'Pose Mode'."""
    bl_idname = "object.export_posed_bone_matrices"
    bl_label = "Export Pose as New Rest Pose"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        if addon_state.asset is None:
            return False
        arm_obj = bpy.data.objects.get(addon_state.asset.name)
        return arm_obj is not None and arm_obj.type == 'ARMATURE'

    def execute(self, context):
        SWOMT = context.scene.SWOMT
        src_path = SWOMT.AssetPath
        arm_obj = bpy.data.objects.get(addon_state.asset.name)

        if arm_obj is None or arm_obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Armature object not found in scene.")
            return {'CANCELLED'}

        # Build pose_bone armature-local matrices for every bone.
        arm_pose_local = {} # bone_name -> Matrix (armature-local)
        for pb in arm_obj.pose.bones:
            arm_pose_local[pb.name] = pb.matrix.copy()

        S = Matrix.Scale(-1.0, 4, Vector((1.0, 0.0, 0.0))) # diag(-1,1,1,1)

        def pose_to_file_world(pose_mat):
            """Convert pose_bone.matrix (Blender armature-local) to MMB file-space world matrix."""
            return S @ pose_mat

        # Determine which bones are actually posed (differ from rest pose).
        POSE_EPSILON = 1e-4

        def _is_posed(pb):
            """Return True if this bone's pose differs from its rest."""
            rest_al = pb.bone.matrix_local
            pose_al = pb.matrix
            for r in range(4):
                for c in range(4):
                    if abs(pose_al[r][c] - rest_al[r][c]) > POSE_EPSILON:
                        return True
            return False

        posed_bones = {pb.name for pb in arm_obj.pose.bones if _is_posed(pb)}

        # Patch skeleton section in a copy of the file bytes.
        mod_file = _mod_file_output(src_path, overwrite=SWOMT.overwrite_existing)

        with open(src_path, 'rb') as f:
            file_data = bytearray(f.read())

        patched_skel = 0
        # Re-walk skeleton section to get file offsets (not stored on asset.bones)
        pos = 0
        version = file_data[3]
        pos += 8
        if version >= 15:
            pos += 4

        bone_count = unpack('<I', bytes(file_data[pos:pos+4]))[0]
        pos += 4

        bone_mat_offsets = [] # file offset of each bone's 64-byte matrix
        bone_parents = []
        for i in range(bone_count):
            nlen = unpack('<H', bytes(file_data[pos:pos+2]))[0]
            pos += 2
            pos += nlen  # skip name
            bone_mat_offsets.append(pos)
            pos += 64  # matrix
            parent_idx = unpack('<H', bytes(file_data[pos:pos+2]))[0]
            bone_parents.append(parent_idx)
            pos += 2

        # Build bone name -> index map
        bone_name_to_idx = {b.name: i for i, b in enumerate(addon_state.asset.bones)}

        # Build file_world for every bone.
        orig_file_local = {}
        _pos2 = 0
        _pos2 += 8
        if file_data[3] >= 15:
            _pos2 += 4
        _bc2 = unpack('<I', bytes(file_data[_pos2:_pos2+4]))[0]
        _pos2 += 4
        for _i in range(_bc2):
            _nlen = unpack('<H', bytes(file_data[_pos2:_pos2+2]))[0]
            _pos2 += 2
            _bname = bytes(file_data[_pos2:_pos2+_nlen]).decode('ascii', errors='replace')
            _pos2 += _nlen
            _raw = unpack('<16f', bytes(file_data[_pos2:_pos2+64]))
            _pos2 += 64
            _pos2 += 2
            orig_file_local[_bname] = Matrix([
                [_raw[0], _raw[4], _raw[8],  _raw[12]],
                [_raw[1], _raw[5], _raw[9],  _raw[13]],
                [_raw[2], _raw[6], _raw[10], _raw[14]],
                [_raw[3], _raw[7], _raw[11], _raw[15]],
            ])

        orig_file_world = {}
        for _i, _b in enumerate(addon_state.asset.bones):
            _pidx = bone_parents[_i]
            if _pidx == 65535:
                orig_file_world[_b.name] = orig_file_local.get(_b.name, Matrix.Identity(4))
            else:
                _pname = addon_state.asset.bones[_pidx].name
                orig_file_world[_b.name] = orig_file_world.get(_pname, Matrix.Identity(4)) @ orig_file_local.get(_b.name, Matrix.Identity(4))

        file_world = {}
        for pb in arm_obj.pose.bones:
            if pb.name in posed_bones:
                file_world[pb.name] = pose_to_file_world(pb.matrix)
            else:
                file_world[pb.name] = orig_file_world.get(pb.name, pose_to_file_world(pb.matrix))

        for bone_name, fw in file_world.items():
            # Only write bones that were actually posed
            if bone_name not in posed_bones:
                continue
            bi = bone_name_to_idx.get(bone_name)
            if bi is None:
                continue
            parent_idx = bone_parents[bi]
            if parent_idx == 65535:
                parent_fw = Matrix.Identity(4)
            else:
                parent_bone_name = addon_state.asset.bones[parent_idx].name
                parent_fw = file_world.get(parent_bone_name, Matrix.Identity(4))

            try:
                new_local = parent_fw.inverted() @ fw
            except ValueError:
                continue

            # The translation delta in file_local must be negated relative to the original.
            orig_local = orig_file_local.get(bone_name)
            if orig_local is not None:
                orig_trans = orig_local.col[3].copy()
                new_trans  = new_local.col[3].copy()
                delta = new_trans - orig_trans
                new_local.col[3] = orig_trans - delta

            # Write new local matrix (row-major 4x4 floats) to file_data
            flat = [new_local[r][c] for c in range(4) for r in range(4)]
            offset = bone_mat_offsets[bi]
            file_data[offset:offset+64] = pack('<16f', *flat)
            patched_skel += 1

        # Patch mesh bone slot (inv_bind) matrices.
        # inv_bind = file_world_matrix.inverted()
        patched_slots = 0
        for mi, mesh in enumerate(addon_state.asset.meshes):
            mesh_bones_list = list(mesh.mesh_bones.keys())
            for slot_idx, skel_idx in enumerate(mesh_bones_list):
                if skel_idx >= len(addon_state.asset.bones):
                    continue
                bone_name = addon_state.asset.bones[skel_idx].name
                # Only update inv_bind for bones that were actually posed
                if bone_name not in posed_bones:
                    continue
                fw = file_world.get(bone_name)
                if fw is None:
                    continue
                try:
                    new_inv_bind = fw.inverted()
                except ValueError:
                    self.report({'WARNING'}, f"Could not invert matrix for bone '{bone_name}' - skipping.")
                    continue
                flat = tuple(new_inv_bind[r][c] for c in range(4) for r in range(4))
                mesh.pending_bone_remaps[slot_idx] = (skel_idx, flat)
                patched_slots += 1

        if patched_skel == 0 and patched_slots == 0:
            self.report({'WARNING'}, "No bones were updated. Was the armature posed?")
            return {'CANCELLED'}

        # Write the skeleton-patched file_data to disk first
        with open(mod_file, 'wb') as f:
            f.write(file_data)

        # Then apply mesh inv_bind patches via the existing header-patch path.
        # _apply_header_patches writes bone remaps directly into the file on disk.
        for mesh in addon_state.asset.meshes:
            BME._apply_header_patches(mod_file, mesh, addon_state.asset, operator=self)

        SWOMT.AssetPath = mod_file
        self.report({'INFO'},
            f"Pose exported: {patched_skel} skeleton bone(s) ({len(posed_bones)} posed), "
            f"{patched_slots} inv_bind slot(s) -> {os.path.basename(mod_file)}")
        return {'FINISHED'}

CLASSES = (
    RemapMeshBone, AddMeshBone, MergeSkeletons, MergeSkeletonsPickBones,
    AddBonesFromVertexGroups, ExportPosedBoneMatrices,
)

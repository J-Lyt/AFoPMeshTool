"""LOD, mesh, UV, and normals operators."""

import bpy

from .. import addon_state
from ..mesh_pipeline.exporter import BME

def _max_weights_for_mesh(mesh, lod0_obj=None):
    """Safe generated-LOD limit: source usage capped by declared capacity."""
    capacity = mesh.influence_capacity()
    if lod0_obj is None and mesh.lods:
        lod0 = mesh.lods[0]
        lod0_obj = BME.find_object_by_name(
            lod0.blender_obj_name or f"{mesh.name}_LOD0")
    if lod0_obj is not None:
        source_limit = lod0_obj.get("mmb_source_influence_limit")
        if isinstance(source_limit, int) and source_limit > 0:
            return min(capacity, source_limit)
    return capacity

def _limit_vertex_weights(obj, limit):
    """Keep the highest vertex-group weights per vertex, removing the lowest and
    normalizing the remaining. Returns the number of vertices that were trimmed."""
    trimmed = 0
    for v in obj.data.vertices:
        entries = [(g.group, g.weight) for g in v.groups]
        if len(entries) <= limit:
            continue
        entries.sort(key=lambda e: e[1], reverse=True)
        for gi, _w in entries[limit:]:
            obj.vertex_groups[gi].remove([v.index])
        keep = entries[:limit]
        total = sum(w for _gi, w in keep)
        if total > 0:
            for gi, w in keep:
                obj.vertex_groups[gi].add([v.index], w / total, 'REPLACE')
        trimmed += 1
    return trimmed

class GenerateLODs(bpy.types.Operator):
    """Generate LOD meshes by decimating the imported LOD0 mesh"""
    bl_idname = 'object.generate_lods'
    bl_label = 'Generate LODs'
    bl_options = {'REGISTER', 'UNDO'}

    mesh_index: bpy.props.IntProperty(options={'HIDDEN'})
    ratio_lod1: bpy.props.FloatProperty(
        name="LOD1 Ratio", default=0.5, min=0.01, max=1.0,
        description="Decimate ratio for LOD1, relative to LOD0")
    ratio_lod2: bpy.props.FloatProperty(
        name="LOD2 Ratio", default=0.25, min=0.01, max=1.0,
        description="Decimate ratio for LOD2, relative to LOD0")
    ratio_lod3: bpy.props.FloatProperty(
        name="LOD3 Ratio", default=0.125, min=0.01, max=1.0,
        description="Decimate ratio for LOD3, relative to LOD0")
    replace_existing: bpy.props.BoolProperty(
        name="Replace Existing LOD Objects", default=True,
        description="Replace LOD 1-3 objects already in the scene")

    @classmethod
    def poll(cls, context):
        return addon_state.asset is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        mesh = addon_state.asset.meshes[self.mesh_index]
        layout.label(text=f"Generate LODs for '{mesh.name}' from LOD0")
        layout.label(
            text=f"Max weights per vertex: {_max_weights_for_mesh(mesh)} "
                 "(source LOD0 / declaration)")
        for li in range(1, min(len(mesh.lods), 4)):
            layout.prop(self, f"ratio_lod{li}")
        layout.prop(self, "replace_existing")

    def execute(self, context):
        if bpy.context.active_object and bpy.context.active_object.mode == 'EDIT':
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        with bpy.context.temp_override(window=window, area=area):
                            bpy.ops.object.mode_set(mode='OBJECT')
                        break

        mesh = addon_state.asset.meshes[self.mesh_index]
        if len(mesh.lods) < 2:
            self.report({'ERROR'}, f"'{mesh.name}' has no LOD1+ entries in the MMB.")
            return {'CANCELLED'}

        lod0 = mesh.lods[0]
        lod0_obj = BME.find_object_by_name(lod0.blender_obj_name or f"{mesh.name}_LOD0")
        if lod0_obj is None or len(lod0_obj.data.vertices) == 0:
            self.report({'ERROR'}, f"Import '{mesh.name}' LOD0 first.")
            return {'CANCELLED'}

        # Place generated LODs in the same collection as LOD0
        if lod0_obj.users_collection:
            collection = lod0_obj.users_collection[0]
        else:
            collection = context.scene.collection

        max_w = _max_weights_for_mesh(mesh, lod0_obj)
        ratios = {1: self.ratio_lod1, 2: self.ratio_lod2, 3: self.ratio_lod3}
        created, skipped = [], []

        for li in range(1, min(len(mesh.lods), 4)):
            lod = mesh.lods[li]
            target_name = f"{mesh.name}_LOD{li}"

            existing = bpy.data.objects.get(lod.blender_obj_name) if lod.blender_obj_name else None
            if existing is None:
                existing = bpy.data.objects.get(target_name)
            if existing is not None:
                if not self.replace_existing:
                    skipped.append(target_name)
                    continue
                old_data = existing.data
                bpy.data.objects.remove(existing, do_unlink=True)
                if old_data is not None and old_data.users == 0:
                    bpy.data.meshes.remove(old_data)

            # Duplicate LOD0: obj.copy() keeps the parent, matrix_parent_inverse,
            # modifiers (incl. Armature) and vertex groups.
            new_obj = lod0_obj.copy()
            collection.objects.link(new_obj)

            # Strip every other modifier so only the decimate modifier is baked.
            # (A visible armature modifier would bake the current pose, and hiding it via show_viewport
            # would instead make the depsgraph return the original un-modified mesh.)
            #
            # Strip BEFORE creating the decimate modifier - as removing collection items
            # invalidates modifier references.
            #
            # Armature modifiers are re-created after the bake.
            arm_mods = [(m.name, m.object) for m in new_obj.modifiers
                        if m.type == 'ARMATURE']
            for m in list(new_obj.modifiers):
                new_obj.modifiers.remove(m)

            dec = new_obj.modifiers.new(name="_gen_lod_decimate", type='DECIMATE')
            dec.decimate_type = 'COLLAPSE'
            dec.ratio = ratios[li]
            dec.use_collapse_triangulate = True
            try:
                dg = context.evaluated_depsgraph_get()
                eval_obj = new_obj.evaluated_get(dg)
                # preserve_all_data_layers keeps vertex groups, UVs, colors and the
                # mmb_vertex_order attribute through the decimate.
                new_data = bpy.data.meshes.new_from_object(
                    eval_obj, preserve_all_data_layers=True, depsgraph=dg)
            finally:
                dec = new_obj.modifiers.get("_gen_lod_decimate")
                if dec is not None:
                    new_obj.modifiers.remove(dec)
                for m_name, m_target in arm_mods:
                    am = new_obj.modifiers.new(name=m_name, type='ARMATURE')
                    am.object = m_target

            new_obj.data = new_data
            new_obj.name = target_name
            new_obj.data.name = target_name

            # Generated LODs carry LOD0 original indices in mmb_vertex_order;
            # consumers (e.g. the .mcloth rewriter) pick their source data by this.
            new_obj["mmb_lod_source"] = 0

            trimmed = _limit_vertex_weights(new_obj, max_w)
            lod.blender_obj_name = new_obj.name
            created.append((li, len(new_data.vertices), trimmed))

        if not created and skipped:
            self.report({'WARNING'},
                f"Failed to generate - objects already exist: {', '.join(skipped)}. "
                f"Enable 'Replace Existing LOD Objects' to overwrite them.")
            return {'CANCELLED'}

        parts = ', '.join(f"LOD{li} ({vc} verts, {tr} trimmed)" for li, vc, tr in created)
        msg = f"Generated {len(created)} LOD(s) for '{mesh.name}' [max {max_w} weights/vert]: {parts}"
        if skipped:
            msg += f" | Skipped existing: {', '.join(skipped)}"
        self.report({'INFO'}, msg)
        return {'FINISHED'}

class RemoveMesh(bpy.types.Operator):
    """Zero out all vertex positions for all LODs of this mesh."""
    bl_idname = "object.remove_mesh"
    bl_label = "Remove Mesh"

    mesh_index: bpy.props.IntProperty()

    @classmethod
    def poll(cls, context):
        return addon_state.asset is not None

    def execute(self, context):
        mesh = addon_state.asset.meshes[self.mesh_index]
        mesh.zeroed_out_in_session = True

        # Remove any already-imported Blender objects in the viewport
        for li, lod in enumerate(mesh.lods):
            obj_name = lod.blender_obj_name if lod.blender_obj_name else f"{mesh.name}_LOD{li}"
            obj = bpy.data.objects.get(obj_name)
            if obj is not None:
                bpy.data.objects.remove(obj, do_unlink=True)

        self.report({'INFO'}, f"'{mesh.name}' ({len(mesh.lods)} LOD(s)) will have their vertex positions zeroed out on Export.")
        return {'FINISHED'}

class RevertMesh(bpy.types.Operator):
    """Revert vertex positions for all LODs of this mesh."""
    bl_idname = "object.revert_mesh"
    bl_label = "Revert Mesh"

    mesh_index: bpy.props.IntProperty()

    @classmethod
    def poll(cls, context):
        return addon_state.asset is not None

    def execute(self, context):
        mesh = addon_state.asset.meshes[self.mesh_index]

        if not mesh.zeroed_out_in_session:
            self.report({'WARNING'}, "Mesh was not zeroed out in this session.")
            return {'CANCELLED'}

        # Remove any already-imported Blender objects in the viewport
        for li, lod in enumerate(mesh.lods):
            obj_name = lod.blender_obj_name if lod.blender_obj_name else f"{mesh.name}_LOD{li}"
            obj = bpy.data.objects.get(obj_name)
            if obj is not None:
                bpy.data.objects.remove(obj, do_unlink=True)

        mesh.zeroed_out_in_session = False
        self.report({'INFO'}, f"Reverted '{mesh.name}' ({len(mesh.lods)} LOD(s)) to original positions.")
        return {'FINISHED'}

def _scale_uvs_uv_items(self, context):
    """Populate UV set enum from the mesh's uv_count."""
    if addon_state.asset is None:
        return []
    try:
        mesh = addon_state.asset.meshes[self.mesh_index]
    except (IndexError, AttributeError):
        return []
    return [(str(i), f"UVMap_{i}", f"UV set {i}") for i in range(mesh.uv_count)]

def _scale_u_update(self, context):
    if self.link:
        self["scale_v"] = self.scale_u

def _scale_v_update(self, context):
    if self.link:
        self["scale_u"] = self.scale_v

class ScaleUVs(bpy.types.Operator):
    """Scale a UV layer on all imported LODs of this mesh"""
    bl_idname = "object.scale_uvs"
    bl_label = "Scale UVs"

    mesh_index: bpy.props.IntProperty()
    uv_set: bpy.props.EnumProperty(
        name="UV Set",
        description="Which UV map to scale",
        items=_scale_uvs_uv_items,
    )
    scale_u: bpy.props.FloatProperty(name="Scale U", default=1.0, min=0.001, max=1000.0, step=100, update=_scale_u_update)
    scale_v: bpy.props.FloatProperty(name="Scale V", default=1.0, min=0.001, max=1000.0, step=100, update=_scale_v_update)
    link: bpy.props.BoolProperty(name="Link", default=True, description="Link Scale U and Scale V together")
    pivot_u: bpy.props.FloatProperty(name="Pivot U", default=0.0, step=100, description="U coordinate to scale from")
    pivot_v: bpy.props.FloatProperty(name="Pivot V", default=1.0, step=100, description="V coordinate to scale from")

    @classmethod
    def poll(cls, context):
        return addon_state.asset is not None

    def invoke(self, context, event):
        self.scale_u = 1.0
        self.scale_v = 1.0
        self.link = True
        self.pivot_u = 0.0
        self.pivot_v = 1.0
        context.window.cursor_warp(context.window.width // 2, context.window.height // 2)
        return context.window_manager.invoke_props_dialog(self, width=260)

    def draw(self, context):
        mesh = addon_state.asset.meshes[self.mesh_index]
        layout = self.layout
        layout.label(text=f"Mesh: {mesh.name}")
        if mesh.uv_count > 1:
            layout.prop(self, "uv_set")
        row = layout.row(align=True)
        row.prop(self, "scale_u")
        row.prop(self, "link", text="", icon='LINKED' if self.link else 'UNLINKED')
        row.prop(self, "scale_v")
        pivot_row = layout.row(align=True)
        pivot_row.prop(self, "pivot_u")
        pivot_row.prop(self, "pivot_v")

    def execute(self, context):
        # Exit edit mode first - UV data changes on obj.data are not reflected while the object is being edited.
        if context.active_object and context.active_object.mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')

        mesh = addon_state.asset.meshes[self.mesh_index]
        uv_index = int(self.uv_set) if self.uv_set else 0
        layer_name = f"UVMap_{uv_index}"
        affected = 0
        seen_data = set() # guard against LODs that share the same mesh data block
        for li, lod in enumerate(mesh.lods):
            obj_name = lod.blender_obj_name if lod.blender_obj_name else f"{mesh.name}_LOD{li}"
            obj = bpy.data.objects.get(obj_name)
            if obj is None or obj.type != 'MESH':
                continue
            if obj.data.name in seen_data:
                continue
            seen_data.add(obj.data.name)
            uv_layer = obj.data.uv_layers.get(layer_name)
            if uv_layer is None:
                continue
            # Scale from the chosen pivot: new = pivot + (uv - pivot) * scale
            pu, pv = self.pivot_u, self.pivot_v
            for loop_uv in uv_layer.data:
                loop_uv.uv = (pu + (loop_uv.uv[0] - pu) * self.scale_u,
                              pv + (loop_uv.uv[1] - pv) * self.scale_v)
            obj.data.update()
            affected += 1
        if affected == 0:
            self.report({'WARNING'}, f"No imported LODs found for '{mesh.name}' with UV layer '{layer_name}'")
        else:
            self.report({'INFO'}, f"Scaled {layer_name} by ({self.scale_u}, {self.scale_v}) from pivot ({self.pivot_u}, {self.pivot_v}) on {affected} LOD(s) of '{mesh.name}'")
        return {'FINISHED'}


class RenameMesh(bpy.types.Operator):
    """Rename a mesh and patch its name in the .mmb file in place"""
    bl_idname = "object.rename_mesh"
    bl_label = "Rename Mesh"

    mesh_index: bpy.props.IntProperty()

    @classmethod
    def poll(cls, context):
        return addon_state.asset is not None

    def invoke(self, context, event):
        old_name = addon_state.asset.meshes[self.mesh_index].name
        # Re-register a Scene property with maxlen set to the original name length.
        # Blender enforces maxlen natively in the text widget — this is the only
        # reliable way to hard-lock the input length dynamically.
        bpy.types.Scene.mmb_rename_input = bpy.props.StringProperty(
            name="New Name",
            maxlen=len(old_name),
        )
        context.scene.mmb_rename_input = old_name
        # Center the dialog on screen
        context.window.cursor_warp(context.window.width // 2, context.window.height // 2)
        return context.window_manager.invoke_props_dialog(self, width=380)

    def draw(self, context):
        mesh = addon_state.asset.meshes[self.mesh_index]
        layout = self.layout
        layout.label(text=f"Original: {mesh.name}")
        layout.label(text=f"Max length: {len(mesh.name)} characters")
        layout.prop(context.scene, "mmb_rename_input", text="New Name")

    def execute(self, context):
        mesh = addon_state.asset.meshes[self.mesh_index]
        old_name = mesh.name
        new_name = context.scene.mmb_rename_input.strip()

        if len(new_name) == 0:
            self.report({'ERROR'}, "Name cannot be empty.")
            return {'CANCELLED'}

        if len(new_name) > len(old_name):
            self.report({'ERROR'}, f"Name too long. Max {len(old_name)} characters.")
            return {'CANCELLED'}

        # Stage the rename — original .mmb is never touched.
        # The patch is written to the _MOD copy when the user exports a LOD.
        mesh.pending_rename_new = new_name
        mesh.name = new_name

        # Rename any already-imported Blender objects so Export can still find them
        for li, lod in enumerate(mesh.lods):
            old_obj_name = lod.blender_obj_name if lod.blender_obj_name else f"{old_name}_LOD{li}"
            new_obj_name = f"{new_name}_LOD{li}"
            obj = bpy.data.objects.get(old_obj_name)
            if obj is not None:
                obj.name = new_obj_name
                if obj.data is not None:
                    obj.data.name = new_obj_name
                lod.blender_obj_name = obj.name  # use obj.name in case Blender de-duped it

        return {'FINISHED'}

class ComputeNormals(bpy.types.Operator):
    """Compute normals for the selected mesh object (Original normals are preserved on export)"""
    bl_idname = "object.compute_normals"
    bl_label = "Compute Normals"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
                obj is not None and
                obj.type == 'MESH' and
                obj.select_get()
        )

    def execute(self, context):
        obj = context.active_object
        BME._compute_normals_for_object(obj)
        self.report({'INFO'}, f"Computed normals for {obj.name}.")
        return {'FINISHED'}

class ClearNormals(bpy.types.Operator):
    """Clear normals for the selected mesh object (Original normals are preserved on export)"""
    bl_idname = "object.clear_custom_normals"
    bl_label = "Clear Normals"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
                obj is not None and
                obj.type == 'MESH' and
                obj.select_get() and
                obj.data.has_custom_normals
        )

    def execute(self, context):
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
        return {'FINISHED'}

CLASSES = (
    GenerateLODs, RemoveMesh, RevertMesh, ScaleUVs, RenameMesh,
    ComputeNormals, ClearNormals,
)

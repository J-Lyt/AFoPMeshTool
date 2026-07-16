"""MMB load, import, export, and batch operators."""

import os
from pathlib import Path

import bpy

from . import addon_state
from .cloth_export import _export_mcloth_for_asset
from .exporter import BME
from .file_utils import _mod_file_output, get_merged_mmb
from .importer import BMI
from .mmb import SkeletalMeshAsset

class BrowseMMBFile(bpy.types.Operator):
    """Load a .mmb file"""
    bl_idname = "object.browse_mmb_file"
    bl_label = "Import .mmb"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.mmb", options={'HIDDEN'})

    def invoke(self, context, event):
        current = context.scene.SWOMT.AssetPath
        if current:
            self.filepath = current
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        context.scene.SWOMT.AssetPath = self.filepath
        return {'FINISHED'}

class LoadMMB(bpy.types.Operator):
    """Reads data from base .mmb file"""
    bl_idname = "object.load_mmb"
    bl_label = "Load"

    def execute(self,context):
        SWOMT = context.scene.SWOMT
        with open(SWOMT.AssetPath, 'rb') as file:
            sk_mesh = SkeletalMeshAsset()
            sk_mesh.parse(file)
            sk_mesh.name = Path(SWOMT.AssetPath).stem
            addon_state.asset = sk_mesh

        return {'FINISHED'}

class ImportLOD(bpy.types.Operator):
    """Imports the given LOD"""
    bl_idname = 'object.import_lod'
    bl_label = 'Import'

    mesh_index: bpy.props.IntProperty()
    lod_index: bpy.props.IntProperty()

    @classmethod
    def poll(cls,context):
        return addon_state.asset is not None

    def execute(self,context):
        sk_mesh = addon_state.asset
        mesh = sk_mesh.meshes[self.mesh_index]
        lod = mesh.lods[self.lod_index]
        SWOMT = context.scene.SWOMT
        merged_mmb = get_merged_mmb(SWOMT["AssetPath"])
        obj = BMI.import_mesh(merged_mmb,
                              skeletal_mesh=sk_mesh,
                              mesh=mesh,
                              lod_index=lod.index)
        is_new_armature = bpy.data.objects.find(sk_mesh.name) == -1
        armature = BMI.find_or_create_skeleton(sk_mesh)
        BMI.parent_obj_to_armature(obj, armature, mesh)
        if is_new_armature:
            BMI.rotate_model(obj,armature)
        return {'FINISHED'}

class ExportLOD(bpy.types.Operator):
    """Exports the given LOD"""
    bl_idname = 'object.export_lod'
    bl_label = 'Export'

    mesh_index: bpy.props.IntProperty()
    lod_index: bpy.props.IntProperty()

    @classmethod
    def poll(cls,context):
        return addon_state.asset is not None

    def execute(self,context):
        if bpy.context.active_object and bpy.context.active_object.mode == 'EDIT':
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        with bpy.context.temp_override(window=window, area=area):
                            bpy.ops.object.mode_set(mode='OBJECT')
                        break

        SWOMT = context.scene.SWOMT
        src_path = SWOMT.AssetPath
        mod_file = _mod_file_output(src_path, overwrite=SWOMT.overwrite_existing)

        # Triangulate the Blender mesh before export so all faces are tris
        mesh = addon_state.asset.meshes[self.mesh_index]
        lod  = mesh.lods[self.lod_index]
        lod_obj_name = lod.blender_obj_name or f"{mesh.name}_LOD{self.lod_index}"
        tri_obj = BME.find_object_by_name(lod_obj_name)
        original_data = {}

        if tri_obj:
            original_data = tri_obj.data
            BME._triangulate_object(tri_obj,
                                    compute_normals=context.scene.SWOMT.compute_normals_on_export)

        try:
            BME._write_mod_file(
                edited_lod_index_per_mesh={self.mesh_index: self.lod_index},
                out_path=mod_file,
            )
        except Exception as e:
            if tri_obj:
                tri_obj.data = original_data
            self.report({'ERROR'}, f"Export failed: {e}")
            return {'CANCELLED'}
        finally:
            # Always restore the original mesh data so the user's mesh is untouched
            if tri_obj:
                export_data = tri_obj.data
                tri_obj.data = original_data
                if export_data != original_data:
                    bpy.data.meshes.remove(export_data)

        # Apply header-level patches for every mesh
        for mesh in addon_state.asset.meshes:
            BME._apply_header_patches(mod_file, mesh, addon_state.asset, operator=self)

        # A single oversized LOD makes the whole mesh uint32. Widen every
        # sibling LOD and set the per-mesh tail flag before cloth processing.
        try:
            BME.promote_mixed_index_widths(mod_file)
        except Exception as e:
            self.report({'ERROR'}, f"uint32 index widening failed: {e}")
            return {'CANCELLED'}

        # Update only when final exported geometry exceeds existing culling bounds.
        try:
            BME.expand_mesh_bounds(mod_file, {self.mesh_index})
        except Exception as e:
            self.report({'ERROR'}, f"mesh bounds update failed: {e}")
            return {'CANCELLED'}

        # Rewrite the paired .mcloth (if any) so the cloth vertex mapping matches this export
        _export_mcloth_for_asset(mod_file, operator=self)

        # Apply staged file rename
        if addon_state.asset.pending_file_rename_new:
            new_file = str(Path(mod_file).parent / (addon_state.asset.pending_file_rename_new + '.mmb'))
            try:
                os.replace(mod_file, new_file)
                _mc_out = os.path.splitext(mod_file)[0] + '.mcloth'
                if os.path.isfile(_mc_out):
                    os.replace(_mc_out, os.path.splitext(new_file)[0] + '.mcloth')
            except Exception as e:
                self.report({'ERROR'}, f"Failed to rename mod file: {e}")
                return {'FINISHED'}
            SWOMT.AssetPath = new_file
            self.report({'INFO'}, f"Exported -> {os.path.basename(new_file)}")
        else:
            SWOMT.AssetPath = mod_file

        return {'FINISHED'}

def _import_all_lods(context, lod_n):
    """Shared import logic for ImportAllLODNs operators."""
    sk_mesh = addon_state.asset
    SWOMT = context.scene.SWOMT
    merged_mmb = get_merged_mmb(SWOMT["AssetPath"])
    is_new_armature = bpy.data.objects.find(sk_mesh.name) == -1
    armature = BMI.find_or_create_skeleton(sk_mesh)
    last_obj = None
    for mi, mesh in enumerate(sk_mesh.meshes):
        if len(mesh.lods) <= lod_n:
            continue
        lod = mesh.lods[lod_n]
        obj = BMI.import_mesh(merged_mmb,
                              skeletal_mesh=sk_mesh,
                              mesh=mesh,
                              lod_index=lod.index)
        BMI.parent_obj_to_armature(obj, armature, mesh)
        last_obj = obj
    if is_new_armature and last_obj is not None:
        BMI.rotate_model(last_obj, armature)
    return {'FINISHED'}

class ImportAllLOD0s(bpy.types.Operator):
    """Imports LOD0 for every mesh in the asset"""
    bl_idname = 'object.import_all_lod0s'
    bl_label = "Import All LOD0's"

    @classmethod
    def poll(cls, context):
        return addon_state.asset is not None

    def execute(self, context):
        return _import_all_lods(context, 0)

class ImportAllLOD1s(bpy.types.Operator):
    """Imports LOD1 for every mesh in the asset"""
    bl_idname = 'object.import_all_lod1s'
    bl_label = "Import All LOD1's"

    @classmethod
    def poll(cls, context):
        return addon_state.asset is not None

    def execute(self, context):
        return _import_all_lods(context, 1)

class ImportAllLOD2s(bpy.types.Operator):
    """Imports LOD2 for every mesh in the asset"""
    bl_idname = 'object.import_all_lod2s'
    bl_label = "Import All LOD2's"

    @classmethod
    def poll(cls, context):
        return addon_state.asset is not None

    def execute(self, context):
        return _import_all_lods(context, 2)

class ImportAllLOD3s(bpy.types.Operator):
    """Imports LOD3 for every mesh in the asset"""
    bl_idname = 'object.import_all_lod3s'
    bl_label = "Import All LOD3's"

    @classmethod
    def poll(cls, context):
        return addon_state.asset is not None

    def execute(self, context):
        return _import_all_lods(context, 3)

class ExportAllLODs(bpy.types.Operator):
    """Exports every LOD that has a Blender object in the scene"""
    bl_idname = 'object.export_all_lods'
    bl_label = "Export All LODs"

    @classmethod
    def poll(cls, context):
        return addon_state.asset is not None

    def execute(self, context):
        if bpy.context.active_object and bpy.context.active_object.mode == 'EDIT':
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        with bpy.context.temp_override(window=window, area=area):
                            bpy.ops.object.mode_set(mode='OBJECT')
                        break

        SWOMT = context.scene.SWOMT
        src_path = SWOMT.AssetPath
        mod_file = _mod_file_output(src_path, overwrite=SWOMT.overwrite_existing)

        # Bake parent inverse and triangulate every LOD object that exists in the scene.
        # Baking must happen before triangulation so vertex positions are correct.
        original_data = {}  # obj_name -> (obj, original_data)
        for m in addon_state.asset.meshes:
            for li, lod in enumerate(m.lods):
                lod_obj_name = lod.blender_obj_name or f"{m.name}_LOD{li}"
                tri_obj = BME.find_object_by_name(lod_obj_name)
                if tri_obj:
                    original_data[lod_obj_name] = (tri_obj, tri_obj.data)
                    BME._triangulate_object(tri_obj,
                                            compute_normals=context.scene.SWOMT.compute_normals_on_export)

        # Export in reverse LOD order (3 -> 0), matching the primary-block
        # cumulative layout. Each LOD is then aligned against already-finalized
        # higher-LOD sizes; later passes cannot invalidate its uint32 size_a.
        # After the first pass has written mod_file, subsequent passes read from
        # it so every LOD level accumulates on top of the previous one.
        exported_any = False
        exported_mesh_indices = set()
        current_src = None  # None -> _write_mod_file reads SWOMT.AssetPath
        try:
            for lod_n in reversed(range(4)):
                edited = {}
                for m in addon_state.asset.meshes:
                    if len(m.lods) <= lod_n:
                        continue
                    lod = m.lods[lod_n]
                    obj_name = lod.blender_obj_name or f"{m.name}_LOD{lod_n}"
                    obj = BME.find_object_by_name(obj_name)
                    if obj is None:
                        continue
                    edited[m.index] = lod_n
                if not edited:
                    continue
                try:
                    BME._write_mod_file(edited_lod_index_per_mesh=edited, out_path=mod_file,
                                        src_path=current_src)
                    exported_any = True
                    exported_mesh_indices.update(edited)
                    current_src = mod_file
                except Exception as e:
                    self.report({'ERROR'}, f"Export LOD{lod_n} failed: {e}")
                    return {'CANCELLED'}
        finally:
            # Always restore every object's original mesh data so the user's mesh is untouched
            for obj_name, (obj, orig_data) in original_data.items():
                export_data = obj.data
                obj.data = orig_data
                if export_data != orig_data:
                    bpy.data.meshes.remove(export_data)

        if not exported_any:
            self.report({'WARNING'}, "No LOD objects found in scene to export.")
            return {'CANCELLED'}

        # Apply header-level patches once after all LODs are written
        for mesh in addon_state.asset.meshes:
            BME._apply_header_patches(mod_file, mesh, addon_state.asset, operator=self)

        # Enforce one primary-index width across every LOD of each mesh.
        try:
            BME.promote_mixed_index_widths(mod_file)
        except Exception as e:
            self.report({'ERROR'}, f"uint32 index widening failed: {e}")
            return {'CANCELLED'}

        # Run once against the complete multi-LOD output so every sibling LOD
        # contributes to the final per-mesh bounds.
        try:
            BME.expand_mesh_bounds(mod_file, exported_mesh_indices)
        except Exception as e:
            self.report({'ERROR'}, f"mesh bounds update failed: {e}")
            return {'CANCELLED'}

        # Rewrite the paired .mcloth (if any) so the cloth vertex mapping matches this export
        _export_mcloth_for_asset(mod_file, operator=self)

        # Apply staged file rename
        if addon_state.asset.pending_file_rename_new:
            new_file = str(Path(mod_file).parent / (addon_state.asset.pending_file_rename_new + '.mmb'))
            try:
                os.replace(mod_file, new_file)
                _mc_out = os.path.splitext(mod_file)[0] + '.mcloth'
                if os.path.isfile(_mc_out):
                    os.replace(_mc_out, os.path.splitext(new_file)[0] + '.mcloth')
            except Exception as e:
                self.report({'ERROR'}, f"Failed to rename mod file: {e}")
                return {'FINISHED'}
            SWOMT.AssetPath = new_file
            self.report({'INFO'}, f"Exported -> {os.path.basename(new_file)}")
        else:
            SWOMT.AssetPath = mod_file

        return {'FINISHED'}

CLASSES = (
    BrowseMMBFile, LoadMMB, ImportLOD, ExportLOD, ImportAllLOD0s,
    ImportAllLOD1s, ImportAllLOD2s, ImportAllLOD3s, ExportAllLODs,
)

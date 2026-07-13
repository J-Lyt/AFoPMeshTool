"""MGraph selection and on-disk asset rename operators."""

import os
import re
import shutil
from pathlib import Path

import bpy

from . import addon_state

class SelectMGraphObject(bpy.types.Operator):
    """Select the MGraphObject file and patch the mesh name inside it"""
    bl_idname = "object.select_mgraphobject"
    bl_label = "Select .mgraphobject"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.mgraphobject", options={'HIDDEN'})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        old_name = context.scene.get('_mmb_rename_old', '')
        new_name = context.scene.get('_mmb_rename_new', '')

        if not old_name or not new_name:
            self.report({'ERROR'}, "Rename info missing. Please rename the mesh again.")
            return {'CANCELLED'}

        try:
            with open(self.filepath, 'rb') as f:
                data = f.read()
        except Exception as e:
            self.report({'ERROR'}, f"Could not read file: {e}")
            return {'CANCELLED'}

        # The mgraphobject uses null-terminated strings stored as \x00value\x00.
        # We search for \x00MeshName_value\x00 so we only match the exact MeshName
        # field — not partial matches like g_res_torso_base_01_f which ends with
        # the same bytes as res_torso_base_01_f.
        #
        # For null-terminated strings, padding nulls go AFTER the new name
        # (new_name\x00 + padding), so the game reads the correct string and
        # stops at the first null terminator.
        old_pattern = b'\x00' + old_name.encode('utf-8') + b'\x00'
        padding = len(old_name) - len(new_name)
        new_pattern = b'\x00' + new_name.encode('utf-8') + b'\x00' + b'\x00' * padding

        count = data.count(old_pattern)
        if count == 0:
            self.report({'WARNING'}, f"MeshName '{old_name}' not found in selected file. Nothing patched.")
            return {'FINISHED'}

        patched = data.replace(old_pattern, new_pattern)

        backup_path = self.filepath + '.bak'
        try:
            import shutil
            shutil.copy2(self.filepath, backup_path)
        except Exception as e:
            self.report({'ERROR'}, f"Could not create backup: {e}")
            return {'CANCELLED'}

        try:
            with open(self.filepath, 'wb') as f:
                f.write(patched)
        except Exception as e:
            self.report({'ERROR'}, f"Could not write file: {e}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Patched {count} MeshName occurrence(s): '{old_name}' -> '{new_name}' in {os.path.basename(self.filepath)} (backup: {os.path.basename(backup_path)})")
        return {'FINISHED'}

class RenameMMBFile(bpy.types.Operator):
    """Rename the loaded .mmb file on disk and update path references in a mgraphobject"""
    bl_idname = "object.rename_mmb_file"
    bl_label = "Rename MMB File"

    @classmethod
    def poll(cls, context):
        return addon_state.asset is not None

    @staticmethod
    def _base_stem(path):
        """Return the stem with any _MOD or _MOD<n> suffix stripped."""
        stem = Path(path).stem
        return re.sub(r'_MOD\d*$', '', stem)

    def invoke(self, context, event):
        old_stem = self._base_stem(context.scene.SWOMT.AssetPath)
        bpy.types.Scene.mmb_file_rename_input = bpy.props.StringProperty(
            name="New Filename",
            maxlen=len(old_stem),
        )
        context.scene.mmb_file_rename_input = old_stem
        context.window.cursor_warp(context.window.width // 2, context.window.height // 2)
        return context.window_manager.invoke_props_dialog(self, width=380)

    def draw(self, context):
        old_stem = self._base_stem(context.scene.SWOMT.AssetPath)
        layout = self.layout
        layout.label(text=f"Original: {old_stem}")
        layout.label(text=f"Must be exactly {len(old_stem)} characters")
        layout.prop(context.scene, "mmb_file_rename_input", text="New Filename")

    def execute(self, context):
        SWOMT = context.scene.SWOMT
        old_path = Path(SWOMT.AssetPath)
        old_stem = self._base_stem(SWOMT.AssetPath)
        new_stem = context.scene.mmb_file_rename_input.strip()

        if not new_stem:
            self.report({'ERROR'}, "Filename cannot be empty.")
            return {'CANCELLED'}
        if len(new_stem) != len(old_stem):
            self.report({'ERROR'}, f"New name must be exactly {len(old_stem)} characters.")
            return {'CANCELLED'}

        # Stage the file rename — original .mmb is never touched.
        # The _MOD copy is renamed on next export.
        addon_state.asset.pending_file_rename_old = old_stem
        addon_state.asset.pending_file_rename_new = new_stem

        # Stash for the mgraphobject file-reference patch (applied immediately)
        context.scene['_mmb_file_old_stem'] = old_stem
        context.scene['_mmb_file_new_stem'] = new_stem

        bpy.ops.object.select_mgraphobject_file_patch('INVOKE_DEFAULT')
        return {'FINISHED'}

class SelectMGraphObjectFilePatch(bpy.types.Operator):
    """Select the MGraphObject file and update all path references to the renamed .mmb"""
    bl_idname = "object.select_mgraphobject_file_patch"
    bl_label = "Select .mgraphobject"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.mgraphobject", options={'HIDDEN'})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        old_stem = context.scene.get('_mmb_file_old_stem', '')
        new_stem = context.scene.get('_mmb_file_new_stem', '')

        if not old_stem or not new_stem:
            self.report({'ERROR'}, "File rename info missing. Please rename the file again.")
            return {'CANCELLED'}

        try:
            with open(self.filepath, 'rb') as f:
                data = bytearray(f.read())
        except Exception as e:
            self.report({'ERROR'}, f"Could not read file: {e}")
            return {'CANCELLED'}

        # Only replace the .mmb file reference — leave .mreflex, .juice,
        # assetName, and any other occurrences of the stem untouched.
        old_b = (old_stem + '.mmb').encode('utf-8')
        new_b = (new_stem + '.mmb').encode('utf-8')

        count = data.count(old_b)
        if count == 0:
            self.report({'WARNING'}, f"'{old_stem}.mmb' not found in selected file. Nothing patched.")
            return {'FINISHED'}

        patched = data.replace(old_b, new_b)

        backup_path = self.filepath + '.bak'
        try:
            import shutil
            shutil.copy2(self.filepath, backup_path)
        except Exception as e:
            self.report({'ERROR'}, f"Could not create backup: {e}")
            return {'CANCELLED'}

        try:
            with open(self.filepath, 'wb') as f:
                f.write(patched)
        except Exception as e:
            self.report({'ERROR'}, f"Could not write file: {e}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Updated {count} .mmb reference(s): '{old_stem}.mmb' -> '{new_stem}.mmb' in {os.path.basename(self.filepath)} (backup: {os.path.basename(backup_path)})")
        return {'FINISHED'}

CLASSES = (SelectMGraphObject, RenameMMBFile, SelectMGraphObjectFilePatch)

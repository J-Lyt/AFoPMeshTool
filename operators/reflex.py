"""Blender operators for paired MReflex dangle-bone physics."""

import os

import bpy

from .. import addon_state
from ..formats.mreflex import (
    DangleNodeUpdate,
    mod_output_path,
    rewrite_dangle_nodes,
)
from ..settings import _load_mreflex_into_settings
from ..mesh_pipeline.files import source_setting_path


def _source_mmb_path(settings):
    return bpy.path.abspath(source_setting_path(
        settings, "AssetPath", "SourceAssetPath"))


class BrowseMReflexFile(bpy.types.Operator):
    """Select an MReflex paired with the loaded MMB."""

    bl_idname = "object.browse_mreflex_file"
    bl_label = "Select .mreflex"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(
        default="*.mreflex", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return addon_state.asset is not None

    def invoke(self, context, event):
        current = context.scene.SWOMT.ReflexPath
        if current:
            self.filepath = bpy.path.abspath(current)
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        settings = context.scene.SWOMT
        path = bpy.path.abspath(self.filepath)
        if not os.path.isfile(path):
            self.report({'ERROR'}, f"MReflex file does not exist: {path}")
            return {'CANCELLED'}
        settings["SourceReflexPath"] = path
        if not _load_mreflex_into_settings(
                settings,
                _source_mmb_path(settings),
                addon_state.asset,
                reflex_path=path):
            self.report({'ERROR'}, settings.reflex_status)
            return {'CANCELLED'}
        return {'FINISHED'}


class ReloadMReflex(bpy.types.Operator):
    """Discard edits and reload the selected MReflex."""

    bl_idname = "object.reload_mreflex"
    bl_label = "Reload Dangle Physics"

    @classmethod
    def poll(cls, context):
        return addon_state.asset is not None

    def execute(self, context):
        settings = context.scene.SWOMT
        selected = bpy.path.abspath(settings.ReflexPath) if settings.ReflexPath else ""
        if not _load_mreflex_into_settings(
                settings,
                _source_mmb_path(settings),
                addon_state.asset,
                reflex_path=selected):
            self.report({'ERROR'}, settings.reflex_status)
            return {'CANCELLED'}
        self.report({'INFO'}, settings.reflex_status)
        return {'FINISHED'}


class SaveMReflex(bpy.types.Operator):
    """Save edited values to a _MOD MReflex output."""

    bl_idname = "object.save_mreflex"
    bl_label = "Save Dangle Physics"

    @classmethod
    def poll(cls, context):
        settings = context.scene.SWOMT
        return (
            addon_state.asset is not None
            and bool(settings.ReflexPath)
            and bool(settings.reflex_nodes)
        )

    def execute(self, context):
        settings = context.scene.SWOMT
        source_path = bpy.path.abspath(source_setting_path(
            settings, "ReflexPath", "SourceReflexPath"))
        if not os.path.isfile(source_path):
            self.report({'ERROR'}, f"MReflex file does not exist: {source_path}")
            return {'CANCELLED'}
        export_dir = (
            bpy.path.abspath(settings.ExportPath)
            if settings.ExportPath else os.path.dirname(source_path)
        )
        if not export_dir or not os.path.isdir(export_dir):
            self.report(
                {'ERROR'},
                f"Export folder does not exist: {export_dir or '(empty)'}")
            return {'CANCELLED'}

        displayed_path = (
            bpy.path.abspath(settings.ReflexPath)
            if settings.ReflexPath else source_path
        )
        destination = os.path.join(
            export_dir, os.path.basename(displayed_path))
        output_path = mod_output_path(
            destination, overwrite=settings.overwrite_existing)
        if (os.path.normcase(os.path.abspath(output_path))
                == os.path.normcase(source_path)):
            output_path = mod_output_path(destination, overwrite=False)

        updates = []
        try:
            for item in settings.reflex_nodes:
                updates.append(DangleNodeUpdate(
                    record_index=item.record_index,
                    node_id=int(item.node_id, 16),
                    gravity=item.gravity,
                    weight=item.weight,
                    spring=item.spring,
                    damping=item.damping,
                    limits=(
                        item.limit_1,
                        item.limit_2,
                        item.limit_3,
                        item.limit_4,
                    ),
                ))
            with open(source_path, "rb") as stream:
                rewritten = rewrite_dangle_nodes(stream.read(), updates)
            temporary = output_path + ".mreflex_tmp"
            try:
                with open(temporary, "wb") as stream:
                    stream.write(rewritten)
                os.replace(temporary, output_path)
            finally:
                if os.path.exists(temporary):
                    os.remove(temporary)
        except Exception as error:
            self.report({'ERROR'}, f"Could not save MReflex: {error}")
            return {'CANCELLED'}

        _load_mreflex_into_settings(
            settings,
            _source_mmb_path(settings),
            addon_state.asset,
            reflex_path=output_path,
        )
        self.report(
            {'INFO'}, f"Saved dangle physics: {os.path.basename(output_path)}")
        return {'FINISHED'}


CLASSES = (BrowseMReflexFile, ReloadMReflex, SaveMReflex)

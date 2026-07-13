"""Runtime asset loading and scene property definitions."""

import os
from pathlib import Path

import bpy

from . import addon_state
from .file_utils import _strip_mod_suffix, get_merged_mmb
from .log import logger, set_debug
from .mmb import SkeletalMeshAsset


def _on_debug_logging_update(self, context):
    set_debug(self.debug_logging)


def apply_debug_logging_preference():
    """Restore the saved add-on logging preference after registration."""
    addon = bpy.context.preferences.addons.get(__package__)
    if addon is not None:
        set_debug(addon.preferences.debug_logging)


@bpy.app.handlers.persistent
def _on_load_post(filepath, *args, **kwargs):
    """Resets the asset when a .blend file is loaded, then re-loads it from the AssetPath if the file still exists."""
    addon_state.asset = None
    try:
        for scene in bpy.data.scenes:
            path = scene.SWOMT.get("AssetPath", "")
            if not path or not os.path.isfile(path):
                continue
            try:
                with open(path, 'rb') as f:
                    sk_mesh = SkeletalMeshAsset()
                    sk_mesh.parse(f)
                    # AssetPath may be an exported '_MOD'; the armature in this
                    # .blend was never renamed, so prefer its un-suffixed name if found.
                    full_stem = Path(path).stem
                    bare_stem = _strip_mod_suffix(full_stem)
                    if bare_stem != full_stem and bpy.data.objects.get(bare_stem) is not None:
                        sk_mesh.name = bare_stem
                    else:
                        sk_mesh.name = full_stem
                    addon_state.asset = sk_mesh
                _check_vert_pos_mmb(sk_mesh, path)
                logger.info("Loaded %s from %s", sk_mesh.name, path)
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)
            break
    except Exception as e:
        logger.exception("Load-post handler failed: %s", e)

def _resolve_asset_name(new_path, old_asset):
    """
    Name a newly-parsed asset for `new_path`.

    If the new filename is just the loaded asset's name plus a '_MOD' suffix, and the armature
    for that asset still exists - keep the loaded asset's name so it still matches the exisitng armature.

    Otherwise, derive the name from the path.
    """
    new_stem = Path(new_path).stem
    if (old_asset is not None and old_asset.name
            and _strip_mod_suffix(new_stem) == old_asset.name
            and bpy.data.objects.get(old_asset.name) is not None):
        return old_asset.name
    return new_stem


def _auto_load_mmb(self, context):
    path = self.AssetPath
    if not path or not os.path.isfile(path):
        return
    try:
        new_name = _resolve_asset_name(path, addon_state.asset)
        with open(path, 'rb') as file:
            sk_mesh = SkeletalMeshAsset()
            sk_mesh.parse(file)
            sk_mesh.name = new_name
            addon_state.asset = sk_mesh
        _check_vert_pos_mmb(sk_mesh, path)
    except Exception as e:
        logger.warning("MMB auto-load failed: %s", e)

def _vert_count_changed():
    """Return True if any imported LOD Blender object has a different vert count than the MMB."""
    if addon_state.asset is None:
        return False
    for m in addon_state.asset.meshes:
        for li, lod in enumerate(m.lods):
            if lod.vertex_count == 0:
                continue
            obj_name = lod.blender_obj_name or f"{m.name}_LOD{li}"
            obj = bpy.data.objects.get(obj_name)
            if obj is not None and len(obj.data.vertices) != 0 and len(obj.data.vertices) != lod.vertex_count:
                return True
    return False

def _check_vert_pos_mmb(sk_mesh, path: str):
    """
    Read LOD0 vertex positions for each mesh directly from the MMB file. (We assume that LODs 1-3 are zeroed)
    If all positions for LOD0 are zero, zeroed_out_in_mmb = True
    """
    try:
        merged = get_merged_mmb(path)
    except Exception as e:
        logger.warning("Could not inspect vertex positions in %s: %s", path, e)
        return

    for mesh in sk_mesh.meshes:
        if not mesh.lods:
            continue
        lod = mesh.lods[0]
        if lod.vertex_count == 0:
            continue
        try:
            raw = mesh.extract_mesh_file(merged)
            positions = lod.get_vertex_positions(raw)
            is_zeroed = all(
                abs(x) < 1e-6 and abs(y) < 1e-6 and abs(z) < 1e-6
                for x, y, z in positions
            )
        except Exception as e:
            logger.warning("Could not inspect vertex positions for %s: %s", mesh.name, e)
            continue

        mesh.zeroed_out_in_mmb = is_zeroed

def _on_compute_normals_on_export_update(self, context):
    """Auto-enable export_normals when compute_normals_on_export is checked."""
    if self.compute_normals_on_export:
        self.export_normals = True

def _on_export_normals_update(self, context):
    """Auto-uncheck compute_normals_on_export, export_vertex_colors and export_uvs when export_normals is unchecked."""
    if not self.export_normals:
        if self.compute_normals_on_export:
            self.compute_normals_on_export = False
        if self.export_vertex_colors:
            self.export_vertex_colors = False
        if self.export_uvs:
            self.export_uvs = False

def _on_export_vertex_colors_update(self, context):
    """Auto-enable export_normals when export_vertex_colors is checked."""
    if self.export_vertex_colors:
        self.export_normals = True

def _on_export_uvs_update(self, context):
    """Auto-enable export_normals when export_uvs is checked."""
    if self.export_uvs:
        self.export_normals = True

def _get_export_normals(self):
    if _vert_count_changed():
        return True
    return self.get("export_normals", False)

def _set_export_normals(self, value):
    if _vert_count_changed():
        return
    old = self.get("export_normals", False)
    self["export_normals"] = value
    if old != value:
        _on_export_normals_update(self, None)

def _get_export_weights(self):
    if _vert_count_changed():
        return True
    return self.get("export_weights", False)

def _set_export_weights(self, value):
    if not _vert_count_changed():
        self["export_weights"] = value

def _get_export_uvs(self):
    if _vert_count_changed():
        return True
    return self.get("export_uvs", False)

def _set_export_uvs(self, value):
    if _vert_count_changed():
        return
    old = self.get("export_uvs", False)
    self["export_uvs"] = value
    if old != value:
        _on_export_uvs_update(self, None)


class AFOPPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    debug_logging: bpy.props.BoolProperty(
        name="Enable Debug Logging",
        default=False,
        description="Write diagnostic messages to Blender's system console",
        update=_on_debug_logging_update,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "debug_logging")
        layout.label(text="Diagnostic messages are written to Blender's system console.")


class SWOMTSettings(bpy.types.PropertyGroup):
    AssetPath: bpy.props.StringProperty(name="Asset Path", update=_auto_load_mmb)
    overwrite_existing: bpy.props.BoolProperty(
        name="Overwrite existing file",
        default=False,
        description="When checked, export overwrites the loaded file instead of creating a new _MOD file",
    )
    mesh_expanded: bpy.props.BoolVectorProperty(size=32, default=tuple([False]*32))
    bone_slots_expanded: bpy.props.BoolVectorProperty(size=32, default=tuple([False]*32))
    compute_normals_on_export: bpy.props.BoolProperty(
        name="Compute Normals on Export",
        default=False,
        description="Recompute normals on export.",
        update=_on_compute_normals_on_export_update,
    )
    export_normals: bpy.props.BoolProperty(
        name="Export Normals",
        description="Write normals into the exported file. When unchecked, the original normals from the .mmb are preserved. Automatically forced on when vert count has changed.",
        get=_get_export_normals,
        set=_set_export_normals,
    )
    export_weights: bpy.props.BoolProperty(
        name="Export Weights",
        description="Write bone weights into the exported file. When unchecked, the original weights from the .mmb are preserved. Automatically forced on when vert count has changed.",
        get=_get_export_weights,
        set=_set_export_weights,
    )
    export_vertex_colors: bpy.props.BoolProperty(
        name="Export Vertex Colors",
        default=False,
        description="Write vertex colors from Blender into the exported file. When unchecked, the original vertex colors from the .mmb are preserved.",
        update=_on_export_vertex_colors_update,
    )
    cloth_donor_radius: bpy.props.FloatProperty(
        name="Cloth Donor Radius",
        default=0.05,
        min=0.001,
        soft_max=0.5,
        precision=3,
        subtype='DISTANCE',
        description="Verts added to a '_CLOTH_RENDER' mesh inherit cloth behavior from the nearest original vertex within this distance.",
    )
    export_uvs: bpy.props.BoolProperty(
        name="Export UVs",
        description="Write UV coordinates from Blender into the exported file. When unchecked, the original UVs from the .mmb are preserved. Automatically forced on when vert count has changed.",
        get=_get_export_uvs,
        set=_set_export_uvs,
    )
    export_options_expanded: bpy.props.BoolProperty(
        name="Export Options",
        default=True,
    )
    force_lod0_mmb_override: bpy.props.StringProperty(
        name="MMB Filename",
        description="Original .mmb filename to look up in mmb_lod_presets.json",
        default="",
    )
    force_lod0_output_path: bpy.props.StringProperty(
        name="Force LOD0 Output Path",
        default="",
    )
    force_lod0_cfg_path: bpy.props.StringProperty(
        name="LOD Presets CFG",
        description="Path to an existing lod_presets.cfg to update. Leave empty to generate a new one alongside the asset.",
        default="",
        subtype="FILE_PATH",
    )

CLASSES = (AFOPPreferences, SWOMTSettings)

"""Runtime asset loading and scene property definitions."""

import os
from pathlib import Path

import bpy

from . import addon_state
from .mesh_pipeline.files import _strip_mod_suffix, get_merged_mmb
from .log import logger, set_debug
from .formats.mcloth import source_path as paired_mcloth_path
from .formats.mmb import SkeletalMeshAsset
from .formats.mreflex import (
    BoneCandidate,
    match_dangle_nodes,
    paired_mreflex_path,
    read_dangle_nodes,
    read_transform_anchors,
    reflect_x_basis,
)


_sdf_search_generation = 0
_SDF_SEARCH_DELAY = 0.35


def _on_debug_logging_update(self, context):
    set_debug(self.debug_logging)


def _on_sdf_search_update(self, context):
    """Apply search text shortly after typing stops instead of on every keypress."""
    global _sdf_search_generation
    _sdf_search_generation += 1
    generation = _sdf_search_generation
    scene_pointer = context.scene.as_pointer() if context and context.scene else None

    def apply_search():
        if generation != _sdf_search_generation:
            return None
        try:
            for scene in bpy.data.scenes:
                if scene_pointer is not None and scene.as_pointer() != scene_pointer:
                    continue
                settings = getattr(scene, "SWOMT", None)
                if settings is not None:
                    settings.sdf_show_all_results = False
                    settings.sdf_search_applied = settings.sdf_search
                    from .operators import sdf as operators_sdf
                    operators_sdf.populate_search_results(scene, settings.sdf_search)
                break
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == "PROPERTIES":
                        area.tag_redraw()
        except (ReferenceError, RuntimeError):
            pass
        return None

    bpy.app.timers.register(apply_search, first_interval=_SDF_SEARCH_DELAY)


def _addon_preferences(context=None):
    """Return this add-on's saved preferences when it is installed/enabled."""
    preferences = (context or bpy.context).preferences
    addon = preferences.addons.get(__package__)
    return addon.preferences if addon is not None else None


def get_default_game_directory(context=None):
    """Return the game directory saved in the add-on preferences."""
    preferences = _addon_preferences(context)
    return preferences.default_game_directory if preferences is not None else ""


def blender_extracted_files_directory():
    """Return the existing Blender data-files location used for SDF extracts."""
    path = bpy.utils.user_resource(
        "DATAFILES", path=os.path.join("afop_mesh_tool", "sdf_cache"), create=False
    )
    if path:
        return path
    return os.path.join(bpy.app.tempdir, "afop_mesh_tool", "sdf_cache")


def blender_sdf_index_cache_directory():
    """Return the Blender data-files location for targeted SDF index caches."""
    path = bpy.utils.user_resource(
        "DATAFILES", path=os.path.join("afop_mesh_tool", "sdf_index_cache"), create=False
    )
    return path or os.path.join(bpy.app.tempdir, "afop_mesh_tool", "sdf_index_cache")


def get_default_extracted_files_directory(context=None):
    """Return the saved extraction folder, falling back to Blender data files."""
    preferences = _addon_preferences(context)
    if preferences is not None:
        return preferences.default_extracted_files_directory or blender_extracted_files_directory()
    return blender_extracted_files_directory()


def apply_debug_logging_preference():
    """Restore the saved add-on logging preference after registration."""
    preferences = _addon_preferences()
    if preferences is not None:
        set_debug(preferences.debug_logging)


def _weighted_bone_indices(sk_mesh, mmb_path):
    """Return skeleton indices that receive a non-zero LOD0 vertex weight."""
    weighted = set()
    merged = get_merged_mmb(mmb_path)
    for mesh in sk_mesh.meshes:
        if not mesh.lods or mesh.lods[0].vertex_count == 0:
            continue
        slots = list(mesh.mesh_bones.keys())
        if not slots:
            continue
        try:
            raw_mesh = mesh.extract_mesh_file(merged)
            for vertex_weights in mesh.lods[0].get_bone_weights(raw_mesh):
                for slot, weight in vertex_weights.items():
                    if weight > 0.0 and 0 <= slot < len(slots):
                        weighted.add(slots[slot])
        except Exception as error:
            logger.warning(
                "Could not inspect bone weights for %s: %s", mesh.name, error)
    return weighted


def _load_mreflex_into_settings(settings, mmb_path, sk_mesh, reflex_path=""):
    """Load a paired MReflex and conservatively label its kind-10 nodes."""
    settings.reflex_nodes.clear()
    settings.reflex_node_index = 0
    path = os.path.abspath(reflex_path) if reflex_path else paired_mreflex_path(mmb_path)
    settings["ReflexPath"] = path
    if path and not settings.get("SourceReflexPath", ""):
        settings["SourceReflexPath"] = path
    if not path or not os.path.isfile(path):
        settings["reflex_status"] = "No paired .mreflex found"
        return False

    try:
        with open(path, "rb") as stream:
            reflex_data = stream.read()
        nodes = read_dangle_nodes(reflex_data)
        matches = {}
        if nodes:
            anchors = read_transform_anchors(reflex_data, nodes)
            weighted = _weighted_bone_indices(sk_mesh, mmb_path)
            bones = []
            for index, bone in enumerate(sk_mesh.bones):
                local_matrix = tuple(
                    float(bone.matrix[row][column])
                    for row in range(4) for column in range(4)
                )
                bones.append(BoneCandidate(
                    index=index,
                    name=bone.name,
                    parent_index=bone.parent_index,
                    reflex_matrix=reflect_x_basis(local_matrix),
                ))
            matches = match_dangle_nodes(
                nodes, bones, weighted, anchors=anchors)

        for node in nodes:
            item = settings.reflex_nodes.add()
            item.record_index = node.record_index
            item.node_id = f"{node.node_id:08X}"
            item.parent_id = f"{node.parent_id:08X}"
            match = matches.get(node.record_index)
            if match is not None:
                item.bone_name = match.bone_name
                item.match_status = "MATCHED"
                item.match_error = match.error
            else:
                item.bone_name = ""
                item.match_status = "UNMATCHED"
                item.match_error = -1.0
            item.gravity = node.gravity
            item.weight = node.weight
            item.spring = node.spring
            item.damping = node.damping
            item.limit_1, item.limit_2, item.limit_3, item.limit_4 = node.limits

        if not nodes:
            settings["reflex_status"] = (
                "Valid .mreflex; no kind-10 dangle nodes")
        else:
            settings["reflex_status"] = (
                f"{len(nodes)} dangle nodes; {len(matches)} uniquely matched")
        logger.info("Loaded MReflex %s (%s)", path, settings.reflex_status)
        return True
    except Exception as error:
        settings["reflex_status"] = f"MReflex load failed: {error}"
        logger.warning("MReflex load failed for %s: %s", path, error)
        return False


@bpy.app.handlers.persistent
def _on_load_post(filepath, *args, **kwargs):
    """Resets the asset when a .blend file is loaded, then re-loads it from the AssetPath if the file still exists."""
    addon_state.asset = None
    try:
        for scene in bpy.data.scenes:
            path = scene.SWOMT.get("AssetPath", "")
            if not path or not os.path.isfile(path):
                continue
            if not scene.SWOMT.get("SourceAssetPath", ""):
                scene.SWOMT["SourceAssetPath"] = path
            source_path = bpy.path.abspath(
                scene.SWOMT.get("SourceAssetPath", "") or path)
            parse_path = source_path if os.path.isfile(source_path) else path
            try:
                with open(parse_path, 'rb') as f:
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
                _check_removed_meshes_mmb(sk_mesh, parse_path)
                cloth_path = scene.SWOMT.get("MClothPath", "")
                resolved_cloth_path = (
                    bpy.path.abspath(cloth_path) if cloth_path else "")
                if (not resolved_cloth_path
                        or not os.path.isfile(resolved_cloth_path)):
                    scene.SWOMT["MClothPath"] = (
                        paired_mcloth_path(path) or "")
                if (scene.SWOMT.get("MClothPath", "")
                        and not scene.SWOMT.get("SourceMClothPath", "")):
                    scene.SWOMT["SourceMClothPath"] = scene.SWOMT["MClothPath"]
                _load_mreflex_into_settings(scene.SWOMT, parse_path, sk_mesh)
                logger.info("Loaded %s from %s", sk_mesh.name, parse_path)
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)
            break
    except Exception as e:
        logger.exception("Load-post handler failed: %s", e)
    try:
        from .operators import sdf as operators_sdf
        operators_sdf.schedule_cached_auto_load(reset=True)
    except Exception as e:
        logger.warning("Could not schedule cached SDF auto-load: %s", e)

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
    path = bpy.path.abspath(self.AssetPath) if self.AssetPath else ""
    if path and not self.get("SourceAssetPath", ""):
        self["SourceAssetPath"] = path
    retained_source = bpy.path.abspath(
        self.get("SourceAssetPath", "") or path) if path else ""
    old_asset = addon_state.asset
    self["banshee_pattern_status"] = ""
    self["MClothPath"] = (paired_mcloth_path(path) or "") if path else ""
    if retained_source and not self.get("SourceMClothPath", ""):
        source_cloth = paired_mcloth_path(retained_source) or ""
        if source_cloth:
            self["SourceMClothPath"] = source_cloth
    self.reflex_nodes.clear()
    self["ReflexPath"] = ""
    self["reflex_status"] = ""

    # Clear the parsed asset immediately so editing the path cannot leave the
    # import/export controls for the previously loaded MMB active.
    addon_state.asset = None
    if path:
        self["ExportPath"] = os.path.dirname(os.path.abspath(path))
    try:
        parse_path = retained_source if os.path.isfile(retained_source) else path
        if not parse_path or not os.path.isfile(parse_path):
            return
        new_name = _resolve_asset_name(parse_path, old_asset)
        with open(parse_path, 'rb') as file:
            sk_mesh = SkeletalMeshAsset()
            sk_mesh.parse(file)
            sk_mesh.name = new_name
            addon_state.asset = sk_mesh
        _check_removed_meshes_mmb(sk_mesh, parse_path)
        _load_mreflex_into_settings(self, parse_path, sk_mesh)
    except Exception as e:
        logger.warning("MMB auto-load failed: %s", e)
    finally:
        if context and context.area:
            context.area.tag_redraw()
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "PROPERTIES":
                    area.tag_redraw()


def _on_source_asset_update(self, context):
    """Mirror the sole user-facing MMB source into the legacy load property."""
    path = bpy.path.abspath(self.SourceAssetPath) if self.SourceAssetPath else ""
    current = bpy.path.abspath(self.AssetPath) if self.AssetPath else ""
    if path != current:
        self.AssetPath = path
        return
    if path:
        _auto_load_mmb(self, context)


def _on_source_mcloth_update(self, context):
    """Mirror the sole user-facing MCloth source into the legacy property."""
    self["MClothPath"] = (
        bpy.path.abspath(self.SourceMClothPath)
        if self.SourceMClothPath else "")


def _on_source_reflex_update(self, context):
    """Mirror and load the sole user-facing MReflex source."""
    if not self.SourceReflexPath:
        self["ReflexPath"] = ""
        self.reflex_nodes.clear()
        self.reflex_node_index = 0
        self["reflex_status"] = "No source .mreflex selected"
        return
    path = bpy.path.abspath(self.SourceReflexPath)
    self["ReflexPath"] = path
    if addon_state.asset is not None and os.path.isfile(path):
        mmb_path = self.SourceAssetPath or self.AssetPath
        _load_mreflex_into_settings(
            self, bpy.path.abspath(mmb_path), addon_state.asset,
            reflex_path=path)

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

def _check_removed_meshes_mmb(sk_mesh, path: str):
    """
    Mark meshes whose LODs are already faceless. Legacy exports that removed a
    mesh by zeroing every LOD0 position are still recognised as removed.
    """
    try:
        merged = get_merged_mmb(path)
    except Exception as e:
        logger.warning("Could not inspect vertex positions in %s: %s", path, e)
        return

    for mesh in sk_mesh.meshes:
        if not mesh.lods:
            continue
        populated_lods = [lod for lod in mesh.lods if lod.vertex_count > 0]
        if populated_lods and all(lod.index_count == 0 for lod in populated_lods):
            mesh.removed_in_mmb = True
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

        mesh.removed_in_mmb = is_zeroed

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

    default_game_directory: bpy.props.StringProperty(
        name="Default Game Directory",
        subtype='DIR_PATH',
        description="Default AFOP folder containing the SDF archives for new and unsaved scenes",
    )
    default_extracted_files_directory: bpy.props.StringProperty(
        name="Default Extracted Files",
        subtype='DIR_PATH',
        description="Default folder for MMB, mcloth, mreflex, and texture files extracted from SDF archives",
        default=blender_extracted_files_directory(),
    )
    debug_logging: bpy.props.BoolProperty(
        name="Enable Debug Logging",
        default=False,
        description="Write diagnostic messages to Blender's system console",
        update=_on_debug_logging_update,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "default_game_directory")
        layout.label(text="Used whenever the current scene has no game directory selected.")
        layout.prop(self, "default_extracted_files_directory")
        layout.label(text="Extracted assets are stored here unless the scene overrides it.")
        cache_row = layout.row(align=True)
        cache_button = cache_row.row(align=True)
        cache_button.ui_units_x = 7.0
        cache_button.operator(
            "object.clear_sdf_index_cache",
            text="Clear Cache",
            icon="TRASH",
        )
        layout.separator()
        layout.prop(self, "debug_logging")
        layout.label(text="Diagnostic messages are written to Blender's system console.")


class SDFAssetListItem(bpy.types.PropertyGroup):
    """One searchable MMB, graph, or compound result from the SDF index."""

    asset_path: bpy.props.StringProperty(options={'SKIP_SAVE'})
    asset_type: bpy.props.StringProperty(options={'SKIP_SAVE'})
    archive_label: bpy.props.StringProperty(options={'SKIP_SAVE'})
    entry_id: bpy.props.IntProperty(default=-1, options={'SKIP_SAVE'})


class MReflexNodeSettings(bpy.types.PropertyGroup):
    """Editable values for one structurally validated kind-10 MReflex node."""

    record_index: bpy.props.IntProperty(options={'HIDDEN', 'SKIP_SAVE'})
    node_id: bpy.props.StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    parent_id: bpy.props.StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    bone_name: bpy.props.StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    match_status: bpy.props.StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    match_error: bpy.props.FloatProperty(
        default=-1.0, options={'HIDDEN', 'SKIP_SAVE'})
    gravity: bpy.props.FloatProperty(
        name="Gravity",
        description=(
            "Strength of the constant downward acceleration.\n"
            "At zero, displaced sections may remain raised"
        ),
        precision=4,
        soft_min=0.0,
        soft_max=98.0,
    )
    weight: bpy.props.FloatProperty(
        name="Force Response",
        description=(
            "How strongly wind and other external forces move the node.\n"
            "Zero prevents dynamic movement; one gives maximum response"
        ),
        precision=4,
        soft_min=0.0,
        soft_max=1.0,
    )
    spring: bpy.props.FloatProperty(
        name="Stiffness",
        description=(
            "How strongly the node resists bending and returns toward its "
            "rest orientation.\n"
            "Very low values can cause continuous movement or twitching"
        ),
        precision=5,
        soft_min=0.0,
        soft_max=10.0,
    )
    damping: bpy.props.FloatProperty(
        name="Motion Retention",
        description=(
            "How much movement is retained.\n"
            "Lower values suppress movement more strongly; higher values "
            "allow motion to continue"
        ),
        precision=5,
        soft_min=0.0,
        soft_max=1.0,
    )
    limit_1: bpy.props.FloatProperty(
        name="Limit 1", subtype='ANGLE', precision=2)
    limit_2: bpy.props.FloatProperty(
        name="Limit 2", subtype='ANGLE', precision=2)
    limit_3: bpy.props.FloatProperty(
        name="Limit 3", subtype='ANGLE', precision=2)
    limit_4: bpy.props.FloatProperty(
        name="Limit 4", subtype='ANGLE', precision=2)


def _get_sdf_game_directory(self):
    """Use the scene override when present, otherwise the saved add-on default."""
    return self.get("sdf_game_directory", "") or get_default_game_directory()


def _set_sdf_game_directory(self, value):
    self["sdf_game_directory"] = value


def _get_sdf_extracted_directory(self):
    """Use the scene extraction folder when present, otherwise the saved default."""
    return self.get("sdf_extracted_directory", "") or get_default_extracted_files_directory()


def _set_sdf_extracted_directory(self, value):
    self["sdf_extracted_directory"] = value


def _get_export_path(self):
    """Default exports to the directory containing the currently loaded asset."""
    stored = self.get("ExportPath", "")
    if stored:
        return stored
    asset_path = self.get("AssetPath", "")
    return os.path.dirname(os.path.abspath(bpy.path.abspath(asset_path))) if asset_path else ""


def _set_export_path(self, value):
    self["ExportPath"] = value


def _banshee_pattern_items(self, context):
    from .operators.patterns import pattern_enum_items
    return pattern_enum_items(self, context)


def _on_sdf_browser_expanded_update(self, context):
    """Load current cached SDF metadata when the game-files panel is opened."""
    if not self.sdf_browser_expanded:
        return
    try:
        from .operators import sdf as operators_sdf
        operators_sdf.schedule_cached_auto_load()
    except Exception as error:
        logger.warning("Could not schedule cached SDF auto-load: %s", error)


class SWOMTSettings(bpy.types.PropertyGroup):
    source_files_expanded: bpy.props.BoolProperty(
        name="Source Files",
        description="Show or hide the source file paths",
        default=True,
    )
    SourceAssetPath: bpy.props.StringProperty(
        name="Source MMB File",
        description="Path of the currently loaded MMB asset",
        update=_on_source_asset_update,
    )
    SourceMClothPath: bpy.props.StringProperty(
        name="Source MCloth File",
        description="Cloth simulation file used when exporting the loaded MMB",
        update=_on_source_mcloth_update,
    )
    SourceReflexPath: bpy.props.StringProperty(
        name="Source MReflex File",
        description="Dangle-bone physics file paired with the loaded MMB",
        update=_on_source_reflex_update,
    )
    AssetPath: bpy.props.StringProperty(
        name="Path of the currently loaded asset",
        update=_auto_load_mmb,
    )
    MClothPath: bpy.props.StringProperty(
        name="MCloth File",
        description="Cloth simulation file used when exporting the loaded MMB",
    )
    ExportPath: bpy.props.StringProperty(
        name="Folder where MMB, MCloth, and MReflex files are exported",
        get=_get_export_path,
        set=_set_export_path,
    )
    ReflexPath: bpy.props.StringProperty(
        name="MReflex File",
        description="Dangle-bone physics file paired with the loaded MMB",
    )
    reflex_nodes: bpy.props.CollectionProperty(
        type=MReflexNodeSettings,
        options={'SKIP_SAVE'},
    )
    reflex_node_index: bpy.props.IntProperty(
        default=0, min=0, options={'SKIP_SAVE'})
    reflex_status: bpy.props.StringProperty(
        options={'HIDDEN', 'SKIP_SAVE'})
    reflex_expanded: bpy.props.BoolProperty(
        name="Dangle Physics",
        default=False,
    )
    sdf_browser_expanded: bpy.props.BoolProperty(
        name="Load from Game Files",
        default=False,
        update=_on_sdf_browser_expanded_update,
    )
    sdf_game_directory: bpy.props.StringProperty(
        name="Game Directory",
        description="AFOP game root folder (e.g. '...\\Ubisoft\\AFOP')",
        get=_get_sdf_game_directory,
        set=_set_sdf_game_directory,
    )
    sdf_extracted_directory: bpy.props.StringProperty(
        name="Extracted Files",
        description="Folder for MMB, MCloth, MReflex, and Texture files extracted from SDF archives",
        get=_get_sdf_extracted_directory,
        set=_set_sdf_extracted_directory,
    )
    sdf_assets: bpy.props.CollectionProperty(
        type=SDFAssetListItem,
        options={'SKIP_SAVE'},
    )
    sdf_show_mmb: bpy.props.BoolProperty(
        name="MMB",
        description="Show MMB mesh assets in search results",
        default=True,
        options={'SKIP_SAVE'},
        update=_on_sdf_search_update,
    )
    sdf_show_mgraphobject: bpy.props.BoolProperty(
        name="MGraphObject",
        description="Show MGraphObject files in search results",
        default=False,
        options={'SKIP_SAVE'},
        update=_on_sdf_search_update,
    )
    sdf_show_mcompoundnode: bpy.props.BoolProperty(
        name="MCompoundNode",
        description="Show MCompoundNode files in search results",
        default=False,
        options={'SKIP_SAVE'},
        update=_on_sdf_search_update,
    )
    sdf_show_rogue: bpy.props.BoolProperty(
        name="Rogue",
        description="Include Rogue assets in search results",
        default=True,
        options={'SKIP_SAVE'},
        update=_on_sdf_search_update,
    )
    sdf_show_dlc1: bpy.props.BoolProperty(
        name="DLC1",
        description="Include DLC1 assets in search results",
        default=True,
        options={'SKIP_SAVE'},
        update=_on_sdf_search_update,
    )
    sdf_show_dlc2: bpy.props.BoolProperty(
        name="DLC2",
        description="Include DLC2 assets in search results",
        default=True,
        options={'SKIP_SAVE'},
        update=_on_sdf_search_update,
    )
    sdf_show_dlc3: bpy.props.BoolProperty(
        name="DLC3",
        description="Include DLC3 assets in search results",
        default=True,
        options={'SKIP_SAVE'},
        update=_on_sdf_search_update,
    )
    sdf_search: bpy.props.StringProperty(
        name="Search MMB, MGraphObject, and MCompoundNode paths",
        options={'SKIP_SAVE', 'TEXTEDIT_UPDATE'},
        update=_on_sdf_search_update,
    )
    sdf_search_applied: bpy.props.StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    sdf_search_result_status: bpy.props.StringProperty(options={'HIDDEN', 'SKIP_SAVE'})
    sdf_show_all_results: bpy.props.BoolProperty(
        default=False,
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    sdf_asset_index: bpy.props.IntProperty(default=-1, min=-1, options={'SKIP_SAVE'})
    sdf_result_generation: bpy.props.IntProperty(default=-1, options={'HIDDEN', 'SKIP_SAVE'})
    sdf_load_as_asset: bpy.props.BoolProperty(
        name="Load as Asset",
        description=(
            "The selected MMB will be loaded in 'MMB File' on import;\n"
            "for a MGraph or MCompound containing several MMBs, the last one will be loaded"
        ),
        default=True,
    )
    sdf_import_materials: bpy.props.BoolProperty(
        name="Import Materials and Textures",
        description="Extract referenced textures, and assign Blender materials to imported meshes",
        default=True,
    )
    banshee_pattern: bpy.props.EnumProperty(
        name="Pattern",
        description="Select a Banshee color pattern from the loaded game assets",
        items=_banshee_pattern_items,
        options={'SKIP_SAVE'},
    )
    banshee_pattern_status: bpy.props.StringProperty(
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    overwrite_existing: bpy.props.BoolProperty(
        name="Overwrite existing file",
        default=False,
        description=(
            "Replace the existing _MOD export; the original source file is "
            "never overwritten"
        ),
    )
    mesh_expanded: bpy.props.BoolVectorProperty(size=32, default=tuple([False]*32))
    bone_slots_expanded: bpy.props.BoolVectorProperty(size=32, default=tuple([False]*32))
    limit_total_vertex_groups: bpy.props.BoolProperty(
        name="Limit Total Vertex Groups",
        default=True,
        description="Limit Number of Weights per Vertex to the source LOD's limit by removing the lowest weights",
    )
    compute_normals_on_export: bpy.props.BoolProperty(
        name="Compute Normals on Export",
        default=False,
        description="Recompute normals on export.",
        update=_on_compute_normals_on_export_update,
    )
    export_normals: bpy.props.BoolProperty(
        name="Export Normals",
        description=(
            "Write normals into the exported file. When unchecked, preserve the original normals.\n"
            "Automatically forced on when vert count has changed."
        ),
        get=_get_export_normals,
        set=_set_export_normals,
    )
    export_weights: bpy.props.BoolProperty(
        name="Export Weights",
        description=(
            "Write bone weights into the exported file. When unchecked, preserve the original weights.\n"
            "Automatically forced on when vert count has changed."
        ),
        get=_get_export_weights,
        set=_set_export_weights,
    )
    export_vertex_colors: bpy.props.BoolProperty(
        name="Export Vertex Colors",
        default=False,
        description="Write vertex colors from Blender into the exported file. When unchecked, preserve the original vertex colors.",
        update=_on_export_vertex_colors_update,
    )
    cloth_donor_radius: bpy.props.FloatProperty(
        name="Cloth Donor Radius",
        default=0.05,
        min=0.001,
        soft_max=0.5,
        precision=3,
        subtype='DISTANCE',
        description="Vertices added to a '_CLOTH_RENDER' mesh inherit cloth behavior from the nearest original vertex within this distance.",
    )
    force_cloth_recook: bpy.props.BoolProperty(
        name="Force Recook",
        default=False,
        description=(
            "Force every mcloth stream to receive a geometry-neutral "
            "SIM-section byte change, invalidating cached cloth mappings"
        ),
    )
    export_uvs: bpy.props.BoolProperty(
        name="Export UVs",
        description=(
            "Write UV coordinates from Blender into the exported file. When unchecked, preserve the original UVs.\n"
            "Automatically forced on when vert count has changed."
        ),
        get=_get_export_uvs,
        set=_set_export_uvs,
    )
    export_options_expanded: bpy.props.BoolProperty(
        name="Export Options",
        default=True,
    )

CLASSES = (
    AFOPPreferences,
    SDFAssetListItem,
    MReflexNodeSettings,
    SWOMTSettings,
)

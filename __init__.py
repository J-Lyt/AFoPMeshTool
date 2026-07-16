# Original author: AlexPo
bl_info = {
    "name": "AFoP Mesh Tool",
    "author": "JasperZebra, J-Lyt, SaintBaron",
    "location": "Scene Properties > AFoP Mesh Tool Panel",
    "version": (0, 1, 89),
    "blender": (5, 0, 0),
    "description": "Imports skeletal meshes from AFoP .mmb files. Supports versions 11-17.",
    "category": "Import-Export",
}

import logging
import os
import shutil


_bootstrap_logger = logging.getLogger("afop_mesh_tool")


# Delete the package cache before importing split modules. Blender updates are
# restart-based, so no loaded submodule is expected to survive this import.
_cache_dir = os.path.join(os.path.dirname(__file__), "__pycache__")
try:
    if os.path.exists(_cache_dir):
        shutil.rmtree(_cache_dir)
except OSError:
    pass


# v0.1.70 and older only download __init__.py, mcloth.py, and data files when
# applying an update. This one-time bridge retrieves missing v0.1.71 modules
# before importing them. Fresh zip installs and all later updates already carry
# the complete manifest, so the network path is not used normally.
_RAW_BASE = "https://raw.githubusercontent.com/J-Lyt/AFoPMeshTool/master"
_REQUIRED_SPLIT_MODULES = (
    "addon_state.py",
    "binary_io.py",
    "blender_mesh_utils.py",
    "cloth_export.py",
    "exporter.py",
    "file_utils.py",
    "importer.py",
    "log.py",
    "mmb.py",
    "operators_bones.py",
    "operators_files.py",
    "operators_io.py",
    "operators_mesh.py",
    "settings.py",
    "ui.py",
    "updater.py",
)


def _bootstrap_split_modules():
    missing = [name for name in _REQUIRED_SPLIT_MODULES
               if not os.path.isfile(os.path.join(os.path.dirname(__file__), name))]
    if not missing:
        return
    import urllib.request

    payloads = {}
    try:
        for filename in missing:
            with urllib.request.urlopen(f"{_RAW_BASE}/{filename}", timeout=30) as request:
                data = request.read()
            compile(data.decode("utf-8"), filename, "exec")
            payloads[filename] = data
    except Exception as error:
        raise ImportError(
            f"AFoP Mesh Tool update is missing {filename} and could not "
            f"download it: {error}. Reinstall the add-on from a complete zip."
        ) from error

    staged = {}
    installed = []
    try:
        for filename, data in payloads.items():
            destination = os.path.join(os.path.dirname(__file__), filename)
            temporary = destination + ".bootstrap_tmp"
            with open(temporary, "wb") as stream:
                stream.write(data)
            staged[filename] = temporary
        for filename in missing:
            destination = os.path.join(os.path.dirname(__file__), filename)
            os.replace(staged[filename], destination)
            installed.append(destination)
            _bootstrap_logger.info("Bootstrapped missing module %s", filename)
    except Exception as error:
        for destination in installed:
            try:
                os.remove(destination)
            except OSError:
                pass
        raise ImportError(
            f"AFoP Mesh Tool could not install its split modules: {error}. "
            "Reinstall the add-on from a complete zip."
        ) from error
    finally:
        for temporary in staged.values():
            try:
                if os.path.exists(temporary):
                    os.remove(temporary)
            except OSError:
                pass


_bootstrap_split_modules()

import bpy

from . import operators_bones
from . import operators_files
from . import operators_io
from . import operators_mesh
from . import settings
from . import ui
from . import updater


# Preserve the historical registration order so saved Blender files and
# operator availability behave exactly as before the module split.
classes = (
    settings.AFOPPreferences,
    settings.SWOMTSettings,
    operators_io.BrowseMMBFile,
    operators_mesh.ComputeNormals,
    operators_mesh.ClearNormals,
    operators_io.ImportAllLOD0s,
    operators_io.ImportAllLOD1s,
    operators_io.ImportAllLOD2s,
    operators_io.ImportAllLOD3s,
    operators_io.ExportAllLODs,
    operators_bones.RemapMeshBone,
    operators_bones.AddMeshBone,
    operators_bones.MergeSkeletons,
    operators_bones.MergeSkeletonsPickBones,
    operators_bones.AddBonesFromVertexGroups,
    updater.CheckForUpdates,
    updater.ApplyUpdate,
    ui.SWOMTPanel,
    operators_io.LoadMMB,
    operators_io.ImportLOD,
    operators_io.ExportLOD,
    operators_mesh.GenerateLODs,
    operators_mesh.ScaleUVs,
    operators_mesh.RenameMesh,
    operators_files.SelectMGraphObject,
    operators_files.RenameMMBFile,
    operators_files.SelectMGraphObjectFilePatch,
    operators_bones.ExportPosedBoneMatrices,
    operators_mesh.RemoveMesh,
    operators_mesh.RevertMesh,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.SWOMT = bpy.props.PointerProperty(type=settings.SWOMTSettings)
    if settings._on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(settings._on_load_post)
    settings.apply_debug_logging_preference()
    updater.start_update_check()


def unregister():
    if settings._on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(settings._on_load_post)
    if hasattr(bpy.types.Scene, "SWOMT"):
        del bpy.types.Scene.SWOMT
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()

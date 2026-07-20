"""Headless Blender smoke test for the MMB import/export pipeline.

Run from the repository root with Blender 5.0 or newer::

    blender --background --factory-startup --python-exit-code 1 \
        --python tools/mesh_pipeline_regression.py -- --mmb D:\\path\\asset.mmb

The source asset is never modified. The exported MMB is written to a temporary
directory, parsed again, and removed when the test finishes.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

import bpy


REPOSITORY = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "afop_mesh_pipeline_regression"


def _arguments():
    values = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmb", required=True, type=Path, help="Source MMB to test")
    return parser.parse_args(values)


def _load_addon_package():
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        REPOSITORY / "__init__.py",
        submodule_search_locations=[str(REPOSITORY)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create an import specification for the add-on")
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _require_finished(label, result):
    if result != {"FINISHED"}:
        raise RuntimeError(f"{label} returned {sorted(result)}")


def main():
    source = _arguments().mmb.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    addon = _load_addon_package()
    addon.register()
    try:
        with tempfile.TemporaryDirectory(prefix="afop_mesh_pipeline_") as directory:
            settings = bpy.context.scene.SWOMT
            settings.AssetPath = str(source)
            settings.ExportPath = directory

            _require_finished("load", bpy.ops.object.load_mmb())
            source_asset = addon.addon_state.asset
            source_shape = tuple(len(mesh.lods) for mesh in source_asset.meshes)

            mesh_count_before = len(bpy.data.meshes)
            _require_finished("LOD0 import", bpy.ops.object.import_all_lod0s())
            imported_count = len(bpy.data.meshes) - mesh_count_before
            if imported_count <= 0:
                raise RuntimeError("LOD0 import did not create any Blender meshes")

            _require_finished("export", bpy.ops.object.export_all_lods())
            exported_path = Path(settings.AssetPath)
            if not exported_path.is_file() or exported_path.parent != Path(directory):
                raise RuntimeError(f"Expected a temporary exported MMB, got {exported_path}")

            from afop_mesh_pipeline_regression.formats import mcloth
            from afop_mesh_pipeline_regression.formats.mmb import SkeletalMeshAsset

            with exported_path.open("rb") as stream:
                exported_asset = SkeletalMeshAsset()
                exported_asset.parse(stream)
            exported_shape = tuple(len(mesh.lods) for mesh in exported_asset.meshes)
            if exported_shape != source_shape:
                raise RuntimeError(
                    f"Exported mesh/LOD shape changed: {source_shape} -> {exported_shape}"
                )

            cloth_note = ""
            source_cloth = source.with_suffix(".mcloth")
            if source_cloth.is_file():
                exported_cloth = exported_path.with_suffix(".mcloth")
                if not exported_cloth.is_file():
                    raise RuntimeError("Paired cloth export was not created")
                source_streams, _ = mcloth.parse_streams(source_cloth.read_bytes())
                exported_streams, _ = mcloth.parse_streams(exported_cloth.read_bytes())
                if len(exported_streams) != len(source_streams):
                    raise RuntimeError(
                        "Exported cloth stream count changed: "
                        f"{len(source_streams)} -> {len(exported_streams)}"
                    )
                cloth_note = f" and preserved {len(exported_streams)} cloth stream(s)"

            print(
                "[PASS] mesh pipeline imported "
                f"{imported_count} LOD0 mesh(es), exported {exported_path.name}, "
                f"parsed it successfully{cloth_note}"
            )
    finally:
        addon.unregister()


if __name__ == "__main__":
    main()

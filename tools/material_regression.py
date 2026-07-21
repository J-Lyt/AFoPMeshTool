"""Headless Blender regression checks for AFoP material importing.

Run from the repository root with Blender 5.0 or newer::

    blender --background --factory-startup --python-exit-code 1 \
        --python tools/material_regression.py

The checks use generated meshes and PNG images only.  No extracted game assets
are required.  A failed assertion raises at the end so the Blender process
returns a non-zero exit code suitable for local automation.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import bpy


REPOSITORY = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "afop_material_regression"
_case_number = 0
_failures = []


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


ADDON = _load_addon_package()
from afop_material_regression import (  # noqa: E402
    material_import,
    operators_sdf,
    updater,
)
from afop_material_regression.formats import (  # noqa: E402
    banshee_patterns, mgraph, shader_schema,
)
from afop_material_regression.materials import registry as material_profile_registry
from afop_material_regression.materials import profiles as material_profiles

_audit_spec = importlib.util.spec_from_file_location(
    "afop_material_audit_tool", REPOSITORY / "tools" / "material_audit.py"
)
if _audit_spec is None or _audit_spec.loader is None:
    raise RuntimeError("Could not load the standalone material-audit report engine")
material_audit = importlib.util.module_from_spec(_audit_spec)
_audit_spec.loader.exec_module(material_audit)


def _material_profile_registry_case():
    registry = material_profile_registry
    check(
        material_import.supported_auxiliary_paths
        is registry.supported_auxiliary_paths,
        "material-import facade bypasses the profile registry",
    )
    expected_traits = {
        "px_character_navi_face.mshader": {"navi_skin", "navi_face"},
        "px_hair2_3color_tousle.mshader": {"hair"},
        "px_wildlife_skin.mshader": {"wildlife_skin"},
        "px_dlc3_medusa_skin.mshader": {"medusa_skin"},
        "px_basic_rustymetal_static.mshader": {"rusty_metal"},
        "px_character_gear_vhq.mshader": {"character_gear_vhq"},
        "px_rustymetal_vcoverlay_rda_dlc3.mshader": {
            "rusty_metal_vcoverlay"
        },
        "px_dlc3_vegetation_static.mshader": {"vegetation"},
    }
    for shader_name, expected in expected_traits.items():
        actual = registry.profile_traits(shader_name)
        check(
            expected.issubset(actual),
            f"profile registry did not classify {shader_name}: {sorted(actual)}",
        )
    hair_binding = {
        "shader": "blue/shaders/px_hair2_3color_tousle.mshader",
        "aux": {
            "HairMaps": "textures/hair_m.dds",
            "DirectionMap": "textures/hair_dir.dds",
            "AO": "textures/hair_ao.dds",
            "DetailNormal": "textures/unrelated_detail.dds",
        },
    }
    check(
        material_import.supported_auxiliary_paths(hair_binding)
        == (
            "textures/hair_m.dds",
            "textures/hair_dir.dds",
            "textures/hair_ao.dds",
        ),
        "hair auxiliary textures are not routed by the registered profile",
    )

    rusty_binding = {
        "shader": "dlc3/shaders/PX_RustyMetal_VCoverlay_RDA_DLC3.mshader",
        "aux": {
            "AlbedoTexture": "textures/blade_albedo.dds",
            "DetailNormal1": "textures/blade_detail_1.dds",
            "DetailNormal2": "textures/blade_detail_2.dds",
            "RustyMetalMask": "textures/blade_mask.dds",
            "PaintTexture": "textures/blade_paint.dds",
        },
    }
    check(
        material_import.supported_auxiliary_paths(rusty_binding)
        == (
            *rusty_binding["aux"].values(),
            *material_profile_registry.RUSTY_METAL_VC_DEFAULT_TEXTURES.values(),
        ),
        "DLC3 rusty-metal auxiliary textures are not all requested",
    )


def _banshee_pattern_parser_case():
    manifest = banshee_patterns.BansheePatternData.loads(b'''\
include blue/gameplay/vanity/fruit/bansheepatterndata.fruit
include "blue/vanity/banshee body.mcolorpattern"
include "blue/vanity/banshee head.mcolorpattern"
include "blue/vanity/banshee body.mpatterncontrol"
include "blue/vanity/banshee head.mpatterncontrol"
BansheePatternData test_pattern < uid=ABCDEF >
{
 myBodyColorPattern < uid=1111 > = "banshee body" AABBCCDD
 myHeadPatternControl < uid=2222 > = "banshee head" 11223344
 myBodyPatternCoat blue/baked/body_pc.dds
 myHeadPatternCoat blue/baked/head_pc.dds
}
''')
    members = manifest.member_paths()
    check(manifest.name == "test_pattern", "Banshee manifest name was not parsed")
    check(
        members[("body", "color")].endswith("banshee body.mcolorpattern")
        and members[("head", "coat")] == "blue/baked/head_pc.dds",
        f"Banshee manifest members were not resolved: {members!r}",
    )
    check(
        manifest.reference_targets() == {
            ("body", "color"): "AABBCCDD",
            ("head", "control"): "11223344",
        },
        "Banshee manifest target UIDs were not parsed",
    )
    colors = banshee_patterns.ColorPattern.loads(b'''\
ColorPattern "test colours" < uid=1234 >
{
 myColor1 0xff804020
 myColor10 0xfe102040
}
''')
    check(
        colors.rgb(0) == (128 / 255.0, 64 / 255.0, 32 / 255.0),
        "ARGB Banshee colour was not converted to RGB",
    )
    control = banshee_patterns.PatternControl.loads(b'''\
PatternControl "test control" < uid=5678 >
{
 myPattern1Invert -1.0
 myPattern2Invert 1.0
 myPattern1LevelControl 1.7
 myPattern2LevelControl 0.3
}
''')
    check(
        (control.invert1, control.invert2, control.level1, control.level2)
        == (-1.0, 1.0, 1.7, 0.3),
        "Banshee pattern controls were not parsed",
    )


def _banshee_graph_pattern_reference_case():
    tree = {
        "nodesById": {
            10: {
                "type": "internal:Compound",
                "filename": "blue/nodes/compounds/wl_banshee_character_compound.mcompoundnode",
            },
            20: {"type": "native:Constant", "valueType": "texture", "value": "textures/body_pc.dds"},
            21: {"type": "native:Constant", "valueType": "texture", "value": "textures/head_pc.dds"},
            22: {"type": "native:Constant", "valueType": "Color Pattern", "value": ["#&BODYCOLOR"]},
            23: {"type": "native:Constant", "valueType": "Color Pattern", "value": ["#&HEADCOLOR"]},
            24: {"type": "native:Constant", "valueType": "Pattern Control", "value": ["#&BODYCONTROL"]},
            25: {"type": "native:Constant", "valueType": "Pattern Control", "value": ["#&HEADCONTROL"]},
        },
        "connectionsById": {
            1: [20, 0, 10, 124], 2: [21, 0, 10, 125],
            3: [22, 0, 10, 126], 4: [23, 0, 10, 127],
            5: [24, 0, 10, 128], 6: [25, 0, 10, 129],
        },
    }

    class FakeDecoder:
        def __init__(self, _data):
            pass

        def root(self):
            return tree

    original = mgraph._Bv2Decoder
    try:
        mgraph._Bv2Decoder = FakeDecoder
        detected = mgraph.banshee_pattern_bindings(b"synthetic")
    finally:
        mgraph._Bv2Decoder = original
    check(
        detected == {
            "body": {
                "coat": "textures/body_pc.dds",
                "color_uid": "BODYCOLOR",
                "control_uid": "BODYCONTROL",
            },
            "head": {
                "coat": "textures/head_pc.dds",
                "color_uid": "HEADCOLOR",
                "control_uid": "HEADCONTROL",
            },
        },
        f"Banshee graph UID references were not mapped by body/head: {detected!r}",
    )


def _nested_update_install_case():
    with tempfile.TemporaryDirectory(prefix="afop_nested_update_") as directory:
        destination = Path(directory)
        payloads = {
            "material_import.py": b"VALUE = 'facade'\n",
            "materials/__init__.py": b"VALUE = 'package'\n",
            "materials/profiles.py": b"VALUE = 'profiles'\n",
        }
        original_plugin_dir = updater._plugin_dir
        original_update_files = updater.UPDATE_FILES
        try:
            updater._plugin_dir = lambda: str(destination)
            updater.UPDATE_FILES = tuple(payloads)
            updater._install_payloads(payloads)
        finally:
            updater._plugin_dir = original_plugin_dir
            updater.UPDATE_FILES = original_update_files

        for relative_path, expected in payloads.items():
            installed = destination / relative_path
            check(installed.is_file(), f"updater did not create {relative_path}")
            check(
                installed.read_bytes() == expected,
                f"updater corrupted nested payload {relative_path}",
            )

_audit_cli_spec = importlib.util.spec_from_file_location(
    "afop_material_audit_cli", REPOSITORY / "tools" / "audit_material_corpus.py"
)
if _audit_cli_spec is None or _audit_cli_spec.loader is None:
    raise RuntimeError("Could not load the standalone material-audit command")
audit_material_corpus = importlib.util.module_from_spec(_audit_cli_spec)
_audit_cli_spec.loader.exec_module(audit_material_corpus)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def run_case(name, callback):
    try:
        callback()
    except Exception as exc:  # Each case reports before the final non-zero exit.
        _failures.append((name, exc, traceback.format_exc()))
        print(f"[FAIL] {name}: {exc}")
    else:
        print(f"[PASS] {name}")


def _make_source_png(directory):
    image = bpy.data.images.new("AFOP regression source", width=2, height=2)
    image.pixels = (
        0.8, 0.2, 0.1, 0.5,
        0.2, 0.8, 0.1, 1.0,
        0.1, 0.2, 0.8, 0.25,
        0.5, 0.5, 0.5, 0.75,
    )
    path = Path(directory) / "source.png"
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)
    return path


def _logical_texture_paths(binding):
    paths = []
    for field in ("d", "n", "m", "a"):
        path = binding.get(field)
        if isinstance(path, str):
            paths.append(path)
    for path in binding.get("aux", {}).values():
        if isinstance(path, str):
            paths.append(path)
    paths.extend(material_import.supported_auxiliary_paths(binding))
    return list(dict.fromkeys(paths))


def _texture_files(binding, directory, source_png):
    result = {}
    for index, logical_path in enumerate(_logical_texture_paths(binding)):
        disk_path = Path(directory) / f"texture_{index:02d}.png"
        shutil.copyfile(source_png, disk_path)
        result[logical_path.casefold()] = str(disk_path)
    return result


def _make_mesh_object(name):
    mesh_data = bpy.data.meshes.new(f"{name}_data")
    mesh_data.from_pydata(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        [],
        [(0, 1, 2)],
    )
    mesh_data.uv_layers.new(name="UVMap_0")
    mesh_data.uv_layers.new(name="UVMap_1")
    colors = mesh_data.color_attributes.new(
        name="Color_0", type="FLOAT_COLOR", domain="POINT"
    )
    for item in colors.data:
        item.color = (1.0, 1.0, 1.0, 1.0)
    obj = bpy.data.objects.new(name, mesh_data)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _assign(binding, temporary_directory, source_png, part_suffix="Mesh"):
    global _case_number
    _case_number += 1
    part_name = f"Case{_case_number}_{part_suffix}"
    obj = _make_mesh_object(f"AFOP_regression_{_case_number}")
    part = SimpleNamespace(
        name=part_name,
        lods=[SimpleNamespace(blender_obj_name=obj.name)],
    )
    asset = SimpleNamespace(name=f"Regression{_case_number}", meshes=[part])
    texture_files = _texture_files(binding, temporary_directory, source_png)
    assigned = material_import.assign_materials(
        asset,
        {part_name: binding},
        texture_files,
        "synthetic/material_regression.mgraphobject",
    )
    check(assigned == 1, f"expected one assigned material, got {assigned}")
    check(len(obj.data.materials) == 1, "material was not attached to the mesh")
    return obj.data.materials[0]


def _principled(material):
    node = next(
        (
            item
            for item in material.node_tree.nodes
            if item.bl_idname == "ShaderNodeBsdfPrincipled"
        ),
        None,
    )
    check(node is not None, "Principled BSDF node is missing")
    return node


def _assert_profile(material, expected):
    actual = material.get("afop_shader_profile")
    check(actual == expected, f"expected profile {expected!r}, got {actual!r}")


def _assert_linked(socket, label):
    check(bool(socket.links), f"{label} is not connected")


def _binding_resolution_cases():
    original_decoder = mgraph._Bv2Decoder
    original_plain = mgraph._bv2_plain
    trees = {}

    class FakeDecoder:
        def __init__(self, data):
            self.data = data

        def root(self):
            return trees[self.data]

    try:
        mgraph._Bv2Decoder = FakeDecoder
        mgraph._bv2_plain = lambda value: value

        trees[b"direct"] = {
            "nodesById": {
                1: {"type": "native:Constant", "value": "textures/body_d.dds"},
                2: {
                    "MeshName": "Banshee_Body",
                    "ShaderFile": "blue/shaders/PX_Test.mshader",
                },
            },
            "connectionsById": {1: [1, 0, 2, 9101]},
        }
        resolved = mgraph._connected_material_role_textures(
            b"direct", {"px_test.mshader": {101: "d"}}
        )
        check(
            resolved == {"banshee_body": {"d": "textures/body_d.dds"}},
            f"unexpected direct binding result: {resolved!r}",
        )

        trees[b"empty-schema"] = {
            "nodesById": {
                1: {"type": "native:Constant", "value": "textures/fur_d.dds"},
                2: {
                    "MeshName": "Fur",
                    "ShaderFile": "blue/shaders/PX_Character_Gear_Fur.mshader",
                },
            },
            "connectionsById": {1: [1, 0, 2, 9120]},
        }
        resolved = mgraph._connected_material_role_textures(
            b"empty-schema", {"px_character_gear_fur.mshader": {}}
        )
        check(
            resolved == {},
            "an authoritative empty shader schema incorrectly used legacy fallback pins",
        )

        compound_path = "blue/graphs/green_moss_clusters_a.mcompoundnode"
        trees[b"parent"] = {
            "nodesById": {
                1: {"type": "native:Constant", "value": "textures/moss_d.dds"},
                2: {"type": "internal:Compound", "filename": compound_path},
            },
            "connectionsById": {1: [1, 0, 2, 103]},
        }
        trees[b"compound"] = {
            "nodesById": {
                1: {"type": "internal:CompoundInputs"},
                2: {
                    "MeshName": "GreenMoss_Clusters_A_LOD0",
                    "ShaderFile": "blue/shaders/PX_Natural_Moss.mshader",
                },
            },
            "connectionsById": {1: [1, 103, 2, 9101]},
        }
        resolved = mgraph._forwarded_compound_role_textures(
            b"parent",
            {compound_path: b"compound"},
            {"px_natural_moss.mshader": {101: "d"}},
        )
        expected = {
            "greenmoss_clusters_a_lod0": {"d": "textures/moss_d.dds"},
        }
        check(resolved == expected, f"unexpected compound binding result: {resolved!r}")
    finally:
        mgraph._Bv2Decoder = original_decoder
        mgraph._bv2_plain = original_plain


def _direct_material_source_discovery_case():
    target_path = "blue/baked/wl_test_01.mmb"
    direct_graph_path = "blue/graphs/wl_test_direct.mgraphobject"
    indirect_graph_path = "blue/graphs/wl_test_indirect.mgraphobject"
    direct_compound_path = "blue/graphs/wl_test_direct.mcompoundnode"
    indirect_compound_path = "blue/graphs/wl_test_indirect.mcompoundnode"
    payloads = {
        direct_graph_path: mgraph.MAGIC + b"direct-graph",
        indirect_graph_path: mgraph.MAGIC + b"indirect-graph",
        direct_compound_path: mgraph.MAGIC + b"direct-compound",
        indirect_compound_path: mgraph.MAGIC + b"indirect-compound",
    }
    nested_payload = mgraph.MAGIC + b"nested-compound"

    class FakeArchive:
        def extract(self, asset):
            return payloads[asset.name]

    archive = FakeArchive()

    def indexed(path):
        return operators_sdf._IndexedAsset(
            archive=archive,
            asset=SimpleNamespace(name=path),
            archive_label="synthetic",
            cache_key="synthetic",
        )

    target = indexed(target_path)
    graph_entries = [indexed(direct_graph_path), indexed(indirect_graph_path)]
    compound_entries = [indexed(direct_compound_path), indexed(indirect_compound_path)]
    original_graphs = operators_sdf._state.graphs
    original_compounds = operators_sdf._state.compounds
    original_referenced_meshes = mgraph.referenced_meshes
    original_compound_data = operators_sdf._referenced_compound_data
    original_richness = operators_sdf._material_source_richness
    traversed = []

    def referenced_meshes(data):
        if data in (payloads[direct_graph_path], payloads[direct_compound_path], nested_payload):
            return [target_path]
        return []

    def compound_data(data, _archive):
        traversed.append(data)
        if data in (payloads[indirect_graph_path], payloads[indirect_compound_path]):
            return {"blue/graphs/nested.mcompoundnode": nested_payload}
        return {}

    try:
        operators_sdf._state.graphs = {
            entry.asset.name.casefold(): [entry] for entry in graph_entries
        }
        operators_sdf._state.compounds = {
            entry.asset.name.casefold(): [entry] for entry in compound_entries
        }
        mgraph.referenced_meshes = referenced_meshes
        operators_sdf._referenced_compound_data = compound_data
        operators_sdf._material_source_richness = lambda _data, _compounds: 1

        graph_options = operators_sdf._material_graph_options(target)
        compound_options = operators_sdf._material_compound_options(target)
        check(
            [item[0].asset.name for item in graph_options] == [direct_graph_path],
            "automatic MMB discovery accepted an indirectly linked graph",
        )
        check(
            [item[0].asset.name for item in compound_options] == [direct_compound_path],
            "automatic MMB discovery accepted an indirectly linked compound",
        )
        check(
            set(traversed) == {
                payloads[direct_graph_path],
                payloads[direct_compound_path],
            },
            "automatic MMB discovery traversed a source before confirming its direct link",
        )

        recursive_references, _sources = operators_sdf._material_source_references(
            payloads[indirect_graph_path], archive
        )
        check(
            recursive_references == [target_path],
            "explicit graph/compound traversal no longer resolves nested MMB references",
        )
    finally:
        operators_sdf._state.graphs = original_graphs
        operators_sdf._state.compounds = original_compounds
        mgraph.referenced_meshes = original_referenced_meshes
        operators_sdf._referenced_compound_data = original_compound_data
        operators_sdf._material_source_richness = original_richness


def _multi_mmb_import_selection_case():
    source_path = "blue/graphs/wildlife_test.mgraphobject"
    direct_path = "blue/baked/wildlife_direct.mmb"
    linked_path = "blue/baked/wildlife_linked.mmb"
    source_data = mgraph.MAGIC + b"source"
    linked_data = mgraph.MAGIC + b"linked"

    class FakeArchive:
        def extract(self, asset):
            check(asset.name == source_path, "unexpected asset extraction during scope test")
            return source_data

    archive = FakeArchive()

    def indexed(path):
        return operators_sdf._IndexedAsset(
            archive=archive,
            asset=SimpleNamespace(name=path),
            archive_label="synthetic",
            cache_key="synthetic",
        )

    source = indexed(source_path)
    direct = indexed(direct_path)
    linked = indexed(linked_path)
    original_entries = operators_sdf._state.entries
    original_referenced_meshes = mgraph.referenced_meshes
    original_compound_data = operators_sdf._referenced_compound_data

    def referenced_meshes(data):
        if data == source_data:
            return [direct_path]
        if data == linked_data:
            return [linked_path]
        return []

    try:
        operators_sdf._state.entries = [direct, linked]
        mgraph.referenced_meshes = referenced_meshes
        operators_sdf._referenced_compound_data = (
            lambda _data, _archive: {
                "blue/graphs/wildlife_linked.mcompoundnode": linked_data
            }
        )

        entries, missing = operators_sdf._referenced_mmb_entries(source)
        check(
            [entry.asset.name for entry in entries] == [direct_path, linked_path],
            "graph import did not resolve both direct and linked MMB references",
        )
        check(not missing, "graph import unexpectedly lost an indexed MMB")
        choices = [
            SimpleNamespace(
                cache_key=direct.cache_key,
                asset_path=direct.asset.name,
                selected=False,
            ),
            SimpleNamespace(
                cache_key=linked.cache_key,
                asset_path=linked.asset.name,
                selected=True,
            ),
        ]
        selected = operators_sdf._selected_mmb_entries(entries, choices)
        check(
            [entry.asset.name for entry in selected] == [linked_path],
            "multi-MMB dialog selection did not filter the resolved import targets",
        )
        check(
            operators_sdf._selected_mmb_entries(entries, []) == entries,
            "single/no-dialog graph import no longer retains every resolved MMB",
        )
    finally:
        operators_sdf._state.entries = original_entries
        mgraph.referenced_meshes = original_referenced_meshes
        operators_sdf._referenced_compound_data = original_compound_data


class _SDFResultCollection(list):
    def add(self):
        item = SimpleNamespace()
        self.append(item)
        return item


def _sdf_search_settings(**overrides):
    values = dict(
        sdf_assets=_SDFResultCollection(),
        sdf_asset_index=-1,
        sdf_search_applied="",
        sdf_search_result_status="",
        sdf_show_all_results=False,
        sdf_show_mmb=True,
        sdf_show_mgraphobject=False,
        sdf_show_mcompoundnode=False,
        sdf_show_rogue=True,
        sdf_show_dlc1=True,
        sdf_show_dlc2=True,
        sdf_show_dlc3=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _sdf_archive_filter_case():
    settings = _sdf_search_settings()
    scene = SimpleNamespace(SWOMT=settings)
    labels = ("rogue", "dlc1", "DLC2", "dlc3", "future_archive")
    entries = [
        (
            operators_sdf._ASSET_MMB,
            operators_sdf._IndexedAsset(
                archive=None,
                asset=SimpleNamespace(name=f"blue/baked/shared_{index}.mmb"),
                archive_label=label,
                cache_key=label,
            ),
        )
        for index, label in enumerate(labels)
    ]
    original_entries = operators_sdf._state.search_entries
    original_generation = operators_sdf._state.generation
    try:
        operators_sdf._state.search_entries = entries
        operators_sdf._state.generation = original_generation + 1
        operators_sdf.populate_search_results(scene, "shared")
        check(
            [item.archive_label for item in settings.sdf_assets] == list(labels),
            "enabled-by-default archive filters hid search results",
        )

        settings.sdf_show_rogue = False
        settings.sdf_show_dlc2 = False
        operators_sdf.populate_search_results(scene, "shared")
        check(
            [item.archive_label for item in settings.sdf_assets]
            == ["dlc1", "dlc3", "future_archive"],
            "archive filters did not hide only the selected Rogue/DLC archives",
        )
    finally:
        operators_sdf._state.search_entries = original_entries
        operators_sdf._state.generation = original_generation


def _sdf_show_all_results_case():
    settings = _sdf_search_settings(sdf_search_applied="shared")
    scene = SimpleNamespace(SWOMT=settings)
    entries = [
        (
            operators_sdf._ASSET_MMB,
            operators_sdf._IndexedAsset(
                archive=None,
                asset=SimpleNamespace(name=f"blue/baked/shared_{index}.mmb"),
                archive_label="rogue",
                cache_key=f"rogue_{index}",
            ),
        )
        for index in range(501)
    ]
    original_entries = operators_sdf._state.search_entries
    original_generation = operators_sdf._state.generation
    try:
        operators_sdf._state.search_entries = entries
        operators_sdf._state.generation = original_generation + 1
        operators_sdf.populate_search_results(scene, settings.sdf_search_applied)
        check(len(settings.sdf_assets) == 500, "search result safety cap changed")
        check(
            settings.sdf_search_result_status.startswith("Showing first 500"),
            "truncated search did not show its result-limit message",
        )

        result = operators_sdf.ShowAllSDFResults.execute(
            None, SimpleNamespace(scene=scene)
        )
        check(result == {"FINISHED"}, "Show All search action did not finish")
        check(len(settings.sdf_assets) == 501, "Show All did not reveal every result")
        check(
            not settings.sdf_search_result_status,
            "Show All left the truncated-result message visible",
        )
    finally:
        operators_sdf._state.search_entries = original_entries
        operators_sdf._state.generation = original_generation


def _direhorse_weakpoint_binding_case():
    body_paths = {
        "d": "blue/baked/wildlife/direhorse/wildlife_direhorse_01_d.dds",
        "n": "blue/baked/wildlife/direhorse/wildlife_direhorse_01_nr.dds",
        "m": "blue/baked/wildlife/direhorse/wildlife_direhorse_01_m.dds",
    }
    eye_paths = {
        "d": "blue/baked/wildlife/banshee/eddie_eye_d.dds",
        "n": "blue/baked/wildlife/banshee/eddie_eye_n.dds",
        "m": None,
    }
    groups = [
        {"base": "wildlife_direhorse_01", "paths": list(body_paths.values()), **body_paths},
        {
            "base": "eddie_eyes",
            "paths": [path for path in eye_paths.values() if path],
            **eye_paths,
        },
    ]
    textures = [
        {"path": path, "kind": role}
        for role, path in body_paths.items()
    ] + [
        {"path": path, "kind": role}
        for role, path in eye_paths.items()
        if path
    ]
    bio_palette = [(float(index) / 10.0, 0.2, 0.4) for index in range(6)]
    wildlife_shader = "blue/shaders/PX_Wildlife_Skin.mshader"
    material_shaders = [
        ("body", wildlife_shader),
        ("wildlife_dirhorse_weakpoint", "blue/shaders/PX_Basic.mshader"),
        ("eyes", "blue/shaders/PX_Wildlife_Eye.mshader"),
    ]
    replacements = {
        "_player_head_sex": lambda _data, _names: None,
        "_graph_materials": lambda _data: material_shaders,
        "_decoded_graph_materials": lambda _data: material_shaders,
        "_connected_material_textures": lambda _data: {},
        "_connected_material_alpha_textures": lambda _data: {},
        "_connected_material_role_textures": lambda _data, _pins=None: {},
        "_connected_material_parameters": lambda _data, _pins=None: {},
        "_forwarded_compound_role_textures": lambda _data, _sources, _pins=None: {},
        "_wildlife_bio_parameters": lambda _data: {
            "body": {"bio_palette": bio_palette, "bio_strength": 1.0}
        },
        "_forwarded_wildlife_bio_parameters": lambda _data, _sources=None: {},
        "_texture_groups": lambda _data: groups,
        "texture_pool": lambda _data: textures,
    }
    originals = {name: getattr(mgraph, name) for name in replacements}
    try:
        for name, replacement in replacements.items():
            setattr(mgraph, name, replacement)
        bindings = mgraph.material_bindings(
            mgraph.MAGIC + b"synthetic",
            ["wildlife_dirhorse_weakpoint", "eyes", "eyes_small", "body"],
            shader_role_pins={
                "px_wildlife_skin.mshader": {101: "d", 102: "n", 103: "m"},
                "px_basic.mshader": {101: "d", 102: "n", 103: "m"},
            },
        )
    finally:
        for name, original in originals.items():
            setattr(mgraph, name, original)

    body = bindings["body"]
    weakpoint = bindings["wildlife_dirhorse_weakpoint"]
    check(weakpoint == body, "Direhorse weakpoint does not inherit the complete body binding")
    check(weakpoint["shader"] == wildlife_shader, "Direhorse weakpoint retained PX_Basic")
    check(weakpoint["d"] == body_paths["d"], "Direhorse weakpoint diffuse is not the body diffuse")
    check(weakpoint["n"] == body_paths["n"], "Direhorse weakpoint normal is not the body normal")
    check(weakpoint["m"] == body_paths["m"], "Direhorse weakpoint mask is not the body mask")
    check(weakpoint.get("bio_palette") == bio_palette, "Direhorse weakpoint lost the body bio palette")


def _wildlife_gameplay_part_alias_case():
    wildlife_shader = "blue/shaders/PX_Wildlife_Skin.mshader"
    body = {
        "shader": wildlife_shader,
        "d": "textures/armoredprey_body_d.dds",
        "n": "textures/armoredprey_body_n.dds",
        "m": "textures/armoredprey_body_m.dds",
        "aux": {"PatternCoat": "textures/armoredprey_pattern.dds"},
        "bio_palette": [(0.0, 0.1, 0.2), (0.1, 0.4, 0.8)],
    }
    head = {
        "shader": wildlife_shader,
        "d": "textures/creature_head_d.dds",
        "n": "textures/creature_head_n.dds",
        "m": "textures/creature_head_m.dds",
    }
    wrong = {
        "shader": "blue/shaders/PX_Emissive_Color.mshader",
        "d": None,
        "n": None,
        "m": None,
    }
    names = [
        "wildlife_armoredprey_01",
        "wildlife_armoredprey_01_armor",
        "wildlife_armoredprey_weakpoint",
        "creature_head",
        "creature_head_armor",
        "creature_head_weakpoint",
    ]
    bindings = {
        names[0]: dict(body),
        names[1]: dict(wrong),
        names[2]: dict(wrong),
        names[3]: dict(head),
        names[4]: dict(wrong),
        names[5]: dict(wrong),
    }
    mgraph._apply_wildlife_part_aliases(names, bindings)

    check(bindings[names[1]] == body, "Armored Prey armor did not inherit its body binding")
    check(bindings[names[2]] == body, "Armored Prey weakpoint did not inherit its body binding")
    check(bindings[names[4]] == head, "part-specific armor did not inherit its base binding")
    check(bindings[names[5]] == head, "part-specific weakpoint did not inherit its base binding")


def _named_mask_and_piercing_binding_case():
    check(
        mgraph._classify_texture("textures/g_hmn_mask_01_d.dds") == "d",
        "asset-family word 'mask' overrode an explicit diffuse suffix",
    )
    check(
        mgraph._classify_texture("textures/g_hmn_mask_01_n.dds") == "n",
        "asset-family word 'mask' overrode an explicit normal suffix",
    )
    check(
        mgraph._classify_texture("textures/g_hmn_mask_01_m.dds") == "m",
        "asset-family word 'mask' overrode an explicit material suffix",
    )

    body = {
        "shader": "dlc3/shaders/px_character_skin_vhq_body.mshader",
        "d": "textures/cus_115_body_d.dds",
        "n": "textures/cus_115_body_n.dds",
        "m": "textures/cus_115_body_m.dds",
    }
    piercing = {
        "shader": body["shader"],
        "d": body["d"],
        "n": body["n"],
        "m": "textures/cus_115_head_m.dds",
    }
    bindings = {"Body": dict(body), "piercing_body": piercing}
    mgraph._apply_named_part_texture_aliases(list(bindings), bindings)
    check(
        bindings["piercing_body"]["m"] == body["m"],
        "piercing_body did not inherit cus_115_body_m.dds",
    )


def _supplemental_graph_override_case():
    glass_name = "g_hmn_mask_01_Glass"
    mask_name = "g_hmn_mask_01_Mask"
    source_data = mgraph.MAGIC + b"cus-03-mask"

    class FakeArchive:
        def extract(self, _asset):
            return source_data

    archive = FakeArchive()
    primary = operators_sdf._IndexedAsset(
        archive=archive,
        asset=SimpleNamespace(name="blue/graph objects/gear/cus_03_gear.mgraphobject"),
        archive_label="synthetic",
        cache_key="synthetic",
    )
    supplemental = operators_sdf._IndexedAsset(
        archive=archive,
        asset=SimpleNamespace(name="blue/graph objects/gear/cus_03_mask.mgraphobject"),
        archive_label="synthetic",
        cache_key="synthetic",
    )
    mesh_entry = operators_sdf._IndexedAsset(
        archive=archive,
        asset=SimpleNamespace(
            name="blue/baked/characterart/npc/custom/cus_03/cus_03_gear.mmb"
        ),
        archive_label="synthetic",
        cache_key="synthetic",
    )
    expected = {
        glass_name: {
            "shader": "blue/shaders/PX_Glass_Simple.mshader",
            "d": "textures/g_rda_accessories_d.dds",
            "n": "textures/g_rda_accessories_n.dds",
            "m": "textures/g_rda_accessories_m.dds",
        },
        mask_name: {
            "shader": "blue/shaders/PX_Character_RDA_Human.mshader",
            "d": "textures/g_hmn_mask_01_d.dds",
            "n": "textures/g_hmn_mask_01_n.dds",
            "m": "textures/g_hmn_mask_01_m.dds",
        },
    }
    bindings = {
        name: {
            "shader": "blue/shaders/px_ch_cloth.mshader",
            "d": "textures/cus_03_upperbody_d.dds",
            "n": "textures/cus_03_upperbody_n.dds",
            "m": "textures/cus_03_upperbody_m.dds",
        }
        for name in expected
    }
    replacements = {
        "compound": operators_sdf._referenced_compound_data,
        "pins": operators_sdf._shader_pins_for_sources,
        "pool": mgraph.texture_pool,
        "materials": mgraph._graph_materials,
        "bindings": mgraph.material_bindings,
    }
    with operators_sdf._state.lock:
        original_graphs = operators_sdf._state.graphs
        operators_sdf._state.graphs = {
            primary.asset.name.casefold(): [primary],
            supplemental.asset.name.casefold(): [supplemental],
        }
    try:
        operators_sdf._referenced_compound_data = lambda _data, _archive: {}
        operators_sdf._shader_pins_for_sources = lambda _data, _sources, _archive: ({}, {})
        mgraph.texture_pool = lambda _data: [{"path": "textures/present.dds"}]
        mgraph._graph_materials = lambda _data: [
            (glass_name, expected[glass_name]["shader"]),
            (mask_name, expected[mask_name]["shader"]),
        ]
        mgraph.material_bindings = (
            lambda _data, names, **_kwargs: {
                name: dict(expected[name]) for name in names if name in expected
            }
        )
        result = operators_sdf._supplemental_material_bindings(
            mesh_entry,
            primary,
            list(expected),
            bindings,
            primary_material_names={"Lowerbody", "Upperbody", "Female_Body"},
        )
    finally:
        with operators_sdf._state.lock:
            operators_sdf._state.graphs = original_graphs
        operators_sdf._referenced_compound_data = replacements["compound"]
        operators_sdf._shader_pins_for_sources = replacements["pins"]
        mgraph.texture_pool = replacements["pool"]
        mgraph._graph_materials = replacements["materials"]
        mgraph.material_bindings = replacements["bindings"]

    check(result[glass_name] == expected[glass_name], "mask glass sibling graph was not authoritative")
    check(result[mask_name] == expected[mask_name], "mask body sibling graph was not authoritative")


def _banshee_layered_texture_fallback_case():
    diffuse = "textures/wildlife_banshee_01_body_d.dds"
    bindings = {
        "Banshee_Body": {
            "shader": "blue/shaders/PX_Wildlife_Skin_Banshee.mshader",
            "d": diffuse,
            "n": "snowdrop/textures/flat/sd_flat_normal_128_n.dds",
            "m": None,
        },
        "Banshee_Eyes": {
            "shader": "blue/shaders/PX_Wildlife_Eye.mshader",
            "d": "textures/banshee_eye_d.dds",
            "n": "textures/banshee_eye_n.dds",
            "m": None,
        },
    }
    indexed = {
        "textures/wildlife_banshee_01_body_nr.dds",
        "textures/wildlife_banshee_01_body_m.dds",
    }
    original_texture_entry = operators_sdf._texture_entry
    try:
        operators_sdf._texture_entry = (
            lambda path, _archive: object() if path.casefold() in indexed else None
        )
        result = operators_sdf._complete_banshee_skin_bindings(
            bindings, object()
        )
    finally:
        operators_sdf._texture_entry = original_texture_entry
    check(
        result["Banshee_Body"]["n"]
        == "textures/wildlife_banshee_01_body_nr.dds",
        "layered Banshee graph did not recover its matching normal texture",
    )
    check(
        result["Banshee_Body"]["m"]
        == "textures/wildlife_banshee_01_body_m.dds",
        "layered Banshee graph did not recover its matching material mask",
    )
    check(
        result["Banshee_Eyes"]["m"] is None,
        "Banshee skin fallback changed an unrelated eye material",
    )


def _cloth_sim_case(temporary_directory, source_png):
    global _case_number
    _case_number += 1
    part_name = f"Case{_case_number}_CLOTH_SIM"
    obj = _make_mesh_object(f"AFOP_regression_{_case_number}")
    asset = SimpleNamespace(
        name="ClothRegression",
        meshes=[SimpleNamespace(
            name=part_name,
            lods=[SimpleNamespace(blender_obj_name=obj.name)],
        )],
    )
    binding = {
        "shader": "blue/shaders/PX_Basic.mshader",
        "d": "textures/cloth_d.dds",
    }
    assigned = material_import.assign_materials(
        asset,
        {part_name: binding},
        _texture_files(binding, temporary_directory, source_png),
        "synthetic/cloth.mgraphobject",
    )
    check(assigned == 0, "CLOTH_SIM unexpectedly received a material")
    check(len(obj.data.materials) == 0, "CLOTH_SIM material slot is not empty")


def _constants_case(directory, source_png):
    binding = {
        "shader": "blue/shaders/PX_Constants.mshader",
        "d": "textures/constants_d.dds",
        "parameters": {
            "myColorMultiply": (0.7, 0.8, 0.9),
            "myMetalness": 0.0,
            "myRoughness": 0.65,
            "myTransmissionOpacity": 0.4,
        },
    }
    material = _assign(binding, directory, source_png)
    _assert_profile(material, "constants_cutout")
    shader = _principled(material)
    _assert_linked(shader.inputs["Base Color"], "constants base color")
    _assert_linked(shader.inputs["Alpha"], "constants alpha")
    check(
        material.get("afop_alpha_source") == "diffuse+transmission-opacity",
        "constants alpha did not preserve authored transmission opacity",
    )


def _human_skin_case(directory, source_png):
    binding = {
        "shader": "blue/shaders/PX_Skin_VHQ.mshader",
        "d": "textures/human_d.dds",
        "n": "textures/human_n.dds",
        "m": "textures/human_m.dds",
        "aux": {
            "Bioluminescence": "textures/human_bio.dds",
            "WrinkleNormal1": "textures/human_wrinkle_n.dds",
        },
        "parameters": {
            "myBaseColorOverlay": (0.4, 0.5, 0.6),
            "myMaterialAlpha": 1.0,
            "myBioBrightness": 1.2,
            "myBioColorStrength": 0.5,
        },
    }
    material = _assign(binding, directory, source_png)
    _assert_profile(material, "human_skin_vhq_static")
    shader = _principled(material)
    check(not shader.inputs["Metallic"].links, "human skin is incorrectly metallic")
    _assert_linked(shader.inputs["Alpha"], "human skin material alpha")
    check(material.get("afop_bio_preview") == "night", "human bio preview is missing")
    check(
        material.node_tree.nodes.get("WrinkleNormal1 (facial animation source)") is not None,
        "human wrinkle source node is missing",
    )


def _character_gear_vhq_case(directory, source_png):
    shared_diffuse = "textures/cus_115_body_d.dds"
    body = _assign(
        {
            "shader": "dlc3/shaders/PX_Character_Skin_VHQ_Body.mshader",
            "d": shared_diffuse,
        },
        directory,
        source_png,
        "Body",
    )
    body_image = body.node_tree.nodes["Diffuse / Albedo"].image
    check(
        body_image.alpha_mode == "STRAIGHT",
        "character body diffuse no longer uses Straight alpha",
    )
    binding = {
        "shader": "blue/shaders/PX_Character_Gear_VHQ.mshader",
        "d": shared_diffuse,
        "n": "textures/cus_115_body_n.dds",
        "m": "textures/cus_115_body_m.dds",
        "aux": {
            "Masks": "textures/cus_115_body_reg_mask.dds",
            "DetailNormal": "textures/flat_detail_n.dds",
        },
        "parameters": {
            "myDetailTiling": (0, 0),
            "myDetailStrength": 1.0,
        },
    }
    material = _assign(binding, directory, source_png, "Rings")
    _assert_profile(material, "character_gear_vhq_static")
    shader = _principled(material)
    _assert_linked(shader.inputs["Base Color"], "character gear color")
    base_link = shader.inputs["Base Color"].links[0]
    gear_image = material.node_tree.nodes["Diffuse / Albedo"].image
    check(
        gear_image.alpha_mode == "NONE",
        "character gear diffuse does not ignore its unused alpha channel",
    )
    check(
        gear_image != body_image,
        "character body and gear unexpectedly share one alpha-configured image",
    )
    check(
        base_link.from_node.name == "Diffuse / Albedo",
        "character gear color is incorrectly darkened by packed AO",
    )
    check(
        shader.inputs["Metallic"].links[0].from_node.name
        == "Character Gear VHQ metalness",
        "character gear region metalness is missing",
    )
    check(
        shader.inputs["Roughness"].links[0].from_node.name
        == "Character Gear VHQ roughness",
        "character gear region roughness is missing",
    )
    _assert_linked(shader.inputs["Normal"], "character gear normal")
    check(
        material.get("afop_character_gear_regions")
        == "textures/cus_115_body_reg_mask.dds",
        "character gear region mask was not retained",
    )


def _rusty_metal_vcoverlay_case(directory, source_png):
    binding = {
        "shader": "dlc3/shaders/PX_RustyMetal_VCoverlay_RDA_DLC3.mshader",
        "aux": {
            "AlbedoTexture": "textures/default_grey_d.dds",
            "DetailNormal1": "textures/blade_detail_1.dds",
            "DetailNormal2": "textures/blade_detail_2.dds",
            "RustyMetalMask": "textures/blade_mask.dds",
            "PaintTexture": "textures/default_white_d.dds",
        },
        "parameters": {
            "myUseAlbedo": 0,
            "myBaseColor": (0.0945, 0.0769, 0.0632),
            "myBaseisMetal": 1,
            "myBaseRoughnessScale": 1.2078,
            "myBaseDetialNormalStrength": 0.1647,
            "myDetail1Tiling": (5, 5),
            "myDetail2Tiling": (5, 5),
            "myRustyMetalTiling": (2, 2),
            "myRColor": (0.0044, 0.0017, 0.0001),
            "myRisMetal": 1,
            "myRRoughnessStrength": 0.7115,
            "myGColor": (0, 1, 0),
            "myGisMetal": 0,
            "myGRoughnessStrength": 0.7,
            "myBColor": (1, 1, 1),
            "myBisMetal": 0,
            "myBRoughnessStrength": 1.4,
        },
    }
    material = _assign(binding, directory, source_png, "WukulaBlade")
    _assert_profile(material, "rusty_metal_vcoverlay_static")
    shader = _principled(material)
    _assert_linked(shader.inputs["Base Color"], "rusty-metal blade base color")
    _assert_linked(shader.inputs["Metallic"], "rusty-metal blade metalness")
    _assert_linked(shader.inputs["Roughness"], "rusty-metal blade roughness")
    _assert_linked(shader.inputs["Normal"], "rusty-metal blade normal")
    check(
        material.get("afop_rusty_metal_mask") == "textures/blade_mask.dds",
        "rusty-metal blade mask was not retained",
    )
    authored_mask = material.node_tree.nodes.get("Material / Mask (packed)")
    authored_mask_uv = authored_mask.inputs["Vector"].links[0].from_node
    check(
        authored_mask_uv.bl_idname == "ShaderNodeUVMap"
        and authored_mask_uv.uv_map == "UVMap_0",
        "rusty-metal authored mask is not sampled directly with UVMap_0",
    )
    check(
        material.get("afop_normal_strength") == 0.1647,
        "rusty-metal blade base detail strength was not applied",
    )
    check(
        material.node_tree.nodes.get("Rusty Metal selected detail normal") is not None,
        "rusty-metal blade detail-normal selector was not built",
    )
    for node_name in (
        "RDA Generic Material (RGBA packed)",
        "Rust / Scratch / Dirt procedural mask",
        "Rust color gradient",
        "Rust procedural normal (packed)",
        "Rusty metal scratch mask",
        "Rusty metal rust mask",
        "Rusty metal dirt mask",
        "Rusty metal procedural relief",
    ):
        check(
            material.node_tree.nodes.get(node_name) is not None,
            f"rusty-metal reconstruction is missing {node_name}",
        )
    rust_gradient = material.node_tree.nodes.get("Rust color gradient")
    check(
        rust_gradient.image.colorspace_settings.name == "Non-Color",
        "linear rust gradient is being sampled as sRGB",
    )


def _character_skin_variant_case(directory, source_png):
    for shader_name in (
        "blue/shaders/px_character_skin_human.mshader",
        "dlc3/shaders/px_character_skin_vhq_body.mshader",
    ):
        check(
            material_audit._runtime_profile(shader_name)
            == ("human_skin", "specialized"),
            f"standalone audit does not recognize {shader_name} as specialized skin",
        )
        binding = {
            "shader": shader_name,
            "d": "textures/character_skin_d.dds",
            "n": "textures/character_skin_n.dds",
            "m": "textures/character_skin_m.dds",
            "parameters": {"myBaseColorOverlay": (0.5, 0.5, 0.5, 0.5)},
        }
        material = _assign(binding, directory, source_png)
        _assert_profile(material, "human_skin_vhq_static")
        shader = _principled(material)
        check(
            not shader.inputs["Metallic"].links,
            f"{shader_name} retained a packed-mask metallic connection",
        )
        check(
            shader.inputs["Metallic"].default_value == 0.0,
            f"{shader_name} did not force skin metalness to zero",
        )


def _hair_case(directory, source_png):
    binding = {
        "shader": "blue/shaders/PX_Hair2_3Color_Tousle.mshader",
        "aux": {
            "HairMaps": "textures/hair_maps.dds",
            "DirectionMap": "textures/hair_direction.dds",
            "AO": "textures/hair_ao.dds",
        },
        "parameters": {
            "myHairColor1": (0.1, 0.05, 0.02),
            "myHairColor2": (0.3, 0.15, 0.05),
            "myHairColor3": (0.7, 0.5, 0.2),
            "myRoughness": 0.6,
        },
    }
    material = _assign(binding, directory, source_png)
    _assert_profile(material, "hair_three_color_static")
    shader = _principled(material)
    _assert_linked(shader.inputs["Base Color"], "hair procedural base color")
    _assert_linked(shader.inputs["Alpha"], "hair cutout")
    check(
        material.get("afop_alpha_source") == "hairmaps-alpha",
        "hair alpha did not use the authored HairMaps alpha channel",
    )
    alpha_link = shader.inputs["Alpha"].links[0]
    check(
        alpha_link.from_node.name == "Hair Maps (root/color/depth/alpha)"
        and alpha_link.from_socket.name == "Alpha",
        "hair opacity is not connected directly from HairMaps alpha",
    )
    check(
        not any(
            node.name in {
                "Hair vertex cutout",
                "Hair map x vertex cutout",
                "Hair deferred opacity R x G",
                "Hair cutout x deferred opacity",
            }
            for node in material.node_tree.nodes
        ),
        "hair retained an extra RGB or vertex opacity multiplier",
    )


def _wildlife_skin_case(directory, source_png):
    parameters = {}
    for coat in (1, 2):
        for index in range(1, 6):
            parameters[f"myCoat{coat}Color{index}"] = (
                0.05 * index,
                0.1 * coat,
                0.08 * index,
            )
    binding = {
        "shader": "blue/shaders/PX_Wildlife_Skin.mshader",
        "d": "textures/wildlife_d.dds",
        "n": "textures/wildlife_n.dds",
        "m": "textures/wildlife_m.dds",
        "aux": {"PatternCoat": "textures/wildlife_pattern.dds"},
        "parameters": parameters,
        "bio_palette": [
            (0.0, 0.1, 0.2),
            (0.0, 0.2, 0.4),
            (0.0, 0.4, 0.6),
            (0.1, 0.6, 0.8),
            (0.3, 0.8, 1.0),
            (0.7, 1.0, 1.0),
        ],
    }
    material = _assign(binding, directory, source_png)
    _assert_profile(material, "wildlife_skin_static_clean")
    shader = _principled(material)
    check(not shader.inputs["Metallic"].links, "wildlife skin is incorrectly metallic")
    _assert_linked(shader.inputs["Alpha"], "wildlife skin alpha")
    emission = shader.inputs.get("Emission Color") or shader.inputs.get("Emission")
    check(emission is not None, "Principled emission input is unavailable")
    _assert_linked(emission, "wildlife bioluminescence")
    check(material.get("afop_bio_preview") == "night", "wildlife bio preview is missing")
    replacement = tuple(
        (index / 20.0, index / 30.0, index / 40.0)
        for index in range(1, 11)
    )
    control = SimpleNamespace(invert1=-1.0, invert2=1.0, level1=1.7, level2=0.3)
    pattern_node = material.node_tree.nodes.get("Wildlife Pattern Coat")
    check(pattern_node is not None, "wildlife pattern texture node is missing")
    original_first_color = tuple(
        sorted(
            material.node_tree.nodes["Wildlife Coat 1 palette"].color_ramp.elements,
            key=lambda element: element.position,
        )[0].color[:3]
    )
    check(
        material_profiles.apply_banshee_pattern(
            material, pattern_node.image, replacement, control,
            "textures/replacement_pc.dds",
        ),
        "Banshee pattern did not apply to an imported wildlife material",
    )
    first_ramp = material.node_tree.nodes.get("Wildlife Coat 1 palette")
    first_color = sorted(
        first_ramp.color_ramp.elements, key=lambda element: element.position
    )[0].color
    check(
        all(
            abs(actual - expected) < 1e-6
            for actual, expected in zip(first_color[:3], replacement[0])
        ),
        "Banshee colour selection did not update the first coat ramp",
    )
    pattern1 = material.node_tree.nodes.get("Wildlife pattern 1")
    check(
        abs(pattern1.inputs["From Max"].default_value - 0.425) < 1e-6,
        "Banshee level control did not update the pattern threshold",
    )
    check(
        material_profiles.restore_banshee_pattern(material),
        "Banshee pattern removal did not restore the imported material",
    )
    restored_first_color = tuple(
        sorted(first_ramp.color_ramp.elements, key=lambda element: element.position)[0]
        .color[:3]
    )
    check(
        all(
            abs(actual - expected) < 1e-6
            for actual, expected in zip(restored_first_color, original_first_color)
        ) and "afop_banshee_original_colors" not in material,
        "Banshee pattern removal did not restore and clear its saved state",
    )

    fallback_binding = dict(binding)
    fallback_binding["parameters"] = {}
    fallback_material = _assign(fallback_binding, directory, source_png)
    fallback_shader = _principled(fallback_material)
    original_base_source = fallback_shader.inputs["Base Color"].links[0].from_node.name
    fallback_pattern = fallback_material.node_tree.nodes.get("Wildlife Pattern Coat")
    check(
        fallback_pattern is not None
        and fallback_material.node_tree.nodes.get("Wildlife Coat 1 palette") is None,
        "wildlife fallback material unexpectedly had authored pattern constants",
    )
    check(
        material_profiles.apply_banshee_pattern(
            fallback_material, fallback_pattern.image, replacement, control,
            "textures/replacement_pc.dds",
        ) and fallback_material.get("afop_banshee_original_pipeline_missing", False),
        "Banshee fallback pattern did not save its original base-color connection",
    )
    check(
        sum(
            node.bl_idname == "ShaderNodeTexImage"
            and node.name.startswith("Wildlife Pattern Coat")
            for node in fallback_material.node_tree.nodes
        ) == 1,
        "Banshee fallback duplicated an already imported pattern texture node",
    )
    check(
        material_profiles.restore_banshee_pattern(fallback_material),
        "Banshee fallback pattern could not be removed",
    )
    check(
        fallback_shader.inputs["Base Color"].links[0].from_node.name
        == original_base_source
        and fallback_material.node_tree.nodes.get("Wildlife Coat 1 palette") is None,
        "Banshee fallback removal did not restore the original node setup",
    )

    legacy_material = _assign(fallback_binding, directory, source_png)
    legacy_shader = _principled(legacy_material)
    legacy_source = legacy_shader.inputs["Base Color"].links[0].from_node.name
    legacy_pattern = legacy_material.node_tree.nodes.get("Wildlife Pattern Coat")
    check(
        material_profiles.apply_banshee_pattern(
            legacy_material, legacy_pattern.image, replacement, control,
            "textures/replacement_pc.dds",
        ),
        "legacy Banshee fallback pattern setup failed",
    )
    legacy_material["afop_banshee_pattern"] = "legacy/test.mbansheepatterndata"
    for key in (
        "afop_banshee_original_pipeline_missing",
        "afop_banshee_original_base_node",
        "afop_banshee_original_base_socket",
        "afop_banshee_original_pattern_coat",
    ):
        if key in legacy_material:
            del legacy_material[key]
    for node in legacy_material.node_tree.nodes:
        if "afop_banshee_pattern_generated" in node:
            del node["afop_banshee_pattern_generated"]
    check(
        material_profiles.restore_banshee_pattern(legacy_material),
        "legacy Banshee pattern could not be removed",
    )
    check(
        legacy_shader.inputs["Base Color"].links[0].from_node.name == legacy_source
        and legacy_material.node_tree.nodes.get("Wildlife Coat 1 palette") is None,
        "legacy Banshee removal did not recover the original node setup",
    )


def _banshee_auto_pattern_case(directory, source_png):
    binding = {
        "shader": "blue/shaders/PX_Wildlife_Skin.mshader",
        "d": "textures/auto_banshee_d.dds",
        "n": "textures/auto_banshee_n.dds",
        "m": "textures/auto_banshee_m.dds",
        "aux": {"PatternCoat": "textures/original_body_pc.dds"},
        "parameters": {},
    }
    material = _assign(binding, directory, source_png, part_suffix="Banshee_Body")
    obj = next(
        obj for obj in bpy.data.objects
        if obj.type == "MESH" and material.name in obj.data.materials
    )
    asset = SimpleNamespace(
        meshes=[SimpleNamespace(
            name="Banshee_Body",
            lods=[SimpleNamespace(blender_obj_name=obj.name)],
        )]
    )
    color_data = b'''ColorPattern "auto" < uid=AABB >\n{\n''' + b"\n".join(
        f"myColor{index} 0xff{index:02x}4060".encode("ascii")
        for index in range(1, 11)
    ) + b"\n}\n"
    control_data = b'''PatternControl "auto" < uid=CCDD >
{
 myPattern1Invert -1.0
 myPattern2Invert 1.0
 myPattern1LevelControl 1.7
 myPattern2LevelControl 0.3
}
'''

    class FakeArchive:
        def extract(self, indexed_asset):
            return color_data if indexed_asset.name == "color" else control_data

    archive = FakeArchive()
    color_entry = SimpleNamespace(archive=archive, asset=SimpleNamespace(name="color"))
    control_entry = SimpleNamespace(archive=archive, asset=SimpleNamespace(name="control"))
    coat_entry = SimpleNamespace(archive=archive, asset=SimpleNamespace(name="coat"))
    pattern_image = material.node_tree.nodes["Wildlife Pattern Coat"].image
    originals = {
        "bindings": mgraph.banshee_pattern_bindings,
        "uid_entry": operators_sdf._banshee_uid_member_entry,
        "member_entry": operators_sdf.banshee_pattern_member_entry,
        "extract": operators_sdf._extract_texture_to_cache,
        "load_image": operators_sdf._load_image,
    }
    try:
        mgraph.banshee_pattern_bindings = lambda _data: {
            "body": {
                "coat": "textures/detected_body_pc.dds",
                "color_uid": "AABB",
                "control_uid": "CCDD",
            }
        }
        operators_sdf._banshee_uid_member_entry = lambda _uid, role, _archive: (
            color_entry if role == "color" else control_entry
        )
        operators_sdf.banshee_pattern_member_entry = (
            lambda _path, _archive, _role: coat_entry
        )
        operators_sdf._extract_texture_to_cache = (
            lambda _entry, _directory: source_png
        )
        operators_sdf._load_image = (
            lambda _disk, _logical, non_color=True: pattern_image
        )
        applied = operators_sdf._apply_detected_banshee_pattern(
            asset,
            b"synthetic graph",
            archive,
            directory,
            "blue/graphs/wl_banshee_character_test.mgraphobject",
        )
    finally:
        mgraph.banshee_pattern_bindings = originals["bindings"]
        operators_sdf._banshee_uid_member_entry = originals["uid_entry"]
        operators_sdf.banshee_pattern_member_entry = originals["member_entry"]
        operators_sdf._extract_texture_to_cache = originals["extract"]
        operators_sdf._load_image = originals["load_image"]
    check(
        applied == 1
        and material.get("afop_banshee_pattern_source") == "graph UID references"
        and material.get("afop_banshee_pattern", "").endswith("test.mgraphobject"),
        "graph-detected Banshee pattern was not applied automatically",
    )


def _medusa_case(directory, source_png):
    binding = {
        "shader": "blue/shaders/PX_DLC3_Medusa_Skin.mshader",
        "d": "textures/medusa_d.dds",
        "n": "textures/medusa_n.dds",
        "m": "textures/medusa_m.dds",
        "aux": {
            "DetailMask": "textures/medusa_detail_m.dds",
            "DetailNormal": "textures/medusa_detail_n.dds",
            "BloodveinColor": "textures/medusa_blood_d.dds",
            "BloodveinNormal": "textures/medusa_blood_n.dds",
            "InnerAlpha": "textures/medusa_inner.dds",
        },
        "bio_palette": [
            (0.0, 0.05, 0.1),
            (0.0, 0.2, 0.3),
            (0.0, 0.4, 0.6),
            (0.2, 0.7, 0.9),
            (0.6, 1.0, 1.0),
        ],
        "bio_procedural": True,
        "bio_strength": 0.5,
    }
    material = _assign(binding, directory, source_png)
    _assert_profile(material, "medusa_skin_static")
    shader = _principled(material)
    check(not shader.inputs["Metallic"].links, "Medusa skin is incorrectly metallic")
    emission = shader.inputs.get("Emission Color") or shader.inputs.get("Emission")
    check(emission is not None, "Principled emission input is unavailable")
    _assert_linked(emission, "Medusa bioluminescence")
    check(material.get("afop_bio_procedural") is True, "Medusa bio mask is not procedural")


def _rock_case(directory, source_png):
    binding = {
        "shader": "blue/shaders/PX_Natural_Rock_Temperate_V2.mshader",
        "n": "textures/rock_unique_n.dds",
        "aux": {
            "SetRockGradient": "textures/rock_gradient.dds",
            "SetRockNormalA": "textures/rock_set_a_n.dds",
            "SetRockNormalB": "textures/rock_set_b_n.dds",
            "Mask": "textures/rock_mask.dds",
        },
    }
    material = _assign(binding, directory, source_png)
    _assert_profile(material, "natural_rock_static_lookup")
    shader = _principled(material)
    _assert_linked(shader.inputs["Base Color"], "natural-rock procedural color")
    check(
        material.node_tree.nodes.get("Natural rock gradient x unique AO") is not None,
        "natural-rock AO reconstruction is missing",
    )
    for node_name in ("Natural Rock Set Normal A", "Natural Rock Set Normal B"):
        node = material.node_tree.nodes.get(node_name)
        check(node is not None, f"{node_name} is missing")
        check(node.projection == "BOX", f"{node_name} is not box projected")


def _terrain_case(directory, source_png):
    material = _assign(
        {"shader": "blue/shaders/PX_TerrainBlend.mshader"},
        directory,
        source_png,
    )
    _assert_profile(material, "terrain_runtime_placeholder")
    shader = _principled(material)
    _assert_linked(shader.inputs["Base Color"], "terrain placeholder color")
    check(
        material.node_tree.nodes.get("Terrain runtime placeholder") is not None,
        "terrain placeholder node is missing",
    )


def _moss_patch_case(directory, source_png):
    binding = {
        "shader": "blue/shaders/PX_Basic_MossPatch.mshader",
        "d": "textures/moss_patch_d.dds",
        "n": "textures/moss_patch_n.dds",
        "m": "textures/moss_patch_m.dds",
        "aux": {"WorldSpaceOverlay": "textures/moss_patch_overlay.dds"},
    }
    material = _assign(binding, directory, source_png)
    _assert_profile(material, "moss_patch_static")
    shader = _principled(material)
    _assert_linked(shader.inputs["Alpha"], "moss-patch alpha")
    overlay = material.node_tree.nodes.get("Moss patch world overlay")
    check(overlay is not None, "moss-patch world overlay is missing")
    check(
        overlay.inputs[1].links[0].from_node.name == "Moss Patch World Overlay",
        "moss-patch projected texture is not the authored overlay base",
    )
    check(
        overlay.inputs[2].links[0].from_node.name == "Diffuse / Albedo",
        "moss-patch diffuse is not the authored overlay layer",
    )


def _moss_card_case(directory, source_png):
    binding = {
        "shader": "blue/shaders/PX_MossCard_Ground.mshader",
        "d": "textures/moss_card_d.dds",
        "n": "textures/moss_card_n.dds",
        "m": "textures/moss_card_m.dds",
        "aux": {
            "ProjectedOverlay": "textures/moss_card_overlay.dds",
            "Bioluminescence": "textures/moss_card_bio.dds",
        },
    }
    material = _assign(binding, directory, source_png)
    _assert_profile(material, "moss_card_static_night")
    shader = _principled(material)
    _assert_linked(shader.inputs["Alpha"], "moss-card alpha")
    check(
        material.get("afop_alpha_source") == "diffuse+material-opacity",
        "moss-card material opacity is not combined with diffuse alpha",
    )
    projected = material.node_tree.nodes.get("Moss projected OverlayNoMask")
    check(projected is not None, "moss-card projected overlay is missing")
    check(
        projected.inputs[1].links[0].from_node.name == "Moss Card Projected Overlay",
        "moss-card projected texture is not the authored overlay base",
    )
    check(
        projected.inputs[2].links[0].from_node.name == "Diffuse / Albedo",
        "moss-card diffuse is not the authored overlay layer",
    )


def _wildlife_gear_case(directory, source_png):
    binding = {
        "shader": "blue/shaders/PX_Wildlife_Gear.mshader",
        "d": "textures/gear_d.dds",
        "n": "textures/gear_n.dds",
        "m": "textures/gear_m.dds",
        "aux": {
            "Regions": "textures/gear_regions.dds",
            "DetailNormal": "textures/gear_detail_n.dds",
        },
        "parameters": {
            "myPrimaryTintA": (0.2, 0.3, 0.4),
            "myPrimaryTintB": (0.4, 0.3, 0.2),
            "mySecondaryTintA": (0.1, 0.2, 0.3),
            "mySecondaryTintB": (0.3, 0.2, 0.1),
            "myDetailStrength": 0.5,
        },
    }
    material = _assign(binding, directory, source_png)
    _assert_profile(material, "wildlife_gear_static")
    shader = _principled(material)
    _assert_linked(shader.inputs["Metallic"], "wildlife-gear region metalness")
    check(
        shader.inputs["Metallic"].links[0].from_node.name == "Wildlife gear metalness",
        "wildlife-gear metallic input uses the wrong reconstruction",
    )
    _assert_linked(shader.inputs["Alpha"], "wildlife-gear alpha")


def _audit_profile_case():
    parsed = shader_schema.parse_shader_source(
        b'''shaderType = "Object";
        MR_Sampler2D Color : MR_Texture0 { texture = "textures/test_d.dds" }
            < pinId = 101, label = "Color" >;
        float2 myTiling < pinId = 105, default = 2.0 >;
        '''
    )
    check(parsed["shader_type"] == "Object", "runtime shader type was not parsed")
    check(
        parsed["samplers"] == [{
            "field": "Color",
            "texture_index": 0,
            "role": "d",
            "default_texture": "textures/test_d.dds",
            "pin_id": 101,
            "label": "Color",
            "graph_connectable": True,
        }],
        f"unexpected runtime shader sampler schema: {parsed['samplers']!r}",
    )
    check(
        parsed["parameters"][0]["field"] == "myTiling"
        and parsed["parameters"][0]["default"] == 2.0,
        f"unexpected runtime shader parameter schema: {parsed['parameters']!r}",
    )
    expected = {
        "PX_Constants.mshader": ("constants", "specialized"),
        "PX_Skin_VHQ.mshader": ("human_skin", "specialized"),
        "PX_Hair2_3Color_Tousle.mshader": ("hair", "specialized"),
        "PX_Natural_Rock_Temperate_V2.mshader": ("natural_rock", "specialized"),
        "PX_TerrainBlend.mshader": ("terrain_runtime", "specialized"),
        "PX_Basic_MossPatch.mshader": ("moss_patch", "specialized"),
        "PX_Wildlife_Gear.mshader": ("wildlife_gear", "specialized"),
        "PX_Character_Gear_VHQ.mshader": (
            "character_gear_vhq", "specialized"
        ),
        "PX_RustyMetal_VCoverlay_RDA_DLC3.mshader": (
            "rusty_metal_vcoverlay", "specialized"
        ),
    }
    for shader_name, profile in expected.items():
        actual = material_audit._runtime_profile(f"blue/shaders/{shader_name}")
        check(actual == profile, f"{shader_name}: expected {profile}, got {actual}")


def _addon_registration_case():
    ADDON.register()
    try:
        check(hasattr(bpy.types.Scene, "SWOMT"), "scene settings were not registered")
        check(
            hasattr(bpy.types, "SWOMT_PT_sdf_archive_filters"),
            "archive-filter popover was not registered",
        )
        check(
            ADDON.ui.SDFArchiveFilterPopover.bl_region_type == "HEADER",
            "archive-filter popover would also appear as a Scene Properties panel",
        )
        check(
            hasattr(bpy.types, "OBJECT_OT_show_all_sdf_results"),
            "Show All search-results action was not registered",
        )
        check(
            hasattr(bpy.types, "OBJECT_OT_apply_banshee_pattern")
            and hasattr(bpy.types, "OBJECT_OT_remove_banshee_pattern")
            and "banshee_pattern"
            in ADDON.settings.SWOMTSettings.bl_rna.properties,
            "Banshee pattern controls were not registered",
        )
        check(
            not hasattr(ADDON.operators_sdf, "AuditSDFMaterials"),
            "the material-audit operator is still coupled to Blender",
        )
    finally:
        ADDON.unregister()
    check(not hasattr(bpy.types.Scene, "SWOMT"), "scene settings were not unregistered")
    check(
        not hasattr(bpy.types, "SWOMT_PT_sdf_archive_filters"),
        "archive-filter popover was not unregistered",
    )
    check(
        not hasattr(bpy.types, "OBJECT_OT_show_all_sdf_results"),
        "Show All search-results action was not unregistered",
    )
    check(
        not hasattr(bpy.types, "OBJECT_OT_apply_banshee_pattern"),
        "Banshee pattern action was not unregistered",
    )
    check(
        not hasattr(bpy.types, "OBJECT_OT_remove_banshee_pattern"),
        "Banshee pattern removal was not unregistered",
    )


def _standalone_audit_case(directory):
    game_directory = Path(directory) / "synthetic_game"
    output_directory = Path(directory) / "synthetic_audit"
    game_directory.mkdir()
    toc_path = game_directory / "sdf.sdftoc"
    toc_path.write_bytes(b"synthetic toc")
    assets = [
        SimpleNamespace(name="blue/baked/test.mmb"),
        SimpleNamespace(name="blue/graphs/test.mgraphobject"),
        SimpleNamespace(name="blue/shaders/test.mshader"),
    ]

    class FakeArchive:
        def __init__(self, _toc_path, _oodle_path):
            pass

        def extract(self, asset):
            if asset.name.endswith(".mgraphobject"):
                return mgraph.MAGIC + b"blue/baked/test.mmb\0"
            check(asset.name.endswith(".mshader"), "unexpected synthetic extraction")
            return b'''shaderType = "Object";
                MR_Sampler2D Color : MR_Texture0
                    < pinId = 101, label = "Color" >;
            '''

    arguments = SimpleNamespace(
        game_directory=game_directory,
        output_directory=output_directory,
        cache_directory=None,
        oodle=None,
        download_oodle=False,
        no_rebuild=False,
    )
    original_archive = audit_material_corpus.SdfArchive
    original_oodle = audit_material_corpus._find_oodle
    original_load = audit_material_corpus._load_targeted_assets
    try:
        audit_material_corpus.SdfArchive = FakeArchive
        audit_material_corpus._find_oodle = lambda _arguments, _root: Path("fake_oodle.dll")
        audit_material_corpus._load_targeted_assets = (
            lambda _archive, _toc, _cache, _no_rebuild: (assets, True)
        )
        audit_material_corpus.run(arguments)
    finally:
        audit_material_corpus.SdfArchive = original_archive
        audit_material_corpus._find_oodle = original_oodle
        audit_material_corpus._load_targeted_assets = original_load

    report_path = output_directory / "afop_material_audit.json"
    check(report_path.is_file(), "standalone audit did not write its JSON report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    check(report["schema_version"] == 5, "standalone audit schema changed unexpectedly")
    check(report["summary"]["shaders"] == 1, "standalone audit lost shader records")
    check(report["summary"]["referenced_mmbs"] == 1, "standalone audit lost MMB paths")


def main():
    with tempfile.TemporaryDirectory(prefix="afop_material_regression_") as directory:
        source_png = _make_source_png(directory)
        cases = [
            ("material profile registry and public facade", _material_profile_registry_case),
            ("Banshee pattern text formats", _banshee_pattern_parser_case),
            ("Banshee graph UID references", _banshee_graph_pattern_reference_case),
            ("nested material-package update install", _nested_update_install_case),
            ("direct and compound binding resolution", _binding_resolution_cases),
            ("direct-only MMB material-source discovery", _direct_material_source_discovery_case),
            ("multi-MMB graph import selection", _multi_mmb_import_selection_case),
            ("Rogue and DLC search-result filters", _sdf_archive_filter_case),
            ("show all truncated search results", _sdf_show_all_results_case),
            ("Direhorse weakpoint inherits body binding", _direhorse_weakpoint_binding_case),
            ("wildlife gameplay parts inherit rendered skin", _wildlife_gameplay_part_alias_case),
            ("named mask and piercing texture families", _named_mask_and_piercing_binding_case),
            ("supplemental graph exact-material override", _supplemental_graph_override_case),
            ("layered Banshee skin texture fallback", _banshee_layered_texture_fallback_case),
            (
                "CLOTH_SIM material exclusion",
                lambda: _cloth_sim_case(directory, source_png),
            ),
            ("constants cutout profile", lambda: _constants_case(directory, source_png)),
            ("human skin profile", lambda: _human_skin_case(directory, source_png)),
            ("character gear VHQ regions", lambda: _character_gear_vhq_case(directory, source_png)),
            ("DLC3 rusty-metal vertex-overlay profile", lambda: _rusty_metal_vcoverlay_case(directory, source_png)),
            ("additional character skin variants", lambda: _character_skin_variant_case(directory, source_png)),
            ("hair cutout profile", lambda: _hair_case(directory, source_png)),
            ("wildlife skin and bioluminescence", lambda: _wildlife_skin_case(directory, source_png)),
            ("automatic Banshee graph pattern", lambda: _banshee_auto_pattern_case(directory, source_png)),
            ("Medusa skin and bioluminescence", lambda: _medusa_case(directory, source_png)),
            ("natural-rock procedural lookup", lambda: _rock_case(directory, source_png)),
            ("terrain runtime placeholder", lambda: _terrain_case(directory, source_png)),
            ("moss-patch overlay and alpha", lambda: _moss_patch_case(directory, source_png)),
            ("moss-card overlay, alpha, and bio", lambda: _moss_card_case(directory, source_png)),
            ("wildlife-gear tint, metal, and alpha", lambda: _wildlife_gear_case(directory, source_png)),
            ("audit/runtime profile agreement", _audit_profile_case),
            ("add-on registration without audit UI", _addon_registration_case),
            (
                "standalone material-audit command",
                lambda: _standalone_audit_case(directory),
            ),
        ]
        for name, callback in cases:
            run_case(name, callback)

    if _failures:
        print("\nAFoP material regression failures:")
        for name, _exception, stack in _failures:
            print(f"\n--- {name} ---\n{stack}")
        raise RuntimeError(
            f"{len(_failures)} of {len(cases)} material regression checks failed"
        )
    print(f"\nAFoP material regression: {len(cases)} checks passed")


if __name__ == "__main__":
    main()

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
    mgraph,
    shader_schema,
)

_audit_spec = importlib.util.spec_from_file_location(
    "afop_material_audit_tool", REPOSITORY / "tools" / "material_audit.py"
)
if _audit_spec is None or _audit_spec.loader is None:
    raise RuntimeError("Could not load the standalone material-audit report engine")
material_audit = importlib.util.module_from_spec(_audit_spec)
_audit_spec.loader.exec_module(material_audit)

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
        material.get("afop_alpha_source") == "hairmaps+vertex-color",
        "hair alpha did not use its packed maps and vertex color",
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
    }
    for shader_name, profile in expected.items():
        actual = material_audit._runtime_profile(f"blue/shaders/{shader_name}")
        check(actual == profile, f"{shader_name}: expected {profile}, got {actual}")


def _addon_registration_case():
    ADDON.register()
    try:
        check(hasattr(bpy.types.Scene, "SWOMT"), "scene settings were not registered")
        check(
            not hasattr(ADDON.operators_sdf, "AuditSDFMaterials"),
            "the material-audit operator is still coupled to Blender",
        )
    finally:
        ADDON.unregister()
    check(not hasattr(bpy.types.Scene, "SWOMT"), "scene settings were not unregistered")


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
            ("direct and compound binding resolution", _binding_resolution_cases),
            ("Direhorse weakpoint inherits body binding", _direhorse_weakpoint_binding_case),
            (
                "CLOTH_SIM material exclusion",
                lambda: _cloth_sim_case(directory, source_png),
            ),
            ("constants cutout profile", lambda: _constants_case(directory, source_png)),
            ("human skin profile", lambda: _human_skin_case(directory, source_png)),
            ("hair cutout profile", lambda: _hair_case(directory, source_png)),
            ("wildlife skin and bioluminescence", lambda: _wildlife_skin_case(directory, source_png)),
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

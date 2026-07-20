"""Shader-profile matching and auxiliary-texture requirements."""

from __future__ import annotations

import os
from dataclasses import dataclass


NAVI_SKIN_SHADERS = frozenset({
    "px_character_navi.mshader",
    "px_character_navi_face.mshader",
    "px_character_navi_npc_face.mshader",
    "px_character_skin_navi_face_ash.mshader",
    "px_character_workbench.mshader",
})
NAVI_DETAIL_SHADERS = frozenset({"px_character_navi.mshader"})
NAVI_FACE_DETAIL_SHADERS = frozenset({
    "px_character_navi_face.mshader",
    "px_character_navi_npc_face.mshader",
    "px_character_skin_navi_face_ash.mshader",
})
BASIC_EMISSIVE_SHADER = "px_basic_emissive.mshader"
MOSS_CARD_SHADERS = frozenset({
    "px_mosscard.mshader",
    "px_mosscard_ground.mshader",
})
HUMAN_SKIN_SHADERS = frozenset({
    "px_skin_vhq.mshader",
    "px_character_skin_human.mshader",
    "px_character_skin_vhq_body.mshader",
})
HAIR_SHADERS = frozenset({"px_hair2_3color_tousle.mshader"})
NATURAL_ROCK_SHADERS = frozenset({"px_natural_rock_temperate_v2.mshader"})
TERRAIN_RUNTIME_SHADERS = frozenset({"px_terrainblend.mshader"})
MOSS_PATCH_SHADERS = frozenset({
    "px_basic_mosspatch.mshader",
    "px_dlc3_basic_mosspatch.mshader",
})
WILDLIFE_GEAR_SHADERS = frozenset({"px_wildlife_gear.mshader"})
CHARACTER_GEAR_VHQ_SHADERS = frozenset({"px_character_gear_vhq.mshader"})
CONSTANT_SHADERS = frozenset({"px_constants.mshader"})
RUSTY_METAL_VC_DEFAULT_TEXTURES = {
    "GenericMaterial": (
        "blue/baked/art/generic_textures/rda/"
        "rda_rusty_metal_vcolor_tiling.dds"
    ),
    "ProceduralMask": (
        "snowdrop/baked/art/[assets]/[techtexture]/tt_rust_mask.dds"
    ),
    "RustGradient": (
        "snowdrop/baked/art/[assets]/[generictexture]/_test/rm_gradient.dds"
    ),
    "RustNormal": (
        "snowdrop/baked/art/[assets]/[techtexture]/tt_rust_n.dds"
    ),
}


@dataclass(frozen=True)
class MaterialProfile:
    """One reusable shader-family matcher in the Blender profile registry."""

    name: str
    exact: frozenset[str] = frozenset()
    prefixes: tuple[str, ...] = ()
    contains: tuple[str, ...] = ()

    def matches(self, shader_name: str) -> bool:
        return (
            shader_name in self.exact
            or any(shader_name.startswith(prefix) for prefix in self.prefixes)
            or any(fragment in shader_name for fragment in self.contains)
        )


PROFILE_REGISTRY = (
    MaterialProfile("navi_skin", NAVI_SKIN_SHADERS),
    MaterialProfile("navi_detail", NAVI_DETAIL_SHADERS),
    MaterialProfile("navi_face", NAVI_FACE_DETAIL_SHADERS),
    MaterialProfile("wildlife_skin", prefixes=("px_wildlife_skin",)),
    MaterialProfile("medusa_skin", prefixes=("px_dlc3_medusa_skin",)),
    MaterialProfile(
        "dragonfly_wing", frozenset({"px_wildlife_dragonflywing.mshader"})
    ),
    MaterialProfile("wildlife_eye", frozenset({"px_wildlife_eye.mshader"})),
    MaterialProfile(
        "eye_parallax",
        frozenset({"px_wildlife_eye.mshader", "px_eye2.mshader"}),
    ),
    MaterialProfile(
        "eye_shell", frozenset({"px_character_eye_shell.mshader"})
    ),
    MaterialProfile("human_skin", HUMAN_SKIN_SHADERS),
    MaterialProfile("hair", HAIR_SHADERS),
    MaterialProfile("natural_rock", NATURAL_ROCK_SHADERS),
    MaterialProfile("terrain_runtime", TERRAIN_RUNTIME_SHADERS),
    MaterialProfile("moss_patch", MOSS_PATCH_SHADERS),
    MaterialProfile("wildlife_gear", WILDLIFE_GEAR_SHADERS),
    MaterialProfile("character_gear_vhq", CHARACTER_GEAR_VHQ_SHADERS),
    MaterialProfile("constants", CONSTANT_SHADERS),
    MaterialProfile("emissive_color", frozenset({"px_emissive_color.mshader"})),
    MaterialProfile("basic_emissive", frozenset({BASIC_EMISSIVE_SHADER})),
    MaterialProfile("rusty_metal", prefixes=("px_basic_rustymetal",)),
    MaterialProfile(
        "rusty_metal_vcoverlay",
        prefixes=("px_rustymetal_vcoverlay",),
    ),
    MaterialProfile("vegetation", contains=("vegetation",)),
    MaterialProfile("moss_card", MOSS_CARD_SHADERS),
    MaterialProfile(
        "basic_blend", frozenset({"px_basic_blendmaterial.mshader"})
    ),
)


def profile_traits(shader_name: str) -> frozenset[str]:
    """Return every applicable profile trait for a normalized shader name."""
    return frozenset(
        profile.name for profile in PROFILE_REGISTRY if profile.matches(shader_name)
    )


def supported_auxiliary_paths(binding):
    """Return auxiliary textures required by the matching Blender profile."""
    shader_name = os.path.basename(binding.get("shader", "")).casefold()
    auxiliary = binding.get("aux", {})
    traits = profile_traits(shader_name)

    if "navi_detail" in traits:
        paths = (auxiliary.get("DetailNormal"),)
    elif "navi_face" in traits:
        paths = (
            auxiliary.get("DetailNormal"),
            auxiliary.get("SarentuScarNormal") or auxiliary.get("Scar"),
        )
    elif "wildlife_skin" in traits:
        paths = (
            auxiliary.get("PatternCoat"), auxiliary.get("DetailNormalMask"),
            auxiliary.get("DetailNormal1"), auxiliary.get("DetailNormal2"),
            auxiliary.get("DetailNormal3"),
        )
    elif "eye_parallax" in traits:
        paths = (auxiliary.get("Height"),)
    elif "eye_shell" in traits:
        paths = (auxiliary.get("NormalTexture"),)
    elif "human_skin" in traits:
        paths = (
            auxiliary.get("Bioluminescence"), auxiliary.get("WrinkleNormal1"),
            auxiliary.get("WrinkleNormal2"), auxiliary.get("WrinkleNormal3"),
            auxiliary.get("WrinkleMask1"), auxiliary.get("WrinkleMask2"),
            auxiliary.get("WrinkleMask3"), auxiliary.get("WrinkleMask4"),
            auxiliary.get("WrinkleMask5"),
        )
    elif "hair" in traits:
        paths = (
            auxiliary.get("HairMaps"), auxiliary.get("DirectionMap"),
            auxiliary.get("AO"),
        )
    elif "natural_rock" in traits:
        paths = (
            auxiliary.get("SetRockGradient"), auxiliary.get("SetRockNormalA"),
            auxiliary.get("SetRockNormalB"), auxiliary.get("Mask"),
        )
    elif "moss_patch" in traits:
        paths = (auxiliary.get("WorldSpaceOverlay"),)
    elif "wildlife_gear" in traits:
        paths = (auxiliary.get("Regions"), auxiliary.get("DetailNormal"))
    elif "character_gear_vhq" in traits:
        paths = (auxiliary.get("Masks"), auxiliary.get("DetailNormal"))
    elif "medusa_skin" in traits:
        paths = (
            auxiliary.get("DetailMask"), auxiliary.get("DetailNormal"),
            auxiliary.get("BloodveinColor"), auxiliary.get("BloodveinNormal"),
            auxiliary.get("InnerAlpha"),
        )
    elif "rusty_metal" in traits:
        paths = (auxiliary.get("DetailNormal"), auxiliary.get("RustyMetalMask"))
    elif "rusty_metal_vcoverlay" in traits:
        paths = (
            auxiliary.get("AlbedoTexture"), auxiliary.get("DetailNormal1"),
            auxiliary.get("DetailNormal2"), auxiliary.get("RustyMetalMask"),
            auxiliary.get("PaintTexture"),
            *RUSTY_METAL_VC_DEFAULT_TEXTURES.values(),
        )
    elif "vegetation" in traits:
        paths = (
            auxiliary.get("DetailA"), auxiliary.get("DetailB"),
            auxiliary.get("DetailNormal"), auxiliary.get("Bioluminescence"),
        )
    elif "moss_card" in traits:
        paths = (auxiliary.get("Bioluminescence"), auxiliary.get("ProjectedOverlay"))
    elif "basic_blend" in traits:
        paths = (
            auxiliary.get("ColorBase"), auxiliary.get("NormalBase"),
            auxiliary.get("MaterialBase"), auxiliary.get("ColorBlend"),
            auxiliary.get("NormalBlend"), auxiliary.get("MaterialBlend"),
            auxiliary.get("BlendMask"), auxiliary.get("DetailNormal"),
        )
    elif "basic_emissive" in traits:
        paths = (auxiliary.get("Emission"), auxiliary.get("DetailNormal"))
    else:
        paths = (
            auxiliary.get("DetailNormal") or auxiliary.get("DetailNormalMap")
            or auxiliary.get("DetailSampler"),
            auxiliary.get("Emission"), auxiliary.get("Emissive"),
        )
    return tuple(path for path in paths if path)

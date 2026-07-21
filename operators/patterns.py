"""Discover and apply Banshee vanity patterns from indexed game archives."""

from __future__ import annotations

import os

import bpy

from .. import addon_state
from ..formats.banshee_patterns import (
    BansheePatternData,
    ColorPattern,
    PatternControl,
)
from ..log import logger
from ..materials.profiles import apply_banshee_pattern, restore_banshee_pattern
from ..materials.textures import _load_image
from . import sdf as operators_sdf


_EMPTY_PATTERN = "__NONE__"
_enum_generation = None
_enum_items = [(_EMPTY_PATTERN, "No patterns found", "Reload the SDF archives", 0)]


def _display_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    prefix = "banshee_pattern_data_"
    if stem.casefold().startswith(prefix):
        stem = stem[len(prefix):]
    return stem.replace("_", " ").strip().title() or os.path.basename(path)


def pattern_enum_items(_self, _context):
    global _enum_generation, _enum_items
    generation, entries = operators_sdf.banshee_pattern_entries()
    signature = (generation, tuple((entry.cache_key, entry.asset.hash) for entry in entries))
    if signature != _enum_generation:
        _enum_generation = signature
        _enum_items = [
            (
                f"{entry.cache_key}:{entry.asset.hash}",
                _display_name(entry.asset.name),
                f"{entry.asset.name} ({entry.archive_label})",
                index,
            )
            for index, entry in enumerate(entries)
        ] or [
            (_EMPTY_PATTERN, "No patterns found", "Reload the SDF archives", 0)
        ]
    return _enum_items


def is_banshee_asset(asset=None):
    asset = addon_state.asset if asset is None else asset
    if asset is None:
        return False
    if "banshee" in str(getattr(asset, "name", "")).casefold():
        return True
    return any(
        "banshee" in str(getattr(mesh, "name", "")).casefold()
        for mesh in getattr(asset, "meshes", ())
    )


def _read_member(manifest_entry, path, role, parser):
    entry = operators_sdf.banshee_pattern_member_entry(
        path, manifest_entry.archive, role
    )
    if entry is None:
        raise LookupError(f"{os.path.basename(path)} was not found in the SDF index")
    return parser.loads(entry.archive.extract(entry.asset)), entry


def _part_materials(part):
    result = []
    seen = set()
    for mesh in addon_state.asset.meshes:
        name = mesh.name.casefold()
        matches = (
            "head" in name
            if part == "head"
            else ("body" in name or "weakpoint" in name)
        )
        if not matches or not mesh.lods:
            continue
        object_name = mesh.lods[0].blender_obj_name or f"{mesh.name}_LOD0"
        obj = bpy.data.objects.get(object_name)
        if obj is None or obj.type != "MESH":
            continue
        for slot in obj.material_slots:
            material = slot.material
            if material is None or material.as_pointer() in seen:
                continue
            shader = os.path.basename(str(material.get("afop_shader", ""))).casefold()
            if not shader.startswith("px_wildlife_skin"):
                continue
            seen.add(material.as_pointer())
            result.append(material)
    return result


def has_applied_pattern():
    if not is_banshee_asset():
        return False
    return any(
        "afop_banshee_pattern" in material
        for part in ("body", "head")
        for material in _part_materials(part)
    )


class ApplyBansheePattern(bpy.types.Operator):
    """Apply a Snowdrop Banshee colour pattern to imported body/head materials."""

    bl_idname = "object.apply_banshee_pattern"
    bl_label = "Apply Banshee Pattern"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return is_banshee_asset() and context.scene is not None

    def execute(self, context):
        settings = context.scene.SWOMT
        identifier = settings.banshee_pattern
        entry = operators_sdf.banshee_pattern_entry(identifier)
        if entry is None:
            self.report({"ERROR"}, "Select a Banshee pattern from the loaded game files.")
            return {"CANCELLED"}
        try:
            manifest = BansheePatternData.loads(entry.archive.extract(entry.asset))
            members = manifest.member_paths()
            part_data = {}
            for part in ("body", "head"):
                color_path = members.get((part, "color"))
                control_path = members.get((part, "control"))
                coat_path = members.get((part, "coat"))
                if not color_path or not control_path or not coat_path:
                    raise LookupError(f"{part.title()} pattern data is incomplete")
                color, _color_entry = _read_member(
                    entry, color_path, "color", ColorPattern
                )
                control, _control_entry = _read_member(
                    entry, control_path, "control", PatternControl
                )
                coat_entry = operators_sdf.banshee_pattern_member_entry(
                    coat_path, entry.archive, "coat"
                )
                if coat_entry is None:
                    raise LookupError(
                        f"{os.path.basename(coat_path)} was not found in the SDF index"
                    )
                disk_path = operators_sdf._extract_texture_to_cache(
                    coat_entry, settings.sdf_extracted_directory
                )
                image = _load_image(disk_path, coat_path, non_color=True)
                part_data[part] = (
                    tuple(color.rgb(index) for index in range(10)),
                    control,
                    image,
                    coat_path,
                )

            applied = 0
            compatible = 0
            missing_parts = []
            for part, values in part_data.items():
                materials = _part_materials(part)
                if not materials:
                    missing_parts.append(part)
                    continue
                compatible += len(materials)
                colors, control, image, coat_path = values
                for material in materials:
                    if apply_banshee_pattern(
                        material, image, colors, control, coat_path
                    ):
                        material["afop_banshee_pattern"] = entry.asset.name
                        applied += 1
            if not applied:
                if compatible:
                    raise LookupError(
                        "The imported Banshee materials are missing the diffuse or "
                        "material-mask nodes required by patterns"
                    )
                raise LookupError(
                    "No imported Banshee body or head wildlife materials were found"
                )
            settings.banshee_pattern_status = (
                f"Applied {manifest.name} to {applied} material"
                f"{'s' if applied != 1 else ''}."
            )
            self.report({"INFO"}, settings.banshee_pattern_status)
            if missing_parts:
                self.report(
                    {"WARNING"},
                    "No compatible " + " or ".join(missing_parts) + " material was loaded.",
                )
            return {"FINISHED"}
        except Exception as error:
            logger.exception("Could not apply Banshee pattern %s", entry.asset.name)
            settings.banshee_pattern_status = str(error)
            self.report({"ERROR"}, f"Could not apply Banshee pattern: {error}")
            return {"CANCELLED"}


class RemoveBansheePattern(bpy.types.Operator):
    """Restore the Banshee materials as they were originally imported."""

    bl_idname = "object.remove_banshee_pattern"
    bl_label = "Remove Banshee Pattern"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return is_banshee_asset() and context.scene is not None

    def execute(self, context):
        restored = 0
        seen = set()
        for part in ("body", "head"):
            for material in _part_materials(part):
                pointer = material.as_pointer()
                if pointer in seen:
                    continue
                seen.add(pointer)
                if restore_banshee_pattern(material):
                    restored += 1
        if not restored:
            self.report({"WARNING"}, "No applied Banshee pattern was found.")
            return {"CANCELLED"}
        status = (
            f"Restored the original pattern on {restored} material"
            f"{'s' if restored != 1 else ''}."
        )
        context.scene.SWOMT.banshee_pattern_status = status
        self.report({"INFO"}, status)
        return {"FINISHED"}


CLASSES = (ApplyBansheePattern, RemoveBansheePattern)

"""Pure-Python inventory of Snowdrop material sources and shader declarations.

The Blender operator in :mod:`operators_sdf` owns archive access.  This module
only consumes extracted bytes, which keeps the analysis deterministic and
allows reports to be regenerated or tested outside Blender.
"""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone

try:
    from . import mgraph
except ImportError:  # Standalone use from the repository directory.
    import mgraph


SCHEMA_VERSION = 5
REPORT_JSON = "afop_material_audit.json"
SHADER_CSV = "afop_shader_inventory.csv"
SOURCE_CSV = "afop_material_sources.csv"
MATERIAL_CSV = "afop_material_bindings.csv"
ISSUE_CSV = "afop_material_issues.csv"
PROFILE_CSV = "afop_profile_coverage.csv"

_SAMPLER_RE = re.compile(
    r"MR_Sampler2D\s+(?P<field>\w+)\s*:\s*MR_Texture(?P<texidx>\d+)\s*"
    r"(?:\{(?P<block>[^{}]*)\}\s*)?"
    r"(?:<(?P<meta>[^<>]*)>\s*)?;",
    re.DOTALL,
)
_TEXTURE_RE = re.compile(r'texture\s*=\s*"([^"]*)"')
_PIN_ID_RE = re.compile(r"pinId\s*=\s*(\d+)")
_LABEL_RE = re.compile(r'label\s*=\s*"([^"]*)"')
_SHADER_TYPE_RE = re.compile(r'shaderType\s*=\s*"([^"]*)"')
_PARAMETER_RE = re.compile(
    r"(?P<type>(?:float|half|int|uint|bool)[1-4]?)\s+"
    r"(?P<field>\w+)\s*<(?P<meta>[^<>]*)>\s*;",
    re.DOTALL,
)
_DEFAULT_RE = re.compile(r"default\s*=\s*([^,>]+)")
_NUMBER_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)


def _normalise_path(value):
    return str(value or "").replace("\\", "/").lstrip("/")


def _record_key(record):
    return (
        _normalise_path(record["path"]).casefold(),
        str(record.get("archive", "")).casefold(),
    )


def _sampler_role(field):
    key = field.casefold()
    if key in {"color", "decal", "diffuse"}:
        return "d"
    if key == "normal":
        return "n"
    if key == "material":
        return "m"
    if "detailnormal" in key or key == "detailsampler":
        return "detail_normal"
    return key


def parse_shader_source(data):
    """Parse sampler and numeric-parameter metadata from an ``.mshader``."""
    text = data.decode("utf-8", "replace") if isinstance(data, bytes) else str(data)
    shader_type = _SHADER_TYPE_RE.search(text)
    samplers = []
    for match in _SAMPLER_RE.finditer(text):
        block = match.group("block") or ""
        metadata = match.group("meta") or ""
        texture = _TEXTURE_RE.search(block)
        pin_id = _PIN_ID_RE.search(metadata)
        label = _LABEL_RE.search(metadata)
        samplers.append({
            "field": match.group("field"),
            "texture_index": int(match.group("texidx")),
            "role": _sampler_role(match.group("field")),
            "default_texture": _normalise_path(texture.group(1)) if texture else "",
            "pin_id": int(pin_id.group(1)) if pin_id else None,
            "label": label.group(1) if label else "",
            "graph_connectable": pin_id is not None,
        })
    parameters = []
    for match in _PARAMETER_RE.finditer(text):
        metadata = match.group("meta") or ""
        pin_id = _PIN_ID_RE.search(metadata)
        if pin_id is None:
            continue
        label = _LABEL_RE.search(metadata)
        default_match = _DEFAULT_RE.search(metadata)
        default = None
        if default_match:
            raw_default = default_match.group(1).strip()
            lowered = raw_default.casefold()
            if lowered in {"true", "false"}:
                default = lowered == "true"
            else:
                values = [float(value) for value in _NUMBER_RE.findall(raw_default)]
                if values:
                    default = values[0] if len(values) == 1 else values
        parameters.append({
            "field": match.group("field"),
            "value_type": match.group("type"),
            "pin_id": int(pin_id.group(1)),
            "label": label.group(1) if label else "",
            "default": default,
        })
    return {
        "shader_type": shader_type.group(1) if shader_type else "",
        "samplers": samplers,
        "parameters": parameters,
    }


def _runtime_profile(shader_path):
    """Describe the fidelity of the add-on's current Blender approximation."""
    name = os.path.basename(_normalise_path(shader_path)).casefold()
    if name in {
        "px_character_navi.mshader",
        "px_character_navi_face.mshader",
        "px_character_navi_npc_face.mshader",
        "px_character_skin_navi_face_ash.mshader",
        "px_character_workbench.mshader",
    }:
        return "navi_skin", "specialized"
    if name.startswith("px_wildlife_skin"):
        return "wildlife_skin", "specialized"
    if name.startswith("px_dlc3_medusa_skin"):
        return "medusa_skin", "specialized"
    if name == "px_wildlife_eye.mshader":
        return "wildlife_eye", "specialized"
    if name == "px_eye2.mshader":
        return "eye_parallax", "specialized"
    if name == "px_character_eye_shell.mshader":
        return "eye_shell", "specialized"
    if name == "px_constants.mshader":
        return "constants", "specialized"
    if name == "px_skin_vhq.mshader":
        return "human_skin", "specialized"
    if name == "px_hair2_3color_tousle.mshader":
        return "hair", "specialized"
    if name == "px_natural_rock_temperate_v2.mshader":
        return "natural_rock", "specialized"
    if name == "px_terrainblend.mshader":
        return "terrain_runtime", "specialized"
    if name in {
        "px_basic_mosspatch.mshader", "px_dlc3_basic_mosspatch.mshader",
    }:
        return "moss_patch", "specialized"
    if name == "px_wildlife_gear.mshader":
        return "wildlife_gear", "specialized"
    if "wildlife_dragonflywing" in name:
        return "wildlife_wing", "specialized"
    if name.startswith("px_basic_rustymetal"):
        return "rusty_metal", "specialized"
    if name == "px_basic_blendmaterial.mshader":
        return "basic_blend", "specialized"
    if name in {"px_mosscard.mshader", "px_mosscard_ground.mshader"}:
        return "moss_card", "specialized"
    if "vegetation" in name:
        return "vegetation", "specialized"
    if name == "px_basic_emissive.mshader":
        return "basic_emissive", "specialized"
    if name == "px_emissive_color.mshader":
        return "emissive_color", "specialized"
    if name:
        return "principled", "generic"
    return "unmapped", "unmapped"


def _sampler_blender_mapping(sampler, profile, support):
    if not sampler["graph_connectable"]:
        texture = sampler.get("default_texture") or "internal shader default"
        return f"Baked default: {texture}", "baked_default"
    if sampler.get("surface_alpha_channel"):
        if sampler.get("surface_alpha_source") == "diffuse":
            return "Principled BSDF / Base Color + Alpha", "connected"
        return "Principled BSDF / Alpha", "connected"
    role = sampler["role"]
    if role == "d":
        return "Principled BSDF / Base Color", "connected"
    if role == "n":
        return "Normal Map / Principled BSDF Normal", "connected"
    if role == "m":
        target = (
            f"{profile} packed material channels"
            if support == "specialized"
            else "Packed R=Metallic, G=Roughness approximation"
        )
        return target, "connected"
    if role == "detail_normal":
        if profile in {
            "basic_emissive", "navi_skin", "wildlife_skin", "medusa_skin",
            "rusty_metal", "vegetation", "basic_blend", "wildlife_gear",
        }:
            return "Tiled RNM detail-normal layer", "connected"
        return "Generic tiled RNM detail-normal layer", "connected_preview"
    if role == "emission" and profile == "basic_emissive":
        return "Principled BSDF / Emission Color + Strength", "connected"
    if role in {"emission", "emissive"}:
        return "Principled BSDF / Emission Color + authored strength", "connected_preview"
    if profile == "wildlife_skin" and role in {
        "patterncoat", "detailnormalmask",
    }:
        return "Wildlife coat/detail-mask profile", "connected"
    if profile in {"wildlife_eye", "eye_parallax"} and role == "height":
        return "Bump approximation of Snowdrop parallax", "connected_preview"
    if profile == "eye_shell" and role == "normaltexture":
        return "Authored eye-shell normal", "connected"
    if profile == "human_skin" and role == "bioluminescence":
        return "Human-skin static night emission", "connected_preview"
    if profile == "human_skin" and role.startswith(("wrinklenormal", "wrinklemask")):
        return "Imported facial-animation wrinkle source", "preview_source"
    if profile == "hair" and role in {"hairmaps", "ao"}:
        return "Three-color hair profile", "connected"
    if profile == "hair" and role == "directionmap":
        return "Imported anisotropic direction source", "preview_source"
    if profile == "natural_rock" and role in {
        "setrockgradient", "setrocknormala", "setrocknormalb", "mask",
    }:
        return "Static natural-rock lookup preview", "connected_preview"
    if profile == "moss_patch" and role == "worldspaceoverlay":
        return "Static moss-patch world overlay", "connected_preview"
    if profile == "wildlife_gear" and role in {"regions", "detail_normal"}:
        return "Wildlife gear region/detail profile", "connected"
    if profile == "basic_blend" and role in {
        "colorbase", "normalbase", "materialbase", "colorblend",
        "normalblend", "materialblend", "blendmask",
    }:
        return "Two-material blend profile", "connected"
    if profile == "moss_card" and role in {
        "bioluminescence", "projectedoverlay",
    }:
        return "Moss-card projected/night profile", "connected_preview"
    if profile == "medusa_skin" and role in {
        "detailmask", "bloodveincolor", "bloodveinnormal", "inneralpha",
    }:
        return "Medusa detail/bloodvein/transmission profile", "connected"
    if profile == "rusty_metal" and role == "rustymetalmask":
        return "Imported for engine-procedural rust profile", "preview_source"
    if profile == "vegetation" and role in {
        "detaila", "detailb", "bioluminescence",
    }:
        return "Vegetation detail/night-preview profile", "connected_preview"
    return "", "unmapped"


def _choose(records, preferred_archive, preferred_cache_key=""):
    if not records:
        return None
    if preferred_cache_key:
        same_toc = [
            record for record in records
            if record.get("cache_key") == preferred_cache_key
        ]
        if same_toc:
            return min(same_toc, key=_record_key)
    preferred = str(preferred_archive or "").casefold()
    same_archive = [
        record for record in records
        if str(record.get("archive", "")).casefold() == preferred
    ]
    return min(same_archive or records, key=_record_key)


def _issue(kind, source, detail, severity="warning"):
    return {
        "severity": severity,
        "kind": kind,
        "source": source,
        "detail": detail,
    }


def build_report(source_records, shader_records, mmb_paths, *, input_signature=None):
    """Build a serializable material corpus report.

    ``source_records`` are extracted mgraphobject/mcompoundnode bytes.  Duplicate
    logical paths are retained per archive and same-archive links are preferred.
    """
    sources = sorted(source_records, key=_record_key)
    shaders = sorted(shader_records, key=_record_key)
    parsed_shaders = {}
    shader_role_pins = {}
    shader_parameter_pins = {}
    for record in shaders:
        parsed = parse_shader_source(record["data"])
        parsed_shaders[id(record)] = parsed
        path = _normalise_path(record["path"]).casefold()
        role_pins = {
            sampler["pin_id"]: (
                sampler["role"]
                if sampler["role"] in {"d", "n", "m"}
                else f"aux:{sampler['field']}"
            )
            for sampler in parsed["samplers"]
            if sampler["pin_id"] is not None
        }
        # An empty sampler declaration is still authoritative.  Preserve it so
        # mesh-local texture-name heuristics cannot invent D/N/M inputs for
        # runtime-only shaders such as PX_TerrainBlend.
        shader_role_pins.setdefault(path, role_pins)
        shader_role_pins.setdefault(os.path.basename(path), role_pins)
        parameter_pins = {
            parameter["pin_id"]: parameter["field"]
            for parameter in parsed["parameters"]
        }
        if parameter_pins:
            shader_parameter_pins.setdefault(path, parameter_pins)
            shader_parameter_pins.setdefault(
                os.path.basename(path), parameter_pins
            )
    mmb_keys = {_normalise_path(path).casefold() for path in mmb_paths}
    compounds = {}
    for record in sources:
        if record.get("kind") == "mcompoundnode":
            compounds.setdefault(_normalise_path(record["path"]).casefold(), []).append(record)

    def compound_closure(source):
        resolved = {}
        missing = []
        pending = [
            (path, source.get("archive", ""), source.get("cache_key", ""))
            for path in mgraph.referenced_compounds(source["data"])
        ]
        while pending:
            path, preferred_archive, preferred_cache_key = pending.pop(0)
            key = _normalise_path(path).casefold()
            if key in resolved or key in {value.casefold() for value in missing}:
                continue
            record = _choose(
                compounds.get(key, ()),
                preferred_archive,
                preferred_cache_key,
            )
            if record is None:
                missing.append(_normalise_path(path))
                continue
            resolved[key] = record
            pending.extend(
                (linked, record.get("archive", ""), record.get("cache_key", ""))
                for linked in mgraph.referenced_compounds(record["data"])
            )
        return resolved, missing

    source_rows = []
    issues = []
    shader_usage = {}
    material_count = 0
    for source in sources:
        path = _normalise_path(source["path"])
        if source["data"][:4] != mgraph.MAGIC:
            issues.append(_issue(
                "invalid_material_source", path,
                "The extracted asset does not begin with BV2 magic.",
                "info",
            ))
        closure, missing_compounds = compound_closure(source)
        all_records = [source, *closure.values()]
        direct_mmbs = [_normalise_path(value) for value in mgraph.referenced_meshes(source["data"])]
        resolved_mmbs = []
        textures = []
        material_pairs = []
        seen_mmbs = set()
        seen_textures = set()
        for record in all_records:
            for value in mgraph.referenced_meshes(record["data"]):
                value = _normalise_path(value)
                if value.casefold() not in seen_mmbs:
                    seen_mmbs.add(value.casefold())
                    resolved_mmbs.append(value)
            for texture in mgraph.texture_pool(record["data"]):
                key = texture["path"].casefold()
                if key not in seen_textures:
                    seen_textures.add(key)
                    textures.append(texture)
            for name, shader in mgraph.material_shader_pairs(record["data"]):
                material_pairs.append((name, shader, record))

        compound_data = {
            record["path"]: record["data"] for record in closure.values()
        }
        names = list(dict.fromkeys(name for name, _shader, _record in material_pairs))
        try:
            bindings = mgraph.material_bindings(
                source["data"], names, compound_sources=compound_data,
                shader_role_pins=shader_role_pins,
                shader_parameter_pins=shader_parameter_pins,
            ) if names else {}
        except Exception as error:
            bindings = {}
            issues.append(_issue(
                "material_binding_error", path,
                f"Could not resolve combined material bindings: {error}",
            ))
        local_bindings = {}
        for record in all_records:
            record_pairs = [
                (name, shader)
                for name, shader, owner in material_pairs if owner is record
            ]
            record_names = list(dict.fromkeys(name for name, _shader in record_pairs))
            if not record_names:
                continue
            try:
                resolved = mgraph.material_bindings(
                    record["data"], record_names,
                    shader_role_pins=shader_role_pins,
                    shader_parameter_pins=shader_parameter_pins,
                )
            except Exception as error:
                resolved = {}
                issues.append(_issue(
                    "material_binding_error", path,
                    f"Could not resolve bindings from {record['path']}: {error}",
                ))
            for name, shader in record_pairs:
                candidate = dict(resolved.get(name, {}))
                candidate["shader"] = candidate.get("shader") or shader
                key = (name.casefold(), shader.casefold())
                existing = local_bindings.get(key, {})
                candidate_score = sum(
                    bool(candidate.get(role))
                    for role in ("d", "n", "m", "a", "bio_palette", "emissive_color")
                ) + len(candidate.get("aux", {}))
                existing_score = sum(
                    bool(existing.get(role))
                    for role in ("d", "n", "m", "a", "bio_palette", "emissive_color")
                ) + len(existing.get("aux", {}))
                if candidate_score > existing_score or not existing:
                    local_bindings[key] = candidate
        materials = []
        reported_materials = set()
        for name, declared_shader, _owner in material_pairs:
            material_key = (name.casefold(), declared_shader.casefold())
            if material_key in reported_materials:
                continue
            reported_materials.add(material_key)
            binding = dict(bindings.get(name, {}))
            local = local_bindings.get(
                material_key, {}
            )
            for role in (
                "shader", "d", "n", "m", "a", "a_channel",
                "emissive_color", "emissive_strength", "bio_palette",
                "bio_procedural", "bio_strength",
            ):
                if not binding.get(role) and local.get(role):
                    binding[role] = local[role]
            auxiliary = dict(local.get("aux", {}))
            auxiliary.update(binding.get("aux", {}))
            parameters = dict(local.get("parameters", {}))
            parameters.update(binding.get("parameters", {}))
            shader = _normalise_path(binding.get("shader") or declared_shader)
            profile, support = _runtime_profile(shader)
            diffuse = _normalise_path(binding.get("d", ""))
            surface_alpha = _normalise_path(binding.get("a", ""))
            surface_alpha_channel = binding.get("a_channel") or ""
            surface_alpha_source = "dedicated" if surface_alpha else ""
            if (
                not surface_alpha
                and diffuse
                and mgraph.diffuse_drives_surface_alpha(shader)
            ):
                surface_alpha = diffuse
                surface_alpha_channel = "alpha"
                surface_alpha_source = "diffuse"
            material = {
                "name": name,
                "shader": shader,
                "diffuse": diffuse,
                "normal": _normalise_path(binding.get("n", "")),
                "material_mask": _normalise_path(binding.get("m", "")),
                "surface_alpha": surface_alpha,
                "surface_alpha_channel": surface_alpha_channel,
                "surface_alpha_source": surface_alpha_source,
                "emissive_color": list(binding.get("emissive_color", ())),
                "emissive_strength": binding.get("emissive_strength", ""),
                "bio_palette": [list(color) for color in binding.get("bio_palette", ())],
                "bio_procedural": binding.get("bio_procedural", ""),
                "bio_strength": binding.get("bio_strength", ""),
                "aux_textures": auxiliary,
                "authored_parameters": parameters,
                "blender_profile": profile,
                "blender_support": support,
            }
            materials.append(material)
            material_count += 1
            if shader:
                usage = shader_usage.setdefault(shader.casefold(), {
                    "path": shader, "sources": set(), "materials": set(),
                    "mmbs": set(),
                })
                usage["sources"].add(path)
                usage["materials"].add(name)
                usage["mmbs"].update(resolved_mmbs)
            else:
                issues.append(_issue(
                    "material_without_shader", path,
                    f"Material {name!r} has no resolved ShaderFile.",
                ))

        for compound in missing_compounds:
            issues.append(_issue(
                "missing_compound", path,
                f"Referenced compound is not indexed: {compound}",
            ))
        for mmb_path in resolved_mmbs:
            if mmb_path.casefold() not in mmb_keys:
                issues.append(_issue(
                    "missing_mmb", path,
                    f"Referenced MMB is not indexed: {mmb_path}",
                ))
        source_rows.append({
            "path": path,
            "kind": source.get("kind", ""),
            "archive": source.get("archive", ""),
            "toc_cache_key": source.get("cache_key", ""),
            "direct_mmbs": direct_mmbs,
            "resolved_mmbs": resolved_mmbs,
            "direct_compounds": [
                _normalise_path(value)
                for value in mgraph.referenced_compounds(source["data"])
            ],
            "resolved_compounds": [record["path"] for record in closure.values()],
            "missing_compounds": missing_compounds,
            "mesh_relevant": bool(resolved_mmbs),
            "textures": textures,
            "materials": materials,
        })

        shaders_by_name = {}
        for material in materials:
            shaders_by_name.setdefault(material["name"].casefold(), set()).add(
                material["shader"].casefold()
            )
        for material_name, shader_names in shaders_by_name.items():
            if len(shader_names) > 1 and material_name != "rtmaterial":
                issues.append(_issue(
                    "conflicting_material_shaders", path,
                    f"Material {material_name!r} resolves to {len(shader_names)} shaders.",
                ))

    source_variants = {}
    for source in source_rows:
        source_variants.setdefault(source["path"].casefold(), []).append(source)
    for variants in source_variants.values():
        if len(variants) < 2:
            continue
        signatures = {
            tuple(sorted(
                (
                    material["name"].casefold(), material["shader"].casefold(),
                    material["diffuse"].casefold(), material["normal"].casefold(),
                    material["material_mask"].casefold(),
                    material["surface_alpha"].casefold(),
                )
                for material in source["materials"]
            ))
            for source in variants
        }
        if len(signatures) > 1:
            issues.append(_issue(
                "material_source_variants", variants[0]["path"],
                f"{len(variants)} indexed copies contain {len(signatures)} material variants.",
                "info",
            ))

    shader_rows = []
    indexed_shader_keys = set()
    for record in shaders:
        path = _normalise_path(record["path"])
        key = path.casefold()
        indexed_shader_keys.add(key)
        parsed = parsed_shaders[id(record)]
        usage = shader_usage.get(key, {})
        profile, support = _runtime_profile(path)
        samplers = parsed["samplers"]
        for sampler in samplers:
            sampler["surface_alpha_channel"] = mgraph.surface_alpha_channel(
                path, sampler["pin_id"]
            )
            sampler["surface_alpha_source"] = (
                "dedicated" if sampler["surface_alpha_channel"] else ""
            )
            if (
                not sampler["surface_alpha_channel"]
                and sampler["role"] == "d"
                and mgraph.diffuse_drives_surface_alpha(path)
            ):
                sampler["surface_alpha_channel"] = "alpha"
                sampler["surface_alpha_source"] = "diffuse"
            target, mapping_status = _sampler_blender_mapping(
                sampler, profile, support
            )
            sampler["blender_target"] = target
            sampler["blender_mapping_status"] = mapping_status
        shader_rows.append({
            "path": path,
            "archive": record.get("archive", ""),
            "toc_cache_key": record.get("cache_key", ""),
            "shader_type": parsed["shader_type"],
            "samplers": samplers,
            "parameters": parsed["parameters"],
            "blender_profile": profile,
            "blender_support": support,
            "used_by_sources": sorted(usage.get("sources", ()), key=str.casefold),
            "used_by_materials": sorted(usage.get("materials", ()), key=str.casefold),
            "used_by_mmbs": sorted(usage.get("mmbs", ()), key=str.casefold),
        })
        if not samplers and usage.get("materials"):
            issues.append(_issue(
                "shader_without_samplers", path,
                "No MR_Sampler2D declarations were parsed.", "info",
            ))

    for shader_key, usage in sorted(shader_usage.items()):
        if shader_key not in indexed_shader_keys:
            issues.append(_issue(
                "missing_shader", next(iter(sorted(usage["sources"]))),
                f"Referenced shader is not indexed: {usage['path']}",
            ))

    profile_names = sorted({
        material["blender_profile"]
        for source in source_rows for material in source["materials"]
    } | {shader["blender_profile"] for shader in shader_rows})
    profile_coverage = []
    for profile in profile_names:
        profile_materials = [
            material for source in source_rows for material in source["materials"]
            if material["blender_profile"] == profile
        ]
        mesh_profile_materials = [
            material for source in source_rows if source["mesh_relevant"]
            for material in source["materials"]
            if material["blender_profile"] == profile
        ]
        profile_shaders = [
            shader for shader in shader_rows
            if shader["blender_profile"] == profile
        ]
        sampler_statuses = {}
        used_sampler_statuses = {}
        for shader in profile_shaders:
            for sampler in shader["samplers"]:
                status = sampler["blender_mapping_status"]
                sampler_statuses[status] = sampler_statuses.get(status, 0) + 1
                if shader["used_by_materials"]:
                    used_sampler_statuses[status] = (
                        used_sampler_statuses.get(status, 0) + 1
                    )
        profile_coverage.append({
            "profile": profile,
            "support": sorted({
                material["blender_support"] for material in profile_materials
            } | {shader["blender_support"] for shader in profile_shaders}),
            "materials": len(profile_materials),
            "mesh_materials": len(mesh_profile_materials),
            "shaders": len(profile_shaders),
            "used_shaders": sum(
                bool(shader["used_by_materials"]) for shader in profile_shaders
            ),
            "sampler_statuses": sampler_statuses,
            "used_sampler_statuses": used_sampler_statuses,
        })

    summary = {
        "sources": len(source_rows),
        "graphobjects": sum(row["kind"] == "mgraphobject" for row in source_rows),
        "compoundnodes": sum(row["kind"] == "mcompoundnode" for row in source_rows),
        "shaders": len(shader_rows),
        "materials": material_count,
        "mesh_sources": sum(row["mesh_relevant"] for row in source_rows),
        "mesh_materials": sum(
            len(row["materials"]) for row in source_rows if row["mesh_relevant"]
        ),
        "auxiliary_texture_bindings": sum(
            len(material["aux_textures"])
            for row in source_rows for material in row["materials"]
        ),
        "mesh_auxiliary_texture_bindings": sum(
            len(material["aux_textures"])
            for row in source_rows if row["mesh_relevant"]
            for material in row["materials"]
        ),
        "used_shaders": sum(bool(row["used_by_materials"]) for row in shader_rows),
        "shader_samplers": sum(len(row["samplers"]) for row in shader_rows),
        "shader_parameters": sum(
            len(row["parameters"]) for row in shader_rows
        ),
        "authored_material_parameters": sum(
            len(material["authored_parameters"])
            for row in source_rows for material in row["materials"]
        ),
        "unmapped_shader_samplers": sum(
            sampler["blender_mapping_status"] in {"unmapped", "not_implemented"}
            for row in shader_rows for sampler in row["samplers"]
        ),
        "baked_default_shader_samplers": sum(
            sampler["blender_mapping_status"] == "baked_default"
            for row in shader_rows for sampler in row["samplers"]
        ),
        "used_shader_samplers": sum(
            len(row["samplers"]) for row in shader_rows if row["used_by_materials"]
        ),
        "used_unmapped_shader_samplers": sum(
            sampler["blender_mapping_status"] in {"unmapped", "not_implemented"}
            for row in shader_rows if row["used_by_materials"]
            for sampler in row["samplers"]
        ),
        "surface_alpha_samplers": sum(
            bool(sampler["surface_alpha_channel"])
            for row in shader_rows for sampler in row["samplers"]
        ),
        "referenced_mmbs": len({
            value.casefold() for row in source_rows for value in row["resolved_mmbs"]
        }),
        "issues": len(issues),
        "issues_by_severity": {
            severity: sum(issue["severity"] == severity for issue in issues)
            for severity in ("warning", "info")
        },
        "profile_coverage": {
            row["profile"]: {
                "materials": row["materials"],
                "mesh_materials": row["mesh_materials"],
                "used_shaders": row["used_shaders"],
                "used_sampler_statuses": row["used_sampler_statuses"],
            }
            for row in profile_coverage
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_signature": sorted(set(input_signature or ())),
        "summary": summary,
        "profile_coverage": profile_coverage,
        "sources": source_rows,
        "shaders": shader_rows,
        "issues": issues,
    }


def _atomic_text(path, write):
    temporary = path + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8", newline="") as stream:
            write(stream)
        os.replace(temporary, path)
    finally:
        try:
            if os.path.exists(temporary):
                os.remove(temporary)
        except OSError:
            pass


def _joined(values):
    return "; ".join(str(value) for value in values)


def write_reports(report, output_directory):
    """Write the master JSON and compact CSV views atomically."""
    os.makedirs(output_directory, exist_ok=True)
    json_path = os.path.join(output_directory, REPORT_JSON)
    _atomic_text(json_path, lambda stream: json.dump(
        report, stream, indent=2, ensure_ascii=False
    ))

    source_fields = (
        "path", "kind", "archive", "toc_cache_key", "mesh_relevant",
        "direct_mmbs", "resolved_mmbs", "direct_compounds",
        "resolved_compounds", "missing_compounds",
        "texture_count", "material_count",
    )
    def write_sources(stream):
        writer = csv.DictWriter(stream, fieldnames=source_fields)
        writer.writeheader()
        for source in report["sources"]:
            writer.writerow({
                **{field: source.get(field, "") for field in source_fields[:5]},
                "direct_mmbs": _joined(source["direct_mmbs"]),
                "resolved_mmbs": _joined(source["resolved_mmbs"]),
                "direct_compounds": _joined(source["direct_compounds"]),
                "resolved_compounds": _joined(source["resolved_compounds"]),
                "missing_compounds": _joined(source["missing_compounds"]),
                "texture_count": len(source["textures"]),
                "material_count": len(source["materials"]),
            })
    _atomic_text(os.path.join(output_directory, SOURCE_CSV), write_sources)

    material_fields = (
        "source", "source_kind", "archive", "toc_cache_key", "mesh_relevant",
        "material", "shader", "diffuse",
        "normal", "material_mask", "surface_alpha", "surface_alpha_channel",
        "surface_alpha_source", "blender_profile", "blender_support",
        "emissive_color", "emissive_strength", "bio_palette",
        "bio_procedural", "bio_strength", "aux_textures",
        "authored_parameters", "referenced_mmbs",
    )
    def write_materials(stream):
        writer = csv.DictWriter(stream, fieldnames=material_fields)
        writer.writeheader()
        for source in report["sources"]:
            for material in source["materials"]:
                writer.writerow({
                    "source": source["path"],
                    "source_kind": source["kind"],
                    "archive": source["archive"],
                    "toc_cache_key": source["toc_cache_key"],
                    "mesh_relevant": source["mesh_relevant"],
                    "material": material["name"],
                    "shader": material["shader"],
                    "diffuse": material["diffuse"],
                    "normal": material["normal"],
                    "material_mask": material["material_mask"],
                    "surface_alpha": material["surface_alpha"],
                    "surface_alpha_channel": material["surface_alpha_channel"],
                    "surface_alpha_source": material["surface_alpha_source"],
                    "blender_profile": material["blender_profile"],
                    "blender_support": material["blender_support"],
                    "emissive_color": json.dumps(material["emissive_color"]),
                    "emissive_strength": material["emissive_strength"],
                    "bio_palette": json.dumps(material["bio_palette"]),
                    "bio_procedural": material["bio_procedural"],
                    "bio_strength": material["bio_strength"],
                    "aux_textures": json.dumps(
                        material["aux_textures"], sort_keys=True
                    ),
                    "authored_parameters": json.dumps(
                        material["authored_parameters"], sort_keys=True
                    ),
                    "referenced_mmbs": _joined(source["resolved_mmbs"]),
                })
    _atomic_text(os.path.join(output_directory, MATERIAL_CSV), write_materials)

    shader_fields = (
        "path", "archive", "toc_cache_key", "shader_type", "blender_profile", "blender_support",
        "sampler_count", "sampler_roles", "sampler_mappings",
        "parameter_count", "parameters", "used_by_sources",
        "used_by_materials", "used_by_mmbs",
    )
    def write_shaders(stream):
        writer = csv.DictWriter(stream, fieldnames=shader_fields)
        writer.writeheader()
        for shader in report["shaders"]:
            writer.writerow({
                **{field: shader.get(field, "") for field in shader_fields[:6]},
                "sampler_count": len(shader["samplers"]),
                "sampler_roles": _joined(
                    f"{item['field']}={item['role']}@{item['pin_id']}"
                    for item in shader["samplers"]
                ),
                "sampler_mappings": _joined(
                    f"{item['field']} -> {item['blender_target'] or 'UNMAPPED'}"
                    for item in shader["samplers"]
                ),
                "parameter_count": len(shader["parameters"]),
                "parameters": _joined(
                    f"{item['field']}={item['default']}@{item['pin_id']}"
                    for item in shader["parameters"]
                ),
                "used_by_sources": _joined(shader["used_by_sources"]),
                "used_by_materials": _joined(shader["used_by_materials"]),
                "used_by_mmbs": _joined(shader["used_by_mmbs"]),
            })
    _atomic_text(os.path.join(output_directory, SHADER_CSV), write_shaders)

    profile_fields = (
        "profile", "support", "materials", "mesh_materials", "shaders",
        "used_shaders", "sampler_statuses", "used_sampler_statuses",
    )
    def write_profiles(stream):
        writer = csv.DictWriter(stream, fieldnames=profile_fields)
        writer.writeheader()
        for profile in report.get("profile_coverage", ()):
            writer.writerow({
                **profile,
                "support": _joined(profile["support"]),
                "sampler_statuses": json.dumps(
                    profile["sampler_statuses"], sort_keys=True
                ),
                "used_sampler_statuses": json.dumps(
                    profile["used_sampler_statuses"], sort_keys=True
                ),
            })
    _atomic_text(os.path.join(output_directory, PROFILE_CSV), write_profiles)

    issue_fields = ("severity", "kind", "source", "detail")
    def write_issues(stream):
        writer = csv.DictWriter(stream, fieldnames=issue_fields)
        writer.writeheader()
        writer.writerows(report["issues"])
    _atomic_text(os.path.join(output_directory, ISSUE_CSV), write_issues)
    return {
        "json": json_path,
        "sources_csv": os.path.join(output_directory, SOURCE_CSV),
        "materials_csv": os.path.join(output_directory, MATERIAL_CSV),
        "shaders_csv": os.path.join(output_directory, SHADER_CSV),
        "profiles_csv": os.path.join(output_directory, PROFILE_CSV),
        "issues_csv": os.path.join(output_directory, ISSUE_CSV),
    }

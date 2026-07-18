"""Snowdrop texture conversion and Blender material construction."""

from __future__ import annotations

import os
import struct

import bpy

from .log import logger


_STF_DXGI = {
    0x04: 87, 0x05: 2, 0x07: 10, 0x0A: 11, 0x0D: 29, 0x0E: 28,
    0x17: 49, 0x19: 56, 0x1B: 61, 0x1C: 72, 0x1D: 71, 0x1E: 75,
    0x1F: 74, 0x20: 78, 0x21: 77, 0x45: 80, 0x46: 81, 0x47: 83,
    0x48: 84, 0x49: 95, 0x4A: 96, 0x4B: 98, 0x4C: 99,
}
_BLOCK_BYTES = {
    71: 8, 72: 8, 74: 16, 75: 16, 77: 16, 78: 16, 80: 8, 81: 8,
    83: 16, 84: 16, 95: 16, 96: 16, 98: 16, 99: 16,
}
_BYTES_PER_PIXEL = {2: 16, 10: 8, 11: 8, 28: 4, 29: 4, 49: 2, 56: 2, 61: 1, 87: 4}


def _diffuse_drives_surface_alpha(shader_name):
    """Whether this Snowdrop shader treats its color texture alpha as opacity.

    This is deliberately shader-gated. AFOP frequently stores roughness, wear,
    and other packed data in otherwise opaque albedo alpha channels, so image
    alpha statistics or the mere presence of an alpha channel are not proof of
    transparency.
    """
    name = os.path.basename(shader_name).casefold()
    if name in {
        "px_basic.mshader",
        "px_character_gear_simple.mshader",
        "px_character_gear_pearl.mshader",
    }:
        return True
    if name.startswith(("px_basic_transmission", "px_basic_transparent")):
        return True
    if name.startswith((
        "px_vegetation",
        "px_dlc1_vegetation",
        "px_dlc2_vegetation",
        "px_dlc3_vegetation",
        "px_harvest_vegetation",
        "px_grass",
        "px_dlc1_grass",
        "px_dlc3_grass",
        "px_mosscard",
        "px_dlc2_mosscard",
        "px_dlc3_mosscard",
        "px_dlc3_mossycard",
    )):
        return True
    if name.startswith((
        "px_decal_",
        "px_character_weapon_decal",
        "px_fx_decal_",
        "deferred-decal-",
    )):
        return True
    if name.startswith((
        "px_hair",
        "px_character_gear_fur",
        "px_wildlife_fur",
    )) or name == "hair.mshader":
        return True
    if name.startswith((
        "px_wildlife_dragonflywing",
        "px_wildlife_fanlizardwing",
        "px_wildlife_gear",
    )):
        return True
    if "glass" in name or "foliage" in name:
        return True
    if "transparent" in name or "transmission alpha" in name:
        return True
    return False


def _mip_entries(data):
    descriptor = struct.unpack_from("<Q", data, 4)[0]
    count = ((descriptor >> 56) & 0x0F) + 1
    if 12 + count * 4 > len(data):
        raise ValueError("truncated STF mip table")
    entries = []
    for index in range(count):
        value = struct.unpack_from("<I", data, 12 + index * 4)[0]
        entries.append(((value >> 23) & 0x3F, (value & 0x7FFFFF) * 16))
    return descriptor, entries


def _build_dds(width, height, dxgi, payload):
    header = bytearray(148)
    header[:4] = b"DDS "
    struct.pack_into("<I", header, 4, 124)
    struct.pack_into("<I", header, 8, 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000)
    struct.pack_into("<I", header, 12, height)
    struct.pack_into("<I", header, 16, width)
    struct.pack_into("<I", header, 20, len(payload))
    struct.pack_into("<I", header, 28, 1)
    struct.pack_into("<I", header, 76, 32)
    struct.pack_into("<I", header, 80, 0x4)
    header[84:88] = b"DX10"
    struct.pack_into("<I", header, 108, 0x1000)
    struct.pack_into("<I", header, 128, dxgi)
    struct.pack_into("<I", header, 132, 3)  # D3D10_RESOURCE_DIMENSION_TEXTURE2D
    struct.pack_into("<I", header, 140, 1)
    return bytes(header) + payload


def texture_to_dds(data, asset):
    """Turn an extracted STF texture into a one-mip, Blender-readable DDS."""
    if data[:4] == b"DDS ":
        return data
    if data[:4] != b"STF\x02":
        raise ValueError("texture has neither STF nor DDS magic")
    descriptor, entries = _mip_entries(data)
    dxgi = _STF_DXGI.get(descriptor & 0xFF)
    width = (descriptor >> 10) & 0x7FFF
    height = (descriptor >> 25) & 0x7FFF
    if dxgi is None or not (0 < width <= 16384 and 0 < height <= 16384):
        raise ValueError("unsupported STF texture descriptor")

    slice_sizes = [item.decompressed_size for item in asset.data_slices]
    header_size = len(data) - sum(slice_sizes)
    resident = -1
    for index, (_mips, expected_size) in enumerate(entries[:len(slice_sizes)]):
        if abs(slice_sizes[index] - expected_size) > 16:
            break
        resident = index
    if resident < 0:
        # Some old/stub descriptors disagree with their slices.  The largest
        # slice is still preferable to silently importing an empty texture.
        resident = max(range(len(slice_sizes)), key=slice_sizes.__getitem__)

    mip_level = sum(entry[0] for entry in entries[resident + 1:])
    mip_width = max(width >> mip_level, 1)
    mip_height = max(height >> mip_level, 1)
    offset = header_size + sum(slice_sizes[:resident])
    resident_data = data[offset:offset + slice_sizes[resident]]
    if dxgi in _BLOCK_BYTES:
        needed = (
            max((mip_width + 3) // 4, 1)
            * max((mip_height + 3) // 4, 1)
            * _BLOCK_BYTES[dxgi]
        )
    else:
        needed = mip_width * mip_height * _BYTES_PER_PIXEL[dxgi]
    if needed > len(resident_data):
        raise ValueError("resident texture mip is truncated")
    return _build_dds(mip_width, mip_height, dxgi, resident_data[:needed])


def _load_image(path, logical_path, non_color=False):
    existing = next(
        (image for image in bpy.data.images if image.get("afop_asset_path") == logical_path),
        None,
    )
    image = existing or bpy.data.images.load(path, check_existing=True)
    image["afop_asset_path"] = logical_path
    if non_color:
        try:
            image.colorspace_settings.name = "Non-Color"
        except (AttributeError, TypeError):
            logger.warning("Could not set Non-Color space for %s", logical_path)
    return image


def _new_image_node(nodes, image, label, x, y):
    node = nodes.new("ShaderNodeTexImage")
    node.image = image
    node.label = label
    node.name = label
    node.location = (x, y)
    return node


def _math_node(nodes, operation, label, x, y, value=None):
    node = nodes.new("ShaderNodeMath")
    node.operation = operation
    node.label = label
    node.name = label
    node.location = (x, y)
    if value is not None:
        node.inputs[1].default_value = value
    return node


def _packed_normal_nodes(
    nodes,
    links,
    image_node,
    *,
    x_channel="Alpha",
    auxiliary_channel="Blue",
):
    """Decode a Snowdrop tangent normal and expose its auxiliary channel.

    Most shaders call ``UnpackNormalAndAO`` (X in alpha, Y in green), while the
    Banshee skin and dragonfly-wing shaders read X/Y directly from red/green.
    Both reconstruct positive Z. A standard Blender Normal Map node interprets
    the stored RGB as XYZ, which is why lowering its strength appeared to help
    some assets without making their normals correct.
    """
    separate = nodes.new("ShaderNodeSeparateColor")
    separate.label = f"Snowdrop packed normal ({x_channel[0]}/G)"
    separate.name = separate.label
    separate.location = (-195, -80)
    links.new(image_node.outputs["Color"], separate.inputs["Color"])

    # Match the engine's signed decode: packed * 2 - (2 * 128 / 255).
    offset = 2.0 * 128.0 / 255.0
    x_scale = _math_node(nodes, "MULTIPLY", "Normal X x2", 35, -5, 2.0)
    x_signed = _math_node(nodes, "SUBTRACT", "Normal X signed", 205, -5, offset)
    y_scale = _math_node(nodes, "MULTIPLY", "Normal Y x2", 35, -105, 2.0)
    y_signed = _math_node(nodes, "SUBTRACT", "Normal Y signed", 205, -105, offset)
    x_output = (
        image_node.outputs["Alpha"]
        if x_channel == "Alpha"
        else separate.outputs[x_channel]
    )
    links.new(x_output, x_scale.inputs[0])
    links.new(x_scale.outputs[0], x_signed.inputs[0])
    links.new(separate.outputs["Green"], y_scale.inputs[0])
    links.new(y_scale.outputs[0], y_signed.inputs[0])

    x_sq = _math_node(nodes, "MULTIPLY", "Normal X squared", 375, -5)
    y_sq = _math_node(nodes, "MULTIPLY", "Normal Y squared", 375, -105)
    links.new(x_signed.outputs[0], x_sq.inputs[0])
    links.new(x_signed.outputs[0], x_sq.inputs[1])
    links.new(y_signed.outputs[0], y_sq.inputs[0])
    links.new(y_signed.outputs[0], y_sq.inputs[1])
    sum_sq = _math_node(nodes, "ADD", "Normal XY squared", 535, -55)
    links.new(x_sq.outputs[0], sum_sq.inputs[0])
    links.new(y_sq.outputs[0], sum_sq.inputs[1])
    remaining = _math_node(nodes, "SUBTRACT", "Normal Z squared", 695, -55)
    remaining.inputs[0].default_value = 1.0
    links.new(sum_sq.outputs[0], remaining.inputs[1])
    clamp = _math_node(nodes, "MAXIMUM", "Clamp normal Z", 855, -55, 0.0)
    root = _math_node(nodes, "SQRT", "Reconstruct normal Z", 1015, -55)
    links.new(remaining.outputs[0], clamp.inputs[0])
    links.new(clamp.outputs[0], root.inputs[0])
    z_half = _math_node(nodes, "MULTIPLY", "Encode normal Z", 1175, -55, 0.5)
    z_encoded = _math_node(nodes, "ADD", "Encode normal Z + 0.5", 1335, -55, 0.5)
    links.new(root.outputs[0], z_half.inputs[0])
    links.new(z_half.outputs[0], z_encoded.inputs[0])

    combine = nodes.new("ShaderNodeCombineColor")
    combine.label = "Reconstructed tangent normal"
    combine.name = combine.label
    combine.location = (1505, -55)
    links.new(x_output, combine.inputs["Red"])
    links.new(separate.outputs["Green"], combine.inputs["Green"])
    links.new(z_encoded.outputs[0], combine.inputs["Blue"])
    auxiliary_output = (
        image_node.outputs["Alpha"]
        if auxiliary_channel == "Alpha"
        else separate.outputs[auxiliary_channel]
    )
    return combine.outputs["Color"], auxiliary_output


def _wildlife_bio_nodes(nodes, links, mask_node, normal_alpha, shader, palette):
    """Build a static nighttime preview of PX_Wildlife_Skin bioluminescence."""
    if not isinstance(palette, (list, tuple)) or len(palette) != 6:
        return False
    emission_input = shader.inputs.get("Emission Color") or shader.inputs.get("Emission")
    if emission_input is None:
        return False

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.label = "Wildlife bioluminescence palette (night preview)"
    ramp.name = ramp.label
    ramp.location = (820, -430)
    ramp.color_ramp.interpolation = "LINEAR"
    positions = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    elements = ramp.color_ramp.elements
    elements[0].position = positions[0]
    elements[0].color = (*palette[0][:3], 1.0)
    elements[-1].position = positions[-1]
    elements[-1].color = (*palette[-1][:3], 1.0)
    for position, color in zip(positions[1:-1], palette[1:-1]):
        element = elements.new(position)
        element.color = (*color[:3], 1.0)
    links.new(mask_node.outputs["Alpha"], ramp.inputs["Fac"])

    bio_output = ramp.outputs["Color"]
    if normal_alpha is not None:
        multiply = nodes.new("ShaderNodeMixRGB")
        multiply.blend_type = "MULTIPLY"
        multiply.inputs[0].default_value = 1.0
        multiply.label = "Apply wildlife bio/AO mask"
        multiply.name = multiply.label
        multiply.location = (1320, -430)
        links.new(bio_output, multiply.inputs[1])
        links.new(normal_alpha, multiply.inputs[2])
        bio_output = multiply.outputs["Color"]
    links.new(bio_output, emission_input)
    strength_input = shader.inputs.get("Emission Strength")
    if strength_input is not None:
        strength_input.default_value = 1.0
    return True


def _medusa_bio_nodes(
    nodes, links, mask_node, mask_separate, shader, palette, procedural, strength
):
    """Build the M-map component of the Medusa bioluminescence shader."""
    if not isinstance(palette, (list, tuple)) or len(palette) != 5:
        return False
    emission_input = shader.inputs.get("Emission Color") or shader.inputs.get("Emission")
    if emission_input is None:
        return False

    bio_factor = mask_node.outputs["Alpha"]
    if procedural:
        multiply = _math_node(
            nodes, "MULTIPLY", "Medusa procedural bio mask (A x B)", 55, -545
        )
        links.new(mask_node.outputs["Alpha"], multiply.inputs[0])
        links.new(mask_separate.outputs["Blue"], multiply.inputs[1])
        bio_factor = multiply.outputs[0]

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.label = "Medusa bioluminescence palette (night preview)"
    ramp.name = ramp.label
    ramp.location = (820, -545)
    ramp.color_ramp.interpolation = "LINEAR"
    positions = (0.0, 0.25, 0.5, 0.75, 1.0)
    elements = ramp.color_ramp.elements
    elements[0].position = positions[0]
    elements[0].color = (*palette[0][:3], 1.0)
    elements[-1].position = positions[-1]
    elements[-1].color = (*palette[-1][:3], 1.0)
    for position, color in zip(positions[1:-1], palette[1:-1]):
        element = elements.new(position)
        element.color = (*color[:3], 1.0)
    links.new(bio_factor, ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], emission_input)
    strength_input = shader.inputs.get("Emission Strength")
    if strength_input is not None:
        strength_input.default_value = float(strength)
    return True


def _enable_alpha_blending(material):
    try:
        material.surface_render_method = "DITHERED"
    except (AttributeError, TypeError, ValueError):
        try:
            material.blend_method = "HASHED"
        except (AttributeError, TypeError, ValueError):
            pass


def assign_materials(skeletal_mesh, bindings, texture_files, source_path, lod_index=0):
    """Create Principled materials and assign one to each imported MMB part."""
    assigned = 0
    for mesh in skeletal_mesh.meshes:
        if (
            len(mesh.lods) <= lod_index
            or mesh.name.casefold().endswith("_cloth_sim")
        ):
            continue
        obj = bpy.data.objects.get(mesh.lods[lod_index].blender_obj_name)
        binding = bindings.get(mesh.name)
        if obj is None or obj.type != "MESH" or not binding:
            continue
        shader_name = os.path.basename(binding.get("shader", "")).casefold()
        is_navi_skin = shader_name in {
            "px_character_navi.mshader",
            "px_character_navi_face.mshader",
            "px_character_workbench.mshader",
        }
        is_wildlife_skin = shader_name.startswith("px_wildlife_skin")
        is_medusa_skin = shader_name.startswith("px_dlc3_medusa_skin")
        is_dragonfly_wing = shader_name == "px_wildlife_dragonflywing.mshader"
        is_emissive_color = shader_name == "px_emissive_color.mshader"

        material_name = f"AFOP_{skeletal_mesh.name}_{mesh.name}"
        material = (
            bpy.data.materials.get(material_name)
            or bpy.data.materials.new(material_name)
        )
        material.use_nodes = True
        material["afop_material_source"] = source_path
        if binding.get("shader"):
            material["afop_shader"] = binding["shader"]
        if source_path.casefold().endswith(".mgraphobject"):
            material["afop_mgraphobject"] = source_path
        elif source_path.casefold().endswith(".mcompoundnode"):
            material["afop_mcompoundnode"] = source_path
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        nodes.clear()
        output = nodes.new("ShaderNodeOutputMaterial")
        output.location = (2200, 40)
        if is_emissive_color:
            color = tuple(binding.get("emissive_color", (0.0, 0.0, 0.0)))
            strength = float(binding.get("emissive_strength", 1.0))
            emission = nodes.new("ShaderNodeEmission")
            emission.label = "Snowdrop Emissive Color"
            emission.name = emission.label
            emission.location = (1950, 40)
            emission.inputs["Color"].default_value = (*color[:3], 1.0)
            emission.inputs["Strength"].default_value = strength
            links.new(emission.outputs["Emission"], output.inputs["Surface"])
            material["afop_emissive_color"] = list(color[:3])
            material["afop_emissive_strength"] = strength
            obj.data.materials.clear()
            obj.data.materials.append(material)
            assigned += 1
            continue
        shader = nodes.new("ShaderNodeBsdfPrincipled")
        shader.location = (1950, 40)
        links.new(shader.outputs["BSDF"], output.inputs["Surface"])

        diffuse_path = binding.get("d")
        diffuse_node = None
        if diffuse_path and diffuse_path.casefold() in texture_files:
            disk_path = texture_files[diffuse_path.casefold()]
            image = _load_image(disk_path, diffuse_path, non_color=False)
            node = _new_image_node(nodes, image, "Diffuse / Albedo", -440, 180)
            diffuse_node = node
            material["afop_diffuse"] = diffuse_path

        normal_path = binding.get("n")
        ao_output = None
        roughness_output = None
        if normal_path and normal_path.casefold() in texture_files:
            disk_path = texture_files[normal_path.casefold()]
            image = _load_image(disk_path, normal_path, non_color=True)
            node = _new_image_node(nodes, image, "Normal (packed)", -440, -80)
            if is_wildlife_skin:
                # PX_Wildlife_Skin reads normal X/Y from R/G and AO from A.
                normal_color, ao_output = _packed_normal_nodes(
                    nodes,
                    links,
                    node,
                    x_channel="Red",
                    auxiliary_channel="Alpha",
                )
            elif is_dragonfly_wing:
                # PX_Wildlife_DragonflyWing reads R/G as X/Y and B as its
                # roughness signal. It does not use this texture as AO.
                normal_color, roughness_output = _packed_normal_nodes(
                    nodes,
                    links,
                    node,
                    x_channel="Red",
                    auxiliary_channel="Blue",
                )
            else:
                normal_color, ao_output = _packed_normal_nodes(nodes, links, node)
            normal = nodes.new("ShaderNodeNormalMap")
            normal.location = (1710, -55)
            normal.inputs["Strength"].default_value = 1.0
            links.new(normal_color, normal.inputs["Color"])
            links.new(normal.outputs["Normal"], shader.inputs["Normal"])
            material["afop_normal"] = normal_path

        mask_path = binding.get("m")
        mask_node = None
        mask_separate = None
        if mask_path and mask_path.casefold() in texture_files:
            disk_path = texture_files[mask_path.casefold()]
            image = _load_image(disk_path, mask_path, non_color=True)
            mask = _new_image_node(nodes, image, "Material / Mask (packed)", -440, -340)
            mask_node = mask
            mask_separate = nodes.new("ShaderNodeSeparateColor")
            mask_separate.label = "R: Metal/SSS  G: Roughness  B: Detail  A: Opacity/Coat"
            mask_separate.name = "Snowdrop material channels"
            mask_separate.location = (-195, -340)
            links.new(mask.outputs["Color"], mask_separate.inputs["Color"])
            if is_wildlife_skin:
                # PX_Wildlife_Skin fixes metalness at zero and builds its
                # roughness from Material.R (plus effects Blender does not
                # reproduce). G/B/A are wildlife pattern/bio masks, not the
                # standard Principled material channels.
                mask_separate.label = "R: Roughness  G/B/A: Wildlife masks"
                links.new(mask_separate.outputs["Red"], shader.inputs["Roughness"])
            elif is_medusa_skin:
                # PX_DLC3_Medusa_Skin fixes metalness at zero. Material.R is
                # roughness, while B/A form its procedural bio selector.
                mask_separate.label = "R: Roughness  G: Transition  B/A: Bioluminescence"
                links.new(mask_separate.outputs["Red"], shader.inputs["Roughness"])
            elif is_navi_skin:
                # Navi body, face, and workbench shaders fix metalness at zero.
                # Material R is instead inverted for skin subsurface response,
                # while A drives a narrow clear-coat range. Preserve those
                # semantics rather than making Navi skin metallic.
                subsurface = shader.inputs.get("Subsurface Weight")
                if subsurface is not None:
                    invert_red = _math_node(
                        nodes, "SUBTRACT", "Navi skin: 1 - material R", 55, -340
                    )
                    invert_red.inputs[0].default_value = 1.0
                    links.new(mask_separate.outputs["Red"], invert_red.inputs[1])
                    sss_scale = _math_node(
                        nodes, "MULTIPLY", "Navi skin subsurface scale", 225, -340, 0.05
                    )
                    links.new(invert_red.outputs[0], sss_scale.inputs[0])
                    links.new(sss_scale.outputs[0], subsurface)
                coat = shader.inputs.get("Coat Weight")
                if coat is not None:
                    coat_scale = _math_node(
                        nodes, "MULTIPLY", "Navi coat range", 55, -440, 0.12
                    )
                    coat_bias = _math_node(
                        nodes, "ADD", "Navi coat base", 225, -440, 0.08
                    )
                    links.new(mask.outputs["Alpha"], coat_scale.inputs[0])
                    links.new(coat_scale.outputs[0], coat_bias.inputs[0])
                    links.new(coat_bias.outputs[0], coat)
                if ao_output is not None:
                    # This shader repurposes the normal texture's packed blue
                    # channel as its authored skin-roughness signal. Use it for
                    # both Principled lobes: Blender's otherwise-unconnected
                    # Coat Roughness defaults to 0.03 and makes the masked coat
                    # look like a wet, near-mirror film.
                    links.new(ao_output, shader.inputs["Roughness"])
                    coat_roughness = shader.inputs.get("Coat Roughness")
                    if coat_roughness is not None:
                        links.new(ao_output, coat_roughness)
            else:
                metallic_output = mask_separate.outputs["Red"]
                if shader_name == "px_character_gear_simple.mshader":
                    metal_threshold = _math_node(
                        nodes, "GREATER_THAN", "Gear metallic threshold", 55, -340, 0.6
                    )
                    links.new(metallic_output, metal_threshold.inputs[0])
                    metallic_output = metal_threshold.outputs[0]
                links.new(metallic_output, shader.inputs["Metallic"])
                links.new(mask_separate.outputs["Green"], shader.inputs["Roughness"])
            material["afop_mask"] = mask_path

        alpha_path = binding.get("a")
        alpha_node = None
        if alpha_path and alpha_path.casefold() in texture_files:
            disk_path = texture_files[alpha_path.casefold()]
            image = _load_image(disk_path, alpha_path, non_color=True)
            alpha_node = _new_image_node(
                nodes, image, "Opacity / Alpha", -440, -570
            )
            material["afop_alpha_texture"] = alpha_path

        bio_palette = binding.get("bio_palette")
        if (
            is_wildlife_skin
            and mask_node is not None
            and bio_palette
            and _wildlife_bio_nodes(
                nodes, links, mask_node, ao_output, shader, bio_palette
            )
        ):
            material["afop_bio_palette"] = [
                component
                for color in bio_palette
                for component in color[:3]
            ]
            material["afop_bio_preview"] = "night"
        elif (
            is_medusa_skin
            and mask_node is not None
            and mask_separate is not None
            and bio_palette
            and _medusa_bio_nodes(
                nodes,
                links,
                mask_node,
                mask_separate,
                shader,
                bio_palette,
                bool(binding.get("bio_procedural", False)),
                float(binding.get("bio_strength", 0.5)),
            )
        ):
            material["afop_bio_palette"] = [
                component
                for color in bio_palette
                for component in color[:3]
            ]
            material["afop_bio_preview"] = "night"
            material["afop_bio_procedural"] = bool(
                binding.get("bio_procedural", False)
            )
            material["afop_bio_strength"] = float(
                binding.get("bio_strength", 0.5)
            )

        if roughness_output is not None:
            links.new(roughness_output, shader.inputs["Roughness"])

        if diffuse_node is not None:
            base_ao_output = diffuse_node.outputs["Alpha"] if is_navi_skin else ao_output
            if base_ao_output is not None:
                ao_mix = nodes.new("ShaderNodeMixRGB")
                ao_mix.blend_type = "MULTIPLY"
                ao_mix.inputs[0].default_value = 1.0
                ao_mix.label = "Apply packed normal AO"
                ao_mix.name = ao_mix.label
                ao_mix.location = (55, 180)
                links.new(diffuse_node.outputs["Color"], ao_mix.inputs[1])
                links.new(base_ao_output, ao_mix.inputs[2])
                links.new(ao_mix.outputs["Color"], shader.inputs["Base Color"])
            else:
                links.new(diffuse_node.outputs["Color"], shader.inputs["Base Color"])

        alpha_output = None
        alpha_source = None
        if alpha_node is not None:
            alpha_channel = str(binding.get("a_channel") or "color").casefold()
            if alpha_channel in {"red", "green", "blue"}:
                alpha_separate = nodes.new("ShaderNodeSeparateColor")
                alpha_separate.label = f"Opacity from {alpha_channel.title()}"
                alpha_separate.name = alpha_separate.label
                alpha_separate.location = (-195, -570)
                links.new(alpha_node.outputs["Color"], alpha_separate.inputs["Color"])
                alpha_output = alpha_separate.outputs[alpha_channel.title()]
            elif alpha_channel == "alpha":
                alpha_output = alpha_node.outputs["Alpha"]
            else:
                alpha_output = alpha_node.outputs["Color"]
            alpha_source = "alpha-texture"
        elif diffuse_node is not None and (
            is_wildlife_skin
            or is_dragonfly_wing
            or _diffuse_drives_surface_alpha(shader_name)
        ):
            alpha_output = diffuse_node.outputs["Alpha"]
            alpha_source = "diffuse"
            if mask_node is not None and shader_name == "px_character_gear_simple.mshader":
                alpha_multiply = _math_node(
                    nodes, "MULTIPLY", "Color alpha x material opacity", 55, -430
                )
                links.new(alpha_output, alpha_multiply.inputs[0])
                links.new(mask_node.outputs["Alpha"], alpha_multiply.inputs[1])
                alpha_output = alpha_multiply.outputs[0]
                alpha_source = "diffuse+material"
        if alpha_output is not None:
            links.new(alpha_output, shader.inputs["Alpha"])
            _enable_alpha_blending(material)
            material["afop_alpha_source"] = alpha_source

        obj.data.materials.clear()
        obj.data.materials.append(material)
        assigned += 1
    return assigned

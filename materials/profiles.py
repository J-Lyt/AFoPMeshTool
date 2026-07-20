"""Blender material-profile construction for Snowdrop shaders."""

from __future__ import annotations

import os

import bpy

from .. import mgraph
from .nodes import (
    _aux_image_node,
    _clamp_node,
    _color_parameter,
    _enable_alpha_blending,
    _five_color_ramp,
    _float_parameter,
    _math_node,
    _medusa_bio_nodes,
    _new_image_node,
    _packed_detail_layer,
    _packed_detail_normal_group,
    _packed_normal_nodes,
    _pattern_component,
    _separate_node,
    _tiling_parameter,
    _vector2_parameter,
    _wildlife_bio_nodes,
)
from .registry import profile_traits
from .textures import _load_image


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
        traits = profile_traits(shader_name)
        is_navi_skin = "navi_skin" in traits
        is_navi_detail = "navi_detail" in traits
        is_wildlife_skin = "wildlife_skin" in traits
        is_medusa_skin = "medusa_skin" in traits
        is_dragonfly_wing = "dragonfly_wing" in traits
        is_wildlife_eye = "wildlife_eye" in traits
        is_eye_parallax = "eye_parallax" in traits
        is_eye_shell = "eye_shell" in traits
        is_human_skin = "human_skin" in traits
        is_hair = "hair" in traits
        is_natural_rock = "natural_rock" in traits
        is_terrain_runtime = "terrain_runtime" in traits
        is_moss_patch = "moss_patch" in traits
        is_wildlife_gear = "wildlife_gear" in traits
        is_constants = "constants" in traits
        is_emissive_color = "emissive_color" in traits
        is_basic_emissive = "basic_emissive" in traits
        is_rustymetal = "rusty_metal" in traits
        is_vegetation = "vegetation" in traits
        is_moss_card = "moss_card" in traits
        is_basic_blend = "basic_blend" in traits
        is_navi_face = "navi_face" in traits

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
        parameters = binding.get("parameters", {})
        authored_roughness = parameters.get("myRoughness")
        if isinstance(authored_roughness, (int, float)):
            shader.inputs["Roughness"].default_value = max(
                0.0, min(1.0, float(authored_roughness))
            )
            material["afop_roughness"] = float(authored_roughness)

        surface_uv_output = None
        base_color_override = None
        profile_alpha_output = None
        profile_alpha_source = None
        if is_basic_emissive or is_rustymetal or is_moss_patch:
            uv_tiling = _vector2_parameter(
                parameters.get(
                    "myUVTiling"
                    if is_basic_emissive
                    else (
                        "myTiling" if is_rustymetal else "myTilingX"
                    )
                ),
                (1.0, 1.0),
            )
            if is_moss_patch:
                uv_tiling = (
                    _float_parameter(parameters.get("myTilingX"), 1.0),
                    _float_parameter(parameters.get("myTilingY"), 1.0),
                )
            uv_offset = _vector2_parameter(
                parameters.get("myUVOffset"), (0.0, 0.0)
            )
            texture_coordinates = nodes.new("ShaderNodeTexCoord")
            texture_coordinates.label = (
                "Basic Emissive UV"
                if is_basic_emissive
                else ("Rusty Metal UV" if is_rustymetal else "Moss Patch UV")
            )
            texture_coordinates.name = texture_coordinates.label
            texture_coordinates.location = (-1050, 410)
            uv_scale = nodes.new("ShaderNodeVectorMath")
            uv_scale.operation = "MULTIPLY"
            uv_scale.label = (
                "Basic Emissive UV tiling"
                if is_basic_emissive
                else ("Rusty Metal UV tiling" if is_rustymetal else "Moss Patch UV tiling")
            )
            uv_scale.name = uv_scale.label
            uv_scale.location = (-850, 410)
            uv_scale.inputs[1].default_value = (*uv_tiling, 1.0)
            uv_add = nodes.new("ShaderNodeVectorMath")
            uv_add.operation = "ADD"
            uv_add.label = (
                "Basic Emissive UV offset"
                if is_basic_emissive
                else ("Rusty Metal UV offset" if is_rustymetal else "Moss Patch UV offset")
            )
            uv_add.name = uv_add.label
            uv_add.location = (-650, 410)
            uv_add.inputs[1].default_value = (*uv_offset, 0.0)
            links.new(texture_coordinates.outputs["UV"], uv_scale.inputs[0])
            links.new(uv_scale.outputs["Vector"], uv_add.inputs[0])
            surface_uv_output = uv_add.outputs["Vector"]
            material["afop_uv_tiling"] = list(uv_tiling)
            material["afop_uv_offset"] = list(uv_offset)

        diffuse_path = binding.get("d")
        diffuse_node = None
        if diffuse_path and diffuse_path.casefold() in texture_files:
            disk_path = texture_files[diffuse_path.casefold()]
            image = _load_image(disk_path, diffuse_path, non_color=False)
            node = _new_image_node(nodes, image, "Diffuse / Albedo", -440, 180)
            if surface_uv_output is not None:
                links.new(surface_uv_output, node.inputs["Vector"])
            diffuse_node = node
            material["afop_diffuse"] = diffuse_path

        normal_path = binding.get("n")
        detail_path = binding.get("aux", {}).get("DetailNormal")
        use_detail_normal = bool(
            (is_navi_detail or is_basic_emissive)
            and detail_path
            and detail_path.casefold() in texture_files
            and binding.get("m")
            and binding["m"].casefold() in texture_files
        )
        ao_output = None
        roughness_output = None
        normal_texture_node = None
        normal = None
        if normal_path and normal_path.casefold() in texture_files:
            disk_path = texture_files[normal_path.casefold()]
            image = _load_image(disk_path, normal_path, non_color=True)
            node = _new_image_node(nodes, image, "Normal (packed)", -440, -80)
            normal_texture_node = node
            if surface_uv_output is not None:
                links.new(surface_uv_output, node.inputs["Vector"])
            if use_detail_normal:
                base_channels = nodes.new("ShaderNodeSeparateColor")
                base_channels.label = "Base packed normal channels"
                base_channels.name = base_channels.label
                base_channels.location = (-195, -80)
                links.new(node.outputs["Color"], base_channels.inputs["Color"])
                normal_color = None
                ao_output = base_channels.outputs["Blue"]
            elif is_wildlife_skin:
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
            normal_strength = parameters.get(
                "myBaseNormalStrength",
                parameters.get("myNormalStrength", 1.0),
            )
            if not isinstance(normal_strength, (int, float)):
                normal_strength = 1.0
            normal_strength = max(0.0, float(normal_strength))
            normal.inputs["Strength"].default_value = normal_strength
            if normal_color is not None:
                links.new(normal_color, normal.inputs["Color"])
            links.new(normal.outputs["Normal"], shader.inputs["Normal"])
            material["afop_normal"] = normal_path
            material["afop_normal_strength"] = normal_strength

        mask_path = binding.get("m")
        mask_node = None
        mask_separate = None
        if mask_path and mask_path.casefold() in texture_files:
            disk_path = texture_files[mask_path.casefold()]
            image = _load_image(disk_path, mask_path, non_color=True)
            mask = _new_image_node(nodes, image, "Material / Mask (packed)", -440, -340)
            mask_node = mask
            if surface_uv_output is not None:
                links.new(surface_uv_output, mask.inputs["Vector"])
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

        if (
            use_detail_normal
            and normal_texture_node is not None
            and normal is not None
            and mask_separate is not None
            and detail_path
            and detail_path.casefold() in texture_files
        ):
            disk_path = texture_files[detail_path.casefold()]
            image = _load_image(disk_path, detail_path, non_color=True)
            detail_node = _new_image_node(
                nodes, image, "Navi Detail Normal (packed)", -440, -720
            )
            texture_coordinates = nodes.new("ShaderNodeTexCoord")
            texture_coordinates.label = "Detail normal UV"
            texture_coordinates.name = texture_coordinates.label
            texture_coordinates.location = (-850, -720)
            tiling = parameters.get("myDetailTiling", (1.0, 1.0))
            if not (
                isinstance(tiling, (list, tuple))
                and len(tiling) >= 2
                and all(isinstance(value, (int, float)) for value in tiling[:2])
            ):
                tiling = (1.0, 1.0)
            uv_scale = nodes.new("ShaderNodeVectorMath")
            uv_scale.operation = "MULTIPLY"
            uv_scale.label = "Detail normal UV tiling"
            uv_scale.name = uv_scale.label
            uv_scale.location = (-650, -720)
            uv_scale.inputs[1].default_value = (
                float(tiling[0]), float(tiling[1]), 1.0
            )
            links.new(texture_coordinates.outputs["UV"], uv_scale.inputs[0])
            links.new(uv_scale.outputs["Vector"], detail_node.inputs["Vector"])

            detail_strength = (
                1.0
                if is_basic_emissive
                else parameters.get("myDetailStrength", 0.5)
            )
            if not isinstance(detail_strength, (int, float)):
                detail_strength = 0.5
            detail_strength = max(0.0, float(detail_strength))
            detail_group = nodes.new("ShaderNodeGroup")
            detail_group.node_tree = _packed_detail_normal_group()
            detail_group.label = (
                "Basic Emissive detail normal (RNM)"
                if is_basic_emissive
                else "Navi detail normal (RNM)"
            )
            detail_group.name = detail_group.label
            detail_group.location = (1505, -300)
            detail_group.inputs["Strength"].default_value = detail_strength
            links.new(
                normal_texture_node.outputs["Color"],
                detail_group.inputs["Base Color"],
            )
            links.new(
                normal_texture_node.outputs["Alpha"],
                detail_group.inputs["Base Alpha"],
            )
            links.new(detail_node.outputs["Color"], detail_group.inputs["Detail Color"])
            links.new(detail_node.outputs["Alpha"], detail_group.inputs["Detail Alpha"])
            links.new(mask_separate.outputs["Blue"], detail_group.inputs["Mask"])
            for link in list(normal.inputs["Color"].links):
                links.remove(link)
            links.new(detail_group.outputs["Normal Color"], normal.inputs["Color"])
            ao_output = detail_group.outputs["AO"]
            material["afop_detail_normal"] = detail_path
            material["afop_detail_normal_strength"] = detail_strength
            material["afop_detail_normal_tiling"] = [
                float(tiling[0]), float(tiling[1])
            ]

        if (
            is_rustymetal
            and normal_texture_node is not None
            and normal is not None
            and mask_separate is not None
        ):
            rust_detail_path = binding.get("aux", {}).get("DetailNormal")
            rust_detail = _aux_image_node(
                nodes, links, texture_files, rust_detail_path,
                "Rusty Metal Detail Normal (packed)", -440, -720,
                tiling=parameters.get("myDetailTiling", (1.0, 1.0)),
            )
            if rust_detail is not None:
                detail = _packed_detail_layer(
                    nodes, links,
                    normal_texture_node.outputs["Color"],
                    normal_texture_node.outputs["Alpha"],
                    rust_detail, rust_detail.outputs["Alpha"],
                    mask_separate.outputs["Blue"],
                    _float_parameter(parameters.get("myDetailStrength"), 1.0),
                    "Rusty Metal detail normal (RNM)", 1450, -300,
                )
                for link in list(normal.inputs["Color"].links):
                    links.remove(link)
                links.new(detail.outputs["Normal Color"], normal.inputs["Color"])
                material["afop_detail_normal"] = rust_detail_path
                material["afop_detail_normal_strength"] = _float_parameter(
                    parameters.get("myDetailStrength"), 1.0
                )
            rust_mask_path = binding.get("aux", {}).get("RustyMetalMask")
            rust_mask = _aux_image_node(
                nodes, links, texture_files, rust_mask_path,
                "Rust / scratch / dirt mask (engine procedural)", -440, -930,
                tiling=parameters.get("myRustyMetalTiling", (1.0, 1.0)),
                uv_map="UVMap_1",
            )
            if rust_mask is not None:
                material["afop_rusty_metal_mask"] = rust_mask_path
            if diffuse_node is not None:
                overlay_color = nodes.new("ShaderNodeRGB")
                overlay_color.label = "Authored rusty-metal color overlay"
                overlay_color.name = overlay_color.label
                overlay_color.location = (-195, 300)
                overlay_color.outputs[0].default_value = (
                    *_color_parameter(parameters.get("myColorOverlay"), (0.5,) * 3),
                    1.0,
                )
                overlay = nodes.new("ShaderNodeMixRGB")
                overlay.blend_type = "OVERLAY"
                overlay.inputs[0].default_value = 1.0
                overlay.label = "Snowdrop OverlayNoMask"
                overlay.name = overlay.label
                overlay.location = (55, 300)
                links.new(diffuse_node.outputs["Color"], overlay.inputs[1])
                links.new(overlay_color.outputs[0], overlay.inputs[2])
                base_color_override = overlay.outputs["Color"]
            material["afop_shader_profile"] = "rusty_metal_static"
            material["afop_profile_limit"] = (
                "Rust/scratch/dirt weather functions require engine scan masks"
            )

        if (
            is_navi_face
            and normal_texture_node is not None
            and normal is not None
            and mask_separate is not None
        ):
            auxiliary = binding.get("aux", {})
            face_detail_path = auxiliary.get("DetailNormal")
            scar_path = auxiliary.get("SarentuScarNormal") or auxiliary.get("Scar")
            scar = None
            scar_channels = None
            if scar_path:
                if shader_name == "px_character_navi_face.mshader":
                    scar_offset = parameters.get("mySarentuScarOffset", (0.0, 0.0))
                    scar_scale = parameters.get("mySarentuScarScale", (0.0, 0.0))
                    scar_rotation = _float_parameter(
                        parameters.get("mySarentuScarRotation"), 0.0
                    )
                else:
                    scar_offset = parameters.get("myScarUVOffset", (0.0, 0.0))
                    scar_scale = parameters.get("myScarScale", (0.0, 0.0))
                    scar_rotation = 0.0
                scar = _aux_image_node(
                    nodes, links, texture_files, scar_path,
                    "Navi Scar Normal / Mask (packed)", -440, -1010,
                    tiling=scar_scale, offset=scar_offset, rotation=scar_rotation,
                )
                if scar is not None:
                    scar_channels = _separate_node(
                        nodes, links, scar.outputs["Color"],
                        "Navi scar packed channels", -195, -1010,
                    )
            face_detail = _aux_image_node(
                nodes, links, texture_files, face_detail_path,
                "Navi Face Detail Normal (packed)", -440, -760,
                tiling=parameters.get("myDetailTiling", (1.0, 1.0)),
            )
            current_color = normal_texture_node.outputs["Color"]
            current_x = normal_texture_node.outputs["Alpha"]
            if face_detail is not None:
                detail_mask = mask_separate.outputs["Blue"]
                if (
                    shader_name == "px_character_navi_face.mshader"
                    and scar_channels is not None
                ):
                    invert_scar = _math_node(
                        nodes, "SUBTRACT", "Scar-free detail mask", 55, -760
                    )
                    invert_scar.inputs[0].default_value = 1.0
                    links.new(scar_channels.outputs["Blue"], invert_scar.inputs[1])
                    detail_multiply = _math_node(
                        nodes, "MULTIPLY", "Face detail x scar mask", 225, -760
                    )
                    links.new(detail_mask, detail_multiply.inputs[0])
                    links.new(invert_scar.outputs[0], detail_multiply.inputs[1])
                    detail_mask = detail_multiply.outputs[0]
                detail = _packed_detail_layer(
                    nodes, links, current_color, current_x,
                    face_detail, face_detail.outputs["Alpha"], detail_mask,
                    _float_parameter(parameters.get("myDetailStrength"), 0.5),
                    "Navi face detail normal (RNM)", 1320, -300,
                )
                current_color = detail.outputs["Normal Color"]
                current_x = _separate_node(
                    nodes, links, current_color,
                    "Navi face combined normal channels", 1500, -430,
                ).outputs["Red"]
                material["afop_detail_normal"] = face_detail_path
            if scar is not None and scar_channels is not None:
                if shader_name == "px_character_navi_face.mshader":
                    scar_mask = scar_channels.outputs["Blue"]
                else:
                    scar_value = nodes.new("ShaderNodeValue")
                    scar_value.label = "Full scar normal contribution"
                    scar_value.outputs[0].default_value = 1.0
                    scar_mask = scar_value.outputs[0]
                scar_layer = _packed_detail_layer(
                    nodes, links, current_color, current_x,
                    scar, scar.outputs["Alpha"], scar_mask, 1.0,
                    "Navi scar normal (RNM)", 1660, -300,
                )
                current_color = scar_layer.outputs["Normal Color"]
                material["afop_scar"] = scar_path
                if (
                    shader_name != "px_character_navi_face.mshader"
                    and diffuse_node is not None
                ):
                    scar_color = nodes.new("ShaderNodeRGB")
                    scar_color.label = "Authored Navi scar color"
                    scar_color.outputs[0].default_value = (
                        *_color_parameter(parameters.get("myScarColor")), 1.0
                    )
                    scar_mix = nodes.new("ShaderNodeMixRGB")
                    scar_mix.label = "Navi scar color mask"
                    scar_mix.name = scar_mix.label
                    links.new(scar_channels.outputs["Blue"], scar_mix.inputs[0])
                    links.new(diffuse_node.outputs["Color"], scar_mix.inputs[1])
                    links.new(scar_color.outputs[0], scar_mix.inputs[2])
                    base_color_override = scar_mix.outputs["Color"]
            for link in list(normal.inputs["Color"].links):
                links.remove(link)
            links.new(current_color, normal.inputs["Color"])
            material["afop_shader_profile"] = "navi_face_scar_detail"

        if (
            is_wildlife_skin
            and normal_texture_node is not None
            and normal is not None
            and mask_separate is not None
            and diffuse_node is not None
        ):
            auxiliary = binding.get("aux", {})
            detail_mask_path = auxiliary.get("DetailNormalMask")
            detail_mask_node = _aux_image_node(
                nodes, links, texture_files, detail_mask_path,
                "Wildlife Detail Normal Mask", -520, -720,
            )
            detail_mask_channels = (
                _separate_node(
                    nodes, links, detail_mask_node.outputs["Color"],
                    "Wildlife detail masks (R/G/B)", -280, -720,
                ) if detail_mask_node is not None else None
            )
            current_color = normal_texture_node.outputs["Color"]
            current_x = _separate_node(
                nodes, links, current_color,
                "Wildlife base packed normal channels", -195, -80,
            ).outputs["Red"]
            detail_records = []
            for index, (path_key, tiling_key, channel) in enumerate((
                ("DetailNormal1", "myDN1Tilling", "Red"),
                ("DetailNormal2", "myDN2Tilling", "Green"),
                ("DetailNormal3", "myDN3Tiling", "Blue"),
            )):
                path = auxiliary.get(path_key)
                tiling_value = _float_parameter(parameters.get(tiling_key), 0.0)
                detail_node = _aux_image_node(
                    nodes, links, texture_files, path,
                    f"Wildlife {path_key} (packed)", -520, -940 - index * 210,
                    tiling=(tiling_value, tiling_value),
                )
                if detail_node is None or detail_mask_channels is None:
                    continue
                detail_channels = _separate_node(
                    nodes, links, detail_node.outputs["Color"],
                    f"{path_key} packed channels", -280, -940 - index * 210,
                )
                layer = _packed_detail_layer(
                    nodes, links, current_color, current_x,
                    detail_node, detail_channels.outputs["Red"],
                    detail_mask_channels.outputs[channel], 1.0,
                    f"Wildlife {path_key} (RNM)", 1120 + index * 210,
                    -300 - index * 140,
                )
                current_color = layer.outputs["Normal Color"]
                current_x = _separate_node(
                    nodes, links, current_color,
                    f"Wildlife combined normal {index + 1}",
                    1260 + index * 210, -480 - index * 140,
                ).outputs["Red"]
                detail_records.append((detail_node, detail_mask_channels.outputs[channel]))
                material[f"afop_{path_key.casefold()}"] = path
                material[f"afop_{tiling_key.casefold()}"] = tiling_value
            if detail_records:
                for link in list(normal.inputs["Color"].links):
                    links.remove(link)
                links.new(current_color, normal.inputs["Color"])

                roughness_value = mask_separate.outputs["Red"]
                detail_sum = None
                for index, (detail_node, detail_mask) in enumerate(detail_records):
                    detail_channels = _separate_node(
                        nodes, links, detail_node.outputs["Color"],
                        f"Wildlife detail roughness {index + 1}",
                        250 + index * 150, -1260,
                    )
                    delta = _math_node(
                        nodes, "SUBTRACT", f"Detail roughness {index + 1} - 0.5",
                        420 + index * 150, -1260, 0.5,
                    )
                    links.new(detail_channels.outputs["Blue"], delta.inputs[0])
                    weighted = _math_node(
                        nodes, "MULTIPLY", f"Masked detail roughness {index + 1}",
                        590 + index * 150, -1260,
                    )
                    links.new(delta.outputs[0], weighted.inputs[0])
                    links.new(detail_mask, weighted.inputs[1])
                    if detail_sum is None:
                        detail_sum = weighted.outputs[0]
                    else:
                        add = _math_node(
                            nodes, "ADD", "Sum wildlife detail roughness",
                            760 + index * 150, -1260,
                        )
                        links.new(detail_sum, add.inputs[0])
                        links.new(weighted.outputs[0], add.inputs[1])
                        detail_sum = add.outputs[0]
                averaged = _math_node(
                    nodes, "MULTIPLY", "Average wildlife detail roughness",
                    1100, -1260, 0.333,
                )
                links.new(detail_sum, averaged.inputs[0])
                rough_add = _math_node(
                    nodes, "ADD", "Wildlife base + detail roughness", 1270, -1260
                )
                links.new(roughness_value, rough_add.inputs[0])
                links.new(averaged.outputs[0], rough_add.inputs[1])
                contrast, brightness = _vector2_parameter(
                    parameters.get("myRoughnessContrastBrightness"), (1.0, 1.0)
                )
                bright = _math_node(
                    nodes, "MULTIPLY", "Wildlife roughness brightness",
                    1440, -1260, brightness,
                )
                links.new(rough_add.outputs[0], bright.inputs[0])
                center = _math_node(
                    nodes, "SUBTRACT", "Center wildlife roughness",
                    1610, -1260, 0.5,
                )
                links.new(bright.outputs[0], center.inputs[0])
                contrast_node = _math_node(
                    nodes, "MULTIPLY", "Wildlife roughness contrast",
                    1780, -1260, contrast,
                )
                links.new(center.outputs[0], contrast_node.inputs[0])
                restore = _math_node(
                    nodes, "ADD", "Restore wildlife roughness midpoint",
                    1950, -1260, 0.5,
                )
                links.new(contrast_node.outputs[0], restore.inputs[0])
                clamp_roughness = _clamp_node(
                    nodes, "Clamp wildlife roughness", 2120, -1260
                )
                links.new(restore.outputs[0], clamp_roughness.inputs["Value"])
                for link in list(shader.inputs["Roughness"].links):
                    links.remove(link)
                links.new(clamp_roughness.outputs[0], shader.inputs["Roughness"])

            pattern_path = auxiliary.get("PatternCoat")
            pattern = _aux_image_node(
                nodes, links, texture_files, pattern_path,
                "Wildlife Pattern Coat", -520, 520, non_color=True,
            )
            has_authored_coat = any(
                f"myCoat{coat}Color{index}" in parameters
                for coat in (1, 2) for index in range(1, 6)
            )
            if pattern is not None and has_authored_coat:
                pattern_channels = _separate_node(
                    nodes, links, pattern.outputs["Color"],
                    "Pattern R/G masks, B/A coat ramps", -280, 520,
                )
                coat1 = _five_color_ramp(
                    nodes, links, pattern_channels.outputs["Blue"],
                    tuple(_color_parameter(parameters.get(f"myCoat1Color{i}")) for i in range(1, 6)),
                    "Wildlife Coat 1 palette", 0, 600,
                )
                coat2 = _five_color_ramp(
                    nodes, links, pattern.outputs["Alpha"],
                    tuple(_color_parameter(parameters.get(f"myCoat2Color{i}")) for i in range(1, 6)),
                    "Wildlife Coat 2 palette", 0, 380,
                )
                pattern1 = _pattern_component(
                    nodes, links, pattern_channels.outputs["Red"],
                    _float_parameter(parameters.get("myPattern1LevelControl"), 0.0),
                    _float_parameter(parameters.get("myInvertPattern1"), 0.0),
                    "Wildlife pattern 1", 230, 620,
                )
                pattern2 = _pattern_component(
                    nodes, links, pattern_channels.outputs["Green"],
                    _float_parameter(parameters.get("myPattern2LevelControl"), 0.0),
                    _float_parameter(parameters.get("myInvertPattern2"), -0.3),
                    "Wildlife pattern 2", 230, 400,
                )
                pattern_add = _math_node(
                    nodes, "ADD", "Combine wildlife patterns", 760, 520
                )
                links.new(pattern1, pattern_add.inputs[0])
                links.new(pattern2, pattern_add.inputs[1])
                pattern_clamp = _clamp_node(
                    nodes, "Clamp wildlife pattern", 930, 520
                )
                links.new(pattern_add.outputs[0], pattern_clamp.inputs["Value"])
                coat_mix = nodes.new("ShaderNodeMixRGB")
                coat_mix.label = "Select wildlife coat"
                coat_mix.name = coat_mix.label
                coat_mix.location = (1100, 520)
                links.new(pattern_clamp.outputs[0], coat_mix.inputs[0])
                links.new(coat1, coat_mix.inputs[1])
                links.new(coat2, coat_mix.inputs[2])
                coat_sqrt = nodes.new("ShaderNodeGamma")
                coat_sqrt.label = "Snowdrop coat sqrt"
                coat_sqrt.inputs["Gamma"].default_value = 0.5
                coat_sqrt.location = (1280, 520)
                diffuse_sqrt = nodes.new("ShaderNodeGamma")
                diffuse_sqrt.label = "Snowdrop diffuse sqrt"
                diffuse_sqrt.inputs["Gamma"].default_value = 0.5
                diffuse_sqrt.location = (1280, 700)
                links.new(coat_mix.outputs["Color"], coat_sqrt.inputs["Color"])
                links.new(diffuse_node.outputs["Color"], diffuse_sqrt.inputs["Color"])
                coat_overlay = nodes.new("ShaderNodeMixRGB")
                coat_overlay.blend_type = "OVERLAY"
                coat_overlay.label = "Wildlife coat overlay"
                coat_overlay.name = coat_overlay.label
                coat_overlay.location = (1470, 600)
                links.new(mask_separate.outputs["Blue"], coat_overlay.inputs[0])
                links.new(diffuse_sqrt.outputs["Color"], coat_overlay.inputs[1])
                links.new(coat_sqrt.outputs["Color"], coat_overlay.inputs[2])
                coat_square = nodes.new("ShaderNodeGamma")
                coat_square.label = "Snowdrop coat square"
                coat_square.inputs["Gamma"].default_value = 2.0
                coat_square.location = (1650, 600)
                links.new(coat_overlay.outputs["Color"], coat_square.inputs["Color"])
                base_color_override = coat_square.outputs["Color"]
                material["afop_pattern_coat"] = pattern_path
            elif pattern is not None:
                material["afop_pattern_coat"] = pattern_path
                material["afop_pattern_coat_status"] = (
                    "Texture imported; coat constants were not resolved, so diffuse was retained"
                )
            material["afop_shader_profile"] = "wildlife_skin_static_clean"
            material["afop_profile_limit"] = (
                "Static clean/night preview; weather, wounds and terrain are engine-driven"
            )

        if (
            is_medusa_skin
            and normal_texture_node is not None
            and normal is not None
            and mask_separate is not None
            and diffuse_node is not None
        ):
            auxiliary = binding.get("aux", {})
            detail_mask_path = auxiliary.get("DetailMask")
            medusa_detail_path = auxiliary.get("DetailNormal")
            blood_color_path = auxiliary.get("BloodveinColor")
            blood_normal_path = auxiliary.get("BloodveinNormal")
            inner_alpha_path = auxiliary.get("InnerAlpha")
            detail_mask = _aux_image_node(
                nodes, links, texture_files, detail_mask_path,
                "Medusa Detail Mask", -520, -720,
            )
            detail_normal = _aux_image_node(
                nodes, links, texture_files, medusa_detail_path,
                "Medusa Detail Normal (R/G packed)", -520, -930,
                tiling=(
                    _float_parameter(parameters.get("myDetailNormal1Tilling"), 1.0),
                ) * 2,
            )
            blood_tiling = _float_parameter(
                parameters.get("myBloodveinTilling"), 1.0
            )
            blood_color = _aux_image_node(
                nodes, links, texture_files, blood_color_path,
                "Medusa Bloodvein Color / Mask", -520, -1140,
                non_color=False, tiling=(blood_tiling, blood_tiling),
            )
            blood_normal = _aux_image_node(
                nodes, links, texture_files, blood_normal_path,
                "Medusa Bloodvein Normal (packed)", -520, -1350,
                tiling=(blood_tiling, blood_tiling),
            )
            inner_alpha = _aux_image_node(
                nodes, links, texture_files, inner_alpha_path,
                "Medusa Inner Alpha / Height", -520, -1560,
            )
            current_color = normal_texture_node.outputs["Color"]
            current_x = normal_texture_node.outputs["Alpha"]
            detail_mask_channels = None
            if detail_mask is not None:
                detail_mask_channels = _separate_node(
                    nodes, links, detail_mask.outputs["Color"],
                    "Medusa detail mask channels", -280, -720,
                )
            if detail_normal is not None and detail_mask_channels is not None:
                detail_channels = _separate_node(
                    nodes, links, detail_normal.outputs["Color"],
                    "Medusa detail packed channels", -280, -930,
                )
                detail_layer = _packed_detail_layer(
                    nodes, links, current_color, current_x,
                    detail_normal, detail_channels.outputs["Red"],
                    detail_mask_channels.outputs["Red"], 1.0,
                    "Medusa detail normal (RNM)", 1280, -300,
                )
                current_color = detail_layer.outputs["Normal Color"]
                current_x = _separate_node(
                    nodes, links, current_color,
                    "Medusa combined detail channels", 1460, -430,
                ).outputs["Red"]
                roughness_mix = nodes.new("ShaderNodeMix")
                roughness_mix.data_type = "FLOAT"
                roughness_mix.label = "Medusa base/detail roughness"
                roughness_mix.name = roughness_mix.label
                roughness_mix.location = (1480, -900)
                links.new(detail_mask_channels.outputs["Red"], roughness_mix.inputs[0])
                links.new(mask_separate.outputs["Red"], roughness_mix.inputs[2])
                links.new(detail_channels.outputs["Blue"], roughness_mix.inputs[3])
                for link in list(shader.inputs["Roughness"].links):
                    links.remove(link)
                links.new(roughness_mix.outputs["Result"], shader.inputs["Roughness"])
                material["afop_detail_normal"] = medusa_detail_path
                material["afop_detail_mask"] = detail_mask_path
            if blood_normal is not None and blood_color is not None:
                blood_layer = _packed_detail_layer(
                    nodes, links, current_color, current_x,
                    blood_normal, blood_normal.outputs["Alpha"],
                    blood_color.outputs["Alpha"], 1.0,
                    "Medusa bloodvein normal (RNM)", 1660, -300,
                )
                current_color = blood_layer.outputs["Normal Color"]
                blood_mix = nodes.new("ShaderNodeMixRGB")
                blood_mix.label = "Medusa bloodvein color"
                blood_mix.name = blood_mix.label
                blood_mix.location = (300, 350)
                links.new(blood_color.outputs["Alpha"], blood_mix.inputs[0])
                links.new(diffuse_node.outputs["Color"], blood_mix.inputs[1])
                links.new(blood_color.outputs["Color"], blood_mix.inputs[2])
                base_color_override = blood_mix.outputs["Color"]
                material["afop_bloodvein_color"] = blood_color_path
                material["afop_bloodvein_normal"] = blood_normal_path
            for link in list(normal.inputs["Color"].links):
                links.remove(link)
            links.new(current_color, normal.inputs["Color"])
            if inner_alpha is not None:
                inner_channels = _separate_node(
                    nodes, links, inner_alpha.outputs["Color"],
                    "Medusa inner-alpha channels", -280, -1560,
                )
                invert_inner = _math_node(
                    nodes, "SUBTRACT", "Medusa inner transmission mask",
                    0, -1560,
                )
                invert_inner.inputs[0].default_value = 1.0
                links.new(inner_channels.outputs["Red"], invert_inner.inputs[1])
                transmission_scale = _math_node(
                    nodes, "MULTIPLY", "Medusa transmission preview",
                    180, -1560, 0.2,
                )
                links.new(invert_inner.outputs[0], transmission_scale.inputs[0])
                transmission = shader.inputs.get("Transmission Weight")
                if transmission is not None:
                    links.new(transmission_scale.outputs[0], transmission)
                material["afop_inner_alpha"] = inner_alpha_path
            subsurface = shader.inputs.get("Subsurface Weight")
            if subsurface is not None:
                subsurface.default_value = 0.05
            coat = shader.inputs.get("Coat Weight")
            if coat is not None:
                coat.default_value = 0.2
            coat_roughness = shader.inputs.get("Coat Roughness")
            if coat_roughness is not None:
                coat_roughness.default_value = 0.7
            material["afop_shader_profile"] = "medusa_skin_static"
            material["afop_profile_limit"] = (
                "Inner parallax/transmission and day-night response are viewport previews"
            )

        if is_eye_parallax:
            height_path = binding.get("aux", {}).get("Height")
            height = _aux_image_node(
                nodes, links, texture_files, height_path,
                "Wildlife Eye Height (parallax source)", -520, -720,
            )
            if height is not None:
                height_channels = _separate_node(
                    nodes, links, height.outputs["Color"],
                    "Wildlife eye height channels", -280, -720,
                )
                bump = nodes.new("ShaderNodeBump")
                bump.label = "Eye parallax height preview"
                bump.name = bump.label
                bump.location = (1660, -180)
                default_height_scale = 20.0 if is_wildlife_eye else 10.0
                bump.inputs["Strength"].default_value = min(
                    1.0,
                    max(
                        0.0,
                        _float_parameter(
                            parameters.get("myHeightScale"), default_height_scale
                        ) * 0.01,
                    ),
                )
                bump.inputs["Distance"].default_value = 0.1
                links.new(height_channels.outputs["Red"], bump.inputs["Height"])
                if normal is not None:
                    links.new(normal.outputs["Normal"], bump.inputs["Normal"])
                for link in list(shader.inputs["Normal"].links):
                    links.remove(link)
                links.new(bump.outputs["Normal"], shader.inputs["Normal"])
                material["afop_height"] = height_path
                material["afop_height_scale"] = _float_parameter(
                    parameters.get("myHeightScale"), default_height_scale
                )
            if is_wildlife_eye:
                shader.inputs["Roughness"].default_value = 0.01
                shader.inputs["Metallic"].default_value = 0.1
                for link in list(shader.inputs["Roughness"].links):
                    links.remove(link)
                for link in list(shader.inputs["Metallic"].links):
                    links.remove(link)
            material["afop_shader_profile"] = (
                "wildlife_eye_parallax_preview"
                if is_wildlife_eye else "eye_parallax_preview"
            )
            material["afop_profile_limit"] = (
                "Snowdrop iterative view-dependent parallax is represented as bump"
            )

        if is_eye_shell:
            shell_color = nodes.new("ShaderNodeRGB")
            shell_color.label = "Authored eye-shell color"
            shell_color.name = shell_color.label
            shell_color.location = (1200, 260)
            shell_color.outputs[0].default_value = (
                *_color_parameter(parameters.get("myColor")), 1.0
            )
            base_color_override = shell_color.outputs[0]
            shell_roughness_output = None
            if obj.data.attributes.get("Color_0") is not None:
                vertex_color = nodes.new("ShaderNodeVertexColor")
                vertex_color.layer_name = "Color_0"
                vertex_color.label = "Eye-shell vertex controls"
                vertex_color.name = vertex_color.label
                vertex_color.location = (760, 420)
                vertex_channels = _separate_node(
                    nodes, links, vertex_color.outputs["Color"],
                    "Eye-shell vertex channels", 960, 420,
                )
                shadow_color = nodes.new("ShaderNodeRGB")
                shadow_color.label = "Authored eye-shell shadow color"
                shadow_color.name = shadow_color.label
                shadow_color.location = (1200, 420)
                shadow_color.outputs[0].default_value = (
                    *_color_parameter(parameters.get("myShadowColor")), 1.0
                )
                shell_shadow = nodes.new("ShaderNodeMixRGB")
                shell_shadow.label = "Eye-shell vertex shadow"
                shell_shadow.name = shell_shadow.label
                shell_shadow.location = (1420, 300)
                links.new(vertex_channels.outputs["Green"], shell_shadow.inputs[0])
                links.new(shell_color.outputs[0], shell_shadow.inputs[1])
                links.new(shadow_color.outputs[0], shell_shadow.inputs[2])
                base_color_override = shell_shadow.outputs["Color"]

                shell_roughness = nodes.new("ShaderNodeMix")
                shell_roughness.data_type = "FLOAT"
                shell_roughness.label = "Eye-shell vertex roughness"
                shell_roughness.name = shell_roughness.label
                shell_roughness.location = (1420, 100)
                links.new(vertex_channels.outputs["Green"], shell_roughness.inputs[0])
                shell_roughness.inputs[2].default_value = max(
                    0.0,
                    min(
                        1.0,
                        _float_parameter(parameters.get("myRoughness"), 0.2),
                    ),
                )
                shell_roughness.inputs[3].default_value = 1.0
                shell_roughness_output = shell_roughness.outputs["Result"]

                shell_alpha = _math_node(
                    nodes, "MULTIPLY", "Eye-shell vertex alpha",
                    1420, -80,
                    _float_parameter(parameters.get("myAlphaMultiplier"), 1.0),
                )
                links.new(vertex_channels.outputs["Red"], shell_alpha.inputs[0])
                profile_alpha_output = shell_alpha.outputs[0]
                profile_alpha_source = "vertex-color-red"
            if diffuse_node is None:
                links.new(base_color_override, shader.inputs["Base Color"])
            shell_normal_path = binding.get("aux", {}).get("NormalTexture")
            shell_normal = _aux_image_node(
                nodes, links, texture_files, shell_normal_path,
                "Eye Shell Normal (packed)", -520, -720,
                tiling=parameters.get("myNormalTiling", (1.0, 1.0)),
            )
            if shell_normal is not None:
                shell_normal_color, _shell_ao = _packed_normal_nodes(
                    nodes, links, shell_normal
                )
                shell_normal_map = nodes.new("ShaderNodeNormalMap")
                shell_normal_map.label = "Eye shell normal intensity"
                shell_normal_map.name = shell_normal_map.label
                shell_normal_map.location = (1660, -180)
                shell_normal_map.inputs["Strength"].default_value = max(
                    0.0,
                    _float_parameter(parameters.get("myNormalIntensity"), 1.0),
                )
                links.new(shell_normal_color, shell_normal_map.inputs["Color"])
                for link in list(shader.inputs["Normal"].links):
                    links.remove(link)
                links.new(shell_normal_map.outputs["Normal"], shader.inputs["Normal"])
                material["afop_normal"] = shell_normal_path
                material["afop_normal_strength"] = _float_parameter(
                    parameters.get("myNormalIntensity"), 1.0
                )
            for link in list(shader.inputs["Roughness"].links):
                links.remove(link)
            for link in list(shader.inputs["Metallic"].links):
                links.remove(link)
            shader.inputs["Roughness"].default_value = max(
                0.0, min(1.0, _float_parameter(parameters.get("myRoughness"), 0.2))
            )
            if shell_roughness_output is not None:
                links.new(shell_roughness_output, shader.inputs["Roughness"])
            shader.inputs["Metallic"].default_value = 0.0
            subsurface = shader.inputs.get("Subsurface Weight")
            if subsurface is not None:
                subsurface.default_value = 0.0004
            coat = shader.inputs.get("Coat Weight")
            if coat is not None:
                coat.default_value = 0.08
            coat_roughness = shader.inputs.get("Coat Roughness")
            if coat_roughness is not None:
                coat_roughness.default_value = 0.7
            material["afop_shader_profile"] = "eye_shell_static"
            material["afop_profile_limit"] = (
                "Snowdrop dither clipping is approximated with Blender hashed alpha"
            )

        if is_constants and diffuse_node is not None:
            constant_tint = nodes.new("ShaderNodeRGB")
            constant_tint.label = "Authored constant color multiplier"
            constant_tint.name = constant_tint.label
            constant_tint.location = (800, 360)
            constant_tint.outputs[0].default_value = (
                *_color_parameter(parameters.get("myColorMultiply"), (0.5,) * 3),
                1.0,
            )
            constant_color = nodes.new("ShaderNodeMixRGB")
            constant_color.blend_type = "MULTIPLY"
            constant_color.inputs[0].default_value = 1.0
            constant_color.label = "Constants shader color"
            constant_color.name = constant_color.label
            constant_color.location = (1080, 300)
            links.new(diffuse_node.outputs["Color"], constant_color.inputs[1])
            links.new(constant_tint.outputs[0], constant_color.inputs[2])
            base_color_override = constant_color.outputs["Color"]
            shader.inputs["Metallic"].default_value = max(
                0.0,
                min(1.0, _float_parameter(parameters.get("myMetalness"), 0.0)),
            )
            shader.inputs["Roughness"].default_value = max(
                0.0,
                min(1.0, _float_parameter(parameters.get("myRoughness"), 0.5)),
            )
            emission_color = _color_parameter(
                parameters.get("myEmission"), (0.0, 0.0, 0.0)
            )
            if any(component > 0.0 for component in emission_color):
                emission_input = (
                    shader.inputs.get("Emission Color") or shader.inputs.get("Emission")
                )
                if emission_input is not None:
                    emission_input.default_value = (*emission_color, 1.0)
                strength_input = shader.inputs.get("Emission Strength")
                if strength_input is not None:
                    strength_input.default_value = 1.0
            material["afop_shader_profile"] = "constants_cutout"

        if is_human_skin and diffuse_node is not None:
            if mask_separate is not None:
                skin_tint = nodes.new("ShaderNodeRGB")
                skin_tint.label = "Authored VHQ skin overlay"
                skin_tint.name = skin_tint.label
                skin_tint.location = (760, 440)
                skin_tint.outputs[0].default_value = (
                    *_color_parameter(
                        parameters.get("myBaseColorOverlay"), (0.5,) * 3
                    ),
                    1.0,
                )
                skin_overlay = nodes.new("ShaderNodeMixRGB")
                skin_overlay.blend_type = "OVERLAY"
                skin_overlay.label = "VHQ skin material-B overlay"
                skin_overlay.name = skin_overlay.label
                skin_overlay.location = (1080, 360)
                links.new(mask_separate.outputs["Blue"], skin_overlay.inputs[0])
                links.new(diffuse_node.outputs["Color"], skin_overlay.inputs[1])
                links.new(skin_tint.outputs[0], skin_overlay.inputs[2])
                base_color_override = skin_overlay.outputs["Color"]

                inverse_sss = _math_node(
                    nodes, "SUBTRACT", "VHQ skin 1 - material R", 1080, -520
                )
                inverse_sss.inputs[0].default_value = 1.0
                links.new(mask_separate.outputs["Red"], inverse_sss.inputs[1])
                sss_scale = _math_node(
                    nodes, "MULTIPLY", "VHQ skin subsurface scale", 1260, -520,
                    0.02 + 0.03 * max(
                        0.0,
                        min(
                            1.0,
                            _float_parameter(
                                parameters.get("mySSS2ONNaviOFFHuman"), 0.0
                            ),
                        ),
                    ),
                )
                links.new(inverse_sss.outputs[0], sss_scale.inputs[0])
                subsurface = shader.inputs.get("Subsurface Weight")
                if subsurface is not None:
                    links.new(sss_scale.outputs[0], subsurface)

                if mask_node is not None:
                    coat_scale = _math_node(
                        nodes, "MULTIPLY", "VHQ skin coat range", 1080, -680, 0.12
                    )
                    coat_bias = _math_node(
                        nodes, "ADD", "VHQ skin coat base", 1260, -680, 0.08
                    )
                    links.new(mask_node.outputs["Alpha"], coat_scale.inputs[0])
                    links.new(coat_scale.outputs[0], coat_bias.inputs[0])
                    coat = shader.inputs.get("Coat Weight")
                    if coat is not None:
                        links.new(coat_bias.outputs[0], coat)
                    coat_rough_scale = _math_node(
                        nodes, "MULTIPLY", "VHQ skin coat roughness range",
                        1080, -820, -0.5,
                    )
                    coat_rough_bias = _math_node(
                        nodes, "ADD", "VHQ skin coat roughness base",
                        1260, -820, 0.7,
                    )
                    links.new(mask_node.outputs["Alpha"], coat_rough_scale.inputs[0])
                    links.new(coat_rough_scale.outputs[0], coat_rough_bias.inputs[0])
                    coat_roughness = shader.inputs.get("Coat Roughness")
                    if coat_roughness is not None:
                        links.new(coat_rough_bias.outputs[0], coat_roughness)

                if _float_parameter(parameters.get("myMaterialAlpha"), 0.0) >= 0.5:
                    profile_alpha_output = mask_separate.outputs["Blue"]
                    profile_alpha_source = "material-blue"
            for link in list(shader.inputs["Metallic"].links):
                links.remove(link)
            shader.inputs["Metallic"].default_value = 0.0
            if ao_output is not None:
                for link in list(shader.inputs["Roughness"].links):
                    links.remove(link)
                links.new(ao_output, shader.inputs["Roughness"])

            skin_aux = binding.get("aux", {})
            bio_path = skin_aux.get("Bioluminescence")
            bio = _aux_image_node(
                nodes, links, texture_files, bio_path,
                "VHQ Skin Bioluminescence", -520, -1420, non_color=False,
            )
            if bio is not None:
                emission_input = (
                    shader.inputs.get("Emission Color") or shader.inputs.get("Emission")
                )
                if emission_input is not None:
                    links.new(bio.outputs["Color"], emission_input)
                strength_input = shader.inputs.get("Emission Strength")
                if strength_input is not None:
                    brightness = _float_parameter(
                        parameters.get("myBioBrightness"), 1.0
                    )
                    strength_input.default_value = min(
                        5.0,
                        max(0.0, brightness)
                        * max(
                            0.0,
                            _float_parameter(
                                parameters.get("myBioColorStrength"), 0.5
                            ),
                        ),
                    )
                material["afop_bioluminescence"] = bio_path
                material["afop_bio_preview"] = "night"
            wrinkle_fields = (
                "WrinkleNormal1", "WrinkleNormal2", "WrinkleNormal3",
                "WrinkleMask1", "WrinkleMask2", "WrinkleMask3",
                "WrinkleMask4", "WrinkleMask5",
            )
            preview_y = -1640
            for field in wrinkle_fields:
                path = skin_aux.get(field)
                node = _aux_image_node(
                    nodes, links, texture_files, path,
                    f"{field} (facial animation source)", -520, preview_y,
                )
                if node is not None:
                    preview_y -= 180
            material["afop_shader_profile"] = "human_skin_vhq_static"
            material["afop_profile_limit"] = (
                "Animated wrinkles, rain and time-of-day bio remain engine-driven"
            )

        if is_hair:
            hair_aux = binding.get("aux", {})
            tiling = parameters.get("myTiling", (1.0, 1.0))
            hair_maps_path = hair_aux.get("HairMaps")
            direction_path = hair_aux.get("DirectionMap")
            ao_path = hair_aux.get("AO")
            hair_maps = _aux_image_node(
                nodes, links, texture_files, hair_maps_path,
                "Hair Maps (root/color/depth/alpha)", -620, 500,
                tiling=tiling,
            )
            direction = _aux_image_node(
                nodes, links, texture_files, direction_path,
                "Hair Direction (anisotropy source)", -620, -700,
                tiling=tiling,
            )
            hair_ao = _aux_image_node(
                nodes, links, texture_files, ao_path,
                "Hair AO", -620, -920, uv_map="UVMap_1",
            )
            if hair_maps is not None:
                hair_channels = _separate_node(
                    nodes, links, hair_maps.outputs["Color"],
                    "Hair map channels", -360, 500,
                )
                colors = [
                    _color_parameter(parameters.get(f"myHairColor{index}"))
                    for index in (1, 2, 3)
                ]
                smoothness = max(
                    0.0,
                    min(
                        1.0,
                        _float_parameter(
                            parameters.get("myColorTransitionSmoothness"), 0.0
                        ),
                    ),
                )
                p1_low = (1.0 - smoothness) * 0.33
                p1_high = (1.0 - smoothness) * 0.34 + smoothness * 0.5
                p2_low = (1.0 - smoothness) * 0.665 + smoothness * 0.5
                p2_high = (1.0 - smoothness) * 0.666 + smoothness
                hair_ramp = nodes.new("ShaderNodeValToRGB")
                hair_ramp.label = "Authored three-color hair gradient"
                hair_ramp.name = hair_ramp.label
                hair_ramp.location = (40, 500)
                hair_ramp.color_ramp.interpolation = "LINEAR"
                elements = hair_ramp.color_ramp.elements
                elements[0].position = 0.0
                elements[0].color = (*colors[0], 1.0)
                elements[-1].position = 1.0
                elements[-1].color = (*colors[2], 1.0)
                for position, color in (
                    (p1_low, colors[0]), (p1_high, colors[1]),
                    (p2_low, colors[1]), (p2_high, colors[2]),
                ):
                    element = elements.new(max(0.0, min(1.0, position)))
                    element.color = (*color, 1.0)
                links.new(hair_channels.outputs["Green"], hair_ramp.inputs["Fac"])

                depth = parameters.get("myDepthColorOverlay", (0.0, 1.0, 0.5, 0.5))
                if not isinstance(depth, (list, tuple)) or len(depth) < 4:
                    depth = (0.0, 1.0, 0.5, 0.5)
                depth_map = nodes.new("ShaderNodeMapRange")
                depth_map.clamp = True
                depth_map.label = "Hair depth overlay"
                depth_map.name = depth_map.label
                depth_map.location = (40, 300)
                depth_map.inputs["From Min"].default_value = float(depth[0])
                depth_map.inputs["From Max"].default_value = float(depth[1])
                depth_map.inputs["To Min"].default_value = float(depth[2])
                depth_map.inputs["To Max"].default_value = float(depth[3])
                links.new(hair_channels.outputs["Blue"], depth_map.inputs["Value"])
                depth_overlay = nodes.new("ShaderNodeMixRGB")
                depth_overlay.blend_type = "OVERLAY"
                depth_overlay.inputs[0].default_value = 1.0
                depth_overlay.label = "Hair depth color overlay"
                depth_overlay.name = depth_overlay.label
                depth_overlay.location = (300, 500)
                links.new(hair_ramp.outputs["Color"], depth_overlay.inputs[1])
                links.new(depth_map.outputs["Result"], depth_overlay.inputs[2])

                root_strength = max(
                    0.0,
                    min(
                        1.0,
                        _float_parameter(parameters.get("myRootDarkening"), 0.0),
                    ),
                )
                root_scale = _math_node(
                    nodes, "MULTIPLY", "Hair root map strength", 520, 300,
                    root_strength,
                )
                root_bias = _math_node(
                    nodes, "ADD", "Hair root map base", 700, 300,
                    1.0 - root_strength,
                )
                links.new(hair_channels.outputs["Red"], root_scale.inputs[0])
                links.new(root_scale.outputs[0], root_bias.inputs[0])
                root_multiply = nodes.new("ShaderNodeMixRGB")
                root_multiply.blend_type = "MULTIPLY"
                root_multiply.inputs[0].default_value = 1.0
                root_multiply.label = "Apply hair root darkening"
                root_multiply.name = root_multiply.label
                root_multiply.location = (880, 500)
                links.new(depth_overlay.outputs["Color"], root_multiply.inputs[1])
                links.new(root_bias.outputs[0], root_multiply.inputs[2])
                hair_color_output = root_multiply.outputs["Color"]
                if hair_ao is not None:
                    hair_ao_channels = _separate_node(
                        nodes, links, hair_ao.outputs["Color"],
                        "Hair AO channels", -360, -920,
                    )
                    ao_multiply = nodes.new("ShaderNodeMixRGB")
                    ao_multiply.blend_type = "MULTIPLY"
                    ao_multiply.inputs[0].default_value = 1.0
                    ao_multiply.label = "Apply hair AO"
                    ao_multiply.name = ao_multiply.label
                    ao_multiply.location = (1080, 500)
                    links.new(hair_color_output, ao_multiply.inputs[1])
                    links.new(hair_ao_channels.outputs["Red"], ao_multiply.inputs[2])
                    hair_color_output = ao_multiply.outputs["Color"]
                base_color_override = hair_color_output
                if diffuse_node is None:
                    links.new(base_color_override, shader.inputs["Base Color"])

                rough_half = _math_node(
                    nodes, "MULTIPLY", "Hair roughness map range", 1080, -360, 0.5
                )
                rough_bias = _math_node(
                    nodes, "ADD", "Hair roughness map base", 1260, -360, 0.5
                )
                rough_value = _math_node(
                    nodes, "MULTIPLY", "Authored hair roughness", 1440, -360,
                    max(
                        0.0,
                        _float_parameter(parameters.get("myRoughness"), 0.45768026),
                    ),
                )
                links.new(hair_channels.outputs["Green"], rough_half.inputs[0])
                links.new(rough_half.outputs[0], rough_bias.inputs[0])
                links.new(rough_bias.outputs[0], rough_value.inputs[0])
                for link in list(shader.inputs["Roughness"].links):
                    links.remove(link)
                links.new(rough_value.outputs[0], shader.inputs["Roughness"])
                for link in list(shader.inputs["Metallic"].links):
                    links.remove(link)
                shader.inputs["Metallic"].default_value = 0.0
                subsurface = shader.inputs.get("Subsurface Weight")
                if subsurface is not None:
                    subsurface.default_value = 0.005

                # HairMaps stores root, colour transition, and depth in RGB;
                # only A is the authored strand cutout. Multiplying opacity by
                # RGB or vertex colour makes most cards nearly invisible.
                profile_alpha_output = hair_maps.outputs["Alpha"]
                profile_alpha_source = "hairmaps-alpha"
                material["afop_hair_maps"] = hair_maps_path
            if direction is not None:
                material["afop_hair_direction"] = direction_path
            if hair_ao is not None:
                material["afop_hair_ao"] = ao_path
            material["afop_shader_profile"] = "hair_three_color_static"
            material["afop_profile_limit"] = (
                "DirectionMap is retained as an anisotropy source; tousle and wetness are engine-driven"
            )

        if is_natural_rock:
            rock_aux = binding.get("aux", {})
            gradient_path = rock_aux.get("SetRockGradient")
            normal_a_path = rock_aux.get("SetRockNormalA")
            normal_b_path = rock_aux.get("SetRockNormalB")
            rock_mask_path = rock_aux.get("Mask")
            rock_gradient = _aux_image_node(
                nodes, links, texture_files, gradient_path,
                "Natural Rock Gradient LUT", -620, 600, non_color=False,
            )
            rock_normal_a = _aux_image_node(
                nodes, links, texture_files, normal_a_path,
                "Natural Rock Set Normal A", -620, 260,
                tiling=(
                    _float_parameter(parameters.get("myNormalATiling"), 0.06),
                    _float_parameter(parameters.get("myNormalATiling"), 0.06),
                ),
            )
            rock_normal_b = _aux_image_node(
                nodes, links, texture_files, normal_b_path,
                "Natural Rock Set Normal B", -620, 20,
                tiling=(
                    _float_parameter(parameters.get("myNormalBTiling"), 0.4),
                    _float_parameter(parameters.get("myNormalBTiling"), 0.4),
                ),
            )
            rock_mask = _aux_image_node(
                nodes, links, texture_files, rock_mask_path,
                "Natural Rock Unique Mask", -620, -220,
            )
            if (
                rock_gradient is not None
                and rock_normal_a is not None
                and rock_normal_b is not None
                and rock_mask is not None
            ):
                # Snowdrop samples the set normals in model-space triplanar
                # projection. UV0 made the lookup coordinates unrelated to
                # the authored shader and produced a visibly wrong LUT color.
                rock_coordinates = nodes.new("ShaderNodeTexCoord")
                rock_coordinates.label = "Natural rock model coordinates"
                rock_coordinates.name = rock_coordinates.label
                rock_coordinates.location = (-1120, 260)
                for texture_node, tiling, label, y in (
                    (
                        rock_normal_a,
                        _float_parameter(parameters.get("myNormalATiling"), 0.06),
                        "Natural rock set A triplanar scale",
                        300,
                    ),
                    (
                        rock_normal_b,
                        _float_parameter(parameters.get("myNormalBTiling"), 0.4),
                        "Natural rock set B triplanar scale",
                        80,
                    ),
                ):
                    texture_node.projection = "BOX"
                    texture_node.projection_blend = 0.2
                    scale = nodes.new("ShaderNodeVectorMath")
                    scale.operation = "SCALE"
                    scale.label = label
                    scale.name = label
                    scale.location = (-900, y)
                    scale.inputs["Scale"].default_value = tiling
                    links.new(rock_coordinates.outputs["Object"], scale.inputs["Vector"])
                    for link in list(texture_node.inputs["Vector"].links):
                        links.remove(link)
                    links.new(scale.outputs["Vector"], texture_node.inputs["Vector"])

                channels_a = _separate_node(
                    nodes, links, rock_normal_a.outputs["Color"],
                    "Natural rock normal-A channels", -340, 260,
                )
                channels_b = _separate_node(
                    nodes, links, rock_normal_b.outputs["Color"],
                    "Natural rock normal-B channels", -340, 20,
                )
                mask_channels = _separate_node(
                    nodes, links, rock_mask.outputs["Color"],
                    "Natural rock mask channels", -340, -220,
                )
                vertex_channels = None
                if (
                    _float_parameter(
                        parameters.get("myUseVertexColorMasking"), 0.0
                    ) >= 0.5
                    and obj.data.attributes.get("Color_0") is not None
                ):
                    vertex = nodes.new("ShaderNodeVertexColor")
                    vertex.layer_name = "Color_0"
                    vertex.label = "Natural rock vertex masks"
                    vertex.name = vertex.label
                    vertex.location = (-760, -420)
                    vertex_channels = _separate_node(
                        nodes, links, vertex.outputs["Color"],
                        "Natural rock vertex-mask channels", -540, -420,
                    )

                def weighted_set_channel(source, vertex_source, amount, label, x, y):
                    strength = nodes.new("ShaderNodeValue")
                    strength.label = f"{label} authored weight"
                    strength.name = strength.label
                    strength.location = (x - 220, y - 100)
                    strength.outputs[0].default_value = max(0.0, float(amount))
                    factor = strength.outputs[0]
                    if vertex_source is not None:
                        vertex_weight = _math_node(
                            nodes, "MULTIPLY", f"{label} vertex weight", x, y - 100
                        )
                        links.new(vertex_source, vertex_weight.inputs[0])
                        links.new(strength.outputs[0], vertex_weight.inputs[1])
                        factor = vertex_weight.outputs[0]
                    weighted = nodes.new("ShaderNodeMixRGB")
                    weighted.label = label
                    weighted.name = label
                    weighted.location = (x, y)
                    weighted.inputs[1].default_value = (0.5, 0.5, 0.5, 1.0)
                    links.new(factor, weighted.inputs[0])
                    links.new(source, weighted.inputs[2])
                    return weighted.outputs["Color"]

                set_a = weighted_set_channel(
                    channels_a.outputs["Red"],
                    vertex_channels.outputs["Red"] if vertex_channels else None,
                    _float_parameter(parameters.get("myNormalAOverlay"), 0.8),
                    "Weighted SetRock A lookup channel", -80, 360,
                )
                set_b = weighted_set_channel(
                    channels_b.outputs["Red"],
                    vertex_channels.outputs["Green"] if vertex_channels else None,
                    _float_parameter(parameters.get("myNormalBOverlay"), 0.5),
                    "Weighted SetRock B lookup channel", -80, 80,
                )
                set_overlay = nodes.new("ShaderNodeMixRGB")
                set_overlay.blend_type = "OVERLAY"
                set_overlay.inputs[0].default_value = 1.0
                set_overlay.label = "Approximate SetRock normal overlay"
                set_overlay.name = set_overlay.label
                set_overlay.location = (180, 220)
                links.new(set_b, set_overlay.inputs[1])
                links.new(set_a, set_overlay.inputs[2])
                mask_overlay = nodes.new("ShaderNodeMixRGB")
                mask_overlay.blend_type = "OVERLAY"
                mask_overlay.inputs[0].default_value = 1.0
                mask_overlay.label = "Natural rock unique-mask overlay"
                mask_overlay.name = mask_overlay.label
                mask_overlay.location = (400, 220)
                links.new(set_overlay.outputs["Color"], mask_overlay.inputs[1])
                links.new(mask_channels.outputs["Red"], mask_overlay.inputs[2])
                lookup_uv = nodes.new("ShaderNodeCombineXYZ")
                lookup_uv.label = "Natural rock gradient lookup UV"
                lookup_uv.name = lookup_uv.label
                lookup_uv.location = (620, 220)
                links.new(mask_overlay.outputs["Color"], lookup_uv.inputs["X"])
                links.new(channels_b.outputs["Red"], lookup_uv.inputs["Y"])
                for link in list(rock_gradient.inputs["Vector"].links):
                    links.remove(link)
                links.new(lookup_uv.outputs["Vector"], rock_gradient.inputs["Vector"])
                base_color_override = rock_gradient.outputs["Color"]
                if ao_output is not None:
                    rock_ao = nodes.new("ShaderNodeMixRGB")
                    rock_ao.blend_type = "MULTIPLY"
                    rock_ao.inputs[0].default_value = 1.0
                    rock_ao.label = "Natural rock gradient x unique AO"
                    rock_ao.name = rock_ao.label
                    rock_ao.location = (1080, 360)
                    links.new(base_color_override, rock_ao.inputs[1])
                    links.new(ao_output, rock_ao.inputs[2])
                    base_color_override = rock_ao.outputs["Color"]
                if diffuse_node is None:
                    links.new(base_color_override, shader.inputs["Base Color"])
                for link in list(shader.inputs["Roughness"].links):
                    links.remove(link)
                links.new(rock_gradient.outputs["Alpha"], shader.inputs["Roughness"])
                material["afop_rock_gradient"] = gradient_path
                material["afop_rock_mask"] = rock_mask_path
            for link in list(shader.inputs["Metallic"].links):
                links.remove(link)
            shader.inputs["Metallic"].default_value = 0.0
            material["afop_shader_profile"] = "natural_rock_static_lookup"
            material["afop_profile_limit"] = (
                "Set normals and gradient use a static box-triplanar reconstruction; procedural moss, terrain and pollution remain engine-driven"
            )

        if is_terrain_runtime:
            terrain_color = nodes.new("ShaderNodeRGB")
            terrain_color.label = "Terrain runtime placeholder"
            terrain_color.name = terrain_color.label
            terrain_color.location = (1420, 260)
            terrain_color.outputs[0].default_value = (0.18, 0.16, 0.12, 1.0)
            links.new(terrain_color.outputs[0], shader.inputs["Base Color"])
            shader.inputs["Roughness"].default_value = 0.85
            shader.inputs["Metallic"].default_value = 0.0
            material["afop_shader_profile"] = "terrain_runtime_placeholder"
            material["afop_profile_limit"] = (
                "PX_TerrainBlend samples the live terrain and has no asset texture samplers"
            )

        if is_moss_patch and diffuse_node is not None:
            patch_overlay_path = binding.get("aux", {}).get("WorldSpaceOverlay")
            patch_overlay = _aux_image_node(
                nodes, links, texture_files, patch_overlay_path,
                "Moss Patch World Overlay", -620, 600, non_color=False,
            )
            if patch_overlay is not None:
                geometry = nodes.new("ShaderNodeNewGeometry")
                geometry.label = "Moss patch world position"
                geometry.name = geometry.label
                geometry.location = (-1120, 600)
                position = _separate_node(
                    nodes, links, geometry.outputs["Position"],
                    "Moss patch world X/Z", -940, 600,
                )
                projected_uv = nodes.new("ShaderNodeCombineXYZ")
                projected_uv.label = "Moss patch projected X/Z UV"
                projected_uv.name = projected_uv.label
                projected_uv.location = (-760, 600)
                links.new(position.outputs["Red"], projected_uv.inputs["X"])
                links.new(position.outputs["Blue"], projected_uv.inputs["Y"])
                projected_scale = nodes.new("ShaderNodeVectorMath")
                projected_scale.operation = "SCALE"
                projected_scale.label = "Moss patch world tiling"
                projected_scale.name = projected_scale.label
                projected_scale.location = (-580, 780)
                projected_scale.inputs["Scale"].default_value = _float_parameter(
                    parameters.get("myWorldSpaceTiling"), 0.0
                )
                links.new(projected_uv.outputs["Vector"], projected_scale.inputs["Vector"])
                for link in list(patch_overlay.inputs["Vector"].links):
                    links.remove(link)
                links.new(projected_scale.outputs["Vector"], patch_overlay.inputs["Vector"])
                patch_color = nodes.new("ShaderNodeMixRGB")
                patch_color.blend_type = "OVERLAY"
                patch_color.inputs[0].default_value = 1.0
                patch_color.label = "Moss patch world overlay"
                patch_color.name = patch_color.label
                patch_color.location = (500, 480)
                # Snowdrop calls OverlayNoMask(projected, color): projected is
                # the base and the authored diffuse is the overlay layer.
                links.new(patch_overlay.outputs["Color"], patch_color.inputs[1])
                links.new(diffuse_node.outputs["Color"], patch_color.inputs[2])
                base_color_override = patch_color.outputs["Color"]
                material["afop_world_overlay"] = patch_overlay_path
            for link in list(shader.inputs["Metallic"].links):
                links.remove(link)
            shader.inputs["Metallic"].default_value = 0.0
            material["afop_shader_profile"] = "moss_patch_static"
            material["afop_profile_limit"] = (
                "Live terrain blending, pollution and burning are engine-driven"
            )

        if is_wildlife_gear and diffuse_node is not None:
            gear_aux = binding.get("aux", {})
            regions_path = gear_aux.get("Regions")
            regions = _aux_image_node(
                nodes, links, texture_files, regions_path,
                "Wildlife Gear Regions", -620, 560,
            )
            region_channels = None
            if regions is not None:
                region_channels = _separate_node(
                    nodes, links, regions.outputs["Color"],
                    "Wildlife gear region channels", -340, 560,
                )
                variant = nodes.new("ShaderNodeMapRange")
                variant.clamp = True
                variant.label = "Gear tint A/B selector"
                variant.name = variant.label
                variant.location = (-80, 720)
                variant.inputs["From Min"].default_value = 0.5
                variant.inputs["From Max"].default_value = 1.0
                links.new(region_channels.outputs["Blue"], variant.inputs["Value"])

                def tint_node(field, label, x, y):
                    node = nodes.new("ShaderNodeRGB")
                    node.label = label
                    node.name = label
                    node.location = (x, y)
                    node.outputs[0].default_value = (
                        *_color_parameter(parameters.get(field), (1.0,) * 3), 1.0
                    )
                    return node

                primary_a = tint_node("myPrimaryTintA", "Primary Tint A", -80, 940)
                primary_b = tint_node("myPrimaryTintB", "Primary Tint B", -80, 1060)
                secondary_a = tint_node("mySecondaryTintA", "Secondary Tint A", -80, 1180)
                secondary_b = tint_node("mySecondaryTintB", "Secondary Tint B", -80, 1300)
                primary_mix = nodes.new("ShaderNodeMixRGB")
                primary_mix.label = "Primary gear tint variant"
                primary_mix.name = primary_mix.label
                primary_mix.location = (180, 980)
                links.new(variant.outputs["Result"], primary_mix.inputs[0])
                links.new(primary_b.outputs[0], primary_mix.inputs[1])
                links.new(primary_a.outputs[0], primary_mix.inputs[2])
                secondary_mix = nodes.new("ShaderNodeMixRGB")
                secondary_mix.label = "Secondary gear tint variant"
                secondary_mix.name = secondary_mix.label
                secondary_mix.location = (180, 1180)
                links.new(variant.outputs["Result"], secondary_mix.inputs[0])
                links.new(secondary_b.outputs[0], secondary_mix.inputs[1])
                links.new(secondary_a.outputs[0], secondary_mix.inputs[2])
                family_mix = nodes.new("ShaderNodeMixRGB")
                family_mix.label = "Primary / secondary gear tint"
                family_mix.name = family_mix.label
                family_mix.location = (420, 1040)
                links.new(region_channels.outputs["Green"], family_mix.inputs[0])
                links.new(primary_mix.outputs["Color"], family_mix.inputs[1])
                links.new(secondary_mix.outputs["Color"], family_mix.inputs[2])
                tint_strength = _math_node(
                    nodes, "MULTIPLY", "Gear tint region strength", 180, 760, 4.0
                )
                links.new(region_channels.outputs["Blue"], tint_strength.inputs[0])
                gear_overlay = nodes.new("ShaderNodeMixRGB")
                gear_overlay.blend_type = "OVERLAY"
                gear_overlay.label = "Wildlife gear region tint"
                gear_overlay.name = gear_overlay.label
                gear_overlay.location = (700, 560)
                links.new(tint_strength.outputs[0], gear_overlay.inputs[0])
                links.new(diffuse_node.outputs["Color"], gear_overlay.inputs[1])
                links.new(family_mix.outputs["Color"], gear_overlay.inputs[2])
                base_color_override = gear_overlay.outputs["Color"]

            detail_path = gear_aux.get("DetailNormal")
            detail = _aux_image_node(
                nodes, links, texture_files, detail_path,
                "Wildlife Gear Detail Normal", -620, -900,
                tiling=parameters.get("myDetailTiling", (1.0, 1.0)),
            )
            if (
                detail is not None
                and normal_texture_node is not None
                and normal is not None
                and mask_separate is not None
            ):
                detail_layer = _packed_detail_layer(
                    nodes, links,
                    normal_texture_node.outputs["Color"],
                    normal_texture_node.outputs["Alpha"],
                    detail, detail.outputs["Alpha"],
                    mask_separate.outputs["Blue"],
                    _float_parameter(parameters.get("myDetailStrength"), 0.0),
                    "Wildlife gear detail normal (RNM)", 1460, -300,
                )
                for link in list(normal.inputs["Color"].links):
                    links.remove(link)
                links.new(detail_layer.outputs["Normal Color"], normal.inputs["Color"])
                ao_output = detail_layer.outputs["AO"]
                material["afop_detail_normal"] = detail_path
            if region_channels is not None and mask_separate is not None:
                region_metal_scale = _math_node(
                    nodes, "MULTIPLY", "Gear metal region R x4", 980, -520, 4.0
                )
                region_metal_gate = _math_node(
                    nodes, "SUBTRACT", "Gear metal region gate", 1160, -520
                )
                region_metal_gate.inputs[0].default_value = 1.0
                metal_result = _math_node(
                    nodes, "MULTIPLY", "Wildlife gear metalness", 1340, -520
                )
                links.new(region_channels.outputs["Red"], region_metal_scale.inputs[0])
                links.new(region_metal_scale.outputs[0], region_metal_gate.inputs[1])
                links.new(mask_separate.outputs["Red"], metal_result.inputs[0])
                links.new(region_metal_gate.outputs[0], metal_result.inputs[1])
                for link in list(shader.inputs["Metallic"].links):
                    links.remove(link)
                links.new(metal_result.outputs[0], shader.inputs["Metallic"])
            material["afop_shader_profile"] = "wildlife_gear_static"
            material["afop_profile_limit"] = (
                "Wet dirt, Fresnel metal response, aiming fade and time-of-day emission are engine-driven"
            )

        if is_vegetation and normal_texture_node is not None and normal is not None:
            auxiliary = binding.get("aux", {})
            detail_a_path = auxiliary.get("DetailA")
            detail_b_path = auxiliary.get("DetailB")
            detail_normal_path = auxiliary.get("DetailNormal")
            bio_path = auxiliary.get("Bioluminescence")
            vegetation_tiling = _float_parameter(
                parameters.get("myDetailTiling"), 5.0
            )
            detail_a = _aux_image_node(
                nodes, links, texture_files, detail_a_path,
                "Vegetation Detail A (packed)", -520, -760,
                tiling=(vegetation_tiling, vegetation_tiling), uv_map="UVMap_1",
            )
            detail_b = _aux_image_node(
                nodes, links, texture_files, detail_b_path,
                "Vegetation Detail B (packed)", -520, -980,
                tiling=(vegetation_tiling, vegetation_tiling), uv_map="UVMap_1",
            )
            single_detail = _aux_image_node(
                nodes, links, texture_files, detail_normal_path,
                "Vegetation Detail Normal (packed)", -520, -1200,
                tiling=parameters.get("myDetailNormalUVTiling", (20.0, 20.0)),
            )
            vegetation_detail = None
            vegetation_detail_x = None
            detail_selection = None
            if detail_a is not None and detail_b is not None and mask_separate is not None:
                detail_selection = nodes.new("ShaderNodeMixRGB")
                detail_selection.label = "Material B selects vegetation detail"
                detail_selection.name = detail_selection.label
                detail_selection.location = (-80, -870)
                links.new(mask_separate.outputs["Blue"], detail_selection.inputs[0])
                links.new(detail_a.outputs["Color"], detail_selection.inputs[1])
                links.new(detail_b.outputs["Color"], detail_selection.inputs[2])
                alpha_selection = nodes.new("ShaderNodeMixRGB")
                alpha_selection.label = "Vegetation detail X selection"
                alpha_selection.location = (-80, -1040)
                links.new(mask_separate.outputs["Blue"], alpha_selection.inputs[0])
                links.new(detail_a.outputs["Alpha"], alpha_selection.inputs[1])
                links.new(detail_b.outputs["Alpha"], alpha_selection.inputs[2])
                vegetation_detail = detail_selection
                vegetation_detail_x = alpha_selection.outputs["Color"]
            elif single_detail is not None:
                vegetation_detail = single_detail
                vegetation_detail_x = single_detail.outputs["Alpha"]
            if vegetation_detail is not None:
                full_mask = nodes.new("ShaderNodeValue")
                full_mask.label = "Vegetation detail enabled"
                full_mask.outputs[0].default_value = 1.0
                detail_layer = _packed_detail_layer(
                    nodes, links,
                    normal_texture_node.outputs["Color"],
                    normal_texture_node.outputs["Alpha"],
                    vegetation_detail, vegetation_detail_x,
                    full_mask.outputs[0],
                    _float_parameter(
                        parameters.get(
                            "myDetailIntensity",
                            parameters.get("myDetailStrength", 1.0),
                        ), 1.0,
                    ),
                    "Vegetation detail normal (RNM)", 1450, -300,
                )
                for link in list(normal.inputs["Color"].links):
                    links.remove(link)
                links.new(detail_layer.outputs["Normal Color"], normal.inputs["Color"])
                material["afop_detail_normal"] = (
                    detail_normal_path or f"{detail_a_path}|{detail_b_path}"
                )
            if diffuse_node is not None:
                tint = nodes.new("ShaderNodeRGB")
                tint.label = "Authored vegetation tint"
                tint.outputs[0].default_value = (
                    *_color_parameter(parameters.get("myTint"), (0.5,) * 3), 1.0
                )
                tint_overlay = nodes.new("ShaderNodeMixRGB")
                tint_overlay.blend_type = "OVERLAY"
                tint_overlay.inputs[0].default_value = 1.0
                tint_overlay.label = "Snowdrop vegetation OverlayNoMask"
                tint_overlay.name = tint_overlay.label
                links.new(diffuse_node.outputs["Color"], tint_overlay.inputs[1])
                links.new(tint.outputs[0], tint_overlay.inputs[2])
                base_color_override = tint_overlay.outputs["Color"]
            bio = _aux_image_node(
                nodes, links, texture_files, bio_path,
                "Vegetation Bioluminescence", -520, -1440, non_color=False,
            )
            if bio is not None:
                hue = nodes.new("ShaderNodeHueSaturation")
                hue.label = "Vegetation bio hue/scale preview"
                hue.name = hue.label
                hue.location = (300, -1440)
                hue.inputs["Hue"].default_value = (
                    0.5 + _float_parameter(parameters.get("myBioHueShift"), 0.0)
                ) % 1.0
                hue.inputs["Value"].default_value = max(
                    0.0,
                    _float_parameter(parameters.get("myBioScale"), 1.0)
                    * _float_parameter(parameters.get("myBioToggle"), 1.0),
                )
                links.new(bio.outputs["Color"], hue.inputs["Color"])
                emission_input = (
                    shader.inputs.get("Emission Color") or shader.inputs.get("Emission")
                )
                if emission_input is not None:
                    links.new(hue.outputs["Color"], emission_input)
                strength_input = shader.inputs.get("Emission Strength")
                if strength_input is not None:
                    strength_input.default_value = 1.0
                material["afop_bioluminescence"] = bio_path
                material["afop_bio_preview"] = "night"
            if mask_separate is not None:
                rough_scale = _math_node(
                    nodes, "MULTIPLY", "Vegetation roughness x 0.65",
                    1250, -1080, 0.65,
                )
                rough_bias = _math_node(
                    nodes, "ADD", "Vegetation roughness + 0.35",
                    1420, -1080, 0.35,
                )
                links.new(mask_separate.outputs["Red"], rough_scale.inputs[0])
                links.new(rough_scale.outputs[0], rough_bias.inputs[0])
                for link in list(shader.inputs["Roughness"].links):
                    links.remove(link)
                links.new(rough_bias.outputs[0], shader.inputs["Roughness"])
                for link in list(shader.inputs["Metallic"].links):
                    links.remove(link)
                shader.inputs["Metallic"].default_value = 0.0
            material["afop_shader_profile"] = "vegetation_static_night"
            material["afop_profile_limit"] = (
                "Wind, pollution, weather and time-of-day branches are engine-driven"
            )

        if is_moss_card and diffuse_node is not None:
            auxiliary = binding.get("aux", {})
            projected_path = auxiliary.get("ProjectedOverlay")
            projected = _aux_image_node(
                nodes, links, texture_files, projected_path,
                "Moss Card Projected Overlay", -520, 500, non_color=False,
            )
            moss_base = diffuse_node.outputs["Color"]
            if projected is not None:
                geometry = nodes.new("ShaderNodeNewGeometry")
                geometry.label = "Moss projected world position"
                geometry.name = geometry.label
                geometry.location = (-1080, 500)
                position_channels = _separate_node(
                    nodes, links, geometry.outputs["Position"],
                    "Moss world X/Z", -900, 500,
                )
                projected_uv = nodes.new("ShaderNodeCombineXYZ")
                projected_uv.label = "Moss projected X/Z UV"
                projected_uv.name = projected_uv.label
                projected_uv.location = (-720, 500)
                links.new(position_channels.outputs["Red"], projected_uv.inputs["X"])
                links.new(position_channels.outputs["Blue"], projected_uv.inputs["Y"])
                projected_scale = nodes.new("ShaderNodeVectorMath")
                projected_scale.operation = "SCALE"
                projected_scale.label = "Moss projected tiling"
                projected_scale.name = projected_scale.label
                projected_scale.location = (-540, 680)
                projected_scale.inputs["Scale"].default_value = _float_parameter(
                    parameters.get("myProjectedTiling"), 1.0
                )
                links.new(projected_uv.outputs["Vector"], projected_scale.inputs["Vector"])
                links.new(projected_scale.outputs["Vector"], projected.inputs["Vector"])
                projected_overlay = nodes.new("ShaderNodeMixRGB")
                projected_overlay.blend_type = "OVERLAY"
                projected_overlay.inputs[0].default_value = 1.0
                projected_overlay.label = "Moss projected OverlayNoMask"
                projected_overlay.name = projected_overlay.label
                projected_overlay.location = (100, 500)
                links.new(projected.outputs["Color"], projected_overlay.inputs[1])
                links.new(diffuse_node.outputs["Color"], projected_overlay.inputs[2])
                moss_base = projected_overlay.outputs["Color"]
                material["afop_projected_overlay"] = projected_path
            tint = nodes.new("ShaderNodeRGB")
            tint.label = "Authored moss-card overlay"
            tint.name = tint.label
            tint.location = (300, 650)
            tint.outputs[0].default_value = (
                *_color_parameter(parameters.get("myColorOverlay"), (0.5,) * 3),
                1.0,
            )
            tint_overlay = nodes.new("ShaderNodeMixRGB")
            tint_overlay.blend_type = "OVERLAY"
            tint_overlay.inputs[0].default_value = 1.0
            tint_overlay.label = "Moss color OverlayNoMask"
            tint_overlay.name = tint_overlay.label
            tint_overlay.location = (500, 500)
            links.new(moss_base, tint_overlay.inputs[1])
            links.new(tint.outputs[0], tint_overlay.inputs[2])
            base_color_override = tint_overlay.outputs["Color"]

            bio_path = auxiliary.get("Bioluminescence")
            bio = _aux_image_node(
                nodes, links, texture_files, bio_path,
                "Moss Card Bioluminescence", -520, -1420, non_color=False,
            )
            if bio is not None:
                bio_overlay = nodes.new("ShaderNodeMixRGB")
                bio_overlay.blend_type = "OVERLAY"
                bio_overlay.inputs[0].default_value = 1.0
                bio_overlay.label = "Moss bio OverlayNoMask"
                bio_overlay.name = bio_overlay.label
                bio_overlay.location = (300, -1420)
                links.new(bio.outputs["Color"], bio_overlay.inputs[1])
                links.new(tint.outputs[0], bio_overlay.inputs[2])
                emission_input = (
                    shader.inputs.get("Emission Color") or shader.inputs.get("Emission")
                )
                if emission_input is not None:
                    links.new(bio_overlay.outputs["Color"], emission_input)
                strength_input = shader.inputs.get("Emission Strength")
                if strength_input is not None:
                    strength_input.default_value = max(
                        0.0,
                        _float_parameter(parameters.get("myBioToggle"), 1.0)
                        * _float_parameter(parameters.get("myBioScale"), 1.0),
                    )
                material["afop_bioluminescence"] = bio_path
                material["afop_bio_preview"] = "night"
            if mask_separate is not None:
                for link in list(shader.inputs["Roughness"].links):
                    links.remove(link)
                for link in list(shader.inputs["Metallic"].links):
                    links.remove(link)
                links.new(mask_separate.outputs["Green"], shader.inputs["Roughness"])
                shader.inputs["Metallic"].default_value = 0.0
            material["afop_shader_profile"] = "moss_card_static_night"
            material["afop_profile_limit"] = (
                "World projection is static; Fresnel, pollution and time-of-day are engine-driven"
            )

        if is_basic_blend:
            auxiliary = binding.get("aux", {})
            base_tiling = parameters.get("myTilingBase", (1.0, 1.0))
            blend_tiling = parameters.get("myTilingBlend", (1.0, 1.0))
            blend_uv_map = (
                "UVMap_1"
                if _float_parameter(parameters.get("myBlendUV2"), 1.0) >= 0.5
                else "UVMap_0"
            )
            color_base = _aux_image_node(
                nodes, links, texture_files, auxiliary.get("ColorBase"),
                "Blend Material Base Color", -760, 720, non_color=False,
                tiling=base_tiling,
            )
            normal_base = _aux_image_node(
                nodes, links, texture_files, auxiliary.get("NormalBase"),
                "Blend Material Base Normal", -760, -520,
                tiling=base_tiling,
            )
            material_base = _aux_image_node(
                nodes, links, texture_files, auxiliary.get("MaterialBase"),
                "Blend Material Base Mask", -760, -760,
                tiling=base_tiling,
            )
            color_blend = _aux_image_node(
                nodes, links, texture_files, auxiliary.get("ColorBlend"),
                "Blend Material Layer Color", -760, 450, non_color=False,
                tiling=blend_tiling, uv_map=blend_uv_map,
            )
            normal_blend = _aux_image_node(
                nodes, links, texture_files, auxiliary.get("NormalBlend"),
                "Blend Material Layer Normal", -760, -1000,
                tiling=blend_tiling, uv_map=blend_uv_map,
            )
            material_blend = _aux_image_node(
                nodes, links, texture_files, auxiliary.get("MaterialBlend"),
                "Blend Material Layer Mask", -760, -1240,
                tiling=blend_tiling, uv_map=blend_uv_map,
            )
            blend_mask = _aux_image_node(
                nodes, links, texture_files, auxiliary.get("BlendMask"),
                "Blend Material Mask", -760, 100, uv_map="UVMap_1",
            )
            blend_factor = None
            if blend_mask is not None:
                blend_mask_channels = _separate_node(
                    nodes, links, blend_mask.outputs["Color"],
                    "Blend material mask channels", -500, 100,
                )
                blend_amount = _math_node(
                    nodes, "MULTIPLY", "Blend material intensity", -300, 100,
                    _float_parameter(parameters.get("myBlendIntensity"), 1.0),
                )
                links.new(blend_mask_channels.outputs["Red"], blend_amount.inputs[0])
                blend_factor = blend_amount.outputs[0]
            if blend_factor is not None and color_base is not None and color_blend is not None:
                color_mix = nodes.new("ShaderNodeMixRGB")
                color_mix.label = "Blend base and layer color"
                color_mix.name = color_mix.label
                color_mix.location = (120, 600)
                links.new(blend_factor, color_mix.inputs[0])
                links.new(color_base.outputs["Color"], color_mix.inputs[1])
                links.new(color_blend.outputs["Color"], color_mix.inputs[2])

                tint = nodes.new("ShaderNodeRGB")
                tint.label = "Authored blend-material tint"
                tint.name = tint.label
                tint.location = (120, 820)
                tint.outputs[0].default_value = (
                    *_color_parameter(parameters.get("myTintOverlay"), (0.5,) * 3),
                    1.0,
                )
                neutral = nodes.new("ShaderNodeRGB")
                neutral.label = "Neutral overlay"
                neutral.outputs[0].default_value = (0.5, 0.5, 0.5, 1.0)
                tint_fade = nodes.new("ShaderNodeMixRGB")
                tint_fade.label = "Fade tint over blend layer"
                tint_fade.location = (320, 820)
                links.new(blend_factor, tint_fade.inputs[0])
                links.new(tint.outputs[0], tint_fade.inputs[1])
                links.new(neutral.outputs[0], tint_fade.inputs[2])
                tint_select = nodes.new("ShaderNodeMixRGB")
                tint_select.label = "Tint base only"
                tint_select.location = (500, 820)
                tint_select.inputs[0].default_value = _float_parameter(
                    parameters.get("myTintBaseOnly"), 1.0
                )
                links.new(tint.outputs[0], tint_select.inputs[1])
                links.new(tint_fade.outputs["Color"], tint_select.inputs[2])
                color_overlay = nodes.new("ShaderNodeMixRGB")
                color_overlay.blend_type = "OVERLAY"
                color_overlay.inputs[0].default_value = 1.0
                color_overlay.label = "Blend material OverlayNoMask"
                color_overlay.name = color_overlay.label
                color_overlay.location = (700, 600)
                links.new(color_mix.outputs["Color"], color_overlay.inputs[1])
                links.new(tint_select.outputs["Color"], color_overlay.inputs[2])
                base_color_override = color_overlay.outputs["Color"]
                if diffuse_node is None:
                    links.new(base_color_override, shader.inputs["Base Color"])

            base_material_channels = None
            blend_material_channels = None
            if material_base is not None:
                base_material_channels = _separate_node(
                    nodes, links, material_base.outputs["Color"],
                    "Base material channels", -300, -760,
                )
            if material_blend is not None:
                blend_material_channels = _separate_node(
                    nodes, links, material_blend.outputs["Color"],
                    "Layer material channels", -300, -1240,
                )
            if (
                blend_factor is not None
                and base_material_channels is not None
                and blend_material_channels is not None
            ):
                roughness_mix = nodes.new("ShaderNodeMix")
                roughness_mix.data_type = "FLOAT"
                roughness_mix.label = "Blend material roughness"
                roughness_mix.location = (1250, -900)
                links.new(blend_factor, roughness_mix.inputs[0])
                links.new(base_material_channels.outputs["Green"], roughness_mix.inputs[2])
                links.new(blend_material_channels.outputs["Green"], roughness_mix.inputs[3])
                metallic_mix = nodes.new("ShaderNodeMix")
                metallic_mix.data_type = "FLOAT"
                metallic_mix.label = "Blend material metalness"
                metallic_mix.location = (1250, -1060)
                links.new(blend_factor, metallic_mix.inputs[0])
                links.new(base_material_channels.outputs["Red"], metallic_mix.inputs[2])
                links.new(blend_material_channels.outputs["Red"], metallic_mix.inputs[3])
                for link in list(shader.inputs["Roughness"].links):
                    links.remove(link)
                for link in list(shader.inputs["Metallic"].links):
                    links.remove(link)
                links.new(roughness_mix.outputs["Result"], shader.inputs["Roughness"])
                links.new(metallic_mix.outputs["Result"], shader.inputs["Metallic"])

            detail_path = auxiliary.get("DetailNormal")
            detail = _aux_image_node(
                nodes, links, texture_files, detail_path,
                "Blend Material Detail Normal", -760, -1480,
                tiling=parameters.get("myDetailTiling", (4.0, 4.0)),
            )
            if (
                normal is None
                and normal_base is not None
                and normal_blend is not None
                and detail is not None
                and blend_factor is not None
                and base_material_channels is not None
            ):
                normal = nodes.new("ShaderNodeNormalMap")
                normal.label = "Blend material tangent normal"
                normal.name = normal.label
                normal.location = (1710, -55)
                normal.inputs["Strength"].default_value = 1.0
                links.new(normal.outputs["Normal"], shader.inputs["Normal"])
                material["afop_normal"] = (
                    f"{auxiliary.get('NormalBase')}|{auxiliary.get('NormalBlend')}"
                )
                material["afop_normal_strength"] = 1.0
            if (
                normal is not None
                and normal_base is not None
                and normal_blend is not None
                and detail is not None
                and blend_factor is not None
                and base_material_channels is not None
            ):
                normal_mix = nodes.new("ShaderNodeMixRGB")
                normal_mix.label = "Blend packed base/layer normals"
                normal_mix.location = (100, -1120)
                links.new(blend_factor, normal_mix.inputs[0])
                links.new(normal_base.outputs["Color"], normal_mix.inputs[1])
                links.new(normal_blend.outputs["Color"], normal_mix.inputs[2])
                normal_x_mix = nodes.new("ShaderNodeMixRGB")
                normal_x_mix.label = "Blend packed normal X"
                normal_x_mix.location = (100, -1300)
                links.new(blend_factor, normal_x_mix.inputs[0])
                links.new(normal_base.outputs["Alpha"], normal_x_mix.inputs[1])
                links.new(normal_blend.outputs["Alpha"], normal_x_mix.inputs[2])
                layer = _packed_detail_layer(
                    nodes, links, normal_mix.outputs["Color"],
                    normal_x_mix.outputs["Color"], detail, detail.outputs["Alpha"],
                    base_material_channels.outputs["Blue"],
                    _float_parameter(parameters.get("myDetailIntensity"), 1.0),
                    "Blend material detail normal (RNM)", 1480, -300,
                )
                for link in list(normal.inputs["Color"].links):
                    links.remove(link)
                links.new(layer.outputs["Normal Color"], normal.inputs["Color"])
                ao_output = layer.outputs["AO"]
                material["afop_detail_normal"] = detail_path
            material["afop_shader_profile"] = "basic_blend_static"
            material["afop_profile_limit"] = (
                "Packed base/layer normal interpolation is a static tangent-space preview"
            )

        has_specialized_detail = (
            is_basic_emissive
            or is_rustymetal
            or is_navi_skin
            or is_wildlife_skin
            or is_medusa_skin
            or is_vegetation
            or is_basic_blend
            or is_wildlife_gear
        )
        generic_detail_path = (
            binding.get("aux", {}).get("DetailNormal")
            or binding.get("aux", {}).get("DetailNormalMap")
            or binding.get("aux", {}).get("DetailSampler")
        )
        if (
            not has_specialized_detail
            and generic_detail_path
            and normal_texture_node is not None
            and normal is not None
        ):
            generic_tiling = _tiling_parameter(
                parameters.get(
                    "myDetailTiling",
                    parameters.get("myDetailNormalTiling", (1.0, 1.0)),
                )
            )
            generic_detail = _aux_image_node(
                nodes, links, texture_files, generic_detail_path,
                "Snowdrop Detail Normal (packed)", -520, -760,
                tiling=generic_tiling,
            )
            if generic_detail is not None:
                if mask_separate is not None:
                    generic_mask = mask_separate.outputs["Blue"]
                else:
                    full_mask = nodes.new("ShaderNodeValue")
                    full_mask.label = "Detail normal enabled"
                    full_mask.outputs[0].default_value = 1.0
                    generic_mask = full_mask.outputs[0]
                generic_layer = _packed_detail_layer(
                    nodes, links,
                    normal_texture_node.outputs["Color"],
                    normal_texture_node.outputs["Alpha"],
                    generic_detail, generic_detail.outputs["Alpha"],
                    generic_mask,
                    _float_parameter(
                        parameters.get(
                            "myDetailStrength",
                            parameters.get("myDetailIntensity", 1.0),
                        ), 1.0,
                    ),
                    "Snowdrop detail normal (RNM)", 1450, -300,
                )
                for link in list(normal.inputs["Color"].links):
                    links.remove(link)
                links.new(generic_layer.outputs["Normal Color"], normal.inputs["Color"])
                ao_output = generic_layer.outputs["AO"]
                material["afop_detail_normal"] = generic_detail_path
                material["afop_detail_normal_tiling"] = list(generic_tiling)
                if "afop_shader_profile" not in material:
                    material["afop_shader_profile"] = "generic_detail_rnm"

        generic_emission_path = (
            binding.get("aux", {}).get("Emission")
            or binding.get("aux", {}).get("Emissive")
        )
        if not is_basic_emissive and generic_emission_path:
            generic_emission = _aux_image_node(
                nodes, links, texture_files, generic_emission_path,
                "Snowdrop Emission", -520, -1440, non_color=False,
            )
            if generic_emission is not None:
                emission_input = (
                    shader.inputs.get("Emission Color") or shader.inputs.get("Emission")
                )
                if emission_input is not None:
                    links.new(generic_emission.outputs["Color"], emission_input)
                emission_strength = 1.0
                for parameter_name in (
                    "myEmissionScale", "myEmissiveIntensity",
                    "myEmissionIntensity", "myEmissiveStrength",
                ):
                    value = parameters.get(parameter_name)
                    if isinstance(value, (int, float)):
                        emission_strength = max(0.0, float(value))
                        break
                strength_input = shader.inputs.get("Emission Strength")
                if strength_input is not None:
                    strength_input.default_value = emission_strength
                material["afop_emission"] = generic_emission_path
                material["afop_emission_scale"] = emission_strength
                if "afop_shader_profile" not in material:
                    material["afop_shader_profile"] = "generic_emission"

        emission_path = binding.get("aux", {}).get("Emission")
        if (
            is_basic_emissive
            and emission_path
            and emission_path.casefold() in texture_files
        ):
            disk_path = texture_files[emission_path.casefold()]
            image = _load_image(disk_path, emission_path, non_color=False)
            emission_texture = _new_image_node(
                nodes, image, "Emission", -440, -980
            )
            if surface_uv_output is not None:
                links.new(surface_uv_output, emission_texture.inputs["Vector"])
            overlay_value = parameters.get(
                "myEmissiveOverlay", (0.5, 0.5, 0.5)
            )
            if not (
                isinstance(overlay_value, (list, tuple))
                and len(overlay_value) >= 3
                and all(
                    isinstance(component, (int, float))
                    for component in overlay_value[:3]
                )
            ):
                overlay_value = (0.5, 0.5, 0.5)
            overlay_color = nodes.new("ShaderNodeRGB")
            overlay_color.label = "Authored emissive overlay"
            overlay_color.name = overlay_color.label
            overlay_color.location = (-195, -1080)
            overlay_color.outputs[0].default_value = (
                float(overlay_value[0]),
                float(overlay_value[1]),
                float(overlay_value[2]),
                1.0,
            )
            overlay = nodes.new("ShaderNodeMixRGB")
            overlay.blend_type = "OVERLAY"
            overlay.inputs[0].default_value = 1.0
            overlay.label = "Snowdrop OverlayNoMask"
            overlay.name = overlay.label
            overlay.location = (55, -980)
            links.new(emission_texture.outputs["Color"], overlay.inputs[1])
            links.new(overlay_color.outputs[0], overlay.inputs[2])
            emission_input = (
                shader.inputs.get("Emission Color")
                or shader.inputs.get("Emission")
            )
            if emission_input is not None:
                links.new(overlay.outputs["Color"], emission_input)
            emission_scale = parameters.get("myEmissionScale", 1.0)
            if not isinstance(emission_scale, (int, float)):
                emission_scale = 1.0
            emission_scale = max(0.0, float(emission_scale))
            strength_input = shader.inputs.get("Emission Strength")
            if strength_input is not None:
                strength_input.default_value = emission_scale
            material["afop_emission"] = emission_path
            material["afop_emissive_overlay"] = [
                float(overlay_value[0]),
                float(overlay_value[1]),
                float(overlay_value[2]),
            ]
            material["afop_emission_scale"] = emission_scale

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
            if base_color_override is not None:
                links.new(base_color_override, shader.inputs["Base Color"])
            elif base_ao_output is not None:
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

        alpha_output = profile_alpha_output
        alpha_source = profile_alpha_source
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
        elif alpha_output is None and diffuse_node is not None and (
            is_wildlife_skin
            or is_dragonfly_wing
            or is_rustymetal
            or is_moss_card
            or mgraph.diffuse_drives_surface_alpha(shader_name)
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
            elif is_moss_card and mask_separate is not None:
                moss_opacity = mask_separate.outputs["Red"]
                if shader_name != "px_mosscard_ground.mshader":
                    moss_opacity_scale = _math_node(
                        nodes, "MULTIPLY", "Moss material opacity x 0.65",
                        55, -430, 0.65,
                    )
                    moss_opacity_bias = _math_node(
                        nodes, "ADD", "Moss material opacity + 0.35",
                        235, -430, 0.35,
                    )
                    links.new(
                        mask_separate.outputs["Red"],
                        moss_opacity_scale.inputs[0],
                    )
                    links.new(
                        moss_opacity_scale.outputs[0], moss_opacity_bias.inputs[0]
                    )
                    moss_opacity = moss_opacity_bias.outputs[0]
                moss_alpha = _math_node(
                    nodes, "MULTIPLY", "Moss cutout x material opacity",
                    415, -430,
                )
                links.new(alpha_output, moss_alpha.inputs[0])
                links.new(moss_opacity, moss_alpha.inputs[1])
                alpha_output = moss_alpha.outputs[0]
                alpha_source = "diffuse+material-opacity"
            elif is_wildlife_gear and mask_separate is not None:
                gear_opacity_scale = _math_node(
                    nodes, "MULTIPLY", "Gear material opacity x 0.5",
                    55, -430, 0.5,
                )
                gear_opacity_bias = _math_node(
                    nodes, "ADD", "Gear material opacity + 0.5",
                    235, -430, 0.5,
                )
                gear_alpha = _math_node(
                    nodes, "MULTIPLY", "Gear cutout x material opacity",
                    415, -430,
                )
                links.new(mask_separate.outputs["Red"], gear_opacity_scale.inputs[0])
                links.new(gear_opacity_scale.outputs[0], gear_opacity_bias.inputs[0])
                links.new(alpha_output, gear_alpha.inputs[0])
                links.new(gear_opacity_bias.outputs[0], gear_alpha.inputs[1])
                alpha_output = gear_alpha.outputs[0]
                alpha_source = "diffuse+material-opacity"
            elif is_constants:
                constant_opacity = _math_node(
                    nodes, "MULTIPLY", "Constants transmission opacity",
                    235, -430,
                    max(
                        0.0,
                        min(
                            1.0,
                            _float_parameter(
                                parameters.get("myTransmissionOpacity"), 1.0
                            ),
                        ),
                    ),
                )
                links.new(alpha_output, constant_opacity.inputs[0])
                alpha_output = constant_opacity.outputs[0]
                alpha_source = "diffuse+transmission-opacity"
        if alpha_output is not None:
            links.new(alpha_output, shader.inputs["Alpha"])
            _enable_alpha_blending(material)
            material["afop_alpha_source"] = alpha_source

        obj.data.materials.clear()
        obj.data.materials.append(material)
        assigned += 1
    return assigned

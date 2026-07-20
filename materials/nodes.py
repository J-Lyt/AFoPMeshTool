"""Reusable Blender nodes and packed-texture helpers for material profiles."""

from __future__ import annotations

import bpy

from .textures import _load_image


_PACKED_DETAIL_GROUP = "AFOP Packed Detail Normal RNM v2"


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


def _clamp_node(nodes, label, x, y, minimum=0.0, maximum=1.0):
    node = nodes.new("ShaderNodeClamp")
    node.label = label
    node.name = label
    node.location = (x, y)
    node.inputs["Min"].default_value = float(minimum)
    node.inputs["Max"].default_value = float(maximum)
    return node

def _vector2_parameter(value, default):
    if (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and all(isinstance(component, (int, float)) for component in value[:2])
    ):
        return float(value[0]), float(value[1])
    return float(default[0]), float(default[1])


def _tiling_parameter(value, default=(1.0, 1.0)):
    if isinstance(value, (int, float)):
        return float(value), float(value)
    return _vector2_parameter(value, default)


def _float_parameter(value, default):
    return float(value) if isinstance(value, (int, float)) else float(default)


def _color_parameter(value, default=(0.0, 0.0, 0.0)):
    if (
        isinstance(value, (list, tuple))
        and len(value) >= 3
        and all(isinstance(component, (int, float)) for component in value[:3])
    ):
        return tuple(float(component) for component in value[:3])
    return tuple(float(component) for component in default[:3])


def _aux_image_node(
    nodes, links, texture_files, logical_path, label, x, y, *, non_color=True,
    tiling=None, offset=None, uv_map="UVMap_0", rotation=0.0,
):
    """Load an auxiliary texture and apply its authored UV transform."""
    if not logical_path or logical_path.casefold() not in texture_files:
        return None
    image = _load_image(
        texture_files[logical_path.casefold()], logical_path, non_color=non_color
    )
    node = _new_image_node(nodes, image, label, x, y)
    if tiling is None and offset is None and not rotation:
        return node

    uv = nodes.new("ShaderNodeUVMap")
    uv.uv_map = uv_map
    uv.label = f"{label} UV ({uv_map})"
    uv.name = uv.label
    uv.location = (x - 620, y)
    vector_output = uv.outputs["UV"]
    if offset is not None:
        add = nodes.new("ShaderNodeVectorMath")
        add.operation = "ADD"
        add.label = f"{label} UV offset"
        add.name = add.label
        add.location = (x - 460, y)
        add.inputs[1].default_value = (*_vector2_parameter(offset, (0.0, 0.0)), 0.0)
        links.new(vector_output, add.inputs[0])
        vector_output = add.outputs["Vector"]
    if tiling is not None:
        scale = nodes.new("ShaderNodeVectorMath")
        scale.operation = "MULTIPLY"
        scale.label = f"{label} UV tiling"
        scale.name = scale.label
        scale.location = (x - 300, y)
        scale.inputs[1].default_value = (*_vector2_parameter(tiling, (1.0, 1.0)), 1.0)
        links.new(vector_output, scale.inputs[0])
        vector_output = scale.outputs["Vector"]
    if rotation:
        rotate = nodes.new("ShaderNodeVectorRotate")
        rotate.rotation_type = "Z_AXIS"
        rotate.label = f"{label} UV rotation"
        rotate.name = rotate.label
        rotate.location = (x - 140, y)
        rotate.inputs["Angle"].default_value = float(rotation)
        links.new(vector_output, rotate.inputs["Vector"])
        vector_output = rotate.outputs["Vector"]
    links.new(vector_output, node.inputs["Vector"])
    return node


def _separate_node(nodes, links, color_socket, label, x, y):
    separate = nodes.new("ShaderNodeSeparateColor")
    separate.label = label
    separate.name = label
    separate.location = (x, y)
    links.new(color_socket, separate.inputs["Color"])
    return separate


def _packed_detail_layer(
    nodes, links, base_color, base_x, detail_node, detail_x, mask_socket,
    strength, label, x, y,
):
    group = nodes.new("ShaderNodeGroup")
    group.node_tree = _packed_detail_normal_group()
    group.label = label
    group.name = label
    group.location = (x, y)
    group.inputs["Strength"].default_value = max(0.0, float(strength))
    links.new(base_color, group.inputs["Base Color"])
    links.new(base_x, group.inputs["Base Alpha"])
    links.new(detail_node.outputs["Color"], group.inputs["Detail Color"])
    links.new(detail_x, group.inputs["Detail Alpha"])
    links.new(mask_socket, group.inputs["Mask"])
    return group


def _five_color_ramp(nodes, links, factor, colors, label, x, y):
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.label = label
    ramp.name = label
    ramp.location = (x, y)
    ramp.color_ramp.interpolation = "LINEAR"
    elements = ramp.color_ramp.elements
    elements[0].position = 0.0
    elements[0].color = (*colors[0], 1.0)
    elements[-1].position = 1.0
    elements[-1].color = (*colors[-1], 1.0)
    for position, color in zip((0.25, 0.5, 0.75), colors[1:-1]):
        element = elements.new(position)
        element.color = (*color, 1.0)
    links.new(factor, ramp.inputs["Fac"])
    return ramp.outputs["Color"]


def _pattern_component(nodes, links, source, level, invert, label, x, y):
    """Snowdrop wildlife pattern threshold with signed inversion."""
    upper = float(level) * 0.25
    mapping = nodes.new("ShaderNodeMapRange")
    mapping.label = label
    mapping.name = label
    mapping.location = (x, y)
    mapping.clamp = True
    mapping.interpolation_type = "SMOOTHSTEP"
    mapping.inputs["From Min"].default_value = upper - 0.25
    mapping.inputs["From Max"].default_value = upper
    mapping.inputs["To Min"].default_value = 0.0
    mapping.inputs["To Max"].default_value = 1.0
    links.new(source, mapping.inputs["Value"])
    multiply = _math_node(nodes, "MULTIPLY", f"{label} signed", x + 180, y)
    multiply.inputs[1].default_value = float(invert)
    links.new(mapping.outputs["Result"], multiply.inputs[0])
    bias = _math_node(
        nodes, "SUBTRACT", f"{label} inversion bias", x + 350, y,
        min(0.0, float(invert)),
    )
    links.new(multiply.outputs[0], bias.inputs[0])
    return bias.outputs[0]


def _packed_detail_normal_group():
    """Build the tangent-space RNM combiner used by supported profiles."""
    existing = bpy.data.node_groups.get(_PACKED_DETAIL_GROUP)
    if existing is not None:
        return existing

    tree = bpy.data.node_groups.new(_PACKED_DETAIL_GROUP, "ShaderNodeTree")
    for name, socket_type in (
        ("Base Color", "NodeSocketColor"),
        ("Base Alpha", "NodeSocketFloat"),
        ("Detail Color", "NodeSocketColor"),
        ("Detail Alpha", "NodeSocketFloat"),
        ("Mask", "NodeSocketFloat"),
        ("Strength", "NodeSocketFloat"),
    ):
        tree.interface.new_socket(
            name=name, in_out="INPUT", socket_type=socket_type
        )
    tree.interface.new_socket(
        name="Normal Color", in_out="OUTPUT", socket_type="NodeSocketColor"
    )
    tree.interface.new_socket(
        name="AO", in_out="OUTPUT", socket_type="NodeSocketFloat"
    )
    inputs = tree.nodes.new("NodeGroupInput")
    inputs.location = (-1250, 0)
    output = tree.nodes.new("NodeGroupOutput")
    output.location = (1250, 0)

    def packed_normal(prefix, color_socket, alpha_socket, y):
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        separate.label = f"{prefix} packed channels"
        separate.location = (-1050, y)
        tree.links.new(color_socket, separate.inputs["Color"])

        offset = 2.0 * 128.0 / 255.0
        x_scale = _math_node(
            tree.nodes, "MULTIPLY", f"{prefix} X x2", -850, y + 80, 2.0
        )
        x_signed = _math_node(
            tree.nodes, "SUBTRACT", f"{prefix} X signed", -680, y + 80, offset
        )
        y_scale = _math_node(
            tree.nodes, "MULTIPLY", f"{prefix} Y x2", -850, y - 20, 2.0
        )
        y_signed = _math_node(
            tree.nodes, "SUBTRACT", f"{prefix} Y signed", -680, y - 20, offset
        )
        tree.links.new(alpha_socket, x_scale.inputs[0])
        tree.links.new(x_scale.outputs[0], x_signed.inputs[0])
        tree.links.new(separate.outputs["Green"], y_scale.inputs[0])
        tree.links.new(y_scale.outputs[0], y_signed.inputs[0])

        x_sq = _math_node(
            tree.nodes, "MULTIPLY", f"{prefix} X squared", -510, y + 80
        )
        y_sq = _math_node(
            tree.nodes, "MULTIPLY", f"{prefix} Y squared", -510, y - 20
        )
        tree.links.new(x_signed.outputs[0], x_sq.inputs[0])
        tree.links.new(x_signed.outputs[0], x_sq.inputs[1])
        tree.links.new(y_signed.outputs[0], y_sq.inputs[0])
        tree.links.new(y_signed.outputs[0], y_sq.inputs[1])
        sum_sq = _math_node(
            tree.nodes, "ADD", f"{prefix} XY squared", -340, y + 30
        )
        tree.links.new(x_sq.outputs[0], sum_sq.inputs[0])
        tree.links.new(y_sq.outputs[0], sum_sq.inputs[1])
        remaining = _math_node(
            tree.nodes, "SUBTRACT", f"{prefix} Z squared", -170, y + 30
        )
        remaining.inputs[0].default_value = 1.0
        tree.links.new(sum_sq.outputs[0], remaining.inputs[1])
        clamp = _math_node(
            tree.nodes, "MAXIMUM", f"{prefix} clamp Z", 0, y + 30, 0.0
        )
        root = _math_node(tree.nodes, "SQRT", f"{prefix} reconstruct Z", 170, y + 30)
        tree.links.new(remaining.outputs[0], clamp.inputs[0])
        tree.links.new(clamp.outputs[0], root.inputs[0])
        z_half = _math_node(
            tree.nodes, "MULTIPLY", f"{prefix} encode Z", 340, y + 30, 0.5
        )
        z_encoded = _math_node(
            tree.nodes, "ADD", f"{prefix} encode Z + 0.5", 510, y + 30, 0.5
        )
        tree.links.new(root.outputs[0], z_half.inputs[0])
        tree.links.new(z_half.outputs[0], z_encoded.inputs[0])
        combine = tree.nodes.new("ShaderNodeCombineColor")
        combine.label = f"{prefix} tangent normal"
        combine.location = (680, y + 30)
        tree.links.new(alpha_socket, combine.inputs["Red"])
        tree.links.new(separate.outputs["Green"], combine.inputs["Green"])
        tree.links.new(z_encoded.outputs[0], combine.inputs["Blue"])
        return combine.outputs["Color"], separate.outputs["Blue"]

    base, base_ao = packed_normal(
        "Base", inputs.outputs["Base Color"], inputs.outputs["Base Alpha"], 300
    )
    detail, detail_ao = packed_normal(
        "Detail", inputs.outputs["Detail Color"], inputs.outputs["Detail Alpha"], -300
    )

    factor = _math_node(tree.nodes, "MULTIPLY", "Detail strength x mask", -250, -520)
    tree.links.new(inputs.outputs["Mask"], factor.inputs[0])
    tree.links.new(inputs.outputs["Strength"], factor.inputs[1])
    clamp_low = _math_node(tree.nodes, "MAXIMUM", "Clamp detail strength low", -80, -520, 0.0)
    clamp_high = _math_node(tree.nodes, "MINIMUM", "Clamp detail strength high", 90, -520, 1.0)
    tree.links.new(factor.outputs[0], clamp_low.inputs[0])
    tree.links.new(clamp_low.outputs[0], clamp_high.inputs[0])
    flat = tree.nodes.new("ShaderNodeRGB")
    flat.label = "Flat tangent normal"
    packed_center = 128.0 / 255.0
    flat.outputs[0].default_value = (
        packed_center, packed_center, 1.0, 1.0
    )
    flat.location = (260, -570)
    weighted = tree.nodes.new("ShaderNodeMixRGB")
    weighted.blend_type = "MIX"
    weighted.label = "Apply authored detail strength"
    weighted.location = (470, -520)
    tree.links.new(clamp_high.outputs[0], weighted.inputs[0])
    tree.links.new(flat.outputs[0], weighted.inputs[1])
    tree.links.new(detail, weighted.inputs[2])

    def vector_node(operation, label, x, y):
        node = tree.nodes.new("ShaderNodeVectorMath")
        node.operation = operation
        node.label = label
        node.name = label
        node.location = (x, y)
        return node

    # Reoriented Normal Mapping, using the packed-normal form from
    # Self Shadow's RNM derivation. This preserves the base surface direction
    # while applying the fine Navi pore detail in tangent space.
    t_mul = vector_node("MULTIPLY", "RNM base x2", 850, 280)
    t_mul.inputs[1].default_value = (2.0, 2.0, 2.0)
    t_add = vector_node("ADD", "RNM base offset", 1020, 280)
    signed_offset = 2.0 * packed_center
    t_add.inputs[1].default_value = (-signed_offset, -signed_offset, 0.0)
    tree.links.new(base, t_mul.inputs[0])
    tree.links.new(t_mul.outputs["Vector"], t_add.inputs[0])

    u_mul = vector_node("MULTIPLY", "RNM detail signed", 850, -280)
    u_mul.inputs[1].default_value = (-2.0, -2.0, 2.0)
    u_add = vector_node("ADD", "RNM detail offset", 1020, -280)
    u_add.inputs[1].default_value = (signed_offset, signed_offset, -1.0)
    tree.links.new(weighted.outputs[0], u_mul.inputs[0])
    tree.links.new(u_mul.outputs["Vector"], u_add.inputs[0])

    dot = vector_node("DOT_PRODUCT", "RNM dot", 1190, 180)
    tree.links.new(t_add.outputs["Vector"], dot.inputs[0])
    tree.links.new(u_add.outputs["Vector"], dot.inputs[1])
    t_scale = vector_node("SCALE", "RNM base weighted", 1360, 180)
    tree.links.new(t_add.outputs["Vector"], t_scale.inputs[0])
    tree.links.new(dot.outputs["Value"], t_scale.inputs[3])
    separate_t = tree.nodes.new("ShaderNodeSeparateXYZ")
    separate_t.location = (1190, 20)
    tree.links.new(t_add.outputs["Vector"], separate_t.inputs[0])
    u_scale = vector_node("SCALE", "RNM detail weighted", 1360, -80)
    tree.links.new(u_add.outputs["Vector"], u_scale.inputs[0])
    tree.links.new(separate_t.outputs["Z"], u_scale.inputs[3])
    subtract = vector_node("SUBTRACT", "RNM combine", 1530, 80)
    tree.links.new(t_scale.outputs["Vector"], subtract.inputs[0])
    tree.links.new(u_scale.outputs["Vector"], subtract.inputs[1])
    normalize = vector_node("NORMALIZE", "RNM normalize", 1700, 80)
    tree.links.new(subtract.outputs["Vector"], normalize.inputs[0])
    encode_scale = vector_node("SCALE", "Encode RNM", 1870, 80)
    encode_scale.inputs[3].default_value = 0.5
    encode_add = vector_node("ADD", "Encode RNM + 0.5", 2040, 80)
    encode_add.inputs[1].default_value = (0.5, 0.5, 0.5)
    tree.links.new(normalize.outputs["Vector"], encode_scale.inputs[0])
    tree.links.new(encode_scale.outputs["Vector"], encode_add.inputs[0])
    tree.links.new(encode_add.outputs["Vector"], output.inputs["Normal Color"])

    minimum_ao = _math_node(tree.nodes, "MINIMUM", "Minimum packed AO", 850, -700)
    tree.links.new(base_ao, minimum_ao.inputs[0])
    tree.links.new(detail_ao, minimum_ao.inputs[1])
    ao_difference = _math_node(
        tree.nodes, "SUBTRACT", "Detail AO difference", 1020, -700
    )
    tree.links.new(minimum_ao.outputs[0], ao_difference.inputs[0])
    tree.links.new(base_ao, ao_difference.inputs[1])
    ao_scaled = _math_node(
        tree.nodes, "MULTIPLY", "Detail AO strength", 1190, -700
    )
    tree.links.new(ao_difference.outputs[0], ao_scaled.inputs[0])
    tree.links.new(clamp_high.outputs[0], ao_scaled.inputs[1])
    ao_result = _math_node(tree.nodes, "ADD", "Combined packed AO", 1360, -700)
    tree.links.new(base_ao, ao_result.inputs[0])
    tree.links.new(ao_scaled.outputs[0], ao_result.inputs[1])
    tree.links.new(ao_result.outputs[0], output.inputs["AO"])
    return tree


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

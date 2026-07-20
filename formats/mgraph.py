"""Focused reader for material references in AFOP graph and compound BV2 files.

The BV2 graph format is reflection-serialized, but its interned string pool is
plain NUL-terminated ASCII.  The graph's material names, shaders, MMB paths,
and texture constants can therefore be recovered without decoding unrelated
node data.  Material-to-texture assignment uses the same structural signals as
the reference extractor: texture-set grouping, material-name overlap, and a
small set of shader semantic hints.
"""

from __future__ import annotations

import os
import re
import struct


MAGIC = b"\xffBV2"
_DDS_RE = re.compile(r"\.dds$", re.IGNORECASE)
_SHADER_RE = re.compile(r"\.mshader$", re.IGNORECASE)
_DEFAULT_NORMAL = (
    "snowdrop/baked/snowdrop/dev/assets/textures/flat/"
    "sd_flat_normal_128_n.dds"
)
_PLAYER_HEAD_MESH_RE = re.compile(
    r"(?:^|[\\/])p_head_\d{2}_([fm])\.mmb$", re.IGNORECASE
)
_PLAYER_HEAD_SHADER = "blue/shaders/PX_Character_Workbench.mshader"
_PLAYER_HEAD_MALE_TEXTURE_ROOT = (
    "blue/baked/characterart/player/head/male/p_head_01_m/textures/"
)
_PLAYER_HEAD_FEMALE_DIFFUSE = (
    "blue/baked/characterart/NPC/RNF/default/head/female/"
    "rnf_head_01_f/textures/rnf_head_01_f_d.dds"
)

# Shader inputs that are dedicated *surface* opacity samplers. Several other
# inputs contain "alpha" in their name (for example Medusa InnerAlpha and
# hair DetailAlpha), but those drive internal shading effects and must not make
# the whole Blender material transparent.
_SURFACE_ALPHA_PINS = {
    # Value is the texture channel that carries opacity. Dedicated grayscale
    # samplers use Color; combined normal/alpha maps use their Alpha channel.
    "px_basic_mosspatch_interior.mshader": {113: "alpha"},
    "px_basicanimal_div2.mshader": {105: "color"},
    "px_character_skin_wildlife_halpha_wingfix.mshader": {207: "color"},
    # Gear fur packs its strand opacity in Color.G rather than texture alpha.
    "px_character_gear_fur.mshader": {120: "green"},
    "px_cloth_clanbanner_avr.mshader": {143: "color"},
    "px_creature_wing.mshader": {314: "color"},
    "px_decal_multipass_quadcolor.mshader": {101: "color"},
    "px_fx_mesh_scrolling_mud.mshader": {172: "color"},
    "px_fx_mesh_scrolling_water.mshader": {172: "color"},
    "px_fx_meshparticle_fluids.mshader": {143: "color"},
    "px_fx_water_foam.mshader": {106: "color"},
    "px_natural_moss.mshader": {113: "alpha"},
    "px_wl_haircard_testhyb.mshader": {105: "color"},
}

# Fallback used when the indexed .mshader declaration is unavailable. Normal
# imports supply authored pin roles dynamically; fur remains here because its
# Color texture is deliberately named *_fur_masks.dds and filename inference
# would otherwise mistake it for the Material sampler.
_DIRECT_TEXTURE_ROLE_PINS = {
    "px_character_gear_fur.mshader": {120: "d", 262: "m", 272: "n"},
}


# Minimal BV2 value decoder used for shader constants connected to material
# nodes. The string-pool scan remains the inexpensive path for textures; this
# decoder is invoked only when a shader needs numeric values that are not text.
_BV2_VECTOR_COUNTS = {2: 1, 10: 4, 12: 4, 14: 3, 16: 3, 18: 2, 20: 2}
_BV2_FLOAT_VECTORS = {10, 14, 18}
_BV2_GUIDS = {22: False, 23: False, 24: True, 25: True}
_BV2_SIGN_BITS = (6, 14, 30)


class _Bv2Token:
    __slots__ = ("u", "s")

    def __init__(self, tag, unsigned):
        self.u = unsigned
        bits = _BV2_SIGN_BITS[tag] if tag < 3 else 32
        self.s = unsigned - (1 << bits) if unsigned >= (1 << (bits - 1)) else unsigned


class _Bv2Struct:
    __slots__ = ("bare", "keys", "values")

    def __init__(self, bare, keys, values):
        self.bare = bare
        self.keys = keys
        self.values = values


class _Bv2Merged:
    __slots__ = ("slots", "value")

    def __init__(self, slots, value):
        self.slots = slots
        self.value = value


class _Bv2Mixed:
    __slots__ = ("items", "keyed")

    def __init__(self, items, keyed):
        self.items = items
        self.keyed = keyed


def _bv2_varint(data, offset):
    first = data[offset]
    tag = first & 3
    if tag == 0:
        return _Bv2Token(0, first >> 2), offset + 1
    if tag == 1:
        value = (first | (data[offset + 1] << 8)) >> 2
        return _Bv2Token(1, value), offset + 2
    if tag == 2:
        value = (
            first
            | (data[offset + 1] << 8)
            | (data[offset + 2] << 16)
            | (data[offset + 3] << 24)
        )
        return _Bv2Token(2, (value >> 2) & 0x3FFFFFFF), offset + 4
    value = (
        data[offset + 1]
        | (data[offset + 2] << 8)
        | (data[offset + 3] << 16)
        | (data[offset + 4] << 24)
    )
    return _Bv2Token(3, value), offset + 5


class _Bv2Decoder:
    def __init__(self, data):
        if data[:4] != MAGIC:
            raise ValueError("not a BV2 file")
        offset = 4
        _version, offset = _bv2_varint(data, offset)
        nstrings, offset = _bv2_varint(data, offset)
        nentities, offset = _bv2_varint(data, offset)
        _header_a, offset = _bv2_varint(data, offset)
        _body_tokens, offset = _bv2_varint(data, offset)
        _header_c, offset = _bv2_varint(data, offset)
        lengths = []
        for _ in range(nstrings.u):
            token, offset = _bv2_varint(data, offset)
            lengths.append(token.u >> 1)
        self.pool = []
        for length in lengths:
            if offset + length >= len(data) or data[offset + length] != 0:
                raise ValueError("invalid BV2 string pool")
            self.pool.append(data[offset:offset + length].decode("utf-8", "replace"))
            offset += length + 1
        tokens = []
        while offset < len(data):
            token, offset = _bv2_varint(data, offset)
            tokens.append(token)
        if len(tokens) < nentities.u:
            raise ValueError("truncated BV2 entity table")
        typecodes = [token.u for token in tokens[-nentities.u:]]
        stream = tokens[:-nentities.u]
        self.entities = {}
        cursor = 0
        for entity_index, typecode in enumerate(typecodes):
            entity_id = entity_index + 3
            if typecode == 0:
                bare_count = stream[cursor].u
                keyed_count = stream[cursor + 1].u
                cursor += 2
                bare = stream[cursor:cursor + bare_count]
                cursor += bare_count
                keys = stream[cursor:cursor + keyed_count]
                cursor += keyed_count
                values = stream[cursor:cursor + keyed_count]
                cursor += keyed_count
                self.entities[entity_id] = _Bv2Struct(bare, keys, values)
            elif typecode in _BV2_GUIDS:
                words = [stream[cursor + index].s & 0xFFFFFFFF for index in range(4)]
                cursor += 4
                self.entities[entity_id] = ("guid", words, _BV2_GUIDS[typecode])
            elif typecode == 6:
                self.entities[entity_id] = bool(stream[cursor].u)
                cursor += 1
            elif typecode == 4:
                low = stream[cursor].s & 0xFFFFFFFF
                high = stream[cursor + 1].s & 0xFFFFFFFF
                cursor += 2
                value = (high << 32) | low
                self.entities[entity_id] = value - (1 << 64) if value >= (1 << 63) else value
            else:
                count = _BV2_VECTOR_COUNTS.get(typecode)
                if count is None:
                    raise ValueError(f"unsupported BV2 typecode {typecode}")
                components = stream[cursor:cursor + count]
                cursor += count
                if typecode in _BV2_FLOAT_VECTORS:
                    values = [
                        struct.unpack("<f", struct.pack("<I", token.s & 0xFFFFFFFF))[0]
                        for token in components
                    ]
                else:
                    values = [token.s for token in components]
                self.entities[entity_id] = values[0] if count == 1 else values
        self.root_id = nentities.u + 2

    def _value(self, token, depth):
        kind = token.u & 3
        if kind == 0:
            return token.s >> 2
        if kind == 1:
            packed = token.u >> 2
            bits = (
                ((packed >> 29) & 1) << 31
                | ((packed & 0x1FFFFFFF) + 0x30000000) & 0x7FFFFFFF
            )
            return struct.unpack("<f", struct.pack("<I", bits))[0]
        if kind == 2:
            return self.pool[token.u >> 3]
        entity_id = token.u >> 2
        if entity_id == 0:
            return None
        if entity_id == 1:
            return True
        if entity_id == 2:
            return False
        return self._materialize(entity_id, depth + 1)

    def _key(self, token, depth):
        kind = token.u & 3
        if kind == 2:
            return self.pool[token.u >> 3]
        if kind == 0:
            return token.s >> 2
        if kind == 3:
            return self._materialize(token.u >> 2, depth + 1)
        raise ValueError("unsupported BV2 key")

    def _materialize(self, entity_id, depth=0):
        if depth > 500:
            raise ValueError("BV2 reference depth exceeded")
        entity = self.entities[entity_id]
        if not isinstance(entity, _Bv2Struct):
            if isinstance(entity, tuple) and entity and entity[0] == "guid":
                _kind, words, ampersand = entity
                word1, word0, word3, word2 = words
                return ("#&" if ampersand else "#") + "%08X%08X%08X%08X" % (
                    word0, word1, word2, word3
                )
            return entity
        items = []
        merged_slots = []
        for position, token in enumerate(entity.bare):
            if (token.u & 3) == 3 and (token.u >> 2) == 0:
                merged_slots.append(position + 1)
                continue
            value = self._value(token, depth)
            if merged_slots:
                items.append(_Bv2Merged(merged_slots, value))
                merged_slots = []
            else:
                items.append(value)
        if merged_slots:
            items.append(_Bv2Merged(merged_slots, None))
        keyed = {
            self._key(key, depth): self._value(value, depth)
            for key, value in zip(entity.keys, entity.values)
        }
        if not keyed and not any(isinstance(item, _Bv2Merged) for item in items):
            return items
        return _Bv2Mixed(items, keyed)

    def root(self):
        return self._materialize(self.root_id)


def _bv2_plain(value):
    if isinstance(value, _Bv2Mixed):
        result = {}
        position = 1
        for item in value.items:
            if isinstance(item, _Bv2Merged):
                plain = _bv2_plain(item.value)
                for slot in item.slots:
                    result[slot] = plain
                if item.slots:
                    position = max(position, max(item.slots) + 1)
            else:
                result[position] = _bv2_plain(item)
                position += 1
        result.update({key: _bv2_plain(item) for key, item in value.keyed.items()})
        return result
    if isinstance(value, dict):
        return {key: _bv2_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_bv2_plain(item) for item in value]
    return value


def _connected_material_constants(data, shader_prefixes, pin_ids):
    """Return connected native constants by material name and shader pin."""
    try:
        tree = _bv2_plain(_Bv2Decoder(data).root())
    except (IndexError, KeyError, TypeError, ValueError):
        return {}
    result = {}

    def visit(value):
        if isinstance(value, dict):
            nodes = value.get("nodesById")
            connections = value.get("connectionsById")
            if isinstance(nodes, dict) and isinstance(connections, (dict, list)):
                connection_values = (
                    list(connections.values()) if isinstance(connections, dict)
                    else connections
                )
                for node_id, node in nodes.items():
                    if not isinstance(node, dict):
                        continue
                    shader = os.path.basename(str(node.get("ShaderFile", ""))).casefold()
                    mesh_name = node.get("MeshName")
                    if not (
                        isinstance(mesh_name, str)
                        and any(shader.startswith(prefix) for prefix in shader_prefixes)
                    ):
                        continue
                    parameters = {}
                    for connection in connection_values:
                        if not (
                            isinstance(connection, list)
                            and len(connection) == 4
                            and connection[2] == node_id
                            and connection[3] - 10000 in pin_ids
                        ):
                            continue
                        source = nodes.get(connection[0])
                        if not isinstance(source, dict) or source.get("type") != "native:Constant":
                            continue
                        parameters[connection[3] - 10000] = source.get("value")
                    keys = {mesh_name.casefold(), mesh_name.rsplit("-", 1)[-1].casefold()}
                    for key in keys:
                        result.setdefault(key, parameters)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(tree)
    return result


def _connected_material_textures(data):
    """Return DDS constants connected directly to each decoded material node."""
    try:
        tree = _bv2_plain(_Bv2Decoder(data).root())
    except (IndexError, KeyError, TypeError, ValueError):
        return {}
    result = {}

    def visit(value):
        if isinstance(value, dict):
            nodes = value.get("nodesById")
            connections = value.get("connectionsById")
            if isinstance(nodes, dict) and isinstance(connections, (dict, list)):
                connection_values = (
                    list(connections.values()) if isinstance(connections, dict)
                    else connections
                )
                for node_id, node in nodes.items():
                    if not isinstance(node, dict):
                        continue
                    mesh_name = node.get("MeshName")
                    shader = node.get("ShaderFile")
                    if not (isinstance(mesh_name, str) and isinstance(shader, str)):
                        continue
                    paths = []
                    for connection in connection_values:
                        if not (
                            isinstance(connection, list)
                            and len(connection) == 4
                            and connection[2] == node_id
                        ):
                            continue
                        source = nodes.get(connection[0])
                        path = source.get("value") if isinstance(source, dict) else None
                        if (
                            isinstance(path, str)
                            and _DDS_RE.search(path)
                            and path.casefold() not in {item.casefold() for item in paths}
                        ):
                            paths.append(path)
                    if paths:
                        keys = {
                            mesh_name.casefold(),
                            mesh_name.rsplit("-", 1)[-1].casefold(),
                        }
                        for key in keys:
                            result.setdefault(key, paths)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(tree)
    return result


def _connected_material_alpha_textures(data):
    """Return dedicated surface-opacity textures connected to material nodes."""
    try:
        tree = _bv2_plain(_Bv2Decoder(data).root())
    except (IndexError, KeyError, TypeError, ValueError):
        return {}
    result = {}

    def first_dds(value):
        if isinstance(value, str):
            return value if _DDS_RE.search(value) else None
        if isinstance(value, dict):
            direct = value.get("value")
            if isinstance(direct, str) and _DDS_RE.search(direct):
                return direct
            for item in value.values():
                found = first_dds(item)
                if found:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = first_dds(item)
                if found:
                    return found
        return None

    def visit(value):
        if isinstance(value, dict):
            nodes = value.get("nodesById")
            connections = value.get("connectionsById")
            if isinstance(nodes, dict) and isinstance(connections, (dict, list)):
                connection_values = (
                    list(connections.values()) if isinstance(connections, dict)
                    else connections
                )
                for node_id, node in nodes.items():
                    if not isinstance(node, dict):
                        continue
                    mesh_name = node.get("MeshName")
                    shader = node.get("ShaderFile")
                    if not (isinstance(mesh_name, str) and isinstance(shader, str)):
                        continue
                    alpha_pins = _SURFACE_ALPHA_PINS.get(
                        os.path.basename(shader).casefold()
                    )
                    if not alpha_pins:
                        continue
                    for connection in connection_values:
                        if not (
                            isinstance(connection, list)
                            and len(connection) == 4
                            and connection[2] == node_id
                            # Texture sampler pin N is serialized at 9000 + N;
                            # numeric shader constants use the separate 10000
                            # range handled by _connected_material_constants.
                            and connection[3] - 9000 in alpha_pins
                        ):
                            continue
                        source = nodes.get(connection[0])
                        path = first_dds(source)
                        if path:
                            keys = {
                                mesh_name.casefold(),
                                mesh_name.rsplit("-", 1)[-1].casefold(),
                            }
                            channel = alpha_pins[connection[3] - 9000]
                            for key in keys:
                                result.setdefault(key, (path, channel))
                            break
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(tree)
    return result


def _connected_material_role_textures(data, shader_role_pins=None):
    """Return exact D/N/M bindings using authored shader sampler pins."""
    try:
        tree = _bv2_plain(_Bv2Decoder(data).root())
    except (IndexError, KeyError, TypeError, ValueError):
        return {}
    result = {}

    def visit(value):
        if isinstance(value, dict):
            nodes = value.get("nodesById")
            connections = value.get("connectionsById")
            if isinstance(nodes, dict) and isinstance(connections, (dict, list)):
                connection_values = (
                    list(connections.values()) if isinstance(connections, dict)
                    else connections
                )
                for node_id, node in nodes.items():
                    if not isinstance(node, dict):
                        continue
                    mesh_name = node.get("MeshName")
                    shader = node.get("ShaderFile")
                    if not (isinstance(mesh_name, str) and isinstance(shader, str)):
                        continue
                    shader_key = shader.replace("\\", "/").lstrip("/").casefold()
                    shader_name = os.path.basename(shader_key)
                    supplied_pins = (shader_role_pins or {}).get(shader_key)
                    if supplied_pins is None:
                        supplied_pins = (shader_role_pins or {}).get(shader_name)
                    role_pins = (
                        dict(supplied_pins)
                        if supplied_pins is not None
                        else dict(_DIRECT_TEXTURE_ROLE_PINS.get(shader_name, {}))
                    )
                    if not role_pins:
                        continue
                    roles = {}
                    for connection in connection_values:
                        if not (
                            isinstance(connection, list)
                            and len(connection) == 4
                            and connection[2] == node_id
                            and connection[3] - 9000 in role_pins
                        ):
                            continue
                        source = nodes.get(connection[0])
                        path = source.get("value") if isinstance(source, dict) else None
                        if isinstance(path, str) and _DDS_RE.search(path):
                            roles.setdefault(role_pins[connection[3] - 9000], path)
                    if roles:
                        keys = {
                            mesh_name.casefold(),
                            mesh_name.rsplit("-", 1)[-1].casefold(),
                        }
                        for key in keys:
                            result.setdefault(key, roles)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(tree)
    return result


def _compound_instance_texture_inputs(data):
    """Return texture values connected to exposed pins of compound instances."""
    try:
        tree = _bv2_plain(_Bv2Decoder(data).root())
    except (IndexError, KeyError, TypeError, ValueError):
        return {}
    result = {}

    def visit(value):
        if isinstance(value, dict):
            nodes = value.get("nodesById")
            connections = value.get("connectionsById")
            if isinstance(nodes, dict) and isinstance(connections, (dict, list)):
                connection_values = (
                    list(connections.values()) if isinstance(connections, dict)
                    else connections
                )
                for node_id, node in nodes.items():
                    if not isinstance(node, dict) or node.get("type") != "internal:Compound":
                        continue
                    filename = node.get("filename")
                    if not isinstance(filename, str):
                        continue
                    key = filename.replace("\\", "/").lstrip("/").casefold()
                    pins = result.setdefault(key, {})
                    for connection in connection_values:
                        if not (
                            isinstance(connection, list)
                            and len(connection) == 4
                            and connection[2] == node_id
                        ):
                            continue
                        source = nodes.get(connection[0])
                        path = source.get("value") if isinstance(source, dict) else None
                        if isinstance(path, str) and _DDS_RE.search(path):
                            pins.setdefault(connection[3], path)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(tree)
    return result


def _forwarded_compound_role_textures(
    data, compound_sources, shader_role_pins=None,
):
    """Resolve parent graph texture inputs through a compound boundary.

    A compound's ``internal:CompoundInputs`` output pin is the same pin used on
    its parent graph's ``internal:Compound`` instance. Following both halves is
    necessary for materials whose diffuse is deliberately supplied by the
    placement graph rather than serialized inside the compound itself.
    """
    instance_inputs = _compound_instance_texture_inputs(data)
    if not instance_inputs:
        return {}
    result = {}

    for logical_path, compound_data in (compound_sources or {}).items():
        key = logical_path.replace("\\", "/").lstrip("/").casefold()
        forwarded = instance_inputs.get(key)
        if not forwarded:
            continue
        try:
            tree = _bv2_plain(_Bv2Decoder(compound_data).root())
        except (IndexError, KeyError, TypeError, ValueError):
            continue

        def visit(value):
            if isinstance(value, dict):
                nodes = value.get("nodesById")
                connections = value.get("connectionsById")
                if isinstance(nodes, dict) and isinstance(connections, (dict, list)):
                    connection_values = (
                        list(connections.values())
                        if isinstance(connections, dict) else connections
                    )
                    for node_id, node in nodes.items():
                        if not isinstance(node, dict):
                            continue
                        mesh_name = node.get("MeshName")
                        shader = node.get("ShaderFile")
                        if not (isinstance(mesh_name, str) and isinstance(shader, str)):
                            continue
                        shader_key = shader.replace("\\", "/").lstrip("/").casefold()
                        shader_name = os.path.basename(shader_key)
                        role_pins = (shader_role_pins or {}).get(shader_key)
                        if role_pins is None:
                            role_pins = (shader_role_pins or {}).get(shader_name)
                        if role_pins is None:
                            role_pins = _DIRECT_TEXTURE_ROLE_PINS.get(shader_name, {})
                        roles = {}
                        for connection in connection_values:
                            if not (
                                isinstance(connection, list)
                                and len(connection) == 4
                                and connection[2] == node_id
                            ):
                                continue
                            shader_pin = connection[3] - 9000
                            role = role_pins.get(shader_pin)
                            source = nodes.get(connection[0])
                            if (
                                role is None
                                or not isinstance(source, dict)
                                or source.get("type") != "internal:CompoundInputs"
                            ):
                                continue
                            path = forwarded.get(connection[1])
                            if path:
                                roles[role] = path
                        if roles:
                            keys = {
                                mesh_name.casefold(),
                                mesh_name.rsplit("-", 1)[-1].casefold(),
                            }
                            for material_key in keys:
                                result.setdefault(material_key, {}).update(roles)
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(tree)
    return result


def _connected_material_parameters(data, shader_parameter_pins=None):
    """Return authored native constants using dynamically parsed shader pins."""
    if not shader_parameter_pins:
        return {}
    try:
        tree = _bv2_plain(_Bv2Decoder(data).root())
    except (IndexError, KeyError, TypeError, ValueError):
        return {}
    result = {}

    def visit(value):
        if isinstance(value, dict):
            nodes = value.get("nodesById")
            connections = value.get("connectionsById")
            if isinstance(nodes, dict) and isinstance(connections, (dict, list)):
                connection_values = (
                    list(connections.values()) if isinstance(connections, dict)
                    else connections
                )
                for node_id, node in nodes.items():
                    if not isinstance(node, dict):
                        continue
                    mesh_name = node.get("MeshName")
                    shader = node.get("ShaderFile")
                    if not (isinstance(mesh_name, str) and isinstance(shader, str)):
                        continue
                    shader_key = shader.replace("\\", "/").lstrip("/").casefold()
                    shader_name = os.path.basename(shader_key)
                    pin_fields = shader_parameter_pins.get(shader_key)
                    if pin_fields is None:
                        pin_fields = shader_parameter_pins.get(shader_name)
                    if not pin_fields:
                        continue
                    parameters = {}
                    for connection in connection_values:
                        if not (
                            isinstance(connection, list)
                            and len(connection) == 4
                            and connection[2] == node_id
                            and connection[3] - 10000 in pin_fields
                        ):
                            continue
                        source = nodes.get(connection[0])
                        if not (
                            isinstance(source, dict)
                            and source.get("type") == "native:Constant"
                        ):
                            continue
                        field = pin_fields[connection[3] - 10000]
                        parameters.setdefault(field, source.get("value"))
                    if parameters:
                        keys = {
                            mesh_name.casefold(),
                            mesh_name.rsplit("-", 1)[-1].casefold(),
                        }
                        for key in keys:
                            result.setdefault(key, parameters)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(tree)
    return result


def _emissive_parameters(data):
    """Return exact color/strength constants for px_emissive_color materials."""
    constants = _connected_material_constants(
        data, ("px_emissive_color.mshader",), {101, 103}
    )
    result = {}
    for name, pins in constants.items():
        color = pins.get(101, (0.0, 0.0, 0.0))
        strength = pins.get(103, 1.0)
        if not (isinstance(color, list) and len(color) >= 3):
            color = (0.0, 0.0, 0.0)
        if not isinstance(strength, (int, float)):
            strength = 1.0
        result[name] = {
            "emissive_color": tuple(float(value) for value in color[:3]),
            "emissive_strength": float(strength),
        }
    return result


def _wildlife_bio_parameters(data):
    """Return the six-color nighttime bio palette for Wildlife Skin nodes."""
    pin_ids = {186, 187, 188, 205, 400}
    constants = _connected_material_constants(
        data, ("px_wildlife_skin",), pin_ids
    )
    result = {}
    for name, pins in constants.items():
        if not any(pin_id in pins for pin_id in pin_ids):
            continue
        palette = [(0.0, 0.0, 0.0)]
        for pin_id in (400, 186, 187, 188, 205):
            color = pins.get(pin_id, (0.0, 0.0, 0.0))
            if not (isinstance(color, list) and len(color) >= 3):
                color = (0.0, 0.0, 0.0)
            palette.append(tuple(float(value) for value in color[:3]))
        result[name] = {"bio_palette": tuple(palette)}
    return result


def _wildlife_compound_bio_interface(data):
    """Map exposed compound pins to Wildlife Skin Bio inputs per material."""
    try:
        tree = _bv2_plain(_Bv2Decoder(data).root())
    except (IndexError, KeyError, TypeError, ValueError):
        return {}
    bio_pins = {186, 187, 188, 205, 400}
    result = {}

    def visit(value):
        if isinstance(value, dict):
            nodes = value.get("nodesById")
            connections = value.get("connectionsById")
            if isinstance(nodes, dict) and isinstance(connections, (dict, list)):
                connection_values = (
                    list(connections.values()) if isinstance(connections, dict)
                    else connections
                )
                for node_id, node in nodes.items():
                    if not isinstance(node, dict):
                        continue
                    mesh_name = node.get("MeshName")
                    shader = str(node.get("ShaderFile", ""))
                    if not (
                        isinstance(mesh_name, str)
                        and _is_wildlife_bio_shader(shader)
                    ):
                        continue
                    exposed_to_shader = {}
                    for connection in connection_values:
                        if not (
                            isinstance(connection, list)
                            and len(connection) == 4
                            and connection[2] == node_id
                            and connection[3] - 10000 in bio_pins
                        ):
                            continue
                        source = nodes.get(connection[0])
                        source_type = (
                            str(source.get("type", ""))
                            if isinstance(source, dict) else ""
                        )
                        if "inputs" in source_type.casefold():
                            exposed_to_shader[connection[1]] = connection[3] - 10000
                    if set(exposed_to_shader.values()) == bio_pins:
                        keys = {
                            mesh_name.casefold(),
                            mesh_name.rsplit("-", 1)[-1].casefold(),
                        }
                        for key in keys:
                            result.setdefault(key, exposed_to_shader)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(tree)
    return result


def _forwarded_wildlife_bio_parameters(data, compound_sources=None):
    """Decode Wildlife Bio Colors forwarded through a wrapper compound."""
    try:
        tree = _bv2_plain(_Bv2Decoder(data).root())
    except (IndexError, KeyError, TypeError, ValueError):
        return {}

    # A graph object stores only the wrapper's exposed pin numbers; the pin
    # names and its internal material connections live in the referenced
    # mcompoundnode. These interfaces were decoded from those compounds.
    compound_interfaces = {
        "wildlife_graph_direhorse_compound.mcompoundnode": {
            "body": {155: 186, 156: 187, 157: 188, 158: 205, 170: 400},
        },
        "wildlife_graph_wild_banshee_compound.mcompoundnode": {
            "banshee_body": {
                119: 186, 120: 187, 121: 188, 122: 205, 123: 400,
            },
            "banshee_head": {
                124: 186, 125: 187, 126: 188, 127: 205, 128: 400,
            },
        },
    }
    for path, compound_data in (compound_sources or {}).items():
        interface = _wildlife_compound_bio_interface(compound_data)
        if interface:
            normalized = str(path).replace("\\", "/").rsplit("/", 1)[-1].casefold()
            compound_interfaces[normalized] = interface
    result = {}

    def visit(value):
        if isinstance(value, dict):
            nodes = value.get("nodesById")
            connections = value.get("connectionsById")
            if isinstance(nodes, dict) and isinstance(connections, (dict, list)):
                connection_values = (
                    list(connections.values()) if isinstance(connections, dict)
                    else connections
                )
                for node_id, node in nodes.items():
                    if not isinstance(node, dict) or node.get("type") != "internal:Compound":
                        continue
                    filename = str(node.get("filename", "")).replace("\\", "/")
                    interface = compound_interfaces.get(filename.rsplit("/", 1)[-1].casefold())
                    if interface is None:
                        continue
                    constants = {}
                    for connection in connection_values:
                        if not (
                            isinstance(connection, list)
                            and len(connection) == 4
                            and connection[2] == node_id
                        ):
                            continue
                        source = nodes.get(connection[0])
                        if isinstance(source, dict) and source.get("type") == "native:Constant":
                            constants[connection[3]] = source.get("value")
                    for material_name, exposed_to_shader in interface.items():
                        shader_values = {
                            shader_pin: constants.get(exposed_pin)
                            for exposed_pin, shader_pin in exposed_to_shader.items()
                        }
                        if not all(
                            isinstance(shader_values.get(pin_id), list)
                            and len(shader_values[pin_id]) >= 3
                            for pin_id in (400, 186, 187, 188, 205)
                        ):
                            continue
                        palette = [(0.0, 0.0, 0.0)] + [
                            tuple(float(component) for component in shader_values[pin_id][:3])
                            for pin_id in (400, 186, 187, 188, 205)
                        ]
                        result[material_name] = {"bio_palette": tuple(palette)}
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(tree)
    return result


def _is_wildlife_bio_shader(shader):
    name = os.path.basename(shader).casefold()
    return name.startswith("px_wildlife_skin") and "static" not in name


def _medusa_bio_parameters(data):
    """Return Medusa bio palettes and a midpoint distance-preview strength."""
    pin_ids = {401, 402, 403, 404, 448, 449}
    constants = _connected_material_constants(
        data, ("px_dlc3_medusa_skin",), pin_ids
    )
    result = {}
    for name, pins in constants.items():
        palette = [(0.0, 0.0, 0.0)]
        for pin_id in (401, 402, 403, 404):
            color = pins.get(pin_id, (0.0, 0.0, 0.0))
            if not (isinstance(color, list) and len(color) >= 3):
                color = (0.0, 0.0, 0.0)
            palette.append(tuple(float(value) for value in color[:3]))
        distance_scale = pins.get(449, (1.0, 1.0))
        if not (isinstance(distance_scale, list) and len(distance_scale) >= 2):
            distance_scale = (1.0, 1.0)
        result[name] = {
            "bio_palette": tuple(palette),
            "bio_procedural": bool(pins.get(448, 0)),
            # The shader multiplies the distance-interpolated value by 0.5.
            # Use its midpoint for a deterministic Blender preview.
            "bio_strength": 0.25 * (
                float(distance_scale[0]) + float(distance_scale[1])
            ),
        }
    return result


def _pool_strings(data):
    """Return printable, NUL-terminated strings in their serialized order."""
    strings = []
    index = 0
    while index < len(data):
        if 32 <= data[index] < 127:
            end = index
            while end < len(data) and 32 <= data[end] < 127:
                end += 1
            if end < len(data) and data[end] == 0 and end - index >= 2:
                strings.append(data[index:end].decode("latin1"))
            index = end + 1
        else:
            index += 1
    for start, value in enumerate(strings):
        if value.endswith("ConstantNodesData"):
            return strings[start:]
    return strings


def referenced_compounds(data):
    """Return distinct compound-node paths referenced by a graph source."""
    result = []
    seen = set()
    for value in _pool_strings(data):
        key = value.casefold()
        if not key.endswith(".mcompoundnode") or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def referenced_meshes(data):
    """Return distinct non-last-LOD MMB paths referenced by a graph."""
    result = []
    seen = set()
    for value in _pool_strings(data):
        key = value.casefold()
        if not key.endswith(".mmb") or "lastlod" in key or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _player_head_sex(data, material_names):
    """Identify layered player-head graphs that share the base 01 skin set."""
    required = {
        "head", "teeth", "eyeleft", "eyeright", "eyeshell",
        "eyelashes", "eyeedge",
    }
    if not required.issubset(name.casefold() for name in material_names):
        return None
    for path in referenced_meshes(data):
        match = _PLAYER_HEAD_MESH_RE.search(path)
        if match:
            return match.group(1).casefold()
    return None


def _classify_texture(path):
    stem = os.path.basename(path).rsplit(".", 1)[0].casefold()
    if stem.endswith(("_alpha", "_opacity")) or "_alpha_" in stem:
        # Dedicated opacity/effect maps are not albedos merely because they do
        # not use the usual D/N/M suffix. Surface-alpha samplers are recovered
        # from their exact shader connection separately.
        return "a"
    if stem.endswith(("_h", "_pc", "_dn_mask", "_reg_mask")):
        # Height, pattern-coat, detail-normal-mask, and region-mask samplers
        # are auxiliary shader inputs. They must not displace the D/N/M set
        # merely because they share the material's name (Banshee head/body is
        # a common example).
        return "x"
    if stem.endswith(("_n_ta", "_normal_ta")):
        return "n"
    # A conventional terminal D/N/M suffix is authoritative even when the
    # asset family itself contains words such as ``mask``.  For example,
    # g_hmn_mask_01_d/n/m is a normal three-texture material set.
    explicit = re.search(r"_([dnm])$", stem)
    if explicit:
        return explicit.group(1)
    if "rgb_mask" in stem or stem.endswith("_mask") or "_mask_" in stem:
        return "m"
    if stem.endswith(("_d_ta", "_ta")):
        return "d"
    if stem.endswith(("_nr", "_nrm")) or "_nr_" in stem:
        return "n"
    if re.search(
        r"_(ao|cavity|height|disp|displacement|spec|specular|gloss|"
        r"metal|metallic|sss|opacity|rough)$",
        stem,
    ):
        return "m"
    if "normal" in stem or "_detail_n" in stem or "_n_" in stem:
        return "n"
    if "mask" in stem or "_m_" in stem or "_rough_" in stem:
        return "m"
    return "d"


def _texture_group_name(path):
    stem = os.path.basename(path).rsplit(".", 1)[0].casefold()
    stem = re.sub(r"(_d_ta|_n_ta|_normal_ta|_rgb_mask|_ta)$", "", stem)
    stem = re.sub(
        r"(_[dnm]|_nr|_nrm|diffuse|_detail_n|_n_imprintstructures|_imprintstructures|"
        r"_mask|_closed_decal_d|_decal_d|_atlas_d)$",
        "",
        stem,
    )
    return re.sub(r"_(d|n|m)$", "", stem)


def texture_pool(data):
    """Return the graph's ordered texture constants and inferred base roles."""
    result = []
    seen = set()
    for value in _pool_strings(data):
        key = value.casefold()
        if "lastlod" in key or not _DDS_RE.search(value) or key in seen:
            continue
        seen.add(key)
        result.append({
            "path": value,
            "kind": _classify_texture(value),
            "base": _texture_group_name(value),
        })
    return result


def _texture_groups(data):
    groups = {}
    for texture in texture_pool(data):
        group = groups.setdefault(
            texture["base"],
            {"base": texture["base"], "d": None, "n": None, "m": None, "paths": []},
        )
        group["paths"].append(texture["path"])
        if texture["kind"] in {"d", "n", "m"} and group[texture["kind"]] is None:
            group[texture["kind"]] = texture["path"]
    return list(groups.values())


_MATERIAL_KEYS = {
    "MeshName", "ModelFile", "ShaderFile", "StaticMeshAnim",
    "VertexShaderFile", "WeatherMask", "_PreventOptimizedMerging",
    "prefab:MeshNormal", "prefab:SplineSnap", "prefab:InstancePosition",
    "Size", "graphTypeSpecific", "ExtraDataFile", "assetName", "meshName",
    "collisionMaterialUID", "prefab:Collision", "OutBoneData",
}
_NON_MATERIAL_NAMES = {"lastLodShader", "lastLodMesh", "Default", "Filler", "prop"}


def _looks_like_material_name(value):
    return (
        value not in _NON_MATERIAL_NAMES
        and value not in _MATERIAL_KEYS
        and ":" not in value
        and "/" not in value
        and not value.casefold().startswith(("lastlod", "mygenerated"))
        and bool(re.search(r"[A-Za-z]", value))
    )


def _decoded_graph_materials(data):
    """Recover unambiguous material shaders from decoded prefab mesh nodes."""
    try:
        tree = _bv2_plain(_Bv2Decoder(data).root())
    except (IndexError, KeyError, TypeError, ValueError):
        return []
    by_name = {}

    def visit(value):
        if isinstance(value, dict):
            name = value.get("MeshName")
            shader = value.get("ShaderFile")
            node_type = str(value.get("type", "")).casefold()
            if (
                isinstance(name, str)
                and isinstance(shader, str)
                and node_type.startswith("prefab:mesh")
                and _SHADER_RE.search(shader)
                and "lastlod" not in shader.casefold().replace(" ", "")
            ):
                shaders = by_name.setdefault(name.casefold(), {})
                shaders.setdefault(shader.casefold(), (name, shader))
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(tree)
    # A placement graph can contain two different models with the same generic
    # mesh name (usually "body"). Only use decoded names whose shader is unique;
    # ambiguous names remain eligible for the established pool fallback.
    return [
        next(iter(shaders.values()))
        for shaders in by_name.values()
        if len(shaders) == 1
    ]


def _graph_materials(data):
    """Recover material-name/shader pairs, preferring decoded prefab nodes."""
    decoded = _decoded_graph_materials(data)
    decoded_keys = {name.casefold() for name, _shader in decoded}
    pool = _pool_strings(data)
    pairs = []
    for index, value in enumerate(pool):
        if not _SHADER_RE.search(value):
            continue
        if "lastlod" in value.casefold().replace(" ", ""):
            # Last-LOD extension shaders describe the distant replacement, not
            # any imported MMB part. Treating one as the sole unmatched shader
            # can otherwise attach it to a real material such as a corpse body.
            continue
        for preceding in reversed(pool[:index]):
            if (
                _looks_like_material_name(preceding)
                and not preceding.casefold().endswith((
                    ".mmb", ".dds", ".juice", ".mphys", ".mcompoundnode"
                ))
            ):
                if preceding.casefold() not in decoded_keys:
                    pairs.append((preceding, value))
                break
    unique = {name.casefold(): (name, shader) for name, shader in decoded}
    for name, shader in pairs:
        unique.setdefault(name.casefold(), (name, shader))
    return list(unique.values())


def material_shader_pairs(data):
    """Return the material-name/ShaderFile pairs recovered from a BV2 source."""
    return _graph_materials(data)


def surface_alpha_channel(shader_path, pin_id):
    """Return the authored surface-opacity channel for a shader sampler pin."""
    shader_name = os.path.basename(shader_path).casefold()
    return _SURFACE_ALPHA_PINS.get(shader_name, {}).get(pin_id, "")


def diffuse_drives_surface_alpha(shader_path):
    """Whether this shader treats the color texture alpha as surface opacity."""
    name = os.path.basename(shader_path).casefold()
    if name in {
        "px_basic.mshader",
        "px_basic_emissive.mshader",
        "px_character_gear_simple.mshader",
        "px_character_gear_pearl.mshader",
        "px_constants.mshader",
        "px_basic_mosspatch.mshader",
        "px_dlc3_basic_mosspatch.mshader",
    }:
        return True
    if name.startswith(("px_basic_transmission", "px_basic_transparent")):
        return True
    if name.startswith("px_basic_rustymetal"):
        return True
    if name.startswith((
        "px_vegetation", "px_dlc1_vegetation", "px_dlc2_vegetation",
        "px_dlc3_vegetation", "px_harvest_vegetation", "px_grass",
        "px_dlc1_grass", "px_dlc3_grass", "px_mosscard",
        "px_dlc2_mosscard", "px_dlc3_mosscard", "px_dlc3_mossycard",
    )):
        return True
    if name.startswith((
        "px_decal_", "px_character_weapon_decal", "px_fx_decal_",
        "deferred-decal-",
    )):
        return True
    if name.startswith((
        "px_hair", "px_character_gear_fur", "px_wildlife_fur",
    )) or name == "hair.mshader":
        return True
    if name.startswith((
        "px_wildlife_dragonflywing", "px_wildlife_fanlizardwing",
        "px_wildlife_gear", "px_wildlife_skin",
    )):
        return True
    return any(fragment in name for fragment in (
        "glass", "foliage", "transparent", "transmission alpha",
    ))


_SHADER_HINTS = {
    "rustymetal": ("metal", "mesh", "rust"),
    "croverlay": ("stripe", "decal", "paint", "warning", "orange"),
    "decal_simple": ("decal", "stripe", "warning", "mask"),
    "decal_imprint": ("decal", "sign", "imprint", "structures"),
    "decal_colormaterial": ("sign", "number", "atlas", "decal"),
    "decal_multipass": ("decal", "sign"),
    "basic": ("paint", "metal", "generic"),
}


def _tokens(value):
    tokens = set()
    for token in re.split(r"[^a-z0-9]+", value.casefold()):
        token = re.sub(r"\d+$", "", token)
        if len(token) >= 3:
            if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
                token = token[:-1]
            tokens.add(token)
            # Medusa graph materials call this part "tentacles", while the
            # corresponding texture family is named "tail".
            if token == "tentacle":
                tokens.add("tail")
    return tokens


def _shader_tokens(value):
    """Material-name tokens retaining tier/part numbers for shader matching."""
    tokens = {
        token for token in re.split(r"[^a-z0-9]+", value.casefold())
        if len(token) >= 2
    }
    for token in tuple(tokens):
        for fragment in ("bomb", "upper", "mouth", "inner", "outer", "light", "led"):
            if fragment in token:
                tokens.add(fragment)
    return tokens


def _semantic_path(textures, role, material_tokens):
    """Best texture of one role for a material when its group lacks that role."""
    ranked = []
    for index, texture in enumerate(textures):
        if texture["kind"] != role:
            continue
        stem = os.path.basename(texture["path"]).rsplit(".", 1)[0].casefold()
        if role == "d" and stem.endswith(("_h", "_pc", "_dn_mask", "_reg_mask")):
            # Height, pattern-coat, and region/detail masks are auxiliary
            # samplers even though their filename does not use _m.
            continue
        path_tokens = _tokens(texture["path"])
        exact = len(material_tokens & path_tokens)
        partial = sum(
            1
            for material_token in material_tokens
            for path_token in path_tokens
            if material_token not in path_tokens
            and path_token not in material_tokens
            and min(len(material_token), len(path_token)) >= 3
            and (material_token in path_token or path_token in material_token)
        )
        score = exact * 4 + partial
        if score:
            ranked.append((score, -index, texture["path"]))
    return max(ranked)[2] if ranked else None


def _apply_wildlife_part_aliases(names, bindings):
    """Make gameplay-only wildlife parts reuse their rendered base material."""
    skin_sources = []
    for index, name in enumerate(names):
        key = name.casefold()
        if key.endswith(("_weakpoint", "_armor")):
            continue
        binding = bindings.get(name)
        shader_name = os.path.basename((binding or {}).get("shader", "")).casefold()
        if shader_name.startswith("px_wildlife_skin"):
            skin_sources.append((index, name, binding))
    if not skin_sources:
        return

    source_by_key = {name.casefold(): (name, binding) for _index, name, binding in skin_sources}
    for target_name in names:
        target_key = target_name.casefold()
        if target_key.endswith("_armor"):
            source = source_by_key.get(target_key[:-6])
            if source is not None:
                bindings[target_name] = dict(source[1])
            continue
        if not target_key.endswith("_weakpoint"):
            continue

        base_key = target_key[:-10]
        source = source_by_key.get(base_key)
        if source is None:
            target_tokens = _tokens(base_key)
            ranked = []
            for source_index, source_name, source_binding in skin_sources:
                source_tokens = _tokens(source_name)
                overlap = len(target_tokens & source_tokens)
                body = int("body" in source_tokens)
                ranked.append((
                    (
                        overlap,
                        body,
                        -len(source_tokens - target_tokens),
                        -source_index,
                    ),
                    source_name,
                    source_binding,
                ))
            best = max(ranked, key=lambda item: item[0]) if ranked else None
            if best is not None and (
                best[0][0] > 0 or best[0][1] or len(skin_sources) == 1
            ):
                source = (best[1], best[2])
        if source is not None:
            bindings[target_name] = dict(source[1])


def _apply_named_part_texture_aliases(names, bindings):
    """Reuse explicit base-part textures for semantic duplicate meshes."""
    binding_by_key = {
        name.casefold(): binding
        for name, binding in bindings.items()
    }
    for target_name in names:
        target_key = target_name.casefold()
        if not target_key.startswith("piercing_"):
            continue
        source = binding_by_key.get(target_key[len("piercing_"):])
        target = bindings.get(target_name)
        if source is None or target is None:
            continue
        for role in ("d", "n", "m"):
            target[role] = source.get(role)


def material_bindings(
    data, material_names, compound_sources=None, shader_role_pins=None,
    shader_parameter_pins=None,
):
    """Map each MMB mesh/material name to diffuse, normal, and mask paths."""
    if data[:4] != MAGIC:
        raise ValueError("material source does not have BV2 magic")
    names = list(material_names)
    player_head_sex = _player_head_sex(data, names)
    material_shaders = _graph_materials(data)
    decoded_source_names = {
        name.casefold() for name, _shader in _decoded_graph_materials(data)
    }
    shader_map = {
        name.casefold(): (name, shader) for name, shader in material_shaders
    }
    for compound_data in (compound_sources or {}).values():
        # A graph wrapper often serializes shader strings without enough local
        # structure to bind them to the correct mesh. Its referenced compound
        # retains the authoritative prefab MeshName/ShaderFile pairs. Override
        # only pool-inferred graph pairs; explicit graph prefab nodes still win.
        for name, shader in _decoded_graph_materials(compound_data):
            key = name.casefold()
            if key not in decoded_source_names:
                shader_map[key] = (name, shader)
    material_shaders = list(shader_map.values())
    connected_textures = _connected_material_textures(data)
    connected_alpha_textures = _connected_material_alpha_textures(data)
    connected_role_textures = _connected_material_role_textures(
        data, shader_role_pins
    )
    connected_parameters = _connected_material_parameters(
        data, shader_parameter_pins
    )
    for compound_data in (compound_sources or {}).values():
        for name, paths in _connected_material_textures(compound_data).items():
            connected_textures.setdefault(name, paths)
        for name, alpha_info in _connected_material_alpha_textures(compound_data).items():
            connected_alpha_textures.setdefault(name, alpha_info)
        for name, roles in _connected_material_role_textures(
            compound_data, shader_role_pins
        ).items():
            combined_roles = connected_role_textures.setdefault(name, {})
            for role, path in roles.items():
                combined_roles.setdefault(role, path)
        for name, parameters in _connected_material_parameters(
            compound_data, shader_parameter_pins
        ).items():
            combined_parameters = connected_parameters.setdefault(name, {})
            for field, value in parameters.items():
                combined_parameters.setdefault(field, value)
    # Parent graph inputs override an exposed compound pin. Without following
    # this boundary a compound-local BIO texture can win the filename fallback
    # when the real Color input lives on the graph instance (green moss is the
    # common example).
    for name, roles in _forwarded_compound_role_textures(
        data, compound_sources, shader_role_pins
    ).items():
        connected_role_textures.setdefault(name, {}).update(roles)
    emissive_parameters = (
        _emissive_parameters(data)
        if any(
            os.path.basename(shader).casefold() == "px_emissive_color.mshader"
            for _name, shader in material_shaders
        )
        else {}
    )
    wildlife_bio_parameters = (
        _wildlife_bio_parameters(data)
        if any(
            _is_wildlife_bio_shader(shader)
            for _name, shader in material_shaders
        )
        else {}
    )
    wildlife_bio_parameters.update(
        _forwarded_wildlife_bio_parameters(data, compound_sources)
    )
    medusa_bio_parameters = (
        _medusa_bio_parameters(data)
        if any(
            os.path.basename(shader).casefold().startswith("px_dlc3_medusa_skin")
            for _name, shader in material_shaders
        )
        else {}
    )
    shader_by_name = {name.casefold(): shader for name, shader in material_shaders}
    resolved_shaders = {
        index: shader_by_name[name.casefold()]
        for index, name in enumerate(names)
        if name.casefold() in shader_by_name
    }
    index_by_name = {name.casefold(): index for index, name in enumerate(names)}
    if player_head_sex is not None:
        # Player-head graphs serialize several alternative nodes named
        # ``head``. The Navi Face node is an intermediate customization layer
        # (and directly binds only the Sarentu scar normal); the graph's final
        # head material is PX_Character_Workbench. Pool order used to select
        # the intermediate node and then borrow the eye texture as face color.
        resolved_shaders[index_by_name["head"]] = _PLAYER_HEAD_SHADER
    # A split *_part mesh is the same source material as its parent. This is
    # stronger evidence than an unrelated leftover graph shader.
    for index, name in enumerate(names):
        if not name.casefold().endswith("_part"):
            continue
        parent_index = index_by_name.get(name[:-5].casefold())
        if parent_index in resolved_shaders:
            resolved_shaders[index] = resolved_shaders[parent_index]
    # User-verified Banshee behavior: the weakpoint renders with the body
    # material despite a separate graph effect node using the same mesh name.
    banshee_body_index = index_by_name.get("banshee_body")
    banshee_weakpoint_index = index_by_name.get("banshee_weakpoint")
    if banshee_body_index in resolved_shaders and banshee_weakpoint_index is not None:
        resolved_shaders[banshee_weakpoint_index] = resolved_shaders[banshee_body_index]
    # Zakru-style multi-eye meshes name the transparent shell geometry
    # *_membrane while the graph calls the same material *_edge.
    for index, name in enumerate(names):
        key = name.casefold()
        if index in resolved_shaders or "eye" not in key or not key.endswith("_membrane"):
            continue
        edge_shader = shader_by_name.get(key[:-9] + "_edge")
        if edge_shader:
            resolved_shaders[index] = edge_shader
    # Known reflection bookkeeping is rejected by _looks_like_material_name.
    # Exact material-name matches are authoritative; if an unknown field still
    # obscures exactly one mesh part and one shader pair, the remaining shader
    # is nevertheless unambiguous and safe to retain as material metadata.
    unmatched_indices = [index for index in range(len(names)) if index not in resolved_shaders]
    matched_shader_keys = {value.casefold() for value in resolved_shaders.values()}
    unmatched_shaders = [
        shader for _name, shader in material_shaders
        if shader.casefold() not in matched_shader_keys
    ]
    if len(unmatched_indices) == len(unmatched_shaders) == 1:
        resolved_shaders[unmatched_indices[0]] = unmatched_shaders[0]
    # Interned shader strings commonly appear only once when sibling materials
    # share a shader. Recover obvious numbered/semantic siblings (Wing1/Wing2,
    # SmallEyes/Eyes) without guessing between unrelated material families.
    exact_shader_names = [
        (index, names[index], shader)
        for index, shader in resolved_shaders.items()
    ]
    semantic_shader_sources = list(exact_shader_names)
    seen_semantic_sources = {
        (source_name.casefold(), shader.casefold())
        for _index, source_name, shader in semantic_shader_sources
    }
    for source_index, (source_name, shader) in enumerate(material_shaders):
        key = (source_name.casefold(), shader.casefold())
        if key not in seen_semantic_sources:
            semantic_shader_sources.append((source_index, source_name, shader))
            seen_semantic_sources.add(key)
    common_name_tokens = _tokens(names[0]) if len(names) > 1 else set()
    for name in names[1:]:
        common_name_tokens &= _tokens(name)
    common_shader_tokens = _shader_tokens(names[0]) if len(names) > 1 else set()
    for name in names[1:]:
        common_shader_tokens &= _shader_tokens(name)
    for index, name in enumerate(names):
        if index in resolved_shaders:
            continue
        target_tokens = _shader_tokens(name)
        ranked_shaders = []
        for source_index, source_name, shader in semantic_shader_sources:
            source_tokens = _shader_tokens(source_name)
            exact = len(target_tokens & source_tokens)
            distinctive = len(
                (target_tokens - common_shader_tokens)
                & (source_tokens - common_shader_tokens)
            )
            partial = sum(
                1
                for target in target_tokens
                for source in source_tokens
                if min(len(target), len(source)) >= 3
                and (target in source or source in target)
            )
            affinity = distinctive * 8 + exact * 2 + partial
            score = affinity * 10 - len(source_tokens - target_tokens)
            if affinity:
                ranked_shaders.append((score, -source_index, shader))
        if ranked_shaders:
            best_score = max(item[0] for item in ranked_shaders)
            best_shaders = {
                shader.casefold(): shader
                for score, _source_index, shader in ranked_shaders
                if score == best_score
            }
            if len(best_shaders) == 1:
                resolved_shaders[index] = next(iter(best_shaders.values()))
    # Wildlife graphs serialize their shared skin shader on the body even
    # though head parts are separate MMB meshes. The decoded reference graph
    # confirms those body/head nodes use the same PX_Wildlife_Skin variant.
    wildlife_skin_sources = [
        (source_index, shader)
        for source_index, _source_name, shader in exact_shader_names
        if os.path.basename(shader).casefold().startswith("px_wildlife_skin")
    ]
    if wildlife_skin_sources:
        wildlife_skin_shader = min(wildlife_skin_sources)[1]
        for index, name in enumerate(names):
            if index not in resolved_shaders and (_tokens(name) & {"body", "head"}):
                resolved_shaders[index] = wildlife_skin_shader
    # The Medusa graph likewise interns one skin shader across body, wings,
    # and tentacles. Retain that shader for every Medusa material even when a
    # sibling node is the only place its ShaderFile string is serialized.
    medusa_skin_sources = [
        (source_index, shader)
        for source_index, _source_name, shader in exact_shader_names
        if os.path.basename(shader).casefold().startswith("px_dlc3_medusa_skin")
    ]
    if medusa_skin_sources:
        medusa_skin_shader = min(medusa_skin_sources)[1]
        for index, name in enumerate(names):
            if index not in resolved_shaders and "medusa" in _tokens(name):
                resolved_shaders[index] = medusa_skin_shader

    groups = [
        group for group in _texture_groups(data)
        if any(group[role] for role in ("d", "n", "m"))
    ]
    textures = texture_pool(data)
    if not names:
        return {}

    # Shared tokens identify the asset prefix only when several material names
    # are present. For a one-part MMB, stripping its entire name would discard
    # the strongest signal available (important for compound nodes containing
    # several unrelated meshes and texture families).
    common = _tokens(names[0]) if len(names) > 1 else set()
    for name in names[1:]:
        common &= _tokens(name)

    pairs = []
    for material_index, name in enumerate(names):
        shader = resolved_shaders.get(material_index, "")
        material_tokens = _tokens(name) - common
        shader_name = os.path.basename(shader).casefold()
        hint_tokens = set()
        for fragment, hints in _SHADER_HINTS.items():
            if fragment in shader_name:
                hint_tokens.update(hints)
        for group in groups:
            group_tokens = set()
            for path in group["paths"]:
                group_tokens.update(_tokens(path))
            semantic_score = 4 * len(material_tokens & group_tokens)
            semantic_score += len(hint_tokens & group_tokens)
            semantic_score += sum(
                3 for token in material_tokens
                if any(token in os.path.basename(path).casefold() for path in group["paths"])
            )
            score = semantic_score + 0.1 * sum(
                bool(group[role]) for role in ("d", "n", "m")
            )
            if semantic_score > 0:
                pairs.append((score, material_index, group["base"], group))
    pairs.sort(key=lambda item: (-item[0], item[1]))

    assigned = {}
    used_groups = set()
    for _score, material_index, base, group in pairs:
        if material_index not in assigned and base not in used_groups:
            assigned[material_index] = group
            used_groups.add(base)
    for _score, material_index, _base, group in pairs:
        assigned.setdefault(material_index, group)

    # Snowdrop sometimes splits one named material into an additional *_part
    # mesh without emitting another material node. Reuse the parent's complete
    # texture set instead of letting the part consume an auxiliary group.
    for material_index, name in enumerate(names):
        if not name.casefold().endswith("_part"):
            continue
        parent_index = index_by_name.get(name[:-5].casefold())
        if parent_index in assigned:
            assigned[material_index] = assigned[parent_index]

    if banshee_body_index in assigned and banshee_weakpoint_index is not None:
        assigned[banshee_weakpoint_index] = assigned[banshee_body_index]

    name_keys = {name.casefold() for name in names}
    is_direhorse = {
        "wildlife_dirhorse_weakpoint", "eyes", "eyes_small", "body"
    }.issubset(name_keys) and any(
        "wildlife_direhorse_" in texture["path"].casefold()
        for texture in textures
    )
    if is_direhorse:
        # This MMB uses generic part names while its body texture family carries
        # only the asset name. The graph's decoded nodes confirm body uses the
        # Direhorse D/NR/M group and both eye meshes share the Eddie eye set.
        # Blender verification also confirms the weakpoint is rendered as part
        # of the body rather than with the graph's generic PX_Basic defaults.
        body_group = next(
            (
                group for group in groups
                if "wildlife_direhorse_" in group["base"] and group["d"]
            ),
            None,
        )
        eye_group = next(
            (
                group for group in groups
                if "eyes" in group["base"] and group["d"]
            ),
            None,
        )
        for material_index, name in enumerate(names):
            key = name.casefold()
            if key == "body" and body_group is not None:
                assigned[material_index] = body_group
            elif key in {"eyes", "eyes_small"} and eye_group is not None:
                assigned[material_index] = eye_group
        body_index = index_by_name["body"]
        weakpoint_index = index_by_name["wildlife_dirhorse_weakpoint"]
        if body_index in assigned:
            assigned[weakpoint_index] = assigned[body_index]
        if body_index in resolved_shaders:
            resolved_shaders[weakpoint_index] = resolved_shaders[body_index]

    default_diffuse = next(
        (texture["path"] for texture in textures if texture["kind"] == "d"), None
    )
    result = {}
    for material_index, name in enumerate(names):
        group = assigned.get(material_index)
        material_tokens = _tokens(name) - common
        shader = resolved_shaders.get(material_index, "")
        shader_name = os.path.basename(shader).casefold()
        shader_key = shader.replace("\\", "/").lstrip("/").casefold()
        role_schema = None
        if shader_role_pins is not None:
            if shader_key in shader_role_pins:
                role_schema = shader_role_pins[shader_key]
            elif shader_name in shader_role_pins:
                role_schema = shader_role_pins[shader_name]
        declared_roles = set(role_schema.values()) if role_schema is not None else set()
        if shader_name == "px_emissive_color.mshader":
            parameters = emissive_parameters.get(name.casefold(), {})
            result[name] = {
                "d": None,
                "n": None,
                "m": None,
                "shader": shader,
                "emissive_color": parameters.get("emissive_color", (0.0, 0.0, 0.0)),
                "emissive_strength": parameters.get("emissive_strength", 1.0),
                "parameters": dict(connected_parameters.get(name.casefold(), {})),
            }
            continue
        diffuse = (group["d"] if group else None) or _semantic_path(
            textures, "d", material_tokens
        )
        normal = (group["n"] if group else None) or _semantic_path(
            textures, "n", material_tokens
        )
        mask = group["m"] if group else None
        direct_paths = connected_textures.get(name.casefold(), ())
        alpha_info = connected_alpha_textures.get(name.casefold())
        if alpha_info is None and name.casefold().endswith("_part"):
            alpha_info = connected_alpha_textures.get(name[:-5].casefold())
        alpha, alpha_channel = alpha_info or (None, None)
        direct_roles = {}
        if name.casefold().endswith("_part"):
            direct_roles.update(
                connected_role_textures.get(name[:-5].casefold(), {})
            )
        direct_roles.update(connected_role_textures.get(name.casefold(), {}))
        if role_schema is None:
            for path in direct_paths:
                if (
                    alpha is not None
                    and alpha_channel == "color"
                    and path.casefold() == alpha.casefold()
                ):
                    continue
                role = _classify_texture(path)
                if role in {"d", "n", "m"}:
                    direct_roles.setdefault(role, path)
        diffuse = direct_roles.get("d", diffuse)
        normal = direct_roles.get("n", normal)
        mask = direct_roles.get("m", mask)
        if role_schema is not None:
            if "d" not in declared_roles:
                diffuse = None
            if "n" not in declared_roles:
                normal = None
            if "m" not in declared_roles:
                mask = None
        auxiliary = {
            role[4:]: path
            for role, path in direct_roles.items()
            if role.startswith("aux:")
        }
        authored_parameters = {}
        if name.casefold().endswith("_part"):
            authored_parameters.update(
                connected_parameters.get(name[:-5].casefold(), {})
            )
        authored_parameters.update(connected_parameters.get(name.casefold(), {}))
        if player_head_sex is not None and name.casefold() == "head":
            # Every p_head morph reuses the same authored base skin maps. The
            # female graph explicitly supplies RNF color plus the shared male
            # normal/material pair; male graphs leave all three implicit.
            # Do not promote the customization scar normal or nearby eye maps
            # into these base roles.
            diffuse = (
                _PLAYER_HEAD_FEMALE_DIFFUSE
                if player_head_sex == "f"
                else _PLAYER_HEAD_MALE_TEXTURE_ROOT + "p_head_01_m_d.dds"
            )
            normal = _PLAYER_HEAD_MALE_TEXTURE_ROOT + "p_head_01_m_n.dds"
            mask = _PLAYER_HEAD_MALE_TEXTURE_ROOT + "p_head_01_m_m.dds"
        if shader_name == "px_wildlife_eye.mshader":
            # Wildlife commonly shares the Banshee/Eddie eye set, whose path
            # has no species or mesh-name affinity. Shader identity is stronger
            # evidence than assigning a nearby head/body texture group.  The
            # Banshee source contains both ``smalleye`` and full-eye families;
            # pool order is not meaningful and previously gave Banshee_Eyes
            # the SmallEyes diffuse.
            wants_small_eye = "small" in name.casefold()

            def eye_texture(role, preferred_names):
                candidates = [
                    texture["path"] for texture in textures
                    if texture["kind"] == role
                    and "eye" in os.path.basename(texture["path"]).casefold()
                ]
                if wants_small_eye:
                    candidates = [
                        path for path in candidates
                        if "smalleye" in os.path.basename(path).casefold()
                    ]
                else:
                    candidates = [
                        path for path in candidates
                        if "smalleye" not in os.path.basename(path).casefold()
                    ]
                if not candidates:
                    return None
                for preferred in preferred_names:
                    for path in candidates:
                        if os.path.basename(path).casefold() == preferred:
                            return path
                return candidates[0]

            diffuse = eye_texture(
                "d", ("wl_banshee_eye_d.dds", "eddie_eye_d.dds")
            ) or diffuse
            normal = eye_texture(
                "n", ("eddie_eye_n.dds", "wl_banshee_eye_n.dds")
            ) or normal
        if shader_name.startswith("px_dlc3_medusa_skin") and "tail" in material_tokens:
            # The tentacle/tail texture set has no dedicated normal map; its
            # Medusa shader node explicitly binds Snowdrop's flat normal.
            normal = next(
                (
                    texture["path"] for texture in textures
                    if os.path.basename(texture["path"]).casefold()
                    == "sd_flat_normal_128_n.dds"
                ),
                normal,
            )
        if "wildlife_dragonflywing" in shader_name:
            wing_group = next(
                (candidate for candidate in groups if "insect_wing" in candidate["base"]),
                None,
            )
            if wing_group is not None:
                diffuse = wing_group["d"] or diffuse
                normal = wing_group["n"] or normal
            else:
                diffuse = next(
                    (
                        texture["path"] for texture in textures
                        if os.path.basename(texture["path"]).casefold()
                        == "sd_gn_default_grey_d.dds"
                    ),
                    diffuse,
                )
                normal = next(
                    (
                        texture["path"] for texture in textures
                        if os.path.basename(texture["path"]).casefold()
                        == "sd_flat_normal_128_n.dds"
                    ),
                    normal,
                )
        # A known shader schema is authoritative about the texture roles that
        # can be authored on its graph node.  In particular dragonfly wings
        # and wildlife eyes expose Color + Normal but no Material sampler; do
        # not let texture-family proximity attach a body/skin packed mask.
        if {"d", "n"}.issubset(declared_roles) and "m" not in declared_roles:
            mask = None
        binding = {
            "d": (
                diffuse
                if role_schema is not None
                else diffuse or default_diffuse
            ),
            "n": (
                normal or _DEFAULT_NORMAL
                if role_schema is None or "n" in declared_roles
                else None
            ),
            "m": mask,
            "a": alpha,
            "a_channel": alpha_channel,
            "shader": shader,
            "aux": auxiliary,
            "parameters": authored_parameters,
        }
        if _is_wildlife_bio_shader(shader):
            bio = wildlife_bio_parameters.get(name.casefold())
            if bio is None and name.casefold().endswith("_part"):
                bio = wildlife_bio_parameters.get(name[:-5].casefold())
            if bio is not None:
                binding.update(bio)
        elif shader_name.startswith("px_dlc3_medusa_skin"):
            bio = medusa_bio_parameters.get(name.casefold())
            if bio is not None:
                binding.update(bio)
        result[name] = binding
    # Wildlife weakpoint and armor graph nodes drive gameplay/VFX state. Their
    # mesh geometry still renders with the corresponding skin material, so copy
    # the complete binding (shader, textures, auxiliaries and bio parameters).
    _apply_wildlife_part_aliases(names, result)
    _apply_named_part_texture_aliases(names, result)
    return result

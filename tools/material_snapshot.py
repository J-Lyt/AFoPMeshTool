"""Capture or compare stable material metadata from the open Blender file.

Examples::

    blender --background asset.blend --python-exit-code 1 \
        --python tools/material_snapshot.py -- --output baseline.json

    blender --background asset.blend --python-exit-code 1 \
        --python tools/material_snapshot.py -- --baseline baseline.json

Only material assignments, AFoP custom properties, node settings, sockets,
links, UV-layer names, and color-attribute names are recorded. Mesh geometry,
image pixels, absolute image paths, animation, and armatures are excluded.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

import bpy


SCHEMA_VERSION = 1
_NUMERIC_SUFFIX = re.compile(r"\.\d{3}$")
_NODE_ATTRIBUTES = (
    "blend_type",
    "clamp",
    "clamp_factor",
    "data_type",
    "distribution",
    "extension",
    "gradient_type",
    "interpolation",
    "interpolation_type",
    "invert",
    "layer_name",
    "mapping",
    "normalize",
    "operation",
    "projection",
    "projection_blend",
    "space",
    "transform_space",
    "uv_map",
    "vector_type",
)


def _arguments():
    values = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the current snapshot here")
    parser.add_argument(
        "--baseline", type=Path,
        help="Compare the current snapshot with this existing JSON baseline",
    )
    parser.add_argument(
        "--source-name",
        help="Stable source label; defaults to the open .blend filename",
    )
    arguments = parser.parse_args(values)
    if arguments.output is None and arguments.baseline is None:
        parser.error("at least one of --output or --baseline is required")
    return arguments


def _clean_number(value):
    value = float(value)
    if abs(value) < 1.0e-9:
        return 0.0
    return round(value, 8)


def _json_value(value):
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return _clean_number(value)
    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]).casefold())
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    to_list = getattr(value, "to_list", None)
    if callable(to_list):
        return [_json_value(item) for item in to_list()]
    try:
        return [_json_value(item) for item in value]
    except TypeError:
        return str(value)


def _socket_identity(sockets, socket):
    index = next(
        (index for index, candidate in enumerate(sockets) if candidate == socket),
        -1,
    )
    return {"index": index, "name": socket.name}


def _socket_defaults(node):
    defaults = []
    for index, socket in enumerate(node.inputs):
        if socket.is_linked or not hasattr(socket, "default_value"):
            continue
        defaults.append({
            "index": index,
            "name": socket.name,
            "value": _json_value(socket.default_value),
        })
    return defaults


def _image_name(image):
    if image is None:
        return None
    logical_path = image.get("afop_asset_path")
    if isinstance(logical_path, str) and logical_path:
        return logical_path.replace("\\", "/")
    filepath = bpy.path.abspath(image.filepath or image.filepath_raw or "")
    return Path(filepath).name if filepath else _NUMERIC_SUFFIX.sub("", image.name)


def _color_ramp(node):
    ramp = getattr(node, "color_ramp", None)
    if ramp is None:
        return None
    return {
        "color_mode": ramp.color_mode,
        "hue_interpolation": ramp.hue_interpolation,
        "interpolation": ramp.interpolation,
        "elements": [
            {
                "position": _clean_number(element.position),
                "color": _json_value(element.color),
            }
            for element in sorted(ramp.elements, key=lambda item: item.position)
        ],
    }


def _curve_mapping(node):
    mapping = getattr(node, "mapping", None)
    curves = getattr(mapping, "curves", None)
    if curves is None:
        return None
    return {
        "clip_min": _json_value(mapping.clip_min),
        "clip_max": _json_value(mapping.clip_max),
        "use_clip": bool(mapping.use_clip),
        "curves": [
            [
                {
                    "location": _json_value(point.location),
                    "handle_type": point.handle_type,
                }
                for point in curve.points
            ]
            for curve in curves
        ],
    }


def _node_snapshot(node):
    result = {
        "name": node.name,
        "type": node.bl_idname,
    }
    if node.label:
        result["label"] = node.label
    attributes = {}
    for name in _NODE_ATTRIBUTES:
        if not hasattr(node, name):
            continue
        value = getattr(node, name)
        if isinstance(value, (bool, str, int, float)):
            attributes[name] = _json_value(value)
    if attributes:
        result["settings"] = attributes
    defaults = _socket_defaults(node)
    if defaults:
        result["unlinked_inputs"] = defaults
    image = _image_name(getattr(node, "image", None))
    if image is not None:
        result["image"] = image
    node_tree = getattr(node, "node_tree", None)
    if node_tree is not None:
        result["node_group"] = node_tree.name
    ramp = _color_ramp(node)
    if ramp is not None:
        result["color_ramp"] = ramp
    curves = _curve_mapping(node)
    if curves is not None:
        result["curve_mapping"] = curves
    return result


def _link_snapshot(link):
    return {
        "from_node": link.from_node.name,
        "from_socket": _socket_identity(link.from_node.outputs, link.from_socket),
        "to_node": link.to_node.name,
        "to_socket": _socket_identity(link.to_node.inputs, link.to_socket),
    }


def _node_tree_snapshot(tree):
    return {
        "nodes": [
            _node_snapshot(node)
            for node in sorted(tree.nodes, key=lambda item: item.name.casefold())
        ],
        "links": sorted(
            (_link_snapshot(link) for link in tree.links),
            key=lambda item: (
                item["to_node"].casefold(), item["to_socket"]["index"],
                item["from_node"].casefold(), item["from_socket"]["index"],
            ),
        ),
    }


def _material_snapshot(material):
    uses_nodes = material.node_tree is not None
    result = {
        "name": material.name,
        "use_nodes": uses_nodes,
        "afop_properties": {
            key: _json_value(material[key])
            for key in sorted(material.keys(), key=str.casefold)
            if key.startswith("afop_")
        },
    }
    for attribute in ("surface_render_method", "blend_method"):
        if hasattr(material, attribute):
            try:
                result[attribute] = getattr(material, attribute)
            except (AttributeError, TypeError):
                pass
    if uses_nodes:
        result["node_tree"] = _node_tree_snapshot(material.node_tree)
    return result


def _referenced_node_groups(materials):
    result = {}

    def visit(tree):
        for node in tree.nodes:
            group = getattr(node, "node_tree", None)
            if group is None or group.name in result:
                continue
            result[group.name] = None
            visit(group)
            result[group.name] = _node_tree_snapshot(group)

    for material in materials:
        if material.node_tree is not None:
            visit(material.node_tree)
    return {
        name: result[name]
        for name in sorted(result, key=str.casefold)
    }


def capture(source_name=None):
    objects = sorted(
        (obj for obj in bpy.data.objects if obj.type == "MESH"),
        key=lambda item: item.name.casefold(),
    )
    materials_by_name = {
        material.name: material
        for obj in objects
        for material in obj.data.materials
        if material is not None
    }
    materials = [
        materials_by_name[name]
        for name in sorted(materials_by_name, key=str.casefold)
    ]
    if source_name is None:
        source_name = Path(bpy.data.filepath).name or "unsaved.blend"
    return {
        "schema_version": SCHEMA_VERSION,
        "source": source_name,
        "objects": [
            {
                "name": obj.name,
                "materials": [
                    material.name if material is not None else None
                    for material in obj.data.materials
                ],
                "uv_layers": [layer.name for layer in obj.data.uv_layers],
                "color_attributes": [
                    attribute.name for attribute in obj.data.color_attributes
                ],
                "polygon_material_indices": sorted({
                    polygon.material_index for polygon in obj.data.polygons
                }),
            }
            for obj in objects
        ],
        "materials": [_material_snapshot(material) for material in materials],
        "node_groups": _referenced_node_groups(materials),
    }


def _serialized(snapshot):
    return json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _write(path, snapshot):
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_serialized(snapshot), encoding="utf-8", newline="\n")
    print(f"[SNAPSHOT] wrote {path}")


def _compare(path, snapshot):
    path = path.resolve()
    expected = json.loads(path.read_text(encoding="utf-8"))
    if expected == snapshot:
        print(f"[PASS] material snapshot matches {path}")
        return
    difference = difflib.unified_diff(
        _serialized(expected).splitlines(),
        _serialized(snapshot).splitlines(),
        fromfile=str(path),
        tofile=f"current:{snapshot['source']}",
        lineterm="",
    )
    print("\n".join(difference))
    raise AssertionError(f"material snapshot differs from {path}")


def main():
    arguments = _arguments()
    snapshot = capture(arguments.source_name)
    if arguments.output is not None:
        _write(arguments.output, snapshot)
    if arguments.baseline is not None:
        _compare(arguments.baseline, snapshot)


if __name__ == "__main__":
    main()

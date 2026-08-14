"""Structural reader and byte-preserving editor for Snowdrop ``.mreflex`` files."""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import struct


MAGIC = 0x012B6441
VERSION = 2
RECORD_TAG = 0x012B3B2C
TGET_MARKER = bytes.fromhex("d7454754")

KIND_DANGLE_NODE = 10
KIND_DANGLE_HEADER_A = 3
KIND_DANGLE_PAYLOAD_SIZE = 284
_ANCHOR_LAYOUTS = {
    0: (0x70, 0x78),
    12: (0x30, 0x38),
}
_FAMILY_PARENT_MAX_ERROR = 0.5
_FAMILY_PARENT_MAX_RATIO = 0.75

_IDENTITY_4X4 = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)
_MOD_SUFFIX_RE = re.compile(r"^(.*?)(_MOD\d*)$", re.IGNORECASE)


class MReflexError(ValueError):
    """Raised when a file is not a structurally valid supported MReflex."""


@dataclass(frozen=True)
class ReflexRecord:
    index: int
    offset: int
    kind: int
    guid: bytes
    header_a: int
    header_b: int
    ordinal: int
    payload_size: int

    @property
    def end_offset(self):
        return self.offset + 40 + self.payload_size


@dataclass(frozen=True)
class DangleNode:
    record_index: int
    record_offset: int
    node_id: int
    parent_id: int
    rest_matrix: tuple
    gravity: float
    weight: float
    spring: float
    damping: float
    limits: tuple


@dataclass(frozen=True)
class DangleNodeUpdate:
    record_index: int
    node_id: int
    gravity: float
    weight: float
    spring: float
    damping: float
    limits: tuple


@dataclass(frozen=True)
class TransformAnchor:
    record_index: int
    record_offset: int
    kind: int
    node_id: int
    rest_matrix: tuple


@dataclass(frozen=True)
class BoneCandidate:
    index: int
    name: str
    parent_index: int
    reflex_matrix: tuple


@dataclass(frozen=True)
class BoneMatch:
    bone_index: int
    bone_name: str
    error: float
    hierarchy_only: bool = False
    family_verified: bool = False


def _u16(data, offset):
    if offset < 0 or offset + 2 > len(data):
        raise MReflexError(f"uint16 at 0x{offset:X} is out of bounds")
    return struct.unpack_from("<H", data, offset)[0]


def _u32(data, offset):
    if offset < 0 or offset + 4 > len(data):
        raise MReflexError(f"uint32 at 0x{offset:X} is out of bounds")
    return struct.unpack_from("<I", data, offset)[0]


def _f32(data, offset):
    if offset < 0 or offset + 4 > len(data):
        raise MReflexError(f"float32 at 0x{offset:X} is out of bounds")
    return struct.unpack_from("<f", data, offset)[0]


def _f32s(data, offset, count):
    size = count * 4
    if offset < 0 or offset + size > len(data):
        raise MReflexError(
            f"{count} float32 values at 0x{offset:X} are out of bounds")
    return struct.unpack_from(f"<{count}f", data, offset)


def _walk_records(data, start, count):
    records = []
    offset = start
    for index in range(count):
        if offset + 40 > len(data):
            raise MReflexError(
                f"record {index} header at 0x{offset:X} is truncated")
        tag, kind = struct.unpack_from("<II", data, offset)
        if tag != RECORD_TAG:
            raise MReflexError(
                f"record {index} at 0x{offset:X} has tag 0x{tag:08X}, "
                f"expected 0x{RECORD_TAG:08X}")
        header_a, header_b, ordinal, payload_size = struct.unpack_from(
            "<IIII", data, offset + 0x18)
        end = offset + 40 + payload_size
        if end > len(data):
            raise MReflexError(
                f"record {index} at 0x{offset:X} extends past EOF")
        records.append(ReflexRecord(
            index=index,
            offset=offset,
            kind=kind,
            guid=bytes(data[offset + 8:offset + 0x18]),
            header_a=header_a,
            header_b=header_b,
            ordinal=ordinal,
            payload_size=payload_size,
        ))
        offset = end
    if offset != len(data):
        raise MReflexError(
            f"record stream ends at 0x{offset:X}, not EOF 0x{len(data):X}")
    return records


def parse_records(data):
    """Validate the container and return every top-level record."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("MReflex data must be bytes-like")
    data = bytes(data)
    if len(data) < 20:
        raise MReflexError("MReflex header is truncated")
    magic, version = struct.unpack_from("<II", data, 0)
    if magic != MAGIC:
        raise MReflexError(
            f"invalid MReflex magic 0x{magic:08X}; expected 0x{MAGIC:08X}")
    if version != VERSION:
        raise MReflexError(
            f"unsupported MReflex version {version}; expected {VERSION}")

    record_count = _u32(data, 0x0C)
    marker_offset = data.find(TGET_MARKER, 20)
    errors = []
    while marker_offset >= 0:
        try:
            table_count = _u16(data, marker_offset - 2)
            record_start = marker_offset + 4 + table_count * 4
            return _walk_records(data, record_start, record_count)
        except MReflexError as error:
            errors.append(str(error))
            marker_offset = data.find(TGET_MARKER, marker_offset + 1)
    detail = f" ({errors[-1]})" if errors else ""
    raise MReflexError(f"no valid TGET record-table boundary found{detail}")


def is_mreflex(data):
    """Return whether *data* is a supported, structurally valid MReflex file."""
    try:
        parse_records(data)
        return True
    except (MReflexError, TypeError, struct.error):
        return False


def _validate_dangle_record(data, record):
    if record.header_a != KIND_DANGLE_HEADER_A:
        raise MReflexError(
            f"kind-10 record {record.index} has header A {record.header_a}, "
            f"expected {KIND_DANGLE_HEADER_A}")
    if record.payload_size != KIND_DANGLE_PAYLOAD_SIZE:
        raise MReflexError(
            f"kind-10 record {record.index} has payload size "
            f"{record.payload_size}, expected {KIND_DANGLE_PAYLOAD_SIZE}")
    offset = record.offset
    if _u32(data, offset + 0x28) != record.ordinal:
        raise MReflexError(
            f"kind-10 record {record.index} has mismatched ordinal copy")
    if _u32(data, offset + 0x2C) != record.header_b:
        raise MReflexError(
            f"kind-10 record {record.index} has mismatched header-B copy")
    parent_id = _u32(data, offset + 0x34)
    if _u32(data, offset + 0x78) != parent_id:
        raise MReflexError(
            f"kind-10 record {record.index} has mismatched parent-ID copies")
    identity = _f32s(data, offset + 0xB4, 16)
    if any(not math.isfinite(actual) or abs(actual - expected) > 1e-6
           for actual, expected in zip(identity, _IDENTITY_4X4)):
        raise MReflexError(
            f"kind-10 record {record.index} has an invalid identity matrix")


def read_dangle_nodes(data):
    """Return every structurally validated kind-10 dangle node."""
    data = bytes(data)
    nodes = []
    for record in parse_records(data):
        if record.kind != KIND_DANGLE_NODE:
            continue
        _validate_dangle_record(data, record)
        offset = record.offset
        gravity_a = _f32(data, offset + 0xA4)
        gravity_b = _f32(data, offset + 0x11C)
        rest_matrix = _f32s(data, offset + 0x38, 16)
        weight = _f32(data, offset + 0xFC)
        spring = _f32(data, offset + 0x110)
        damping = _f32(data, offset + 0x114)
        limits = _f32s(data, offset + 0x120, 4)
        if not math.isclose(gravity_a, gravity_b, rel_tol=0.0, abs_tol=1e-6):
            raise MReflexError(
                f"kind-10 record {record.index} has mismatched gravity copies")
        if not all(math.isfinite(value) for value in (
                *rest_matrix, gravity_a, weight, spring, damping, *limits)):
            raise MReflexError(
                f"kind-10 record {record.index} contains a non-finite value")
        nodes.append(DangleNode(
            record_index=record.index,
            record_offset=offset,
            node_id=_u32(data, offset + 0x30),
            parent_id=_u32(data, offset + 0x34),
            rest_matrix=rest_matrix,
            gravity=gravity_a,
            weight=weight,
            spring=spring,
            damping=damping,
            limits=limits,
        ))
    return nodes


def read_transform_anchors(data, nodes=None):
    """Return non-kind-10 transforms referenced as dangle-node parents.

    Kinds 0 and 12 use different offsets for the graph ID and local transform.
    Only records whose ID is referenced by a validated kind-10 node are
    exposed, and malformed/non-affine candidates are ignored conservatively.
    """
    data = bytes(data)
    records = parse_records(data)
    nodes = tuple(nodes) if nodes is not None else read_dangle_nodes(data)
    referenced_ids = {node.parent_id for node in nodes}
    anchors = []
    for record in records:
        layout = _ANCHOR_LAYOUTS.get(record.kind)
        if layout is None:
            continue
        id_offset, matrix_offset = layout
        if record.offset + matrix_offset + 64 > record.end_offset:
            continue
        node_id = _u32(data, record.offset + id_offset)
        if node_id not in referenced_ids:
            continue
        matrix = _f32s(data, record.offset + matrix_offset, 16)
        if not all(math.isfinite(value) for value in matrix):
            continue
        if any(
            abs(actual - expected) > 1e-5
            for actual, expected in zip(matrix[12:16], (0.0, 0.0, 0.0, 1.0))
        ):
            continue
        anchors.append(TransformAnchor(
            record_index=record.index,
            record_offset=record.offset,
            kind=record.kind,
            node_id=node_id,
            rest_matrix=matrix,
        ))
    return anchors


def _require_finite(name, values):
    for value in values:
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must contain only finite values")


def rewrite_dangle_nodes(data, updates):
    """Patch kind-10 parameters while preserving all other bytes and file size."""
    source = bytes(data)
    nodes = read_dangle_nodes(source)
    by_index = {node.record_index: node for node in nodes}
    output = bytearray(source)
    seen = set()
    for update in updates:
        if update.record_index in seen:
            raise ValueError(
                f"duplicate update for record {update.record_index}")
        seen.add(update.record_index)
        node = by_index.get(update.record_index)
        if node is None:
            raise ValueError(
                f"record {update.record_index} is not a validated kind-10 node")
        if int(update.node_id) != node.node_id:
            raise ValueError(
                f"record {update.record_index} node ID changed from "
                f"0x{node.node_id:08X} to 0x{int(update.node_id):08X}")
        limits = tuple(float(value) for value in update.limits)
        if len(limits) != 4:
            raise ValueError("each dangle-node update requires four limits")
        values = (
            float(update.gravity),
            float(update.weight),
            float(update.spring),
            float(update.damping),
            *limits,
        )
        _require_finite(f"record {update.record_index}", values)
        offset = node.record_offset
        struct.pack_into("<f", output, offset + 0xA4, values[0])
        struct.pack_into("<f", output, offset + 0xFC, values[1])
        struct.pack_into("<f", output, offset + 0x110, values[2])
        struct.pack_into("<f", output, offset + 0x114, values[3])
        struct.pack_into("<f", output, offset + 0x11C, values[0])
        struct.pack_into("<4f", output, offset + 0x120, *limits)

    rewritten = bytes(output)
    if len(rewritten) != len(source):
        raise AssertionError("MReflex rewrite changed the file size")
    read_dangle_nodes(rewritten)
    return rewritten


def _matrix_error(left, right):
    if len(left) != 16 or len(right) != 16:
        raise ValueError("bone and node transforms must contain 16 values")
    return max(abs(float(a) - float(b)) for a, b in zip(left, right))


def reflect_x_basis(matrix):
    """Convert a row-major MMB local matrix to the MReflex coordinate basis."""
    values = tuple(float(value) for value in matrix)
    if len(values) != 16:
        raise ValueError("matrix must contain 16 values")
    signs = (-1.0, 1.0, 1.0, 1.0)
    return tuple(
        signs[row] * signs[column] * values[row * 4 + column]
        for row in range(4)
        for column in range(4)
    )


def _bone_name_variant(name):
    """Normalize separators and remove only a leading side marker."""
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    tokens = [
        token.casefold()
        for token in re.split(r"[^A-Za-z0-9]+", separated)
        if token
    ]
    if tokens and tokens[0] in {
            "l", "r", "c", "left", "right", "center"}:
        tokens.pop(0)
    return "_".join(tokens)


def _bone_name_family(name):
    """Normalize side/index variants into a conservative bone-family key."""
    normalized = []
    for token in _bone_name_variant(name).split("_"):
        token = re.sub(r"\d+$", "", token)
        if token:
            normalized.append(token)
    return "_".join(normalized)


def match_dangle_nodes(
        nodes, bones, weighted_bone_indices, anchors=(), tolerance=1e-4):
    """Conservatively map nodes to uniquely compatible weighted MMB bones.

    Node transforms supply weighted-bone candidates. Referenced kind-0/kind-12
    anchors may match any MMB bone, including unweighted attachment bones.
    A uniquely matched anchor may recover one stale child transform only when
    it has exactly one weighted direct child. A second provenance tier can
    verify an unused member of an established bone-name family when its stale
    anchor is uniquely closest to that member's parent. Graph links, MMB
    hierarchy, and one-to-one assignment otherwise only remove candidates;
    nodes that do not reduce to one safe candidate are omitted.
    """
    weighted = set(int(index) for index in weighted_bone_indices)
    bone_by_index = {bone.index: bone for bone in bones}
    weighted_bones = {
        index: bone for index, bone in bone_by_index.items()
        if index in weighted
    }
    candidates = {}
    errors = {}
    node_keys = {}
    for node in nodes:
        key = ("node", node.record_index)
        node_keys[node.record_index] = key
        options = set()
        for bone in weighted_bones.values():
            error = _matrix_error(node.rest_matrix, bone.reflex_matrix)
            errors[node.record_index, bone.index] = error
            if error <= tolerance:
                options.add(bone.index)
        candidates[key] = options

    anchor_keys = {}
    for anchor in anchors:
        key = ("anchor", anchor.record_index)
        anchor_keys[anchor.record_index] = key
        candidates[key] = {
            bone.index for bone in bone_by_index.values()
            if _matrix_error(anchor.rest_matrix, bone.reflex_matrix) <= tolerance
        }

    id_entities = {}
    for node in nodes:
        id_entities.setdefault(node.node_id, []).append(
            node_keys[node.record_index])
    for anchor in anchors:
        id_entities.setdefault(anchor.node_id, []).append(
            anchor_keys[anchor.record_index])

    parent_entity = {}
    for node in nodes:
        possible = id_entities.get(node.parent_id, ())
        if len(possible) == 1:
            parent_entity[node_keys[node.record_index]] = possible[0]

    hierarchy_only = set()
    anchor_entities = set(anchor_keys.values())
    for child, parent in parent_entity.items():
        if parent not in anchor_entities or len(candidates[parent]) != 1:
            continue
        parent_bone = next(iter(candidates[parent]))
        compatible = {
            index for index in candidates[child]
            if bone_by_index[index].parent_index == parent_bone
        }
        if compatible:
            continue
        eligible_children = {
            index for index in weighted_bones
            if bone_by_index[index].parent_index == parent_bone
        }
        if len(eligible_children) == 1:
            candidates[child] = eligible_children
            hierarchy_only.add(child)

    def propagate_constraints():
        changed = True
        while changed:
            changed = False
            for child, parent in parent_entity.items():
                child_options = candidates[child]
                parent_options = candidates[parent]
                filtered_children = {
                    index for index in child_options
                    if bone_by_index[index].parent_index in parent_options
                }
                filtered_parents = parent_options & {
                    bone_by_index[index].parent_index
                    for index in filtered_children
                }
                if filtered_children != child_options:
                    candidates[child] = filtered_children
                    changed = True
                if filtered_parents != parent_options:
                    candidates[parent] = filtered_parents
                    changed = True

            singleton_owners = {}
            for key, options in candidates.items():
                if len(options) == 1:
                    singleton_owners.setdefault(
                        next(iter(options)), []).append(key)
            for bone_index, owners in singleton_owners.items():
                if len(owners) != 1:
                    continue
                owner = owners[0]
                for key, options in candidates.items():
                    if key == owner or bone_index not in options:
                        continue
                    candidates[key] = options - {bone_index}
                    changed = True

    def resolved_assignments():
        singleton_owners = {}
        for key, options in candidates.items():
            if len(options) == 1:
                singleton_owners.setdefault(
                    next(iter(options)), []).append(key)
        return {
            owners[0]: bone_index
            for bone_index, owners in singleton_owners.items()
            if len(owners) == 1
        }

    propagate_constraints()

    resolved = resolved_assignments()
    matched_family_counts = {}
    for node in nodes:
        bone_index = resolved.get(node_keys[node.record_index])
        if bone_index is None:
            continue
        bone_name = bone_by_index[bone_index].name
        family = _bone_name_family(bone_name)
        if family:
            matched_family_counts[family] = (
                matched_family_counts.get(family, 0) + 1)
    established_families = {
        family for family, count in matched_family_counts.items()
        if count >= 2
    }

    family_verified = set()
    if established_families:
        used_bones = {
            resolved[key]
            for key in node_keys.values()
            if key in resolved
        }
        unused_family_bones = {
            index: bone for index, bone in weighted_bones.items()
            if (
                index not in used_bones
                and _bone_name_family(bone.name) in established_families
            )
        }
        anchor_by_key = {
            anchor_keys[anchor.record_index]: anchor
            for anchor in anchors
        }
        proposals = {}
        for node in nodes:
            child = node_keys[node.record_index]
            if child in resolved:
                continue
            parent = parent_entity.get(child)
            anchor = anchor_by_key.get(parent)
            if anchor is None or not unused_family_bones:
                continue
            ranked_parents = sorted(
                (
                    _matrix_error(
                        anchor.rest_matrix, bone.reflex_matrix),
                    bone.index,
                )
                for bone in bone_by_index.values()
            )
            if len(ranked_parents) < 2:
                continue
            best_error, best_parent = ranked_parents[0]
            second_error = ranked_parents[1][0]
            if (
                best_error > _FAMILY_PARENT_MAX_ERROR
                or second_error <= 0.0
                or best_error / second_error > _FAMILY_PARENT_MAX_RATIO
            ):
                continue
            eligible_children = {
                index for index, bone in unused_family_bones.items()
                if bone.parent_index == best_parent
            }
            if len(eligible_children) != 1:
                continue
            candidate = next(iter(eligible_children))
            proposals[child] = (candidate, parent)

        bone_owners = {}
        parent_owners = {}
        for child, (bone_index, parent) in proposals.items():
            bone_owners.setdefault(bone_index, []).append(child)
            parent_owners.setdefault(parent, []).append(child)
        for child, (bone_index, parent) in proposals.items():
            if (
                len(bone_owners[bone_index]) != 1
                or len(parent_owners[parent]) != 1
            ):
                continue
            candidates[child] = {bone_index}
            for anchor_entity in anchor_entities:
                if (
                    anchor_entity != parent
                    and bone_index in candidates[anchor_entity]
                ):
                    candidates[anchor_entity] -= {bone_index}
            family_verified.add(child)

    matches = {}
    resolved = resolved_assignments()
    for node in nodes:
        key = node_keys[node.record_index]
        bone_index = resolved.get(key)
        if bone_index is None:
            continue
        bone = bone_by_index[bone_index]
        matches[node.record_index] = BoneMatch(
            bone_index=bone_index,
            bone_name=bone.name,
            error=errors[node.record_index, bone_index],
            hierarchy_only=key in hierarchy_only,
            family_verified=key in family_verified,
        )
    return matches


def paired_mreflex_path(mmb_path):
    """Return an existing same-name (or pre-``_MOD``) MReflex partner."""
    path = Path(os.path.abspath(mmb_path))
    candidates = [path.with_suffix(".mreflex")]
    match = _MOD_SUFFIX_RE.match(path.stem)
    if match:
        candidates.append(path.with_name(match.group(1) + ".mreflex"))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return ""


def mod_output_path(source_path, overwrite=False):
    """Choose a non-destructive ``_MOD`` path for any MReflex source."""
    source = Path(os.path.abspath(source_path))
    match = _MOD_SUFFIX_RE.match(source.stem)
    base_stem = match.group(1) if match else source.stem
    candidate = source.with_name(base_stem + "_MOD" + source.suffix)
    if overwrite or not candidate.exists():
        return str(candidate)
    index = 1
    while True:
        candidate = source.with_name(
            f"{base_stem}_MOD{index}{source.suffix}")
        if not candidate.exists():
            return str(candidate)
        index += 1

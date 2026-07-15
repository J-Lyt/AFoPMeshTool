"""Blender-side skeletal mesh exporter and MMB header patcher."""

import io
import operator
from struct import pack, unpack

import bmesh
import bpy
import numpy as np
from mathutils import Vector

from . import addon_state
from .binary_io import bp
from .blender_mesh_utils import (
    bake_parent_inverse,
    compute_normals_for_object,
    find_object_by_name,
    triangulate_object,
)
from .cloth_export import _sim_free_slot_flags
from .log import logger
from .mmb import SkeletalMeshAsset, _resolve_uv_encoding


def _patch_mesh_tail_delta(file_data, mesh, vertex_delta=0, index_delta=0,
                           data_delta=0, flag=None):
    """Apply known semantic deltas to a mesh's aggregate tail fields.

    The tail data total is not simply sum(lod.data_size): stock files may
    exclude 4-24 bytes of terminal/padding data. Updating by typed-payload
    delta preserves that source-specific convention.
    """
    off = getattr(mesh, 'index_width_flag_offset', 0)
    if off <= 0 or off + 16 > len(file_data):
        raise ValueError(f"'{mesh.name}' has no valid mesh-tail offset")
    old_flag, total_vc, total_ic, total_data = unpack(
        '<4I', file_data[off:off + 16])
    new_values = (
        old_flag if flag is None else flag,
        total_vc + vertex_delta,
        total_ic + index_delta,
        total_data + data_delta,
    )
    if any(v < 0 or v > 0xFFFFFFFF for v in new_values):
        raise ValueError(f"'{mesh.name}' mesh-tail update is outside uint32 range")
    file_data[off:off + 16] = pack('<4I', *new_values)
    mesh.index_width_flag = new_values[0]
    mesh.total_vertex_count = new_values[1]
    mesh.total_index_count = new_values[2]
    mesh.total_data_size = new_values[3]


def _sync_asset_layout(parsed_asset):
    """Refresh binary offsets/counts without discarding export-time cloth state."""
    current = addon_state.asset
    if current is None or len(current.meshes) != len(parsed_asset.meshes):
        return
    current.size = parsed_asset.size
    for old_mesh, new_mesh in zip(current.meshes, parsed_asset.meshes):
        for attr in (
                'name_offset', 'name_length', 'lod_count', 'lod_info_type',
                'vertex_stride', 'normals_stride', 'u_count_offset',
                'bone_table_end_offset', 'index_width_flag_offset',
                'index_width_flag', 'total_vertex_count_offset',
                'total_vertex_count', 'total_index_count_offset',
                'total_index_count', 'total_data_size_offset',
                'total_data_size', 'tail_extra_u32'):
            setattr(old_mesh, attr, getattr(new_mesh, attr))
        old_mesh.mesh_bone_file_offsets = list(new_mesh.mesh_bone_file_offsets)
        if len(old_mesh.lods) != len(new_mesh.lods):
            continue
        for old_lod, new_lod in zip(old_mesh.lods, new_mesh.lods):
            for attr in (
                    'start_offset', 'vertex_count', 'index_count', 'size_a',
                    'vertex_data_offset_a', 'vertex_data_offset_b',
                    'face_block_offset', 'data_offset_file_pos', 'data_offset',
                    'data_size', 'is_header_lod', 'lod_unk',
                    'index_width_bytes'):
                setattr(old_lod, attr, getattr(new_lod, attr))


class BlenderMeshExporter:
    _compute_normals_for_object = staticmethod(compute_normals_for_object)
    find_object_by_name = staticmethod(find_object_by_name)
    _bake_parent_inverse = staticmethod(bake_parent_inverse)
    _triangulate_object = staticmethod(triangulate_object)

    @staticmethod
    def promote_mixed_index_widths(file_path: str):
        """Widen mixed-width meshes to uniform uint32 index buffers.

        The engine's width flag is per mesh, while size_a records the matching
        width in every LOD header. If one LOD becomes uint32, all non-empty
        sibling LODs are widened. Non-index bytes and face trailers are copied
        verbatim; primary/secondary offsets and asset_size are patched as one
        transaction. Returns the widened mesh names.
        """
        with open(file_path, 'rb') as f:
            original = bytearray(f.read())
        parsed = SkeletalMeshAsset()
        parsed.parse(io.BytesIO(bytes(original)))

        widen_meshes = set()
        flag_only_meshes = set()
        for mi, mesh in enumerate(parsed.meshes):
            widths = {lod.index_width_bytes for lod in mesh.lods
                      if lod.index_width_bytes in (2, 4)}
            invalid = [lod.index for lod in mesh.lods
                       if lod.index_count > 0 and lod.index_width_bytes not in (2, 4)]
            if invalid:
                raise ValueError(
                    f"'{mesh.name}' has ambiguous index width in LOD(s) {invalid}")
            if any(lod.vertex_count > 65535 and lod.index_width_bytes == 2
                   for lod in mesh.lods):
                raise ValueError(
                    f"'{mesh.name}' contains a uint16 LOD above 65535 vertices")
            if 2 in widths and (4 in widths or mesh.index_width_flag == 1):
                widen_meshes.add(mi)
            elif widths == {4} and mesh.index_width_flag != 1:
                flag_only_meshes.add(mi)

        if not widen_meshes and not flag_only_meshes:
            _sync_asset_layout(parsed)
            return []

        blocks = {}
        new_width = {}
        widening_delta = {mi: 0 for mi in widen_meshes}
        for mi, mesh in enumerate(parsed.meshes):
            old_higher = 0
            new_higher = 0
            for lod in reversed(mesh.lods):
                if lod.data_size <= 0:
                    continue
                key = (mi, lod.index)
                intra_voa = lod.vertex_data_offset_a - old_higher
                intra_vob = lod.vertex_data_offset_b - old_higher
                intra_fb = lod.face_block_offset - old_higher
                old_intra_fb = intra_fb
                if not (0 <= intra_voa <= intra_vob <= intra_fb <= lod.data_size):
                    raise ValueError(
                        f"'{mesh.name}' LOD{lod.index} has invalid primary offsets")
                block = bytes(original[lod.data_offset:lod.data_offset + lod.data_size])
                if len(block) != lod.data_size:
                    raise ValueError(f"'{mesh.name}' LOD{lod.index} is truncated")
                width = lod.index_width_bytes
                if width not in (2, 4):
                    # Empty face domains have no width-dependent bytes.
                    width = 4 if mesh.index_width_flag == 1 else 2
                old_face_end = old_intra_fb + lod.index_count * width
                if mi in widen_meshes and lod.index_count > 0 and width == 2:
                    face_end = intra_fb + lod.index_count * 2
                    if face_end > len(block):
                        raise ValueError(
                            f"'{mesh.name}' LOD{lod.index} index buffer is truncated")
                    narrow = block[intra_fb:face_end]
                    wide = np.frombuffer(narrow, dtype='<u2').astype('<u4').tobytes()
                    # Preserve all bytes before the face buffer and every trailer
                    # byte after it. Do not invent alignment after the trailer.
                    block = block[:intra_fb] + wide + block[face_end:]
                    width = 4
                    widening_delta[mi] += lod.index_count * 2
                if mi in widen_meshes and lod.index_count > 0 and width == 4:
                    # face_block_offset is cumulative across reverse-ordered
                    # sibling blocks. A widened higher LOD can shift the next
                    # face buffer to 2 mod 4 even when its own intra-block
                    # layout was aligned. Insert only the required pre-index
                    # sentinel padding; trailers remain byte-identical.
                    alignment = (-(new_higher + intra_fb)) % 4
                    if alignment:
                        padding = (b'\xfa\x7f' * 2)[:alignment]
                        block = block[:intra_fb] + padding + block[intra_fb:]
                        intra_fb += alignment
                blocks[key] = {
                    'bytes': block,
                    'old_offset': lod.data_offset,
                    'old_size': lod.data_size,
                    'intra_voa': intra_voa,
                    'intra_vob': intra_vob,
                    'intra_fb': intra_fb,
                    'old_intra_fb': old_intra_fb,
                    'old_face_end': old_face_end,
                }
                new_width[key] = width
                old_higher += lod.data_size
                new_higher += len(block)

        order = sorted((info['old_offset'], key) for key, info in blocks.items())
        if not order:
            raise ValueError('MMB has no primary LOD blocks to widen')
        for (prev_off, prev_key), (next_off, _next_key) in zip(order, order[1:]):
            previous_end = prev_off + blocks[prev_key]['old_size']
            if next_off < previous_end:
                raise ValueError(
                    f'Primary LOD blocks overlap at 0x{next_off:X}')

        data_start = order[0][0]
        new_offsets = {}
        out = bytearray(original[:data_start])
        cursor = data_start
        for old_offset, key in order:
            info = blocks[key]
            out += original[cursor:old_offset]
            new_offsets[key] = len(out)
            out += info['bytes']
            cursor = old_offset + info['old_size']
        out += original[cursor:]
        total_shift = len(out) - len(original)

        new_fields = {}
        for mi, mesh in enumerate(parsed.meshes):
            new_higher = 0
            for lod in reversed(mesh.lods):
                key = (mi, lod.index)
                if key not in blocks:
                    continue
                info = blocks[key]
                width = new_width[key]
                voa = new_higher + info['intra_voa']
                vob = new_higher + info['intra_vob']
                fb = new_higher + info['intra_fb']
                if fb % width:
                    raise ValueError(
                        f"'{mesh.name}' LOD{lod.index} face offset is not uint{width * 8}-aligned")
                new_fields[key] = (fb // width, voa, vob, fb, len(info['bytes']))
                new_higher += len(info['bytes'])

        for mi, mesh in enumerate(parsed.meshes):
            for lod in mesh.lods:
                key = (mi, lod.index)
                if key not in new_fields:
                    continue
                size_a, voa, vob, fb, data_size = new_fields[key]
                pos = lod.start_offset + lod.lod_field_offset
                out[pos + 8:pos + 12] = pack('<I', size_a)
                out[pos + 12:pos + 16] = pack('<I', voa)
                out[pos + 16:pos + 20] = pack('<I', vob)
                out[pos + 20:pos + 24] = pack('<I', fb)
                out[pos + 24:pos + 28] = pack('<I', new_offsets[key])
                out[pos + 28:pos + 32] = pack('<I', data_size)

        old_asset_size = unpack('<I', original[4:8])[0]
        header_growth = sum(
            len(blocks[key]['bytes']) - blocks[key]['old_size']
            for old_offset, key in order if old_offset < old_asset_size)
        out[4:8] = pack('<I', old_asset_size + header_growth)

        if total_shift:
            growth_events = [
                (old_offset + blocks[key]['old_size'],
                 len(blocks[key]['bytes']) - blocks[key]['old_size'])
                for old_offset, key in order
                if len(blocks[key]['bytes']) != blocks[key]['old_size']
            ]
            for mesh in parsed.meshes:
                if mesh.lod_info_type != 2:
                    continue
                for lod in mesh.lods:
                    pos = lod.start_offset + lod.lod_field_offset + 56
                    sec2_offset = unpack('<I', original[pos:pos + 4])[0]
                    if sec2_offset:
                        container = next(
                            ((old_offset, key) for old_offset, key in order
                             if old_offset <= sec2_offset
                             < old_offset + blocks[key]['old_size']),
                            None)
                        if container is not None:
                            old_offset, key = container
                            info = blocks[key]
                            relative = sec2_offset - old_offset
                            block_delta = len(info['bytes']) - info['old_size']
                            if (block_delta and info['old_intra_fb'] <= relative
                                    < info['old_face_end']):
                                raise ValueError(
                                    f"'{mesh.name}' LOD{lod.index} sec2 points into "
                                    "a widened primary index buffer")
                            if relative >= info['old_face_end']:
                                relative += block_delta
                            new_sec2_offset = new_offsets[key] + relative
                        else:
                            shift = sum(delta for event_offset, delta in growth_events
                                        if event_offset <= sec2_offset)
                            new_sec2_offset = sec2_offset + shift
                        out[pos:pos + 4] = pack('<I', new_sec2_offset)

        for mi, mesh in enumerate(parsed.meshes):
            widths = {new_width[(mi, lod.index)] for lod in mesh.lods
                      if (mi, lod.index) in new_width and lod.index_count > 0}
            if len(widths) > 1:
                raise ValueError(f"'{mesh.name}' is still mixed-width after widening")
            flag = 1 if widths == {4} else 0
            data_delta = widening_delta.get(mi, 0)
            _patch_mesh_tail_delta(
                out, mesh,
                vertex_delta=sum(lod.vertex_count for lod in mesh.lods)
                             - mesh.total_vertex_count,
                index_delta=sum(lod.index_count for lod in mesh.lods)
                            - mesh.total_index_count,
                data_delta=data_delta,
                flag=flag)

        validated = SkeletalMeshAsset()
        validated.parse(io.BytesIO(bytes(out)))
        for mesh in validated.meshes:
            widths = {lod.index_width_bytes for lod in mesh.lods
                      if lod.index_width_bytes in (2, 4)}
            expected_flag = 1 if widths == {4} else 0 if widths == {2} else None
            if len(widths) > 1 or (expected_flag is not None
                                   and mesh.index_width_flag != expected_flag):
                raise ValueError(f"'{mesh.name}' failed uint32 widening validation")

        with open(file_path, 'wb') as f:
            f.write(out)
        _sync_asset_layout(validated)
        widened_names = [parsed.meshes[mi].name for mi in sorted(widen_meshes)]
        for name in widened_names:
            logger.info("Widened all %s LOD index buffers to uint32", name)
        return widened_names

    @staticmethod
    def _write_mod_file(edited_lod_index_per_mesh: dict, out_path: str, src_path: str = None):
        """
        For each mesh/LOD pair in edited_lod_index_per_mesh, the vertex, normal,
        and face data are rewritten from the current Blender mesh. All other LODs
        and the full file header are copied verbatim from the source file.

        If the new data fits within the original block it is written in-place.
        If the new data is larger, the file is restructured: bytes are inserted at
        the end of the edited LOD's block, and all affected offsets are updated.

        :param edited_lod_index_per_mesh: dict mapping mesh_index -> lod_index to rewrite.
               A value of -1 means copy all LODs verbatim (no Blender mesh needed).
        :param out_path: destination file path to write.
        :param src_path: file to copy unedited data from. Defaults to SWOMT.AssetPath.
               ExportAllLODs passes the partially-written mod file here so each LOD
               level accumulates on top of the previous one.
        """
        SWOMT = bpy.context.scene.SWOMT
        if src_path is None:
            src_path = SWOMT.AssetPath

        with open(src_path, 'rb') as src:
            file_data = bytearray(src.read())

        # Sync lod.data_offset values from the source file (_MOD or original).
        for m in addon_state.asset.meshes:
            for lod in m.lods:
                fp = lod.data_offset_file_pos
                lod.data_offset = unpack('<I', file_data[fp:fp+4])[0]

        # Read asset.size (bytes 4-8) - boundary between header and streaming sections
        asset_size = unpack('<I', file_data[4:8])[0]

        for mesh_index, lod_index in edited_lod_index_per_mesh.items():
            if lod_index < 0:
                continue
            mesh = addon_state.asset.meshes[mesh_index]
            if lod_index >= len(mesh.lods):
                continue
            lod = mesh.lods[lod_index]

            obj = BME.find_object_by_name(mesh.name + f"_LOD{lod_index}")
            if obj is None:
                continue
            if len(obj.data.vertices) == 0:
                continue  # zero-vert mesh - auto-zero section will zero positions in file_data

            new_vert_count  = len(obj.data.vertices)
            new_index_count = len(obj.data.polygons) * 3

            # --- Slot-preserving layout for cloth render AND sim meshes ---
            # The cloth runtime keys to the original vertex numbering: renumbering
            # shreds the cloth in-game even with a fully consistent .mcloth.
            # So every vertex keeps its ORIGINAL index slot - deleted vertices remain
            # as face-less orphans copied verbatim.
            # RENDER: new vertices are appended after the original slots.
            # SIM: growing the sim counts crashes the engine (count-tied state we
            # cannot see), so new sim verts REUSE orphaned slots and new tris
            # reuse phantom tri slots - the sim BUDGET is the vanilla counts.
            _is_cloth_render = mesh.name.endswith('_CLOTH_RENDER')
            _is_cloth_sim = mesh.name.endswith('_CLOTH_SIM')
            slot_map = None # blender vertex index -> final slot index
            slot_final_vc = new_vert_count
            lod.exported_append_sources = {}
            lod.exported_append_base = 0
            lod.exported_sim_reused = set()
            lod.exported_sim_moved = set()
            lod.exported_sim_valid_tris = None
            if _is_cloth_render or _is_cloth_sim:
                _orig_vc_file = unpack('<I', file_data[lod.start_offset:lod.start_offset + 4])[0]
                if _orig_vc_file > 0:
                    _attr = obj.data.attributes.get('mmb_vertex_order')
                    # mmb_vertex_order values are valid slot claims only
                    # when they refer to THIS LOD's original numbering.
                    # Generated LODs carry LOD0 indices (mmb_lod_source) and from-scratch replacements
                    # have no attribute: all their vertices are appended instead,
                    # after the full set of orphaned original slots.
                    _can_claim = (_attr is not None
                                  and obj.get('mmb_lod_source', lod_index) == lod_index)
                    slot_map = {}
                    _claimed = set()
                    _extras = []
                    for _vi in range(new_vert_count):
                        _ov = _attr.data[_vi].value if _can_claim else -1
                        if 0 <= _ov < _orig_vc_file and _ov not in _claimed:
                            slot_map[_vi] = _ov
                            _claimed.add(_ov)
                        else:
                            _extras.append(_vi)
                    if _is_cloth_sim and _extras:
                        # Budget reuse: assign new verts to orphaned slots,
                        # FREE (simulating) slots first so new geometry
                        # simulates where possible.
                        _orphans = [s for s in range(_orig_vc_file) if s not in _claimed]
                        if len(_extras) > len(_orphans):
                            raise ValueError(
                                f"'{mesh.name}' sim budget exceeded: {len(_extras)} new "
                                f"vertices but only {len(_orphans)} deleted slot(s) to "
                                f"reuse (vanilla budget {_orig_vc_file}). Delete "
                                f"{len(_extras) - len(_orphans)} more sim vertices first.")
                        _free_flags = _sim_free_slot_flags(_orig_vc_file)
                        _orphans.sort(key=lambda s: (not _free_flags[s]) if _free_flags else 0)
                        for _k, _vi in enumerate(_extras):
                            _slot = _orphans[_k]
                            slot_map[_vi] = _slot
                            lod.exported_sim_reused.add(_slot)
                        slot_final_vc = _orig_vc_file
                    else:
                        for _k, _vi in enumerate(_extras):
                            _slot = _orig_vc_file + _k
                            slot_map[_vi] = _slot
                            # Record each appended slot's source vertex for the
                            # .mcloth row synthesis. Built here, from the EXPORT-time
                            # vertex set: seam splitting may have appended duplicate
                            # vertices that the restored Blender mesh doesn't have.
                            if _attr is not None:
                                lod.exported_append_sources[_slot] = _attr.data[_vi].value
                        slot_final_vc = _orig_vc_file + len(_extras)
                    lod.exported_append_base = _orig_vc_file

            # The preserve-bytes path below copies a normals block sized for the
            # original count, so rebuild whenever this LOD's count changed.
            # Slot mode also rebuilds: rows are written into slot order.
            export_normals = (SWOMT.export_normals or new_vert_count != lod.vertex_count or slot_map is not None)

            # Build new data blocks
            verts_buf = io.BytesIO()
            BME.write_vertices(verts_buf, mesh, lod_index)

            norms_buf = io.BytesIO()
            if export_normals:
                BME.write_normals(norms_buf, mesh, lod_index)
            else:
                # Preserve the original normals bytes from the source file unchanged.
                orig_higher_size = sum(
                    mesh.lods[li].data_size
                    for li in range(lod_index + 1, len(mesh.lods))
                )
                with open(SWOMT.AssetPath, 'rb') as orig_src:
                    orig_file_data = orig_src.read()
                orig_data_offset = unpack('<I', orig_file_data[lod.data_offset_file_pos:lod.data_offset_file_pos + 4])[0]
                file_vob_orig = unpack('<I', orig_file_data[lod.start_offset + lod.lod_field_offset + 16:lod.start_offset + lod.lod_field_offset + 20])[0]
                file_fb_orig = unpack('<I', orig_file_data[lod.start_offset + lod.lod_field_offset + 20:lod.start_offset + lod.lod_field_offset + 24])[0]
                orig_intra_vob = file_vob_orig - orig_higher_size
                orig_intra_fb  = file_fb_orig  - orig_higher_size
                orig_normals_size = orig_intra_fb - orig_intra_vob
                orig_abs_vob = orig_data_offset + orig_intra_vob
                norms_buf.write(orig_file_data[orig_abs_vob:orig_abs_vob + orig_normals_size])

            faces_buf = io.BytesIO()
            # Detect uint32 indices
            orig_sa_pre = unpack('<I', file_data[lod.start_offset + lod.lod_field_offset + 8: lod.start_offset + lod.lod_field_offset + 12])[0]
            orig_fb_pre = unpack('<I', file_data[lod.start_offset + lod.lod_field_offset + 20: lod.start_offset + lod.lod_field_offset + 24])[0]
            orig_uses_uint32_pre = (
                orig_fb_pre > 0 and orig_sa_pre * 4 == orig_fb_pre)
            use_uint32_faces = orig_uses_uint32_pre or slot_final_vc > 65535
            BME.write_triangles(faces_buf, mesh, lod_index, force_uint32=use_uint32_faces)

            vd = verts_buf.getvalue()
            nd = norms_buf.getvalue()
            fd = faces_buf.getvalue()

            if slot_map is not None:
                # Write the Blender-order rows into original-slot order.
                # Unclaimed slots (deleted vertices) keep their original bytes.
                _orig_vc_file = unpack('<I', file_data[lod.start_offset:lod.start_offset + 4])[0]
                _higher = sum(mesh.lods[li].data_size
                              for li in range(lod_index + 1, len(mesh.lods)))
                _abs_voa = lod.data_offset + (lod.vertex_data_offset_a - _higher)
                _abs_vob = lod.data_offset + (lod.vertex_data_offset_b - _higher)
                vs = mesh.vertex_stride
                ns = mesh.normals_stride

                _vd_out = bytearray(slot_final_vc * vs)
                _vd_out[:_orig_vc_file * vs] = file_data[_abs_voa:_abs_voa + _orig_vc_file * vs]
                _nd_out = bytearray(slot_final_vc * ns)
                _nd_out[:_orig_vc_file * ns] = file_data[_abs_vob:_abs_vob + _orig_vc_file * ns]
                for _vi, _slot in slot_map.items():
                    _vd_out[_slot * vs:(_slot + 1) * vs] = vd[_vi * vs:(_vi + 1) * vs]
                    _nd_out[_slot * ns:(_slot + 1) * ns] = nd[_vi * ns:(_vi + 1) * ns]
                # Record sim slots whose POSITION changed vs the source, here
                # where the source bytes (file_data) are still intact - the
                # exported mmb may overwrite the source file (chained _MOD
                # exports), so the mcloth layer can't recompute this afterward.
                if _is_cloth_sim and mesh.position_type == 1:
                    for _slot in slot_map.values():
                        if _slot >= _orig_vc_file or _slot in lod.exported_sim_reused:
                            continue
                        _o = _abs_voa + _slot * vs
                        _op = unpack('<fff', file_data[_o:_o + 12])
                        _np = unpack('<fff', _vd_out[_slot * vs:_slot * vs + 12])
                        if ((_np[0]-_op[0])**2 + (_np[1]-_op[1])**2
                                + (_np[2]-_op[2])**2) > 2.5e-9:  # 0.05mm
                            lod.exported_sim_moved.add(_slot)
                vd = bytes(_vd_out)
                nd = bytes(_nd_out)

                _isz = 4 if use_uint32_faces else 2
                _ifmt = '<I' if use_uint32_faces else '<H'
                if _is_cloth_sim:
                    # Keep the original sim triangle buffer layout (kept tris
                    # stay at their slots so 0x1129 refs and per-tri tables
                    # stay valid). New triangles REUSE phantom tri slots -
                    # the tri budget is the vanilla count (growing it crashes
                    # the engine).
                    _orig_ic = unpack('<I', file_data[lod.start_offset + lod.lod_field_offset + 4:lod.start_offset + lod.lod_field_offset + 8])[0]
                    _abs_fb0 = lod.data_offset + (lod.face_block_offset - _higher)
                    _orig_face_bytes = bytearray(file_data[_abs_fb0:_abs_fb0 + _orig_ic * _isz])
                    _orig_tri_slots = {}
                    for _t in range(_orig_ic // 3):
                        _tv = tuple(sorted(unpack(_ifmt, _orig_face_bytes[(_t * 3 + _j) * _isz:(_t * 3 + _j + 1) * _isz])[0]
                                           for _j in range(3)))
                        _orig_tri_slots.setdefault(_tv, []).append(_t)
                    _matched_tris = set()
                    _new_tris = []
                    for _p in obj.data.polygons:
                        _pv = list(_p.vertices)
                        for _fi in range(1, len(_pv) - 1):
                            _sv = (slot_map[_pv[0]], slot_map[_pv[_fi]], slot_map[_pv[_fi + 1]])
                            _key = tuple(sorted(_sv))
                            _cand = _orig_tri_slots.get(_key)
                            if _cand:
                                _t = _cand.pop()
                                _matched_tris.add(_t) # kept original tri
                                # A tri whose slot key coincides with a
                                # DELETED tri keeps that tri's ORIGINAL
                                # winding bytes - arbitrary for the new
                                # geometry. Refresh the winding from the
                                # Blender face when it touches a reused slot.
                                if any(v in lod.exported_sim_reused
                                       for v in _sv):
                                    _orig_face_bytes[
                                        _t * 3 * _isz:(_t + 1) * 3 * _isz] = (
                                        pack(_ifmt, _sv[0])
                                        + pack(_ifmt, _sv[2])
                                        + pack(_ifmt, _sv[1]))
                            else:
                                _new_tris.append(_sv)
                    _written_tris = set()
                    if _new_tris:
                        _phantoms = [t for tl in _orig_tri_slots.values() for t in tl]
                        _phantoms.sort()
                        if len(_new_tris) > len(_phantoms):
                            raise ValueError(
                                f"'{mesh.name}' sim triangle budget exceeded: "
                                f"{len(_new_tris)} new triangles but only "
                                f"{len(_phantoms)} deleted slot(s) to reuse "
                                f"(vanilla budget {_orig_ic // 3}). Delete "
                                f"{len(_new_tris) - len(_phantoms)} more sim faces first.")
                        for _sv, _t in zip(_new_tris, _phantoms):
                            # engine winding (v0, v2, v1)
                            _orig_face_bytes[_t * 3 * _isz:(_t + 1) * 3 * _isz] = (
                                pack(_ifmt, _sv[0]) + pack(_ifmt, _sv[2])
                                + pack(_ifmt, _sv[1]))
                            _written_tris.add(_t)
                    # Leftover phantom tris still referencing a REUSED slot
                    # would stretch from their old neighborhood to the new
                    # geometry - visible sliver faces AND long sliver edges
                    # fed to the constraint cooker (which then ties the new
                    # region to the old one). Retarget each reused reference
                    # to the surviving vert nearest the DELETED vert's
                    # original position: the tri stays a local, inert sliver.
                    if lod.exported_sim_reused and mesh.position_type == 1:
                        _leftover = {t for tl in _orig_tri_slots.values()
                                     for t in tl} - _written_tris
                        _surv = sorted(_claimed)
                        def _fpos(s):
                            return unpack('<fff', file_data[
                                _abs_voa + s * vs:_abs_voa + s * vs + 12])
                        _spos = {s: _fpos(s) for s in _surv}
                        _near = {}
                        for s in lod.exported_sim_reused:
                            p = _fpos(s)
                            _near[s] = min(
                                _surv, key=lambda u: (
                                    (p[0]-_spos[u][0])**2
                                    + (p[1]-_spos[u][1])**2
                                    + (p[2]-_spos[u][2])**2)) \
                                if _surv else s
                        _retg = 0
                        for _t in _leftover:
                            _tv = [unpack(_ifmt, _orig_face_bytes[
                                (_t*3+_j)*_isz:(_t*3+_j+1)*_isz])[0]
                                for _j in range(3)]
                            if any(v in lod.exported_sim_reused for v in _tv):
                                for _j in range(3):
                                    _orig_face_bytes[
                                        (_t*3+_j)*_isz:(_t*3+_j+1)*_isz] = \
                                        pack(_ifmt, _near.get(_tv[_j],
                                                              _tv[_j]))
                                _retg += 1
                        if _retg:
                            logger.debug(
                                "%s: %d leftover phantom triangle(s) retargeted "
                                "off reused slots",
                                mesh.name, _retg)
                    fd = bytes(_orig_face_bytes)
                    new_index_count = len(fd) // _isz
                    # Tris that exist in the CURRENT sim mesh (kept originals +
                    # rewritten slots). Leftover phantoms are excluded: render
                    # rows referencing them must be re-attached or released.
                    lod.exported_sim_valid_tris = _matched_tris | _written_tris
                else:
                    # Render: remap face indices from Blender to slot numbering.
                    _fd_out = bytearray(len(fd))
                    for _o in range(0, len(fd), _isz):
                        _bvi = unpack(_ifmt, fd[_o:_o + _isz])[0]
                        _fd_out[_o:_o + _isz] = pack(_ifmt, slot_map[_bvi])
                    fd = bytes(_fd_out)

                new_vert_count = slot_final_vc

            if lod.vertex_end_bytes:
                vd += lod.vertex_end_bytes
            if export_normals and lod.normals_end_bytes:
                nd += lod.normals_end_bytes
            if lod.faces_end_bytes:
                fd += lod.faces_end_bytes

            # Signal the .mcloth rewriter to keep every row (identity mapping)
            lod.exported_slot_identity = (slot_final_vc if slot_map is not None else 0)

            # higher_size: sum of data_size for LODs with index -> lod_index in this mesh.
            higher_size = sum(
                mesh.lods[li].data_size
                for li in range(lod_index + 1, len(mesh.lods))
            )

            # Intra-block positions (relative to lod.data_offset).
            # These are preserved from the original and must NOT be included in new_data_size.
            intra_voa = lod.vertex_data_offset_a - higher_size

            # New intra-block positions based on actual buffer sizes.
            new_intra_vob = intra_voa + len(vd)
            new_intra_fb  = new_intra_vob + len(nd)
            if use_uint32_faces:
                alignment = (-(higher_size + new_intra_fb)) % 4
                if alignment:
                    nd += (b'\xfa\x7f' * 2)[:alignment]
                    new_intra_fb += alignment

            # new_data_size is the only writable region (vd+nd+fd), excluding the preserved prefix bytes.
            file_lod_data_size = unpack('<I', file_data[lod.start_offset + lod.lod_field_offset + 28:lod.start_offset + lod.lod_field_offset + 32])[0]
            orig_data_size   = file_lod_data_size - intra_voa  # writable region in source

            # Preserve any trailing bytes after face data (e.g. fa7f sentinel padding).
            orig_idx_size = 4 if orig_uses_uint32_pre else 2
            # Use ORIGINAL vd/nd/fd sizes to correctly locate trailing bytes.
            orig_vc = unpack('<I', file_data[lod.start_offset:lod.start_offset + 4])[0]
            orig_ic = unpack('<I', file_data[lod.start_offset + lod.lod_field_offset + 4:lod.start_offset + lod.lod_field_offset + 8])[0]
            orig_vd_size = orig_vc * mesh.vertex_stride
            orig_nd_size = orig_vc * mesh.normals_stride
            orig_fd_size = orig_ic * orig_idx_size
            orig_trailing_size = orig_data_size - orig_vd_size - orig_nd_size - orig_fd_size
            if orig_trailing_size > 0:
                trailing_abs = lod.data_offset + intra_voa + orig_vd_size + orig_nd_size + orig_fd_size
                trailing_bytes = bytes(file_data[trailing_abs:trailing_abs + orig_trailing_size])
            else:
                trailing_bytes = b''
                orig_trailing_size = 0

            new_data_size    = len(vd) + len(nd) + len(fd) + len(trailing_bytes)
            delta = new_data_size - orig_data_size  # bytes added (negative = shrink)

            # Insertion point is at the end of the full block (including prefix)
            insert_at   = lod.data_offset + file_lod_data_size

            if delta > 0:
                # Growing: insert delta bytes into file_data at insert_at
                file_data[insert_at:insert_at] = b'\x00' * delta
            elif delta < 0:
                # Shrinking: remove the freed bytes from file_data
                shrink_start = lod.data_offset + intra_voa + new_data_size
                del file_data[shrink_start:insert_at]

            if delta != 0:
                # Update data_offset for every LOD in every mesh whose block starts after the insertion point
                for other_mesh in addon_state.asset.meshes:
                    for other_lod in other_mesh.lods:
                        if other_lod.data_offset > lod.data_offset:
                            fp = other_lod.data_offset_file_pos
                            old_val = unpack('<I', file_data[fp:fp+4])[0]
                            file_data[fp:fp+4] = pack('<I', old_val + delta)

                # Update asset.size if the edited LOD lives in the header section
                if lod.data_offset < asset_size:
                    asset_size += delta
                    file_data[4:8] = pack('<I', asset_size)

                # Update voa/vob/fb AND size_a for lower-indexed LODs in this mesh.
                # size_a must stay fb//2 (fb//4 for uint32 indices) or the mesh
                # corrupts in-game.
                for li in range(0, lod_index):
                    other_lod = mesh.lods[li]
                    so = other_lod.start_offset
                    fo = other_lod.lod_field_offset
                    old_fb = unpack('<I', file_data[so + fo + 20:so + fo + 24])[0]
                    old_sa = unpack('<I', file_data[so + fo + 8:so + fo + 12])[0]
                    for field_off in (12, 16, 20):  # voa, vob, fb
                        old = unpack('<I', file_data[so + fo + field_off:so + fo + field_off + 4])[0]
                        file_data[so + fo + field_off:so + fo + field_off + 4] = pack('<I', old + delta)
                    other_uses_uint32 = (old_fb > 0 and old_sa == old_fb // 4)
                    new_sa = (old_fb + delta) // 4 if other_uses_uint32 else (old_fb + delta) // 2
                    file_data[so + fo + 8:so + fo + 12] = pack('<I', new_sa)

                # Shift the second-section ABSOLUTE offsets - field [5] of the 28
                # extra LOD header bytes (see the LOD list notes in Mesh.parse),
                # located at data_offset field + 32. Those blocks live after all
                # primary LOD blocks, so resizing any LOD moves them. '>=' matters:
                # the first block starts exactly at insert_at when LOD0 grows.
                for other_mesh in addon_state.asset.meshes:
                    if getattr(other_mesh, 'lod_info_type', 0) != 2:
                        continue
                    for other_lod in other_mesh.lods:
                        sp = other_lod.data_offset_file_pos + 32
                        old_val = unpack('<I', file_data[sp:sp + 4])[0]
                        if old_val >= insert_at:
                            file_data[sp:sp + 4] = pack('<I', old_val + delta)

            # Write the new vertex/normal/face data at their correct positions.
            abs_voa     = lod.data_offset + intra_voa
            new_abs_vob = lod.data_offset + new_intra_vob
            new_abs_fb  = lod.data_offset + new_intra_fb

            file_data[abs_voa    :abs_voa     + len(vd)] = vd
            file_data[new_abs_vob:new_abs_vob + len(nd)] = nd
            file_data[new_abs_fb :new_abs_fb  + len(fd)] = fd
            if trailing_bytes:
                trail_abs = new_abs_fb + len(fd)
                file_data[trail_abs:trail_abs + len(trailing_bytes)] = trailing_bytes

            # Patch the edited LOD's own header fields.
            so = lod.start_offset
            fo = lod.lod_field_offset
            new_vob_for_header = higher_size + new_intra_vob
            new_fb_for_header  = higher_size + new_intra_fb
            # data_size in the header = intra_voa (prefix) + writable region
            new_file_data_size = intra_voa + new_data_size
            # size_a: fb//2 for uint16, fb//4 for uint32.
            # Use uint32 if original mesh used it OR if vert count exceeds uint16 range.
            use_uint32_faces = orig_uses_uint32_pre or new_vert_count > 65535
            new_size_a = new_fb_for_header // 4 if use_uint32_faces else new_fb_for_header // 2
            file_data[so + 0: so + 4] = pack('<I', new_vert_count)
            file_data[so + fo + 4: so + fo + 8] = pack('<I', new_index_count)
            file_data[so + fo + 8: so + fo + 12] = pack('<I', new_size_a)  # size_a
            # voa (so+fo+12) is unchanged
            file_data[so + fo + 16: so + fo + 20] = pack('<I', new_vob_for_header)  # vob
            file_data[so + fo + 20: so + fo + 24] = pack('<I', new_fb_for_header)  # fb
            # data_offset (so+fo+24) is unchanged for the edited LOD itself
            file_data[so + fo + 28: so + fo + 32] = pack('<I', new_file_data_size)  # data_size

            # Update the in-memory LOD so re-import in the same session uses correct values.
            lod.vertex_count         = new_vert_count
            lod.index_count          = new_index_count
            lod.vertex_data_offset_b = new_vob_for_header
            lod.face_block_offset    = new_fb_for_header
            lod.data_size            = new_file_data_size

            # For other LODs: update data_offset and reversed-buffer offsets
            if delta != 0:
                for other_mesh in addon_state.asset.meshes:
                    for other_lod in other_mesh.lods:
                        if other_lod.data_offset > lod.data_offset:
                            # Read the patched value from file_data
                            fp = other_lod.data_offset_file_pos
                            other_lod.data_offset = unpack('<I', file_data[fp:fp+4])[0]

                for li in range(0, lod_index):
                    other_lod = mesh.lods[li]
                    other_lod_so = other_lod.start_offset
                    other_lod_fo = other_lod.lod_field_offset
                    other_lod.vertex_data_offset_a = unpack('<I', file_data[other_lod_so + other_lod_fo + 12:other_lod_so + other_lod_fo + 16])[0]
                    other_lod.vertex_data_offset_b = unpack('<I', file_data[other_lod_so + other_lod_fo + 16:other_lod_so + other_lod_fo + 20])[0]
                    other_lod.face_block_offset = unpack('<I', file_data[other_lod_so + other_lod_fo + 20:other_lod_so + other_lod_fo + 24])[0]
                    other_lod.size_a = unpack('<I', file_data[other_lod_so + other_lod_fo + 8:other_lod_so + other_lod_fo + 12])[0]

            # Keep the per-mesh aggregate tail correct for ordinary edits. If
            # this LOD just crossed the uint16 limit the mesh can be temporarily
            # mixed-width; the final widening pass handles its siblings and
            # changes the flag only after every requested LOD has been written.
            new_index_width = 4 if use_uint32_faces else 2
            semantic_data_delta = (
                (new_vert_count - orig_vc)
                * (mesh.vertex_stride + mesh.normals_stride)
                + new_index_count * new_index_width
                - orig_ic * orig_idx_size
            )
            _patch_mesh_tail_delta(
                file_data, mesh,
                vertex_delta=new_vert_count - orig_vc,
                index_delta=new_index_count - orig_ic,
                data_delta=semantic_data_delta)
            lod.size_a = new_size_a
            lod.index_width_bytes = new_index_width

        with open(out_path, 'wb') as out:
            out.write(file_data)


    @staticmethod
    def _apply_header_patches(file_path: str, mesh, skeletal_mesh: SkeletalMeshAsset, operator=None):
        """
        Apply all staged header-level patches to an already-written mod file:
        bone remaps, bone additions, mesh rename, and zero-out.
        """
        # --- Bone remaps ---
        if mesh.pending_bone_remaps:
            with open(file_path, 'rb+') as f:
                mesh_bones_list = list(mesh.mesh_bones.items())
                for slot_idx, (new_skel_idx, new_matrix) in mesh.pending_bone_remaps.items():
                    if slot_idx < len(mesh.mesh_bone_file_offsets):
                        index_offset = mesh.mesh_bone_file_offsets[slot_idx]
                        matrix_offset = index_offset - 64
                        f.seek(matrix_offset)
                        f.write(pack('<16f', *new_matrix))
                        f.seek(index_offset)
                        f.write(pack('<H', new_skel_idx))
            new_mesh_bones = {}
            for slot_idx, (old_skel_idx, matrix) in enumerate(mesh_bones_list):
                remap = mesh.pending_bone_remaps.get(slot_idx)
                if remap is not None:
                    new_skel_idx, new_matrix = remap
                    new_mesh_bones[new_skel_idx] = new_matrix
                else:
                    new_mesh_bones[old_skel_idx] = matrix
            mesh.mesh_bones = new_mesh_bones
            mesh.pending_bone_remaps = {}

        # Bone additions
        if mesh.pending_bone_additions:
            n = len(mesh.pending_bone_additions)
            insert_at = mesh.bone_table_end_offset
            inserted_bytes = n * 66  # 64-byte matrix + 2-byte index per slot

            new_slot_bytes = b''
            for new_skel_idx, new_matrix in mesh.pending_bone_additions:
                new_slot_bytes += pack('<16f', *new_matrix)
                new_slot_bytes += pack('<H', new_skel_idx)

            with open(file_path, 'rb') as f:
                file_data = bytearray(f.read())
            file_data[insert_at:insert_at] = new_slot_bytes

            old_u = unpack('<H', file_data[mesh.u_count_offset:mesh.u_count_offset + 2])[0]
            file_data[mesh.u_count_offset:mesh.u_count_offset + 2] = pack('<H', old_u + n)

            # The bone table lives in the header section - grow asset.size to match.
            old_asset_size = unpack('<I', file_data[4:8])[0]
            file_data[4:8] = pack('<I', old_asset_size + inserted_bytes)

            for other_mesh in skeletal_mesh.meshes:
                for lod in other_mesh.lods:
                    if lod.data_offset_file_pos > insert_at:
                        field_pos = lod.data_offset_file_pos + inserted_bytes
                    else:
                        field_pos = lod.data_offset_file_pos
                    old_val = unpack('<I', file_data[field_pos:field_pos + 4])[0]
                    if old_val > 0:
                        file_data[field_pos:field_pos + 4] = pack('<I', old_val + inserted_bytes)
                    # Shift the second-section absolute offset (field [5], data_offset field
                    # + 32; see Mesh.parse) too.
                    if getattr(other_mesh, 'lod_info_type', 0) == 2:
                        sp = field_pos + 32
                        sec2_val = unpack('<I', file_data[sp:sp + 4])[0]
                        if sec2_val > 0:
                            file_data[sp:sp + 4] = pack('<I', sec2_val + inserted_bytes)
                    lod.data_offset += inserted_bytes
                    lod.data_offset_file_pos = field_pos

            with open(file_path, 'wb') as f:
                f.write(file_data)

            for i in range(len(mesh.mesh_bone_file_offsets)):
                if mesh.mesh_bone_file_offsets[i] >= insert_at:
                    mesh.mesh_bone_file_offsets[i] += inserted_bytes
            for si, (new_skel_idx, new_matrix) in enumerate(mesh.pending_bone_additions):
                new_index_offset = insert_at + si * 66 + 64
                mesh.mesh_bone_file_offsets.append(new_index_offset)
                mesh.mesh_bones[new_skel_idx] = new_matrix
            mesh.bone_table_end_offset = insert_at + inserted_bytes
            mesh.pending_bone_additions = []

        # Zero out all LODs
        if mesh.zeroed_out_in_session:
            with open(file_path, 'rb+') as f:
                for lod in mesh.lods:
                    for v in range(lod.vertex_count):
                        f.seek(lod.data_offset + v * mesh.vertex_stride)
                        lod.write_vertex_position(f, pos=(0.0, 0.0, 0.0), scale=None if mesh.position_type == 1 else 1)

        # Mesh name rename
        if mesh.pending_rename_new:
            padded = (mesh.pending_rename_new.encode('utf-8')
                      + b'\x00' * (mesh.name_length - len(mesh.pending_rename_new)))
            try:
                with open(file_path, 'rb+') as f:
                    f.seek(mesh.name_offset + 2)
                    f.write(padded)
            except Exception as e:
                if operator:
                    operator.report({'ERROR'}, f"Failed to patch mesh name in mod file: {e}")

    @staticmethod
    def normalize_weights(raw_weights, max_bones):
        """
        Normalize so the weights sum to 1.0.
        Normalize before encoding to ensure encode_weights_u8/u16 only need to correct the rounding error.
        """
        sw = raw_weights[:max_bones]
        total = sum(w for _, w in sw)
        if total <= 0.0:
            return sw
        inv = 1.0 / total
        return [(s, w * inv) for s, w in sw]

    @staticmethod
    def encode_weights_u8(sw):
        """
        Encode pairs to uint8, then fix the integer sum to exactly 255.
        Applies one-unit adjustments iteratively, always picking the entry
        with the largest rounding error that can actually absorb the step
        without clamping.
        """
        encoded = [(s, int(round(w * 255))) for s, w in sw]
        diff = 255 - sum(e for _, e in encoded)
        if diff == 0:
            return encoded
        step = 1 if diff > 0 else -1
        for _ in range(abs(diff)):
            best_err = -1
            best_idx = -1
            for i, (s, e) in enumerate(encoded):
                if step == 1 and e >= 255:
                    continue
                if step == -1 and e <= 0:
                    continue
                err = abs((sw[i][1] * 255) - e)
                if err > best_err:
                    best_err = err
                    best_idx = i
            if best_idx == -1:
                break
            s, e = encoded[best_idx]
            encoded[best_idx] = (s, e + step)
        return encoded

    @staticmethod
    def encode_weights_u16(sw):
        """
        Encode pairs to uint16, then fix the integer sum to exactly 32767.
        Applies one-unit adjustments iteratively, always picking the entry
        with the largest rounding error that can actually absorb the step
        without clamping.
        """
        encoded = [(s, int(round(w * 32767))) for s, w in sw]
        diff = 32767 - sum(e for _, e in encoded)
        if diff == 0:
            return encoded
        step = 1 if diff > 0 else -1
        for _ in range(abs(diff)):
            best_err = -1
            best_idx = -1
            for i, (s, e) in enumerate(encoded):
                if step == 1 and e >= 32767:
                    continue
                if step == -1 and e <= 0:
                    continue
                err = abs((sw[i][1] * 32767) - e)
                if err > best_err:
                    best_err = err
                    best_idx = i
            if best_idx == -1:
                break
            s, e = encoded[best_idx]
            encoded[best_idx] = (s, e + step)
        return encoded

    @staticmethod
    def convert_coordinate(co):
        return Vector((co[0] * -1, co[1], co[2]))

    @staticmethod
    def write_vertices(file, mesh:SkeletalMeshAsset.Mesh, lod_index=0):
        f = file
        obj = BME.find_object_by_name(mesh.name + f"_LOD{lod_index}")
        lod: SkeletalMeshAsset.Mesh.LOD = mesh.lods[lod_index]
        if obj:
            data = obj.data
            bm = bmesh.new()
            bm.from_mesh(data)
            bm.verts.ensure_lookup_table()
            stride = mesh.vertex_stride
            pos_length = 8 if mesh.position_type == 0 else 12
            mesh_bones = list(mesh.mesh_bones.keys())  # skeleton bone indices in mesh-slot order

            # Map: bone name -> mesh slot index
            name_to_mesh_slot = {}
            for slot, skel_idx in enumerate(mesh_bones):
                if skel_idx < len(addon_state.asset.bones):
                    name_to_mesh_slot[addon_state.asset.bones[skel_idx].name] = slot

            # Map: Blender vertex group index -> mesh slot index (matched by bone name)
            vgroup_to_mesh_slot = {}
            for vg in obj.vertex_groups:
                if vg.name in name_to_mesh_slot:
                    vgroup_to_mesh_slot[vg.index] = name_to_mesh_slot[vg.name]

            # Fallback: if no vertex groups matched by name (e.g. bone names differ between
            # models), attempt to match by the numeric suffix of the group name against the
            # mesh-slot index, then fall back to treating the group index itself as the mesh
            # slot.  This prevents weights from being silently dropped on export.
            if not vgroup_to_mesh_slot and obj.vertex_groups:
                logger.warning(
                    "No vertex groups on %s matched bone names in mesh %s; "
                    "attempting index-based fallback mapping",
                    obj.name, mesh.name)
                import re as _re
                for vg in obj.vertex_groups:
                    # Try to parse a trailing integer from the group name (e.g. "Bone_7" -> 7)
                    m = _re.search(r'(\d+)$', vg.name)
                    if m:
                        slot = int(m.group(1))
                    else:
                        slot = vg.index
                    if slot < len(mesh_bones):
                        vgroup_to_mesh_slot[vg.index] = slot
                if vgroup_to_mesh_slot:
                    logger.info(
                        "Index-based fallback produced %d group mappings for %s",
                        len(vgroup_to_mesh_slot), obj.name)
                else:
                    logger.error(
                        "Index-based fallback failed for %s; weights will be zero. "
                        "Check that vertex group names match skeleton bone names or "
                        "contain a bone index suffix",
                        obj.name)

            # For int16 positions, read per-vertex w scale from the original source file.
            # Always read from SWOMT.AssetPath using the original data_offset from the
            # file header. lod.data_offset may have been corrupted in-memory by a prior
            # mesh export in the same _write_mod_file call (ExportAllLODs exports all
            # meshes at a given LOD level in one call).
            SWOMT = bpy.context.scene.SWOMT
            src_path = SWOMT.AssetPath
            with open(src_path, 'rb') as _f:
                _f.seek(lod.data_offset_file_pos)
                orig_data_offset = unpack('<I', _f.read(4))[0]

            if mesh.position_type == 0:
                original_w = []
                higher_size_w = sum(
                    mesh.lods[li].data_size
                    for li in range(lod_index + 1, len(mesh.lods))
                )
                intra_voa_w = lod.vertex_data_offset_a - higher_size_w
                abs_voa_w_read = orig_data_offset + intra_voa_w
                with open(src_path, 'rb') as src:
                    src.seek(abs_voa_w_read + 6)  # skip x,y,z int16s to reach w
                    for _ in range(lod.vertex_count):
                        original_w.append(unpack('<h', src.read(2))[0])
                        src.seek(stride - 2, 1)
                if len(data.vertices) != lod.vertex_count:
                    # Replacement mesh: derive a scale from the new mesh's actual
                    # extents so every coordinate fits in [-scale, scale] without
                    # overflow or silent skipping.  Averaging the original w values
                    # is unreliable when the replacement is a different size/position.
                    if data.vertices:
                        max_coord = max(
                            max(abs(v.co.x), abs(v.co.y), abs(v.co.z))
                            for v in data.vertices
                        )
                    else:
                        max_coord = 1.0
                    fallback_scale = max(1, int(max_coord) + 1)
                    original_w = [fallback_scale] * len(data.vertices)
            else:
                original_w = [None] * len(data.vertices)

            # Read original weight+index bytes per vertex when THIS LOD's vert count
            # is unchanged (they are indexed by vertex index) and export_weights is unchecked.
            export_weights = (bpy.context.scene.SWOMT.export_weights
                              or len(data.vertices) != lod.vertex_count)
            weight_bytes_per_vert = stride - pos_length

            # Compute abs_voa once for weight reading (use orig_data_offset from file)
            higher_size_w = sum(
                mesh.lods[li].data_size
                for li in range(lod_index + 1, len(mesh.lods))
            )
            intra_voa_w = lod.vertex_data_offset_a - higher_size_w
            abs_voa_w   = orig_data_offset + intra_voa_w

            orig_weight_bytes = None
            # Per-vertex data needed when export_weights=True and stride=32
            orig_stride32_data = None
            # Per-vertex original index bytes for packed-uint8 strides (stride=16 int16, else-branch)
            # Used to preserve the original bone index in zero-weight padding slots, which the
            # game may use as secondary bone references independent of weight.
            orig_index_bytes = None
            # Detect stride=32 layout variant
            stride32_use_u16 = stride == 32 and len(mesh.mesh_bones) > 256

            # Layout C: 12x uint8 weights, 12x uint8 indices.
            stride32_layout_c = False
            if stride == 32 and not stride32_use_u16 and lod.vertex_count > 0:
                _n_slots = len(mesh.mesh_bones)
                with open(src_path, 'rb') as _src_lc:
                    _src_lc.seek(abs_voa_w + pos_length)
                    _peek24 = _src_lc.read(24)
                _a_w_check = unpack('<8H', _peek24[:16])
                if sum(_a_w_check) != 32767:
                    _c_i_check = list(_peek24[12:24])
                    if _n_slots <= 256 and all(0 <= x < _n_slots for x in _c_i_check):
                        stride32_layout_c = True

            if not export_weights and weight_bytes_per_vert > 0:
                orig_weight_bytes = []
                with open(src_path, 'rb') as src:
                    for vi in range(lod.vertex_count):
                        src.seek(abs_voa_w + vi * stride + pos_length)
                        orig_weight_bytes.append(src.read(weight_bytes_per_vert))
            elif export_weights:
                # Read the original weight/index data from the source file for stride=32
                # meshes, where the layout variant (A/B/C) and active slot count must be
                # preserved. Other strides always write weights sorted by descending weight.
                if stride == 32 and lod.vertex_count > 0:
                    orig_stride32_data = []
                    with open(src_path, 'rb') as src:
                        for vi in range(lod.vertex_count):
                            src.seek(abs_voa_w + vi * stride + pos_length)
                            if stride32_use_u16:
                                # Layout B
                                all_w = list(src.read(6))
                                src.seek(2, 1) # skip 2 padding bytes
                                all_i = list(unpack('<6H', src.read(12)))
                            elif stride32_layout_c:
                                # Layout C
                                all_w = list(src.read(12))
                                all_i = list(src.read(12))
                            else:
                                # Layout A
                                all_w = unpack('<8H', src.read(16))
                                all_i = list(src.read(8))
                            orig_stride32_data.append((all_w, all_i))
                elif stride not in (12, 20, 32, 36, 40, 44) and lod.vertex_count > 0:
                    # Packed uint8 strides (stride=16 int16, stride=24, and any other else-branch
                    # strides): read the original index bytes so zero-weight padding slots can
                    # preserve the original bone index rather than being zeroed out.
                    wc_idx = int(weight_bytes_per_vert / 2)
                    orig_index_bytes = []
                    with open(src_path, 'rb') as src:
                        for vi in range(lod.vertex_count):
                            src.seek(abs_voa_w + vi * stride + pos_length + wc_idx)
                            orig_index_bytes.append(src.read(wc_idx))

            for vi, v in enumerate(bm.verts):
                stride_start = f.tell()

                # Write position
                lod.write_vertex_position(
                    f,
                    pos=BME.convert_coordinate(v.co),
                    scale=original_w[vi],
                )
                f.seek(stride_start + pos_length)

                # Write bone weights - use original bytes if vert count is unchanged
                if orig_weight_bytes is not None:
                    f.write(orig_weight_bytes[vi])
                    f.seek(stride_start + stride)
                    continue

                # Gather bone weights
                raw_weights = []
                if weight_bytes_per_vert > 0:
                    vertex = data.vertices[v.index]
                    for vge in vertex.groups:
                        slot = vgroup_to_mesh_slot.get(vge.group)
                        if slot is not None and vge.weight > 0.0:
                            raw_weights.append((slot, vge.weight))
                    raw_weights.sort(key=lambda x: x[1], reverse=True)

                    if not raw_weights:
                        raise ValueError(
                            f"'{mesh.name}' LOD{lod_index} has vertices with no bone weights. "
                            f"Assign bone weights to all vertices before exporting."
                        )

                # Write bone weights
                #
                # *** If a new `elif stride == N:` is added to the branch below, or an
                # existing stride's index width (bp.uint8 vs bp.uint16) is changed,
                # update '_UINT8_INDEX_LIMITED_STRIDES/_UINT16_NON_LIMITED_STRIDES' to match.
                # Otherwise, the 256-slot-reuse logic in Add Bone Slots will stop catching meshes that need it. ***
                if weight_bytes_per_vert == 0:
                    pass
                elif stride == 12:
                    # 4x uint8 bone slot indices, no weight bytes - Weight 1.0 on first index
                    for _ in range(4):
                        f.write(bp.uint8(raw_weights[0][0] if raw_weights else 0))

                elif stride == 16:
                    if pos_length == 12:
                        # Float XYZ position (12b) + 1x uint8 bone slot index + 3b padding.
                        # Same as stride=12 (weight 1.0 on first index) but with float positions.
                        slot = raw_weights[0][0] if raw_weights else 0
                        f.write(bp.uint8(slot))
                        f.write(b'\x00\x00\x00')  # 3 padding bytes
                    else:
                        # Int16 position (8b) + 4x uint8_norm weights + 4x uint8 indices
                        wc = 4
                        sw = BME.normalize_weights(raw_weights, wc)
                        enc = BME.encode_weights_u8(sw)
                        for i in range(wc):
                            f.write(bp.uint8(enc[i][1] if i < len(enc) else 0))
                        orig_idx = orig_index_bytes[vi] if (orig_index_bytes is not None and vi < len(orig_index_bytes)) else None
                        for i in range(wc):
                            if i < len(enc):
                                f.write(bp.uint8(enc[i][0]))
                            else:
                                f.write(bp.uint8(orig_idx[i] if orig_idx is not None else 0))

                elif stride == 20:
                    max_bones = 4
                    sw = BME.normalize_weights(raw_weights, max_bones)
                    enc = BME.encode_weights_u16(sw)
                    for _, e in enc:
                        f.write(bp.uint16(e))
                    for _ in range(max_bones - len(enc)):
                        f.write(bp.uint16(0))
                    for s, _ in enc:
                        f.write(bp.uint8(s))
                    for _ in range(max_bones - len(enc)):
                        f.write(bp.uint8(0))

                elif stride == 32:
                    vert_count_matches = len(data.vertices) == lod.vertex_count
                    if orig_stride32_data is not None and vert_count_matches and vi < len(orig_stride32_data):
                        orig_all_w, orig_all_i = orig_stride32_data[vi]
                        if stride32_use_u16:
                            # Layout B
                            remaining = {s: w for s, w in BME.normalize_weights(raw_weights, 6)}
                            # fill zero-weight slots with any added-bone weights that original indices don't cover.
                            all_i = list(orig_all_i)
                            extra = sorted(remaining.keys() - set(all_i),
                                           key=lambda s: remaining[s], reverse=True)
                            for k in range(6):
                                if all_i[k] not in remaining and extra:
                                    all_i[k] = extra.pop(0)
                            # Pre-encode all 6 weights as a batch to fix rounding sum
                            ordered = [(all_i[k], remaining.pop(all_i[k], 0.0)) for k in range(6)]
                            enc = BME.encode_weights_u8(ordered)
                            for _, e in enc:
                                f.write(bp.uint8(e))
                            f.write(b'\x00\x00') # 2 padding bytes
                            # Write 6 uint16 indices
                            for idx in all_i:
                                f.write(bp.uint16(idx))
                        elif stride32_layout_c:
                            # Layout C
                            remaining = {s: w for s, w in BME.normalize_weights(raw_weights, 12)}
                            all_i = list(orig_all_i)
                            extra = sorted(remaining.keys() - set(all_i),
                                           key=lambda s: remaining[s], reverse=True)
                            for k in range(12):
                                if all_i[k] not in remaining and extra:
                                    all_i[k] = extra.pop(0)
                            # Pre-encode all 12 weights as a batch to fix rounding sum
                            ordered = [(all_i[k], remaining.pop(all_i[k], 0.0)) for k in range(12)]
                            enc = BME.encode_weights_u8(ordered)
                            for _, e in enc:
                                f.write(bp.uint8(e))
                            for bone in all_i:
                                f.write(bp.uint8(bone))
                        else:
                            # Layout A
                            weight_slots = range(8) if sum(orig_all_w) == 32767 else range(4)
                            remaining = {s: w for s, w in BME.normalize_weights(raw_weights, len(weight_slots))}
                            # Inject added-bone weights into zero-weight positions.
                            all_i = list(orig_all_i)
                            empty_slots = [k for k in weight_slots if orig_all_w[k] == 0]
                            extra = sorted(remaining.keys() - set(all_i),
                                           key=lambda s: remaining[s], reverse=True)
                            for k in empty_slots:
                                if not extra:
                                    break
                                all_i[k] = extra.pop(0)
                            # Pre-encode active weight slots as a batch to fix rounding sum
                            ordered = [(all_i[slot], remaining.pop(all_i[slot], 0.0)) for slot in weight_slots]
                            enc = BME.encode_weights_u16(ordered)
                            for _, e in enc:
                                f.write(bp.uint16(e))
                            for slot in range(len(weight_slots), 8):
                                f.write(bp.uint16(orig_all_w[slot]))
                            for bone in all_i:
                                f.write(bp.uint8(bone))
                    else:
                        # Vert count changed: write by weight desc
                        if stride32_use_u16:
                            # Layout B
                            sw = BME.normalize_weights(raw_weights, 6)
                            enc = BME.encode_weights_u8(sw)
                            for _, e in enc:
                                f.write(bp.uint8(e))
                            for _ in range(6 - len(enc)):
                                f.write(bp.uint8(0))
                            f.write(b'\x00\x00')
                            for s, _ in sw[:6]:
                                f.write(bp.uint16(s))
                            for _ in range(6 - min(len(sw), 6)):
                                f.write(bp.uint16(0))
                        elif stride32_layout_c:
                            # Layout C
                            sw = BME.normalize_weights(raw_weights, 12)
                            enc = BME.encode_weights_u8(sw)
                            for _, e in enc:
                                f.write(bp.uint8(e))
                            for _ in range(12 - len(enc)):
                                f.write(bp.uint8(0))
                            for s, _ in sw:
                                f.write(bp.uint8(s))
                            for _ in range(12 - len(sw)):
                                f.write(bp.uint8(0))
                        else:
                            # Layout A
                            sw = BME.normalize_weights(raw_weights, 8)
                            enc = BME.encode_weights_u16(sw)
                            for _, e in enc:
                                f.write(bp.uint16(e))
                            for _ in range(8 - len(enc)):
                                f.write(bp.uint16(0))
                            for s, _ in sw:
                                f.write(bp.uint8(s))
                            for _ in range(8 - len(sw)):
                                f.write(bp.uint8(0))

                elif stride == 36:
                    max_bones = 8
                    sw = BME.normalize_weights(raw_weights, max_bones)
                    enc = BME.encode_weights_u16(sw)
                    for _, e in enc:
                        f.write(bp.uint16(e))
                    for _ in range(max_bones - len(enc)):
                        f.write(bp.uint16(0))
                    for s, _ in sw:
                        f.write(bp.uint8(s))
                    for _ in range(max_bones - len(sw)):
                        f.write(bp.uint8(0))

                elif stride == 40:
                    max_bones = 8
                    sw = BME.normalize_weights(raw_weights, max_bones)
                    enc = BME.encode_weights_u16(sw)
                    for _, e in enc:
                        f.write(bp.uint16(e))
                    for _ in range(max_bones - len(enc)):
                        f.write(bp.uint16(0))
                    for s, _ in sw:
                        f.write(bp.uint16(s))
                    for _ in range(max_bones - len(sw)):
                        f.write(bp.uint16(0))

                elif stride == 44:
                    max_bones = 12
                    sw = BME.normalize_weights(raw_weights, max_bones)
                    enc = BME.encode_weights_u8(sw)
                    for i in range(max_bones):
                        f.write(bp.uint8(enc[i][1] if i < len(enc) else 0))
                    for i in range(max_bones):
                        f.write(bp.uint16(enc[i][0] if i < len(enc) else 0))

                else:
                    wc = int((stride - pos_length) / 2)
                    sw = BME.normalize_weights(raw_weights, wc)
                    enc = BME.encode_weights_u8(sw)
                    for i in range(wc):
                        f.write(bp.uint8(enc[i][1] if i < len(enc) else 0))
                    orig_idx = orig_index_bytes[vi] if (orig_index_bytes is not None and vi < len(orig_index_bytes)) else None
                    for i in range(wc):
                        if i < len(enc):
                            f.write(bp.uint8(enc[i][0]))
                        else:
                            f.write(bp.uint8(orig_idx[i] if orig_idx is not None else 0))

                # Advance to next vertex slot
                f.seek(stride_start + stride)
            bm.free()



    @staticmethod
    def write_normals(file, mesh:SkeletalMeshAsset.Mesh, lod_index=0):
        f = file
        obj = BME.find_object_by_name(mesh.name + f"_LOD{lod_index}")
        lod: SkeletalMeshAsset.Mesh.LOD = mesh.lods[lod_index]

        # Per-vertex preservation data (colors/UVs/trailing/tangents) must be read from the
        # LOD whose numbering mmb_vertex_order refers to - generated LODs carry LOD0 indices (mmb_lod_source).
        # Vertex colors in particular hold the cloth driven-mask, so a wrong source scrambles it.
        pres_li = lod_index
        if obj is not None:
            _src = obj.get('mmb_lod_source', lod_index)
            if isinstance(_src, int) and 0 <= _src < len(mesh.lods):
                pres_li = _src
        pres_lod = mesh.lods[pres_li]
        if obj:
            data = obj.data

            # Save custom split normals before calc_tangents() - it resets loop normals
            # to geometry-derived values, discarding any custom normals on the mesh.
            saved_loop_normals = None
            if data.has_custom_normals:
                saved_loop_normals = [l.normal.copy() for l in data.loops]

            bm = bmesh.new()
            bm.from_mesh(data)
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            data.loops.data.calc_tangents()

            # Restore custom normals after calc_tangents() reset them
            if saved_loop_normals is not None:
                data.normals_split_custom_set(saved_loop_normals)

            # Gather per-vertex normal, tangent, bitangent-sign, and UV data from loops
            NTB = [(Vector((1.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0)), 1.0)] * len(data.vertices)
            all_uvs = [[(0.0, 0.0)] * len(data.vertices) for _ in range(mesh.uv_count)]

            # Detect UV Convention
            uv_centred_u = []
            uv_centred_v = []
            for ui in range(mesh.uv_count):
                # Prefer attributes
                cu_attr = data.attributes.get(f'mmb_uv{ui}_centred_u')
                cv_attr = data.attributes.get(f'mmb_uv{ui}_centred_v')
                if cu_attr is not None:
                    uv_centred_u.append(bool(cu_attr.data[0].value))
                elif ui < len(data.uv_layers):
                    uv_centred_u.append(False)
                else:
                    uv_centred_u.append(False)
                if cv_attr is not None:
                    uv_centred_v.append(bool(cv_attr.data[0].value))
                elif ui < len(data.uv_layers):
                    v_vals = [data.uv_layers[ui].data[li].uv[1] for li in range(len(data.loops))]
                    v_min, v_max = min(v_vals), max(v_vals)
                    uv_centred_v.append(v_min < -0.05 and abs((v_min + v_max) / 2.0 - 0.5) < 0.15)
                else:
                    uv_centred_v.append(False)

            uv_accum = [[None] * len(data.vertices) for _ in range(mesh.uv_count)]

            # For vertices shared between faces with different UVs (seams), average all loop UV values
            for ui in range(mesh.uv_count):
                if ui >= len(data.uv_layers):
                    continue
                uv_data = data.uv_layers[ui].data
                for li, loop in enumerate(data.loops):
                    vi = loop.vertex_index
                    u_bl, v_bl = uv_data[li].uv
                    u = u_bl
                    v = v_bl - 0.5 if uv_centred_v[ui] else 1 - v_bl
                    if uv_accum[ui][vi] is None:
                        uv_accum[ui][vi] = [u, v, 1]
                    else:
                        uv_accum[ui][vi][0] += u
                        uv_accum[ui][vi][1] += v
                        uv_accum[ui][vi][2] += 1
            for ui in range(mesh.uv_count):
                for vi in range(len(data.vertices)):
                    if uv_accum[ui][vi] is not None:
                        u_sum, v_sum, count = uv_accum[ui][vi]
                        all_uvs[ui][vi] = (u_sum / count, v_sum / count)

            for l in data.loops:
                flip = -1.0 if l.bitangent_sign == -1 else 1.0
                NTB[l.vertex_index] = (l.normal, l.tangent, flip)

            SWOMT = bpy.context.scene.SWOMT
            export_uvs = SWOMT.export_uvs or len(data.vertices) != lod.vertex_count

            if mesh.normal_type == 0:
                # int8_norm format:
                #   4 bytes: normal as (x, y, z) int8_norm + w int8 sign (always -1)
                #   4 bytes: tangent packed as X10Y10Z10W2
                #   color_count * 4 bytes: vertex colors as uint8_norm RGBA
                #   uv_count * 4 bytes: UVs as pairs of int16_norm

                # Read original nw and tangent bytes from source for all vertices.
                SWOMT = bpy.context.scene.SWOMT
                # Always read nw and tangent bytes from the original asset file.
                vert_count_unchanged = len(data.vertices) == lod.vertex_count
                with open(SWOMT.AssetPath, 'rb') as orig_src:
                    orig_file_bytes = orig_src.read()
                # Higher-LOD data sizes must come from the ORIGINAL file too:
                # in-memory sizes already reflect earlier passes of a multi-LOD
                # export, which would corrupt the preservation read offsets.
                higher_size = sum(
                    unpack('<I', orig_file_bytes[
                        mesh.lods[li].start_offset + mesh.lods[li].lod_field_offset + 28:
                        mesh.lods[li].start_offset + mesh.lods[li].lod_field_offset + 32])[0]
                    for li in range(pres_li + 1, len(mesh.lods))
                )
                orig_do  = unpack('<I', orig_file_bytes[pres_lod.data_offset_file_pos:pres_lod.data_offset_file_pos+4])[0]
                orig_vob = unpack('<I', orig_file_bytes[pres_lod.start_offset + pres_lod.lod_field_offset + 16:pres_lod.start_offset + pres_lod.lod_field_offset + 20])[0]
                orig_voa = unpack('<I', orig_file_bytes[pres_lod.start_offset + pres_lod.lod_field_offset + 12:pres_lod.start_offset + pres_lod.lod_field_offset + 16])[0]
                abs_vob_src = orig_do + (orig_vob - higher_size)
                abs_voa_src = orig_do + (orig_voa - higher_size)
                orig_vc_src = unpack('<I', orig_file_bytes[pres_lod.start_offset:pres_lod.start_offset+4])[0]
                ns = mesh.normals_stride
                vs = mesh.vertex_stride

                # Read all original nw, tangent, and color bytes.
                orig_nw       = []
                orig_tangents = []
                orig_colors   = []  # list of raw 4*color_count byte chunks per vertex
                orig_uvs      = []  # list of raw 4*uv_count byte chunks per vertex
                orig_trailing = []  # any extra bytes after color+normal+tangent+UVs

                # color(4*cc) | normal(4) | tangent(4) | UV (per-set, variable width)
                color_count = mesh.color_count if getattr(mesh, 'color_in_normals', True) else 0
                _normal_block = 8  # tangent(4) + normal(4)
                _color_prefix = 4 * color_count

                # Detect per-UV-set encoding from the original file bytes.
                # _uv_is_float32[ui] : 8-byte float pair
                # _uv_wide[ui]       : signed int16/4096, no modulo, native range ~[-8,8]
                # _uv_compact[ui]    : uv_unorm; else int16_norm.
                _uv_is_float32 = []
                _uv_wide = []
                _uv_compact = []
                _uv_u_divisor = []
                _uv_v_divisor = []
                _uv_field_offs = []
                _cur = _color_prefix + _normal_block
                _exp_divs = getattr(mesh, 'uv_divisors', None)
                for _ui in range(mesh.uv_count):
                    _div = _exp_divs[_ui] if (_exp_divs is not None and _ui < len(_exp_divs)) else None
                    _plausible = 0
                    _compact_ok = True
                    if orig_vc_src > 0:
                        for _ni in range(orig_vc_src):
                            _o = abs_vob_src + _ni * ns + _cur
                            _fv = unpack('<f', orig_file_bytes[_o:_o + 4])[0]
                            if _fv == _fv and (_fv == 0.0 or 1e-4 < abs(_fv) < 500):
                                _plausible += 1
                            if _div is None:
                                _rs = unpack('<h', orig_file_bytes[_o:_o + 2])[0]
                                if abs(_rs) > 8191:
                                    _compact_ok = False
                        _plausible_f32 = (_plausible / orig_vc_src) > 0.90
                    else:
                        _plausible_f32 = False
                    _enc = _resolve_uv_encoding(_div, _plausible_f32, _compact_ok)
                    _uv_field_offs.append(_cur)
                    _is_f32 = (_enc == 'float32')
                    _uv_is_float32.append(_is_f32)
                    if _is_f32:
                        _uv_wide.append(False); _uv_compact.append(False)
                        _uv_u_divisor.append(1.0); _uv_v_divisor.append(1.0)
                        _cur += 8
                    elif _enc == 'wide':
                        _uv_wide.append(True); _uv_compact.append(False)
                        _uv_u_divisor.append(4096.0); _uv_v_divisor.append(4096.0)
                        _cur += 4
                    else:
                        _uv_wide.append(False)
                        _uv_u_divisor.append(1.0); _uv_v_divisor.append(1.0)
                        _uv_compact.append(_enc == 'compact')
                        _cur += 4
                _total_uv_bytes = _cur - (_color_prefix + _normal_block)
                uv_off_in_stride = _color_prefix + _normal_block
                written_per_vert = _color_prefix + _normal_block + _total_uv_bytes
                trailing_per_vert = ns - written_per_vert
                for ni in range(orig_vc_src):
                    off = abs_vob_src + ni * ns
                    normal_off = off + _color_prefix
                    tangent_off = normal_off + 4
                    uv_off = tangent_off + 4
                    orig_nw.append(unpack('<b', orig_file_bytes[normal_off + 3:normal_off + 4])[0])
                    orig_tangents.append(orig_file_bytes[tangent_off:tangent_off + 4])
                    if color_count > 0:
                        orig_colors.append(orig_file_bytes[off:off + 4 * color_count])
                    if mesh.uv_count > 0:
                        orig_uvs.append(orig_file_bytes[uv_off:uv_off + _total_uv_bytes])
                    if trailing_per_vert > 0:
                        trail_off = off + written_per_vert
                        orig_trailing.append(orig_file_bytes[trail_off:trail_off + trailing_per_vert])

                # Build per-vertex source index using mmb_vertex_order attribute if present
                orig_idx_attr = data.attributes.get("mmb_vertex_order")
                if orig_idx_attr is not None:
                    orig_idx_for_vi = {vi: orig_idx_attr.data[vi].value
                                       for vi in range(len(data.vertices))}
                else:
                    orig_idx_for_vi = None

                # Build position -> source index map as fallback for meshes
                orig_pos_to_ni = {}
                if orig_idx_for_vi is None:
                    for ni in range(orig_vc_src):
                        pos_key = orig_file_bytes[abs_voa_src + ni*vs : abs_voa_src + ni*vs + 6]
                        if pos_key not in orig_pos_to_ni:
                            orig_pos_to_ni[pos_key] = ni

                for vi, v in enumerate(data.vertices):
                    normal  = NTB[v.index][0]
                    tangent = NTB[v.index][1]
                    flip    = NTB[v.index][2]

                    # Determine source vertex index for nw/tangent lookup
                    if orig_idx_for_vi is not None:
                        # Use stored original index directly (survives separate/join)
                        orig_vi = orig_idx_for_vi.get(vi, vi)
                        src_vi = min(orig_vi, orig_vc_src - 1)
                    else:
                        # Fallback
                        src_vi = min(vi, orig_vc_src - 1)

                    read_orig_nw = vert_count_unchanged

                    # Layout: color(4*color_count) | normal(4) | tangent(4) | UV(4*uv_count)

                    # Write vertex color: preserve original bytes unless Export Vertex Colors is on.
                    if color_count > 0:
                        if SWOMT.export_vertex_colors:
                            for ci in range(color_count):
                                layer = bm.verts.layers.float_color.get(f"Color_{ci}")
                                if layer is not None:
                                    vertex_color = bm.verts[v.index][layer]
                                    for c in vertex_color:
                                        f.write(bp.uint8_norm(c))
                                elif orig_colors and src_vi < len(orig_colors):
                                    # Preserve original bytes verbatim
                                    f.write(orig_colors[src_vi][ci * 4:ci * 4 + 4])
                                else:
                                    f.write(b'\x00' * 4)
                        elif orig_colors and src_vi < len(orig_colors):
                            # Preserve original bytes verbatim
                            f.write(orig_colors[src_vi])
                        else:
                            # No source - write zeros
                            f.write(b'\x00' * (4 * color_count))

                    # Write normal as int8 (scale to [-127,127], flip x)
                    def clamp_i8(val):
                        return max(-127, min(127, int(round(val * 127))))
                    f.write(bp.int8(clamp_i8(normal[0] * -1)))
                    f.write(bp.int8(clamp_i8(normal[1])))
                    f.write(bp.int8(clamp_i8(normal[2])))
                    nw_val = orig_nw[src_vi] if read_orig_nw else -1
                    f.write(bp.int8(nw_val))

                    # Write tangent
                    if orig_tangents:
                        f.write(orig_tangents[src_vi])
                    else:
                        f.write(bp.X10Y10Z10W2(tangent[0] * -1, tangent[1], tangent[2], max(0, int(flip))))

                    # Write UVs
                    if mesh.uv_count > 0:
                        if export_uvs:
                            for ui in range(mesh.uv_count):
                                u, v_uv = all_uvs[ui][v.index]
                                if _uv_is_float32[ui]:
                                    f.write(bp.float(u))
                                    f.write(bp.float(v_uv))
                                elif _uv_wide[ui]:
                                    # signed int16 = round(u * 4096), no folding/clamping
                                    # to [0,1] since wide encoding represents the full
                                    # tiled value directly.
                                    _u_raw = max(-32768, min(32767, int(round(u * _uv_u_divisor[ui]))))
                                    _v_raw = max(-32768, min(32767, int(round(v_uv * _uv_v_divisor[ui]))))
                                    f.write(pack('<h', _u_raw))
                                    f.write(pack('<h', _v_raw))
                                elif _uv_compact[ui]:
                                    f.write(bp.uv_unorm_u(max(0.0, min(1.0, u))))
                                    f.write(bp.uv_unorm_v(max(0.0, min(1.0, v_uv))))
                                else:
                                    f.write(bp.int16_norm(max(-1.0, min(1.0, u))))
                                    f.write(bp.int16_norm(max(-1.0, min(1.0, v_uv))))
                        elif orig_uvs and src_vi < len(orig_uvs):
                            f.write(orig_uvs[src_vi])
                        else:
                            f.write(b'\x00' * _total_uv_bytes)

                    # Preserve any trailing bytes
                    if orig_trailing and src_vi < len(orig_trailing):
                        f.write(orig_trailing[src_vi])
                    elif trailing_per_vert > 0:
                        f.write(b'\x00' * trailing_per_vert)

            else:
                # float format (normal_type == 1 or 2).
                # normal_type 1: color(4*cc) | normal(12f) | tangent(12f) | sign(4f) | UV(4*uv)
                SWOMT = bpy.context.scene.SWOMT
                vert_count_unchanged = len(data.vertices) == lod.vertex_count
                with open(SWOMT.AssetPath, 'rb') as orig_src:
                    orig_file_bytes = orig_src.read()
                # Higher-LOD data sizes must come from the ORIGINAL file too:
                # in-memory sizes already reflect earlier passes of a multi-LOD
                # export, which would corrupt the preservation read offsets.
                higher_size = sum(
                    unpack('<I', orig_file_bytes[
                        mesh.lods[li].start_offset + mesh.lods[li].lod_field_offset + 28:
                        mesh.lods[li].start_offset + mesh.lods[li].lod_field_offset + 32])[0]
                    for li in range(pres_li + 1, len(mesh.lods))
                )
                orig_do  = unpack('<I', orig_file_bytes[pres_lod.data_offset_file_pos:pres_lod.data_offset_file_pos+4])[0]
                orig_vob = unpack('<I', orig_file_bytes[pres_lod.start_offset+pres_lod.lod_field_offset+16:pres_lod.start_offset+pres_lod.lod_field_offset+20])[0]
                abs_vob_src = orig_do + (orig_vob - higher_size)
                orig_vc_src = unpack('<I', orig_file_bytes[pres_lod.start_offset:pres_lod.start_offset+4])[0]
                ns = mesh.normals_stride

                orig_colors = []
                orig_uvs    = []
                orig_trailing = []
                # color(4*cc) | normal(12) | tangent(12) | sign(4) | UV
                color_count = mesh.color_count if getattr(mesh, 'color_in_normals', True) else 0
                color_off_in_stride = 0
                uv_off_in_stride    = 4 * color_count + 12 + 12 + 4
                # Per-set encoding from the binary divisor table (this float-normal
                # path stores every UV set as a 4-byte int form).
                _uv_wide = []
                _uv_compact = []
                _uv_u_divisor = []
                _uv_v_divisor = []
                _exp_divs = getattr(mesh, 'uv_divisors', None)
                for _ui in range(mesh.uv_count):
                    _off0 = uv_off_in_stride + _ui * 4
                    _div = _exp_divs[_ui] if (_exp_divs is not None and _ui < len(_exp_divs)) else None
                    _compact_ok = True
                    if _div is None:
                        for _ni in range(orig_vc_src):
                            _off = abs_vob_src + _ni * ns + _off0
                            _rs = unpack('<h', orig_file_bytes[_off:_off + 2])[0]
                            if abs(_rs) > 8191:
                                _compact_ok = False
                                break
                    # 4-byte fixed stride here, so a float32 divisor cannot widen the
                    # layout; fall back to int16_norm in that (unseen) case.
                    _enc = _resolve_uv_encoding(_div, False, _compact_ok)
                    if _enc == 'wide':
                        _uv_wide.append(True); _uv_compact.append(False)
                        _uv_u_divisor.append(4096.0); _uv_v_divisor.append(4096.0)
                    else:
                        _uv_wide.append(False)
                        _uv_u_divisor.append(1.0); _uv_v_divisor.append(1.0)
                        _uv_compact.append(_enc == 'compact')
                written_per_vert = uv_off_in_stride + 4 * mesh.uv_count
                trailing_per_vert = ns - written_per_vert
                for ni in range(orig_vc_src):
                    off = abs_vob_src + ni * ns
                    if color_count > 0:
                        orig_colors.append(orig_file_bytes[off + color_off_in_stride:off + color_off_in_stride + 4 * color_count])
                    if mesh.uv_count > 0:
                        orig_uvs.append(orig_file_bytes[off + uv_off_in_stride:off + uv_off_in_stride + 4 * mesh.uv_count])
                    if trailing_per_vert > 0:
                        trail_off = off + written_per_vert
                        orig_trailing.append(orig_file_bytes[trail_off:trail_off + trailing_per_vert])

                orig_idx_attr = data.attributes.get("mmb_vertex_order")
                orig_idx_for_vi = ({vi: orig_idx_attr.data[vi].value for vi in range(len(data.vertices))}
                                   if orig_idx_attr is not None else None)

                for vi, v in enumerate(data.vertices):
                    normal  = NTB[v.index][0]
                    tangent = NTB[v.index][1]
                    v_flip  = NTB[v.index][2]
                    src_vi  = min(orig_idx_for_vi.get(vi, vi) if orig_idx_for_vi else vi, orig_vc_src - 1)

                    # Colors first
                    if color_count > 0:
                        if SWOMT.export_vertex_colors:
                            for ci in range(color_count):
                                layer = bm.verts.layers.float_color.get(f"Color_{ci}")
                                if layer is not None:
                                    vertex_color = bm.verts[v.index][layer]
                                    for c in vertex_color:
                                        f.write(bp.uint8_norm(c))
                                elif orig_colors and src_vi < len(orig_colors):
                                    f.write(orig_colors[src_vi][ci * 4:ci * 4 + 4])
                                else:
                                    f.write(b'\x00' * 4)
                        elif orig_colors and src_vi < len(orig_colors):
                            f.write(orig_colors[src_vi])
                        else:
                            f.write(b'\x00' * (4 * color_count))

                    f.write(bp.float(normal[0] * -1))
                    f.write(bp.float(normal[1]))
                    f.write(bp.float(normal[2]))
                    f.write(bp.float(tangent[0] * -1))
                    f.write(bp.float(tangent[1]))
                    f.write(bp.float(tangent[2]))
                    f.write(bp.float(v_flip))

                    # UVs
                    if mesh.uv_count > 0:
                        if export_uvs:
                            for ui in range(mesh.uv_count):
                                u, v_uv = all_uvs[ui][v.index]
                                if _uv_wide[ui]:
                                    _u_raw = max(-32768, min(32767, int(round(u * _uv_u_divisor[ui]))))
                                    _v_raw = max(-32768, min(32767, int(round(v_uv * _uv_v_divisor[ui]))))
                                    f.write(pack('<h', _u_raw))
                                    f.write(pack('<h', _v_raw))
                                elif _uv_compact[ui]:
                                    f.write(bp.uv_unorm_u(max(0.0, min(1.0, u))))
                                    f.write(bp.uv_unorm_v(max(0.0, min(1.0, v_uv))))
                                else:
                                    f.write(bp.int16_norm(max(-1.0, min(1.0, u))))
                                    f.write(bp.int16_norm(max(-1.0, min(1.0, v_uv))))
                        elif orig_uvs and src_vi < len(orig_uvs):
                            f.write(orig_uvs[src_vi])
                        else:
                            f.write(b'\x00' * (4 * mesh.uv_count))

                    # Preserve any trailing bytes
                    if orig_trailing and src_vi < len(orig_trailing):
                        f.write(orig_trailing[src_vi])
                    elif trailing_per_vert > 0:
                        f.write(b'\x00' * trailing_per_vert)

            bm.free()


    @staticmethod
    def write_triangles(file, mesh:SkeletalMeshAsset.Mesh, lod_index=0, force_uint32=False):
        f = file
        obj = BME.find_object_by_name(mesh.name + f"_LOD{lod_index}")
        lod: SkeletalMeshAsset.Mesh.LOD = mesh.lods[lod_index]
        if obj:
            data = obj.data
            use_uint32 = force_uint32 or len(data.vertices) > 65535
            bm = bmesh.new()
            bm.from_mesh(data)
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            for p in data.polygons:
                if use_uint32:
                    f.write(bp.uint32(p.vertices[0]))
                    f.write(bp.uint32(p.vertices[2]))
                    f.write(bp.uint32(p.vertices[1]))
                else:
                    f.write(bp.uint16(p.vertices[0]))
                    f.write(bp.uint16(p.vertices[2]))
                    f.write(bp.uint16(p.vertices[1]))
            bm.free()

BME = BlenderMeshExporter

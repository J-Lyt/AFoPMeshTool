"""Scene properties panel for the add-on."""

import bpy

from . import addon_state, bl_info, updater
from .settings import _vert_count_changed

class SWOMTPanel(bpy.types.Panel):
    """Creates a Panel in the Scene Properties window"""
    bl_label = "AFoP Mesh Tool | Version {}.{}.{}".format(*bl_info["version"])
    bl_idname = "OBJECT_PT_swomtpanel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "scene"


    def draw(self,context):
        SWOMT = context.scene.SWOMT

        layout = self.layout

        # Update status bar
        if updater._update_status is None and updater._update_error is None:
            row = layout.row()
            row.operator("object.check_for_updates", text="Check for Updates", icon="FILE_REFRESH")
        elif updater._update_error:
            row = layout.row()
            row.label(text=f"Update check failed", icon="ERROR")
            row.operator("object.check_for_updates", text="Retry", icon="FILE_REFRESH")
        elif updater._update_status == "up_to_date":
            row = layout.row()
            row.label(text="Tool is up to date", icon="CHECKMARK")
            row.operator("object.check_for_updates", text="", icon="FILE_REFRESH")
        else:
            # update available
            box = layout.box()
            box.label(text=f"Update available: {updater._update_status}", icon="INFO")
            row = box.row()
            row.operator("object.apply_update", text="Update Now", icon="IMPORT")
            row.operator("object.check_for_updates", text="", icon="FILE_REFRESH")

        layout.separator()
        row = layout.row(align=True)
        row.prop(SWOMT, "AssetPath", text="Asset Path")
        row.operator("object.browse_mmb_file", text="", icon="FILE_FOLDER")
        layout.prop(SWOMT, "overwrite_existing")

        layout.separator()
        row = layout.row()
        row.operator("object.compute_normals", text="Compute Normals", icon="NORMALS_FACE")
        row.operator("object.clear_custom_normals", text="Clear Normals", icon="REMOVE")

        layout.separator()
        row = layout.row()
        if addon_state.asset:
            row.label(text=addon_state.asset.name)
            row.operator("object.rename_mmb_file", text="", icon="GREASEPENCIL")
            layout.separator()
            layout.label(text="Import", icon='IMPORT')
            imp_row = layout.row(align=True)
            for lod_n in range(4):
                any_lod_exists = any(len(m.lods) > lod_n for m in addon_state.asset.meshes if m.lods)
                btn = imp_row.row(align=True)
                btn.enabled = any_lod_exists
                btn.operator(f"object.import_all_lod{lod_n}s", text=f"LOD{lod_n}")
            layout.separator()
            layout.label(text="Export", icon='EXPORT')
            exp_row = layout.row()
            exp_row.scale_y = 1.5
            exp_row.enabled = any(
                len(m.lods) > lod_n and bpy.data.objects.get(
                    m.lods[lod_n].blender_obj_name if m.lods[lod_n].blender_obj_name else f"{m.name}_LOD{lod_n}"
                ) is not None
                for m in addon_state.asset.meshes if m.lods
                for lod_n in range(len(m.lods))
            )
            exp_row.operator("object.export_all_lods", text="Export All LODs")
            pose_row = layout.row()
            pose_row.scale_y = 1.2
            arm_obj = bpy.data.objects.get(addon_state.asset.name) if addon_state.asset else None
            pose_row.enabled = arm_obj is not None and arm_obj.type == 'ARMATURE'
            pose_row.operator("object.export_posed_bone_matrices",
                              text="Export Pose as New Rest Pose", icon="ARMATURE_DATA")
            force_row = layout.row(align=True)
            cfg_selected = bool(SWOMT.force_lod0_cfg_path.strip())
            btn_text = "Update 'lod_presets.cfg' file (Force LOD0)" if cfg_selected else "Generate 'lod_presets.cfg' file (Force LOD0)"
            force_row.operator("object.force_lod0", text=btn_text, icon='FILE_NEW')
            force_row.operator("object.browse_lod_presets_cfg", text="", icon='FILEBROWSER')
            if cfg_selected:
                force_row.operator("object.clear_lod_presets_cfg", text="", icon='REMOVE')
            if SWOMT.force_lod0_output_path:
                info_box = layout.box()
                if cfg_selected:
                    info_box.label(text="Updated file:", icon='INFO')
                    info_box.label(text=SWOMT.force_lod0_output_path)
                else:
                    info_box.label(text="Place generated file at:", icon='INFO')
                    info_box.label(text=r"...\AFOP\rogue\modules\core\graphobject\lod_presets.cfg")

            # Export Options collapsible box
            forced = _vert_count_changed()
            box = layout.box()
            row = box.row()
            row.prop(SWOMT, "export_options_expanded",
                     icon='TRIA_DOWN' if SWOMT.export_options_expanded else 'TRIA_RIGHT',
                     icon_only=True, emboss=False)
            row.label(text="Export Options")
            if SWOMT.export_options_expanded:
                box.prop(SWOMT, "compute_normals_on_export")
                for prop_name in ("export_normals", "export_weights", "export_uvs"):
                    prop_row = box.row()
                    prop_row.enabled = not forced
                    prop_row.prop(SWOMT, prop_name)
                    if forced:
                        prop_row.label(text="", icon='LOCKED')
                box.prop(SWOMT, "export_vertex_colors")
                if addon_state.asset and any(m.name.endswith('_CLOTH_RENDER')
                                 for m in addon_state.asset.meshes):
                    box.prop(SWOMT, "cloth_donor_radius")

            if forced:
                tip_row = layout.row()
                tip_row.label(text="Tip: Transfer Weights from original mesh", icon='INFO')
            for mi, m in enumerate(addon_state.asset.meshes):
                expanded = SWOMT.mesh_expanded[mi] if mi < 32 else True
                mesh_row = layout.row()
                mesh_box = mesh_row.box()
                name_row = mesh_box.row()
                name_row.prop(SWOMT, "mesh_expanded", index=mi, text="",
                              icon='TRIA_DOWN' if expanded else 'TRIA_RIGHT', emboss=False)
                name_row.label(text=m.name, icon="MESH_ICOSPHERE")
                gen_lods_op = name_row.operator("object.generate_lods", text="", icon="MOD_DECIM")
                gen_lods_op.mesh_index = mi
                scale_uv_op = name_row.operator("object.scale_uvs", text="", icon="UV")
                scale_uv_op.mesh_index = mi
                rename_op = name_row.operator("object.rename_mesh", text="", icon="GREASEPENCIL")
                rename_op.mesh_index = mi
                remove_row = name_row.row()
                remove_row.enabled = not (m.zeroed_out_in_session or m.zeroed_out_in_mmb)
                remove_op = remove_row.operator("object.remove_mesh", text="", icon="X")
                remove_op.mesh_index = mi
                revert_row = name_row.row()
                revert_row.enabled = m.zeroed_out_in_session
                revert_op = revert_row.operator("object.revert_mesh", text="", icon="LOOP_BACK")
                revert_op.mesh_index = mi
                if expanded:
                    for li,l in enumerate(m.lods):
                        row = mesh_box.row()
                        icon = "CON_SIZELIKE"
                        if m.zeroed_out_in_session or m.zeroed_out_in_mmb:
                            icon = "STRIP_COLOR_01"
                        row.label(text=f"LOD{li} - {l.vertex_count}", icon=icon)
                        lod_import_button = row.operator("object.import_lod")
                        lod_import_button.lod_index = li
                        lod_import_button.mesh_index = mi
                        obj_name = l.blender_obj_name if l.blender_obj_name else f"{m.name}_LOD{li}"
                        lod_export_row = row.row()
                        lod_export_row.enabled = bpy.data.objects.get(obj_name) is not None
                        lod_export_button = lod_export_row.operator("object.export_lod")
                        lod_export_button.lod_index = li
                        lod_export_button.mesh_index = mi
                    # Bone slot remap section
                    if m.mesh_bones:
                        bs_expanded = SWOMT.bone_slots_expanded[mi] if mi < 32 else True
                        bone_header = mesh_box.row()
                        bone_header.prop(SWOMT, "bone_slots_expanded", index=mi, text="",
                                         icon='TRIA_DOWN' if bs_expanded else 'TRIA_RIGHT', emboss=False)
                        bone_header.label(text="Bone Slots", icon="BONE_DATA")
                        add_op = bone_header.operator("object.add_mesh_bone", text="", icon="ADD")
                        add_op.mesh_index = mi
                        vg_op = bone_header.operator("object.add_bones_from_vertex_groups", text="", icon="GROUP_VERTEX")
                        vg_op.mesh_index = mi
                        bone_header.operator("object.merge_skeletons", text="", icon="ARMATURE_DATA")
                        if bs_expanded:
                            bone_box = mesh_box.box()
                            mesh_bones_list = list(m.mesh_bones.keys())
                            added_indices = {idx for idx, _ in m.pending_bone_additions}
                            for si, skel_idx in enumerate(mesh_bones_list):
                                bone_name = addon_state.asset.bones[skel_idx].name if skel_idx < len(addon_state.asset.bones) else str(skel_idx)
                                is_added = skel_idx in added_indices
                                pending_remap = m.pending_bone_remaps.get(si)
                                if is_added:
                                    label = bone_name + " +"
                                elif pending_remap is not None:
                                    label = bone_name + " *"
                                else:
                                    label = bone_name
                                slot_row = bone_box.row()
                                slot_row.label(text=f"[{si}] {label}", icon="GROUP_BONE")
                                remap_op = slot_row.operator("object.remap_mesh_bone", text="Remap")
                                remap_op.mesh_index = mi
                                remap_op.slot_index = si
                                if is_added:
                                    slot_row.enabled = False

CLASSES = (SWOMTPanel,)

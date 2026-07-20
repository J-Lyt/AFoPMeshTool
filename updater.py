"""Self-update support for the complete add-on package."""

import os
import re
import threading
import urllib.request

import bpy

RAW_BASE = "https://raw.githubusercontent.com/J-Lyt/AFoPMeshTool/master-refactor-dev"
RAW_INIT_URL = f"{RAW_BASE}/__init__.py"

CODE_FILES = (
    "__init__.py",
    "addon_state.py",
    "binary_io.py",
    "blender_mesh_utils.py",
    "cloth_export.py",
    "exporter.py",
    "file_utils.py",
    "importer.py",
    "log.py",
    "material_import.py",
    "materials/__init__.py",
    "materials/nodes.py",
    "materials/profiles.py",
    "materials/registry.py",
    "materials/textures.py",
    "meshlet.py",
    "mgraph.py",
    "mcloth.py",
    "mmb.py",
    "oodle_helper.py",
    "operators_bones.py",
    "operators_files.py",
    "operators_io.py",
    "operators_mesh.py",
    "operators_sdf.py",
    "sdf_reader.py",
    "sdf_toc.py",
    "settings.py",
    "shader_schema.py",
    "ui.py",
    "updater.py",
)
UPDATE_FILES = CODE_FILES

_update_status = None
_update_error = None


def _plugin_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _download_bytes(url):
    request = urllib.request.urlopen(url, timeout=30)
    return request.read()


def _version_from_source(source):
    match = re.search(
        rb'"version"\s*:\s*\((\d+),\s*(\d+),\s*(\d+)\)', source)
    if not match:
        return None
    return tuple(int(value) for value in match.groups())


def _fetch_remote_version():
    try:
        return _version_from_source(_download_bytes(RAW_INIT_URL)[:4096])
    except Exception:
        return None


def _local_version():
    try:
        with open(os.path.join(_plugin_dir(), "__init__.py"), "rb") as stream:
            return _version_from_source(stream.read(4096))
    except OSError:
        return None


def _check_update_thread():
    global _update_status, _update_error
    remote = _fetch_remote_version()
    local = _local_version()
    if remote is None or local is None:
        _update_error = "Could not read update version information."
        return
    _update_error = None
    _update_status = (f"v{remote[0]}.{remote[1]}.{remote[2]} available"
                      if remote > local else "up_to_date")


def start_update_check():
    threading.Thread(target=_check_update_thread, daemon=True).start()


def _validate_payloads(payloads):
    for filename in CODE_FILES:
        data = payloads[filename]
        try:
            source = data.decode("utf-8")
            compile(source, filename, "exec")
        except Exception as error:
            raise ValueError(f"Invalid Python file {filename}: {error}") from error
def _install_payloads(payloads):
    """Stage every file, then replace the package with rollback on failure."""
    plugin_dir = _plugin_dir()
    staged = {}
    originals = {}
    replaced = []
    try:
        for filename, data in payloads.items():
            destination = os.path.join(plugin_dir, filename)
            temporary = destination + ".update_tmp"
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            if os.path.isfile(destination):
                with open(destination, "rb") as stream:
                    originals[filename] = stream.read()
            else:
                originals[filename] = None
            with open(temporary, "wb") as stream:
                stream.write(data)
            staged[filename] = temporary

        for filename in UPDATE_FILES:
            destination = os.path.join(plugin_dir, filename)
            os.replace(staged[filename], destination)
            replaced.append(filename)
    except Exception:
        for filename in reversed(replaced):
            destination = os.path.join(plugin_dir, filename)
            original = originals[filename]
            if original is None:
                try:
                    os.remove(destination)
                except OSError:
                    pass
            else:
                with open(destination, "wb") as stream:
                    stream.write(original)
        raise
    finally:
        for temporary in staged.values():
            try:
                if os.path.exists(temporary):
                    os.remove(temporary)
            except OSError:
                pass


class CheckForUpdates(bpy.types.Operator):
    """Check GitHub for plugin updates."""

    bl_idname = "object.check_for_updates"
    bl_label = "Check for Updates"

    def execute(self, context):
        global _update_status, _update_error
        _update_status = None
        _update_error = None
        start_update_check()
        self.report({"INFO"}, "Checking for updates...")
        return {"FINISHED"}


class ApplyUpdate(bpy.types.Operator):
    """Download and install one mutually compatible package snapshot."""

    bl_idname = "object.apply_update"
    bl_label = "Update Now"

    def execute(self, context):
        try:
            payloads = {
                filename: _download_bytes(f"{RAW_BASE}/{filename}")
                for filename in UPDATE_FILES
            }
            _validate_payloads(payloads)
            _install_payloads(payloads)
        except Exception as error:
            self.report({"ERROR"}, f"Update failed; existing files were preserved: {error}")
            return {"CANCELLED"}

        global _update_status, _update_error
        _update_status = None
        _update_error = None
        self.report({"INFO"}, "Updated! Restart Blender to apply.")
        return {"FINISHED"}


CLASSES = (CheckForUpdates, ApplyUpdate)

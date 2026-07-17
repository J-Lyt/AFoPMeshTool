"""Browse AFOP SDF archives and import cached MMB assets directly into Blender."""

from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
import threading
from dataclasses import dataclass
from pathlib import PurePosixPath

import bpy

from . import addon_state, oodle_helper
from .log import logger
from .mmb import SkeletalMeshAsset
from . import sdf_toc
from .sdf_reader import SdfArchive


_OODLE_NAMES = (
    oodle_helper.OODLE_DLL_NAME,
    "oo2core_9_win64.dll",
    "oo2core_8_win64.dll",
    "oo2core_7_win64.dll",
    "oo2core_win64.dll",
)
_MAX_SEARCH_RESULTS = 500
_INDEX_CACHE_VERSION = 1


class _PrimitiveUnpickler(pickle.Unpickler):
    """Reject global/class lookups; cache payloads contain primitives only."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError("SDF index cache contains a non-primitive object")


@dataclass(frozen=True)
class _IndexedAsset:
    archive: SdfArchive
    asset: object
    archive_label: str
    cache_key: str


class _Cancelled(Exception):
    pass


class _SdfState:
    def __init__(self):
        self.lock = threading.Lock()
        self.generation = 0
        self.phase = "idle"
        self.status = "Choose the game folder containing the SDF archives."
        self.warning = ""
        self.progress = 0.0
        self.entries: list[_IndexedAsset] = []
        self.sidecars: dict[str, list[_IndexedAsset]] = {}


_state = _SdfState()


def _set_progress(generation, phase, status, progress):
    with _state.lock:
        if generation != _state.generation:
            raise _Cancelled()
        _state.phase = phase
        _state.status = status
        _state.progress = max(0.0, min(1.0, float(progress)))


def get_ui_status():
    with _state.lock:
        return {
            "generation": _state.generation,
            "phase": _state.phase,
            "status": _state.status,
            "warning": _state.warning,
            "progress": _state.progress,
            "count": len(_state.entries),
        }


def _discover_archives(root):
    toc_paths = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            lower = filename.lower()
            full_path = os.path.join(dirpath, filename)
            if lower.endswith(".sdftoc"):
                toc_paths.append(full_path)
    return sorted(toc_paths, key=str.casefold)


def _find_addon_oodle():
    """Return a supported Oodle DLL already installed beside the add-on."""
    addon_directory = os.path.dirname(os.path.abspath(__file__))
    for wanted in _OODLE_NAMES:
        candidate = os.path.join(addon_directory, wanted)
        if os.path.isfile(candidate):
            logger.info("Using Oodle DLL from add-on folder: %s", candidate)
            return candidate
    return None


def _archive_cache_key(toc_path):
    stat = os.stat(toc_path)
    identity = f"{os.path.abspath(toc_path)}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha1(identity.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def _index_cache_path(cache_dir, toc_path):
    identity = os.path.normcase(os.path.abspath(toc_path))
    digest = hashlib.sha1(identity.encode("utf-8", "surrogatepass")).hexdigest()
    return os.path.join(cache_dir, f"sdf_mesh_index_{digest}.pickle")


def _pack_asset(asset):
    slices = []
    for data_slice in asset.data_slices:
        slices.append((
            data_slice.decompressed_size,
            data_slice.compressed_size,
            data_slice.is_compressed,
            data_slice.is_oodle,
            data_slice.is_encrypted,
            data_slice.offset,
            data_slice.index,
            tuple(data_slice.page_sizes) if data_slice.page_sizes is not None else None,
        ))
    return (asset.name, asset.hash, asset.dds_index, asset.unk, tuple(slices))


def _unpack_asset(record):
    name, asset_hash, dds_index, unk, slice_records = record
    asset = sdf_toc.Asset(name=name, hash=asset_hash, dds_index=dds_index, unk=unk)
    for values in slice_records:
        (decompressed_size, compressed_size, is_compressed, is_oodle,
         is_encrypted, offset, index, page_sizes) = values
        asset.data_slices.append(sdf_toc.DataSlice(
            decompressed_size=decompressed_size,
            compressed_size=compressed_size,
            is_compressed=is_compressed,
            is_oodle=is_oodle,
            is_encrypted=is_encrypted,
            offset=offset,
            index=index,
            page_sizes=list(page_sizes) if page_sizes is not None else None,
        ))
    return asset


def _read_targeted_index_cache(cache_dir, toc_path):
    cache_path = _index_cache_path(cache_dir, toc_path)
    stat = os.stat(toc_path)
    try:
        with open(cache_path, "rb") as stream:
            payload = _PrimitiveUnpickler(stream).load()
        if not isinstance(payload, dict):
            return None
        if payload.get("version") != _INDEX_CACHE_VERSION:
            return None
        if payload.get("toc_path") != os.path.normcase(os.path.abspath(toc_path)):
            return None
        if payload.get("toc_size") != stat.st_size or payload.get("toc_mtime_ns") != stat.st_mtime_ns:
            return None
        records = payload.get("assets")
        if not isinstance(records, (tuple, list)):
            return None
        return [_unpack_asset(record) for record in records]
    except (OSError, EOFError, pickle.PickleError, TypeError, ValueError, AttributeError):
        return None


def _write_targeted_index_cache(cache_dir, toc_path, assets):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = _index_cache_path(cache_dir, toc_path)
    temporary = cache_path + ".tmp"
    stat = os.stat(toc_path)
    payload = {
        "version": _INDEX_CACHE_VERSION,
        "toc_path": os.path.normcase(os.path.abspath(toc_path)),
        "toc_size": stat.st_size,
        "toc_mtime_ns": stat.st_mtime_ns,
        "assets": tuple(_pack_asset(asset) for asset in assets),
    }
    try:
        with open(temporary, "wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, cache_path)
    finally:
        try:
            if os.path.exists(temporary):
                os.remove(temporary)
        except OSError:
            pass


def _archive_label(root, toc_path):
    relative = os.path.relpath(toc_path, root).replace(os.sep, "/")
    if "/" not in relative:
        return os.path.basename(root.rstrip("\\/"))
    return relative.split("/", 1)[0]


def _load_archives_worker(root, generation, index_cache_dir=None, allow_rebuild=True):
    try:
        _set_progress(generation, "loading", "Scanning for SDF archives...", 0.01)
        toc_paths = _discover_archives(root)
        if not toc_paths:
            if not allow_rebuild:
                _finish_cached_auto_load_unavailable(generation)
                return
            raise FileNotFoundError(f"No .sdftoc files were found under {root}")
        oodle_path = _find_addon_oodle()
        if oodle_path is None:
            _set_progress(generation, "loading", "Downloading Oodle DLL...", 0.02)
            try:
                oodle_path = oodle_helper.ensure_oodle_dll(
                    os.path.dirname(os.path.abspath(__file__))
                )
                logger.info("Downloaded Oodle DLL to add-on folder: %s", oodle_path)
            except oodle_helper.OodleDownloadError as error:
                logger.warning("Automatic Oodle DLL download failed: %s", error)
                if not allow_rebuild:
                    _finish_cached_auto_load_unavailable(
                        generation,
                        "Oodle DLL is missing. Reload SDF Archives to retry the download.",
                    )
                    return
                raise FileNotFoundError(
                    "Could not find Oodle in the add-on folder, and the automatic "
                    f"download failed: {error}"
                ) from error

        cached_assets = {}
        if not allow_rebuild:
            if not index_cache_dir:
                _finish_cached_auto_load_unavailable(generation)
                return
            for toc_path in toc_paths:
                assets = _read_targeted_index_cache(index_cache_dir, toc_path)
                if assets is None:
                    _finish_cached_auto_load_unavailable(generation)
                    return
                cached_assets[toc_path] = assets

        entries = []
        sidecars = {}
        errors = []
        archive_count = len(toc_paths)
        loaded_count = 0
        cached_count = 0
        rebuilt_count = 0
        for archive_index, toc_path in enumerate(toc_paths):
            with _state.lock:
                if generation != _state.generation:
                    raise _Cancelled()
            label = _archive_label(root, toc_path)
            base_progress = archive_index / archive_count
            span = 1.0 / archive_count
            try:
                archive = SdfArchive(toc_path, oodle_path)
                targeted_assets = (
                    cached_assets[toc_path] if not allow_rebuild else
                    (_read_targeted_index_cache(index_cache_dir, toc_path)
                     if index_cache_dir else None)
                )

                if targeted_assets is None:
                    def decrypt_progress(done, total, sample=None):
                        fraction = done / total if total else 0.0
                        _set_progress(
                            generation,
                            "loading",
                            f"Decrypting {label}",
                            base_progress + span * fraction * 0.45,
                        )

                    def parse_progress(done, total, sample=None):
                        fraction = done / total if total else 0.0
                        detail = sample or label
                        _set_progress(
                            generation,
                            "loading",
                            f"Indexing {detail}",
                            base_progress + span * (0.45 + fraction * 0.55),
                        )

                    archive.load(progress=decrypt_progress, parse_progress=parse_progress)
                    targeted_assets = [
                        asset for asset in archive.assets
                        if asset.name.casefold().endswith((".mmb", ".mcloth"))
                    ]
                    if index_cache_dir:
                        try:
                            _write_targeted_index_cache(index_cache_dir, toc_path, targeted_assets)
                        except OSError as error:
                            logger.warning("Could not write SDF index cache for %s: %s", toc_path, error)
                    archive.assets = []
                    rebuilt_count += 1
                else:
                    _set_progress(
                        generation,
                        "loading",
                        f"Loading cached index for {label}",
                        base_progress + span,
                    )
                    cached_count += 1

                cache_key = _archive_cache_key(toc_path)
                for asset in targeted_assets:
                    lower_name = asset.name.casefold()
                    indexed = _IndexedAsset(archive, asset, label, cache_key)
                    if lower_name.endswith(".mmb"):
                        entries.append(indexed)
                    elif lower_name.endswith(".mcloth"):
                        sidecars.setdefault(lower_name, []).append(indexed)
                loaded_count += 1
            except _Cancelled:
                raise
            except Exception as error:
                if not allow_rebuild:
                    logger.warning("Cached SDF auto-load skipped for %s: %s", toc_path, error)
                    _finish_cached_auto_load_unavailable(generation)
                    return
                logger.exception("Failed to index SDF archive %s: %s", toc_path, error)
                errors.append(f"{label}: {error}")

        entries.sort(key=lambda item: (item.asset.name.casefold(), item.archive_label.casefold()))
        if not loaded_count:
            detail = errors[0] if errors else "No archive could be loaded"
            raise RuntimeError(detail)

        with _state.lock:
            if generation != _state.generation:
                raise _Cancelled()
            _state.entries = entries
            _state.sidecars = sidecars
            _state.phase = "ready"
            _state.progress = 1.0
            if rebuilt_count:
                _state.status = (
                    f"Found {len(entries):,} MMB files in {loaded_count} SDF archive"
                    f"{'s' if loaded_count != 1 else ''}. "
                    f"Used {cached_count} cached {'index' if cached_count == 1 else 'indices'}; "
                    f"rebuilt {rebuilt_count}."
                )
            else:
                _state.status = ""
                logger.info("SDF archives loaded from cache.")
            _state.warning = (
                f"{len(errors)} archive{'s' if len(errors) != 1 else ''} could not be read."
                if errors else ""
            )
    except _Cancelled:
        return
    except Exception as error:
        if not allow_rebuild:
            logger.warning("Cached SDF auto-load skipped: %s", error)
            _finish_cached_auto_load_unavailable(generation)
            return
        logger.exception("SDF indexing failed: %s", error)
        with _state.lock:
            if generation != _state.generation:
                return
            _state.entries = []
            _state.sidecars = {}
            _state.phase = "error"
            _state.progress = 0.0
            _state.status = str(error)
            _state.warning = ""


def _finish_cached_auto_load_unavailable(
    generation,
    status="Reload SDF Archives to build or refresh the index cache.",
):
    """Leave SDF unloaded when startup caches are absent or stale; never rebuild."""
    with _state.lock:
        if generation != _state.generation:
            return
        _state.entries = []
        _state.sidecars = {}
        _state.phase = "idle"
        _state.progress = 0.0
        _state.status = status
        _state.warning = ""


def _populate_scene_results(generation):
    with _state.lock:
        if generation != _state.generation:
            return
    for scene in bpy.data.scenes:
        settings = getattr(scene, "SWOMT", None)
        if settings is None or settings.sdf_result_generation == generation:
            continue
        settings.sdf_assets.clear()
        settings.sdf_asset_index = -1
        settings.sdf_search_result_status = ""
        settings.sdf_result_generation = generation
        if settings.sdf_search_applied.strip():
            populate_search_results(scene, settings.sdf_search_applied)


def populate_search_results(scene, search):
    """Put only a bounded set of matches into Blender's redraw-heavy RNA list."""
    settings = getattr(scene, "SWOMT", None)
    if settings is None:
        return
    terms = search.casefold().split()
    settings.sdf_assets.clear()
    settings.sdf_asset_index = -1
    if not terms:
        settings.sdf_search_result_status = ""
        return

    with _state.lock:
        entries = _state.entries
        generation = _state.generation

    matches = []
    truncated = False
    for entry_id, entry in enumerate(entries):
        haystack = f"{entry.asset.name} {entry.archive_label}".casefold()
        if not all(term in haystack for term in terms):
            continue
        if len(matches) >= _MAX_SEARCH_RESULTS:
            truncated = True
            break
        matches.append((entry_id, entry.asset.name, entry.archive_label))

    with _state.lock:
        if generation != _state.generation:
            return
    for entry_id, asset_path, archive_label in matches:
        item = settings.sdf_assets.add()
        item.name = f"{entry_id}:{asset_path}"
        item.asset_path = asset_path
        item.archive_label = archive_label
        item.entry_id = entry_id
    settings.sdf_asset_index = 0 if matches else -1
    if truncated:
        settings.sdf_search_result_status = (
            f"Showing first {_MAX_SEARCH_RESULTS} matches. Refine the search to narrow results."
        )
    else:
        settings.sdf_search_result_status = ""


def _tag_properties_redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "PROPERTIES":
                area.tag_redraw()


def _index_timer():
    snapshot = get_ui_status()
    if snapshot["phase"] in {"ready", "error"}:
        _populate_scene_results(snapshot["generation"])
    _tag_properties_redraw()
    return 0.2 if snapshot["phase"] == "loading" else None


def _ensure_timer():
    if not bpy.app.timers.is_registered(_index_timer):
        bpy.app.timers.register(_index_timer, first_interval=0.1)


def _start_archive_load(root, allow_rebuild):
    """Start archive loading, optionally restricting it to current caches only."""
    root = bpy.path.abspath(root).rstrip("\\/")
    if os.path.isfile(root):
        root = os.path.dirname(root)
    if not root or not os.path.isdir(root):
        return False

    with _state.lock:
        _state.generation += 1
        generation = _state.generation
        _state.phase = "loading"
        _state.status = "Scanning for SDF archives..."
        _state.warning = ""
        _state.progress = 0.0
        _state.entries = []
        _state.sidecars = {}
    for scene in bpy.data.scenes:
        settings = getattr(scene, "SWOMT", None)
        if settings is not None:
            settings.sdf_result_generation = -1
            settings.sdf_assets.clear()
            settings.sdf_asset_index = -1
            settings.sdf_search_result_status = ""

    from .settings import blender_sdf_index_cache_directory
    threading.Thread(
        target=_load_archives_worker,
        args=(root, generation, blender_sdf_index_cache_directory(), allow_rebuild),
        name="AFoP SDF cached index" if not allow_rebuild else "AFoP SDF index",
        daemon=True,
    ).start()
    _ensure_timer()
    return True


def _cached_auto_load_timer():
    """Load current SDF metadata on startup only when every TOC cache is valid."""
    with _state.lock:
        if _state.phase in {"loading", "ready"}:
            return None
    found_directory = False
    for scene in bpy.data.scenes:
        settings = getattr(scene, "SWOMT", None)
        if settings is None:
            continue
        if settings.sdf_game_directory:
            found_directory = True
            _start_archive_load(settings.sdf_game_directory, allow_rebuild=False)
        break
    if not found_directory:
        with _state.lock:
            if _state.phase == "idle":
                _state.status = "Choose the game folder containing the SDF archives."
    return None


def schedule_cached_auto_load(reset=False):
    """Schedule a non-rebuilding cached load after registration or file load."""
    if reset:
        with _state.lock:
            _state.generation += 1
            _state.phase = "idle"
            _state.status = "Checking for current SDF index caches..."
            _state.warning = ""
            _state.progress = 0.0
            _state.entries = []
            _state.sidecars = {}
        for scene in bpy.data.scenes:
            settings = getattr(scene, "SWOMT", None)
            if settings is not None:
                settings.sdf_result_generation = -1
                settings.sdf_assets.clear()
                settings.sdf_asset_index = -1
                settings.sdf_search_result_status = ""
    if not bpy.app.timers.is_registered(_cached_auto_load_timer):
        bpy.app.timers.register(_cached_auto_load_timer, first_interval=0.25)


def shutdown():
    with _state.lock:
        _state.generation += 1
        _state.phase = "idle"
        _state.entries = []
        _state.sidecars = {}
    if bpy.app.timers.is_registered(_index_timer):
        bpy.app.timers.unregister(_index_timer)
    if bpy.app.timers.is_registered(_cached_auto_load_timer):
        bpy.app.timers.unregister(_cached_auto_load_timer)


def _entry_for_id(entry_id):
    with _state.lock:
        if 0 <= entry_id < len(_state.entries):
            return _state.entries[entry_id]
    return None


def _cache_root(cache_key, extracted_directory):
    root = os.path.join(bpy.path.abspath(extracted_directory), cache_key)
    os.makedirs(root, exist_ok=True)
    return root


def _safe_asset_parts(asset_name):
    path = PurePosixPath(asset_name.replace("\\", "/").lstrip("/"))
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe asset path in SDF index: {asset_name}")
    return path.parts


def _extract_to_cache(entry, extracted_directory=None, destination=None):
    if destination is None:
        if not extracted_directory:
            from .settings import get_default_extracted_files_directory
            extracted_directory = get_default_extracted_files_directory()
        destination = os.path.join(
            _cache_root(entry.cache_key, extracted_directory),
            *_safe_asset_parts(entry.asset.name),
        )
    if os.path.isfile(destination) and os.path.getsize(destination) > 0:
        return destination, False
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temporary = destination + ".extract_tmp"
    try:
        data = entry.archive.extract(entry.asset)
        with open(temporary, "wb") as stream:
            stream.write(data)
        os.replace(temporary, destination)
    finally:
        try:
            if os.path.exists(temporary):
                os.remove(temporary)
        except OSError:
            pass
    return destination, True


def _matching_sidecar(entry):
    sidecar_name = entry.asset.name.rsplit(".", 1)[0] + ".mcloth"
    with _state.lock:
        candidates = list(_state.sidecars.get(sidecar_name.casefold(), ()))
    for candidate in candidates:
        if candidate.archive is entry.archive:
            return candidate
    return candidates[0] if candidates else None


class SDFAssetList(bpy.types.UIList):
    """Searchable list of MMB asset paths found in loaded SDF archives."""

    bl_idname = "SWOMT_UL_sdf_assets"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            row.label(text=item.asset_path, icon="MESH_DATA")
            toc_column = row.column(align=True)
            toc_column.alignment = 'RIGHT'
            toc_column.label(text=item.archive_label)
        else:
            layout.label(text="", icon="MESH_DATA")

class BrowseSDFDirectory(bpy.types.Operator):
    """Choose the AFOP folder containing SDF archives."""

    bl_idname = "object.browse_sdf_directory"
    bl_label = "Choose Game Folder"

    directory: bpy.props.StringProperty(subtype="DIR_PATH")

    def invoke(self, context, event):
        current = bpy.path.abspath(context.scene.SWOMT.sdf_game_directory)
        if current:
            self.directory = current
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        context.scene.SWOMT.sdf_game_directory = self.directory
        result = bpy.ops.object.index_sdf_archives()
        return result if "CANCELLED" in result else {"FINISHED"}


class BrowseSDFExtractedDirectory(bpy.types.Operator):
    """Choose where assets extracted from SDF archives are stored."""

    bl_idname = "object.browse_sdf_extracted_directory"
    bl_label = "Choose Extracted Files Folder"

    directory: bpy.props.StringProperty(subtype="DIR_PATH")

    def invoke(self, context, event):
        current = bpy.path.abspath(context.scene.SWOMT.sdf_extracted_directory)
        if current:
            self.directory = current
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        context.scene.SWOMT.sdf_extracted_directory = self.directory
        return {"FINISHED"}


class ClearSDFIndexCache(bpy.types.Operator):
    """Delete targeted SDF index caches without removing extracted assets."""

    bl_idname = "object.clear_sdf_index_cache"
    bl_label = "Clear SDF Index Cache"
    bl_description = "Delete cached MMB/mcloth indexes; extracted asset files are preserved"

    def execute(self, context):
        from .settings import blender_sdf_index_cache_directory

        cache_dir = blender_sdf_index_cache_directory()
        removed = 0
        errors = []
        try:
            filenames = os.listdir(cache_dir)
        except FileNotFoundError:
            filenames = []
        except OSError as error:
            self.report({"ERROR"}, f"Could not read SDF index cache: {error}")
            return {"CANCELLED"}

        for filename in filenames:
            if not (filename.startswith("sdf_mesh_index_")
                    and filename.endswith((".pickle", ".pickle.tmp"))):
                continue
            path = os.path.join(cache_dir, filename)
            try:
                os.remove(path)
                removed += 1
            except OSError as error:
                errors.append(f"{filename}: {error}")

        if errors:
            self.report(
                {"WARNING"},
                f"Removed {removed} cache file(s); {len(errors)} could not be removed.",
            )
        elif removed:
            self.report({"INFO"}, f"Removed {removed} SDF index cache file(s).")
        else:
            self.report({"INFO"}, "The SDF index cache is already empty.")
        return {"FINISHED"}


class IndexSDFArchives(bpy.types.Operator):
    """Decrypt and index MMB files in the selected AFOP SDF archives."""

    bl_idname = "object.index_sdf_archives"
    bl_label = "Reload SDF Archives"

    def execute(self, context):
        if not _start_archive_load(
            context.scene.SWOMT.sdf_game_directory, allow_rebuild=True
        ):
            self.report({"ERROR"}, "Choose an existing game folder first.")
            return {"CANCELLED"}
        self.report({"INFO"}, "Loading SDF archives in the background...")
        return {"FINISHED"}


class ImportSDFMMB(bpy.types.Operator):
    """Import the selected MMB and optionally retain it as the current asset."""

    bl_idname = "object.import_sdf_mmb"
    bl_label = "Import Selected MMB"

    import_lod0: bpy.props.BoolProperty(default=True, options={"HIDDEN"})

    @classmethod
    def description(cls, context, properties):
        if not properties.import_lod0:
            return "Extract the selected MMB and load it in Asset Path"
        return "Import LOD0 from the selected MMB"

    @classmethod
    def poll(cls, context):
        settings = getattr(context.scene, "SWOMT", None)
        return settings is not None and 0 <= settings.sdf_asset_index < len(settings.sdf_assets)

    def execute(self, context):
        settings = context.scene.SWOMT
        item = settings.sdf_assets[settings.sdf_asset_index]
        entry = _entry_for_id(item.entry_id)
        if entry is None:
            self.report({"ERROR"}, "The SDF index changed; reload the archives and try again.")
            return {"CANCELLED"}

        load_as_asset = not self.import_lod0 or settings.sdf_load_as_asset
        temporary_directory = None
        try:
            if load_as_asset:
                mmb_path, extracted = _extract_to_cache(
                    entry, extracted_directory=settings.sdf_extracted_directory
                )
            else:
                temporary_directory = tempfile.TemporaryDirectory(prefix="afop_sdf_import_")
                temporary_path = os.path.join(
                    temporary_directory.name,
                    *_safe_asset_parts(entry.asset.name),
                )
                mmb_path, extracted = _extract_to_cache(entry, destination=temporary_path)
        except Exception as error:
            if temporary_directory is not None:
                temporary_directory.cleanup()
            logger.exception("Could not extract SDF asset %s: %s", item.asset_path, error)
            self.report({"ERROR"}, f"MMB extraction failed: {error}")
            return {"CANCELLED"}

        if load_as_asset:
            sidecar = _matching_sidecar(entry)
        else:
            sidecar = None
        if sidecar is not None:
            mcloth_path = os.path.splitext(mmb_path)[0] + ".mcloth"
            try:
                _extract_to_cache(sidecar, destination=mcloth_path)
            except Exception as error:
                logger.warning("Could not extract paired mcloth for %s: %s", item.asset_path, error)
                self.report({"WARNING"}, f"MMB loaded, but paired mcloth extraction failed: {error}")

        try:
            with open(mmb_path, "rb") as stream:
                probe = SkeletalMeshAsset()
                probe.parse(stream)
            probe.name = os.path.splitext(os.path.basename(mmb_path))[0]
        except Exception as error:
            if temporary_directory is not None:
                temporary_directory.cleanup()
            self.report({"ERROR"}, f"The extracted MMB could not be parsed: {error}")
            return {"CANCELLED"}

        if not self.import_lod0:
            settings.AssetPath = mmb_path
            if addon_state.asset is None:
                self.report({"ERROR"}, "The extracted MMB could not be parsed.")
                return {"CANCELLED"}
            action = "Extracted and loaded" if extracted else "Loaded cached"
            self.report({"INFO"}, f"{action} {item.asset_path}")
            return {"FINISHED"}

        try:
            from .operators_io import _import_all_lods
            if load_as_asset:
                settings.AssetPath = mmb_path
                if addon_state.asset is None:
                    self.report({"ERROR"}, "The extracted MMB could not be parsed.")
                    return {"CANCELLED"}
                result = _import_all_lods(context, 0)
            else:
                result = _import_all_lods(
                    context,
                    0,
                    skeletal_mesh=probe,
                    asset_path=mmb_path,
                )
        except Exception as error:
            logger.exception("Could not import extracted SDF MMB %s: %s", mmb_path, error)
            self.report({"ERROR"}, f"LOD0 import failed: {error}")
            return {"CANCELLED"}
        finally:
            if temporary_directory is not None:
                temporary_directory.cleanup()

        action = "Imported and loaded" if load_as_asset else "Imported"
        self.report({"INFO"}, f"{action} LOD0 from {item.asset_path}")
        return result


CLASSES = (
    SDFAssetList,
    BrowseSDFDirectory,
    BrowseSDFExtractedDirectory,
    ClearSDFIndexCache,
    IndexSDFArchives,
    ImportSDFMMB,
)

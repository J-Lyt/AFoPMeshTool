"""Browse AFOP SDF archives and import cached MMB assets directly into Blender."""

import hashlib
import os
import pickle
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import PurePosixPath

import bpy

from .. import addon_state, material_import, mgraph, shader_schema
from ..log import logger
from ..mmb import SkeletalMeshAsset
from ..sdf import oodle as oodle_helper
from ..sdf import toc as sdf_toc
from ..sdf.reader import SdfArchive


_OODLE_NAMES = (
    oodle_helper.OODLE_DLL_NAME,
    "oo2core_9_win64.dll",
    "oo2core_8_win64.dll",
    "oo2core_7_win64.dll",
    "oo2core_win64.dll",
)
_MAX_SEARCH_RESULTS = 500
_INDEX_CACHE_VERSION = 4
_ASSET_MMB = "MMB"
_ASSET_MGRAPH = "MGRAPHOBJECT"
_ASSET_MCOMPOUND = "MCOMPOUNDNODE"
_MATERIAL_SOURCE_AUTO = "__AUTO__"
_material_source_dialog_items = [
    (_MATERIAL_SOURCE_AUTO, "Automatic", "Use the highest-ranked material source", 0)
]


def _material_source_enum_items(_self, _context):
    return _material_source_dialog_items


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
        self.search_entries: list[tuple[str, _IndexedAsset]] = []
        self.sidecars: dict[str, list[_IndexedAsset]] = {}
        self.graphs: dict[str, list[_IndexedAsset]] = {}
        self.compounds: dict[str, list[_IndexedAsset]] = {}
        self.shaders: dict[str, list[_IndexedAsset]] = {}
        self.textures: dict[str, list[_IndexedAsset]] = {}


_state = _SdfState()

_shader_schema_lock = threading.Lock()
_shader_pin_cache = {}


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
        dds_block = payload.get("dds_block", b"")
        if not isinstance(dds_block, bytes):
            return None
        return [_unpack_asset(record) for record in records], dds_block
    except (OSError, EOFError, pickle.PickleError, TypeError, ValueError, AttributeError):
        return None


def _write_targeted_index_cache(cache_dir, toc_path, assets, dds_block):
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
        # Texture payloads live in sdfdata, but their STF/DDS descriptors live
        # only in this TOC block. Retain it so cached startup imports can decode
        # textures without decrypting and rebuilding the whole file table.
        "dds_block": bytes(dds_block),
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
        search_entries = []
        sidecars = {}
        graphs = {}
        compounds = {}
        shaders = {}
        textures = {}
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
                cached_target = (
                    cached_assets[toc_path] if not allow_rebuild else
                    (_read_targeted_index_cache(index_cache_dir, toc_path)
                     if index_cache_dir else None)
                )

                if cached_target is None:
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
                        if asset.name.casefold().endswith(
                            (
                                ".mmb", ".mcloth", ".mgraphobject",
                                ".mcompoundnode", ".mshader", ".dds",
                            )
                        )
                    ]
                    if index_cache_dir:
                        try:
                            _write_targeted_index_cache(
                                index_cache_dir,
                                toc_path,
                                targeted_assets,
                                archive.dds_block,
                            )
                        except OSError as error:
                            logger.warning("Could not write SDF index cache for %s: %s", toc_path, error)
                    archive.assets = []
                    rebuilt_count += 1
                else:
                    targeted_assets, cached_dds_block = cached_target
                    archive.dds_block = cached_dds_block
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
                        search_entries.append((_ASSET_MMB, indexed))
                    elif lower_name.endswith(".mcloth"):
                        sidecars.setdefault(lower_name, []).append(indexed)
                    elif lower_name.endswith(".mgraphobject"):
                        graphs.setdefault(lower_name, []).append(indexed)
                        search_entries.append((_ASSET_MGRAPH, indexed))
                    elif lower_name.endswith(".mcompoundnode"):
                        compounds.setdefault(lower_name, []).append(indexed)
                        search_entries.append((_ASSET_MCOMPOUND, indexed))
                    elif lower_name.endswith(".mshader"):
                        shaders.setdefault(lower_name, []).append(indexed)
                    elif lower_name.endswith(".dds"):
                        textures.setdefault(lower_name, []).append(indexed)
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
        search_entries.sort(
            key=lambda item: (
                item[1].asset.name.casefold(),
                item[0],
                item[1].archive_label.casefold(),
            )
        )
        if not loaded_count:
            detail = errors[0] if errors else "No archive could be loaded"
            raise RuntimeError(detail)

        with _state.lock:
            if generation != _state.generation:
                raise _Cancelled()
            _state.entries = entries
            _state.search_entries = search_entries
            _state.sidecars = sidecars
            _state.graphs = graphs
            _state.compounds = compounds
            _state.shaders = shaders
            _state.textures = textures
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
            _state.search_entries = []
            _state.sidecars = {}
            _state.graphs = {}
            _state.compounds = {}
            _state.shaders = {}
            _state.textures = {}
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
        _state.search_entries = []
        _state.sidecars = {}
        _state.graphs = {}
        _state.compounds = {}
        _state.shaders = {}
        _state.textures = {}
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

    enabled_types = set()
    if settings.sdf_show_mmb:
        enabled_types.add(_ASSET_MMB)
    if settings.sdf_show_mgraphobject:
        enabled_types.add(_ASSET_MGRAPH)
    if settings.sdf_show_mcompoundnode:
        enabled_types.add(_ASSET_MCOMPOUND)

    with _state.lock:
        search_entries = _state.search_entries
        generation = _state.generation

    matches = []
    truncated = False
    for entry_id, (asset_type, entry) in enumerate(search_entries):
        if asset_type not in enabled_types:
            continue
        haystack = f"{entry.asset.name} {entry.archive_label}".casefold()
        if not all(term in haystack for term in terms):
            continue
        if len(matches) >= _MAX_SEARCH_RESULTS:
            truncated = True
            break
        matches.append((entry_id, asset_type, entry.asset.name, entry.archive_label))

    with _state.lock:
        if generation != _state.generation:
            return
    for entry_id, asset_type, asset_path, archive_label in matches:
        item = settings.sdf_assets.add()
        item.name = f"{entry_id}:{asset_path}"
        item.asset_path = asset_path
        item.asset_type = asset_type
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

    with _shader_schema_lock:
        _shader_pin_cache.clear()

    with _state.lock:
        _state.generation += 1
        generation = _state.generation
        _state.phase = "loading"
        _state.status = "Scanning for SDF archives..."
        _state.warning = ""
        _state.progress = 0.0
        _state.entries = []
        _state.search_entries = []
        _state.sidecars = {}
        _state.graphs = {}
        _state.compounds = {}
        _state.shaders = {}
        _state.textures = {}
    for scene in bpy.data.scenes:
        settings = getattr(scene, "SWOMT", None)
        if settings is not None:
            settings.sdf_result_generation = -1
            settings.sdf_assets.clear()
            settings.sdf_asset_index = -1
            settings.sdf_search_result_status = ""

    from ..settings import blender_sdf_index_cache_directory
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
            _state.search_entries = []
            _state.sidecars = {}
            _state.graphs = {}
            _state.compounds = {}
            _state.shaders = {}
            _state.textures = {}
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
        _state.search_entries = []
        _state.sidecars = {}
        _state.graphs = {}
        _state.compounds = {}
        _state.shaders = {}
        _state.textures = {}
    with _shader_schema_lock:
        _shader_pin_cache.clear()
    if bpy.app.timers.is_registered(_index_timer):
        bpy.app.timers.unregister(_index_timer)
    if bpy.app.timers.is_registered(_cached_auto_load_timer):
        bpy.app.timers.unregister(_cached_auto_load_timer)


def _search_entry_for_id(entry_id):
    with _state.lock:
        if 0 <= entry_id < len(_state.search_entries):
            return _state.search_entries[entry_id]
    return None, None


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
            from ..settings import get_default_extracted_files_directory
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


def _prefer_archive(candidates, archive):
    """Keep exact logical-name resolution deterministic across game packs."""
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.archive is not archive,
            candidate.archive_label.casefold(),
            candidate.asset.name.casefold(),
        ),
    )


def _material_family_reference(mesh_path, graph_path, references):
    """Accept the confirmed base_01 -> base graph -> base_bind rig pattern."""
    mesh_stem = os.path.splitext(os.path.basename(mesh_path))[0].casefold()
    graph_stem = os.path.splitext(os.path.basename(graph_path))[0].casefold()
    match = re.fullmatch(r"(.+)_\d+", mesh_stem)
    if match is None or graph_stem != match.group(1):
        return False
    bind_stem = graph_stem + "_bind"
    return any(
        os.path.splitext(os.path.basename(reference))[0].casefold() == bind_stem
        for reference in references
    )


def _ranked_material_graph_candidates(entry):
    """Name-ranked graph candidates; contents provide final authority."""
    mesh_name = entry.asset.name.casefold()
    mesh_stem = os.path.splitext(os.path.basename(mesh_name))[0]
    mesh_parts = tuple(
        token for token in re.split(r"[^a-z0-9]+", mesh_stem) if token
    )
    mesh_tokens = {token for token in _source_name_tokens(mesh_stem) if len(token) >= 4}
    with _state.lock:
        all_graphs = [candidate for values in _state.graphs.values() for candidate in values]

    candidates = []
    for candidate in all_graphs:
        graph_stem = os.path.splitext(os.path.basename(candidate.asset.name))[0].casefold()
        graph_parts = tuple(
            token for token in re.split(r"[^a-z0-9]+", graph_stem) if token
        )
        if graph_stem == mesh_stem:
            name_score = 3
            token_affinity = 0
        elif graph_stem.startswith(mesh_stem) or mesh_stem.startswith(graph_stem):
            name_score = 2
            token_affinity = 0
        else:
            graph_tokens = {
                token for token in _source_name_tokens(graph_stem) if len(token) >= 4
            }
            exact_overlap = mesh_tokens & graph_tokens
            partial_overlap = {
                (mesh_token, graph_token)
                for mesh_token in mesh_tokens
                for graph_token in graph_tokens
                if mesh_token not in exact_overlap
                and graph_token not in exact_overlap
                and min(len(mesh_token), len(graph_token)) >= 4
                and (mesh_token in graph_token or graph_token in mesh_token)
            }
            token_affinity = len(exact_overlap) * 4 + len(partial_overlap)
            # Custom-character assets commonly split one MMB across sibling
            # graphs such as cus_03_gear and cus_03_mask.  Their useful family
            # identity consists of short tokens that the general affinity gate
            # deliberately ignores. Admit only the structured <family>_<id>
            # prefix here; an exact material-name match remains authoritative
            # when the supplemental graph is inspected.
            shared_numbered_family = (
                len(mesh_parts) >= 3
                and len(graph_parts) >= 3
                and mesh_parts[:2] == graph_parts[:2]
                and any(character.isdigit() for character in mesh_parts[1])
            )
            if not token_affinity and shared_numbered_family:
                token_affinity = 2
            if not token_affinity:
                # A full graph-library content scan would make every import
                # unexpectedly expensive. Filename token/substring affinity is
                # only a candidate gate; an exact MMB reference is still
                # required below for these alias-named placement graphs.
                continue
            name_score = 1
        candidates.append((name_score, token_affinity, candidate))

    candidates.sort(
        key=lambda item: (
            -item[0],
            -item[1],
            item[2].archive is not entry.archive,
            item[2].asset.name.casefold(),
        )
    )
    return candidates


def _material_graph_options(entry):
    """Return graphs that directly reference the exact or family MMB."""
    mesh_name = entry.asset.name.casefold()
    options = []
    successful_paths = set()
    for name_score, token_affinity, candidate in _ranked_material_graph_candidates(entry):
        logical_key = candidate.asset.name.casefold()
        if logical_key in successful_paths:
            continue
        try:
            data = candidate.archive.extract(candidate.asset)
            if data[:4] != mgraph.MAGIC:
                continue
            references = _direct_material_source_references(data)
            if mesh_name not in references and not _material_family_reference(
                mesh_name, candidate.asset.name, references
            ):
                continue
            compound_sources = _referenced_compound_data(data, candidate.archive)
            richness = _material_source_richness(data, compound_sources)
            if richness == 0:
                continue
            successful_paths.add(logical_key)
            options.append((candidate, data, richness, name_score, token_affinity))
        except Exception as error:
            logger.debug("Could not inspect material graph %s: %s", candidate.asset.name, error)
    options.sort(
        key=lambda item: (
            -item[3],
            -item[4],
            -item[2],
            item[0].archive is not entry.archive,
            item[0].asset.name.casefold(),
        )
    )
    return options


def _material_graph(entry, selected_path=None):
    """Return the selected or highest-ranked plausible material graph."""
    mesh_name = entry.asset.name.casefold()
    if selected_path and selected_path != _MATERIAL_SOURCE_AUTO:
        with _state.lock:
            selected = list(_state.graphs.get(selected_path.casefold(), ()))
        for candidate in _prefer_archive(selected, entry.archive):
            try:
                data = candidate.archive.extract(candidate.asset)
                if data[:4] != mgraph.MAGIC:
                    continue
                source_references, compound_sources = _material_source_references(
                    data, candidate.archive
                )
                references = {path.casefold() for path in source_references}
                if (
                    mesh_name not in references
                    and not _material_family_reference(
                        mesh_name, candidate.asset.name, references
                    )
                ) or not _material_source_richness(data, compound_sources):
                    continue
                return candidate, data
            except Exception as error:
                logger.debug(
                    "Could not inspect selected material graph %s: %s",
                    candidate.asset.name,
                    error,
                )
        return None, None

    best = None
    for name_score, token_affinity, candidate in _ranked_material_graph_candidates(entry):
        if name_score == 1 and best is not None:
            # A conventional exact/prefix-name graph already succeeded; do not
            # spend time probing weaker placement-name aliases as well.
            break
        try:
            data = candidate.archive.extract(candidate.asset)
            if data[:4] != mgraph.MAGIC:
                continue
            references = _direct_material_source_references(data)
            exact_reference = mesh_name in references
            family_reference = _material_family_reference(
                mesh_name, candidate.asset.name, references
            )
            if not (exact_reference or family_reference):
                continue
            if name_score == 1 and not exact_reference:
                continue
            compound_sources = _referenced_compound_data(data, candidate.archive)
            richness = _material_source_richness(data, compound_sources)
            if richness == 0:
                continue
            if name_score == 1:
                # Alias-name candidates are admitted only by an exact embedded
                # MMB reference, so the first successful one is authoritative.
                return candidate, data
            score = (
                name_score * 10000
                + int(exact_reference) * 5000
                + int(family_reference) * 2500
                + token_affinity * 100
                + richness
            )
            if best is None or score > best[0]:
                best = (score, candidate, data)
        except Exception as error:
            logger.debug("Could not inspect material graph %s: %s", candidate.asset.name, error)
    return (best[1], best[2]) if best else (None, None)


def _source_name_tokens(path):
    return {
        token
        for token in re.split(r"[^a-z0-9]+", path.casefold())
        if len(token) >= 2
    }


def _ranked_material_compound_candidates(entry):
    """Return filename-ranked compound candidates for the selected MMB."""
    mesh_name = entry.asset.name.casefold()
    mesh_tokens = _source_name_tokens(mesh_name)
    with _state.lock:
        all_compounds = [
            candidate
            for values in _state.compounds.values()
            for candidate in values
        ]

    candidates = []
    for candidate in all_compounds:
        overlap = len(mesh_tokens & _source_name_tokens(candidate.asset.name))
        if overlap:
            candidates.append((overlap, candidate))
    candidates.sort(
        key=lambda item: (
            -item[0],
            item[1].archive is not entry.archive,
            item[1].asset.name.casefold(),
        )
    )
    return candidates


def _material_compound_options(entry):
    """Return candidate compounds that directly reference this exact MMB."""
    mesh_name = entry.asset.name.casefold()
    options = []
    successful_paths = set()
    for overlap, candidate in _ranked_material_compound_candidates(entry):
        logical_key = candidate.asset.name.casefold()
        if logical_key in successful_paths:
            continue
        try:
            data = candidate.archive.extract(candidate.asset)
            if data[:4] != mgraph.MAGIC:
                continue
            references = _direct_material_source_references(data)
            if mesh_name not in references:
                continue
            compound_sources = _referenced_compound_data(data, candidate.archive)
            richness = _material_source_richness(data, compound_sources)
            if richness == 0:
                continue
            successful_paths.add(logical_key)
            options.append((candidate, data, richness, overlap))
        except Exception as error:
            logger.debug(
                "Could not inspect material compound %s: %s",
                candidate.asset.name,
                error,
            )
    options.sort(
        key=lambda item: (
            -item[3],
            -item[2],
            item[0].archive is not entry.archive,
            item[0].asset.name.casefold(),
        )
    )
    return options


def _material_compound(entry, selected_path=None):
    """Find a BV2 compound chain that references the selected MMB."""
    mesh_name = entry.asset.name.casefold()
    if selected_path and selected_path != _MATERIAL_SOURCE_AUTO:
        with _state.lock:
            selected = list(_state.compounds.get(selected_path.casefold(), ()))
        for candidate in _prefer_archive(selected, entry.archive):
            try:
                data = candidate.archive.extract(candidate.asset)
                if data[:4] != mgraph.MAGIC:
                    continue
                source_references, compound_sources = _material_source_references(
                    data, candidate.archive
                )
                references = {path.casefold() for path in source_references}
                if (
                    mesh_name not in references
                    or not _material_source_richness(data, compound_sources)
                ):
                    continue
                return candidate, data
            except Exception as error:
                logger.debug(
                    "Could not inspect selected material compound %s: %s",
                    candidate.asset.name,
                    error,
                )
        return None, None

    best = None
    best_overlap = None
    for overlap, candidate in _ranked_material_compound_candidates(entry):
        if best_overlap is not None and overlap < best_overlap:
            break
        try:
            data = candidate.archive.extract(candidate.asset)
            if data[:4] != mgraph.MAGIC:
                continue
            references = _direct_material_source_references(data)
            if mesh_name not in references:
                continue
            compound_sources = _referenced_compound_data(data, candidate.archive)
            richness = _material_source_richness(data, compound_sources)
            if richness == 0:
                continue
            score = overlap * 10000 + richness
            if best is None or score > best[0]:
                best = (score, candidate, data)
                best_overlap = overlap
        except Exception as error:
            logger.debug(
                "Could not inspect material compound %s: %s",
                candidate.asset.name,
                error,
            )
    return (best[1], best[2]) if best else (None, None)


def _material_source_options(entry):
    """Return graph and compound choices in automatic-priority order."""
    options = []
    for candidate, data, richness, name_score, affinity in _material_graph_options(entry):
        options.append((
            "Graph",
            candidate,
            data,
            richness,
            name_score,
            affinity,
        ))
    for candidate, data, richness, overlap in _material_compound_options(entry):
        options.append((
            "Compound",
            candidate,
            data,
            richness,
            overlap,
            0,
        ))
    return options


def _texture_entry(logical_path, preferred_archive):
    with _state.lock:
        candidates = list(_state.textures.get(logical_path.casefold(), ()))
    ordered = _prefer_archive(candidates, preferred_archive)
    return ordered[0] if ordered else None


def _shader_pins_for_sources(data, compound_sources, preferred_archive):
    """Read authored sampler and constant pin IDs for a source chain."""
    shader_paths = []
    seen = set()
    for source_data in (data, *(compound_sources or {}).values()):
        for _name, shader_path in mgraph.material_shader_pairs(source_data):
            key = shader_path.replace("\\", "/").lstrip("/").casefold()
            if key not in seen:
                seen.add(key)
                shader_paths.append((key, shader_path))

    result = ({}, {})
    for key, shader_path in shader_paths:
        with _state.lock:
            candidates = list(_state.shaders.get(key, ()))
        ordered = _prefer_archive(candidates, preferred_archive)
        if not ordered:
            continue
        candidate = ordered[0]
        cache_key = (candidate.cache_key, key)
        with _shader_schema_lock:
            cached = _shader_pin_cache.get(cache_key)
        if cached is None:
            try:
                parsed = shader_schema.parse_shader_source(
                    candidate.archive.extract(candidate.asset)
                )
                role_pins = {
                    sampler["pin_id"]: (
                        sampler["role"]
                        if sampler["role"] in {"d", "n", "m"}
                        else f"aux:{sampler['field']}"
                    )
                    for sampler in parsed["samplers"]
                    if sampler["pin_id"] is not None
                }
                parameter_pins = {
                    parameter["pin_id"]: parameter["field"]
                    for parameter in parsed["parameters"]
                }
                schema_known = True
            except Exception as error:
                logger.debug("Could not read shader schema %s: %s", shader_path, error)
                role_pins = {}
                parameter_pins = {}
                schema_known = False
            cached = (role_pins, parameter_pins, schema_known)
            with _shader_schema_lock:
                _shader_pin_cache[cache_key] = cached
        role_pins, parameter_pins, schema_known = cached
        if schema_known:
            result[0][key] = role_pins
            result[0].setdefault(os.path.basename(key), role_pins)
        if parameter_pins:
            result[1][key] = parameter_pins
            result[1].setdefault(os.path.basename(key), parameter_pins)
    return result


def _referenced_compound_data(data, preferred_archive):
    """Extract the complete linked-compound closure for one BV2 source."""
    result = {}
    pending = [
        (logical_path, preferred_archive)
        for logical_path in mgraph.referenced_compounds(data)
    ]
    seen = set()
    while pending:
        logical_path, parent_archive = pending.pop(0)
        key = logical_path.casefold()
        if key in seen:
            continue
        seen.add(key)
        with _state.lock:
            candidates = list(_state.compounds.get(key, ()))
        for candidate in _prefer_archive(candidates, parent_archive):
            try:
                compound_data = candidate.archive.extract(candidate.asset)
                if compound_data[:4] == mgraph.MAGIC:
                    result[logical_path] = compound_data
                    pending.extend(
                        (nested_path, candidate.archive)
                        for nested_path in mgraph.referenced_compounds(compound_data)
                    )
                    break
            except Exception as error:
                logger.debug(
                    "Could not inspect referenced compound %s: %s",
                    candidate.asset.name,
                    error,
                )
    return result


def _material_source_references(data, preferred_archive):
    """Return MMB references from a graph/compound and all linked compounds."""
    compound_sources = _referenced_compound_data(data, preferred_archive)
    references = []
    seen = set()
    for source_data in (data, *compound_sources.values()):
        for logical_path in mgraph.referenced_meshes(source_data):
            key = logical_path.casefold()
            if key in seen:
                continue
            seen.add(key)
            references.append(logical_path)
    return references, compound_sources


def _direct_material_source_references(data):
    """Return normalized MMB references stored directly in one BV2 source."""
    return {path.casefold() for path in mgraph.referenced_meshes(data)}


def _material_source_richness(data, compound_sources):
    """Count distinct texture constants across a linked material-source chain."""
    paths = {
        texture["path"].casefold()
        for source_data in (data, *compound_sources.values())
        for texture in mgraph.texture_pool(source_data)
    }
    return len(paths)


def _supplemental_material_bindings(
    entry,
    primary_entry,
    material_names,
    bindings,
    primary_material_names=None,
):
    """Fill composite-placement gaps from a strongly related species graph."""
    primary_keys = {
        name.casefold()
        for name in (
            material_names
            if primary_material_names is None
            else primary_material_names
        )
    }
    unresolved = {
        name.casefold(): name
        for name in material_names
        if (
            not bindings.get(name, {}).get("shader")
            or name.casefold() not in primary_keys
        )
    }
    if not unresolved:
        return bindings

    for _name_score, _affinity, candidate in _ranked_material_graph_candidates(entry):
        if candidate.asset.name.casefold() == primary_entry.asset.name.casefold():
            continue
        try:
            data = candidate.archive.extract(candidate.asset)
            if data[:4] != mgraph.MAGIC or not mgraph.texture_pool(data):
                continue
            source_materials = {
                name.casefold(): name for name, _shader in mgraph._graph_materials(data)
            }
            exact = set(unresolved) & set(source_materials)

            graph_stem = os.path.splitext(os.path.basename(candidate.asset.name))[0].casefold()
            species_aliases = {
                key for key in unresolved
                if key in _source_name_tokens(graph_stem) and "body" in source_materials
            }
            if not exact and not species_aliases:
                continue

            compound_sources = _referenced_compound_data(data, candidate.archive)
            shader_role_pins, shader_parameter_pins = _shader_pins_for_sources(
                data, compound_sources, candidate.archive
            )
            supplemental = mgraph.material_bindings(
                data,
                material_names,
                compound_sources=compound_sources,
                shader_role_pins=shader_role_pins,
                shader_parameter_pins=shader_parameter_pins,
            )
            for key in exact:
                name = unresolved[key]
                if supplemental.get(name, {}).get("shader"):
                    bindings[name] = supplemental[name]
                    unresolved.pop(key, None)
            body_name = source_materials.get("body")
            if body_name is not None:
                body_binding = mgraph.material_bindings(
                    data,
                    [body_name],
                    compound_sources=compound_sources,
                    shader_role_pins=shader_role_pins,
                    shader_parameter_pins=shader_parameter_pins,
                ).get(body_name)
                if body_binding and body_binding.get("shader"):
                    for key in species_aliases:
                        name = unresolved.get(key)
                        if name is not None:
                            bindings[name] = dict(body_binding)
                            unresolved.pop(key, None)
            if not unresolved:
                break
        except Exception as error:
            logger.debug(
                "Could not inspect supplemental material graph %s: %s",
                candidate.asset.name,
                error,
            )
    return bindings


def _material_parent_graph_for_compound(entry, compound_entry):
    """Return the best MMB-related graph that instantiates a compound."""
    compound_key = compound_entry.asset.name.replace("\\", "/").lstrip("/").casefold()
    cross_archive = None
    for _name_score, _affinity, candidate in _ranked_material_graph_candidates(entry):
        try:
            data = candidate.archive.extract(candidate.asset)
        except Exception as error:
            logger.debug(
                "Could not inspect parent material graph %s: %s",
                candidate.asset.name,
                error,
            )
            continue
        if data[:4] != mgraph.MAGIC:
            continue
        references = {
            path.replace("\\", "/").lstrip("/").casefold()
            for path in mgraph.referenced_compounds(data)
        }
        if compound_key in references:
            if candidate.archive is compound_entry.archive:
                return candidate, data
            if cross_archive is None:
                cross_archive = (candidate, data)
    return cross_archive or (None, None)


def _extract_texture_to_cache(entry, extracted_directory):
    destination = os.path.join(
        _cache_root(entry.cache_key, extracted_directory),
        *_safe_asset_parts(entry.asset.name),
    )
    if os.path.isfile(destination):
        try:
            with open(destination, "rb") as stream:
                if stream.read(4) == b"DDS ":
                    return destination
        except OSError:
            pass
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temporary = destination + ".extract_tmp"
    try:
        converted = material_import.texture_to_dds(
            entry.archive.extract(entry.asset), entry.asset
        )
        with open(temporary, "wb") as stream:
            stream.write(converted)
        os.replace(temporary, destination)
    finally:
        try:
            if os.path.exists(temporary):
                os.remove(temporary)
        except OSError:
            pass
    return destination


def _import_materials_for_entry(
    entry, skeletal_mesh, extracted_directory, material_source_path=None
):
    material_names = [
        mesh.name
        for mesh in skeletal_mesh.meshes
        if not mesh.name.casefold().endswith("_cloth_sim")
    ]
    selected_key = (material_source_path or _MATERIAL_SOURCE_AUTO).casefold()
    if selected_key.endswith(".mcompoundnode"):
        source_entry, source_data = _material_compound(
            entry, selected_path=material_source_path
        )
    else:
        source_entry, source_data = _material_graph(
            entry,
            selected_path=(
                material_source_path if selected_key.endswith(".mgraphobject") else None
            ),
        )
    if source_entry is None and selected_key == _MATERIAL_SOURCE_AUTO.casefold():
        source_entry, source_data = _material_compound(entry)
    elif source_entry is None:
        raise LookupError(
            f"The selected material source no longer references this MMB: "
            f"{material_source_path}"
        )
    if source_entry is None:
        raise LookupError(
            "No matching mgraphobject or mcompoundnode was found in the loaded SDF archives"
        )
    source_is_graph = source_entry.asset.name.casefold().endswith(".mgraphobject")
    binding_data = source_data
    binding_archive = source_entry.archive
    if not source_is_graph:
        parent_entry, parent_data = _material_parent_graph_for_compound(
            entry, source_entry
        )
        if parent_entry is not None:
            # Preserve the selected compound as the user-facing source, while
            # using its best parent graph to resolve exposed material inputs.
            binding_data = parent_data
            binding_archive = parent_entry.archive
    compound_sources = (
        _referenced_compound_data(binding_data, binding_archive)
        if binding_data is not source_data or source_is_graph
        else {}
    )
    shader_role_pins, shader_parameter_pins = _shader_pins_for_sources(
        binding_data, compound_sources, binding_archive
    )
    bindings = mgraph.material_bindings(
        binding_data, material_names, compound_sources=compound_sources,
        shader_role_pins=shader_role_pins,
        shader_parameter_pins=shader_parameter_pins,
    )
    if source_is_graph:
        primary_material_names = {
            name
            for source_data in (binding_data, *compound_sources.values())
            for name, _shader in mgraph._graph_materials(source_data)
        }
        bindings = _supplemental_material_bindings(
            entry,
            source_entry,
            material_names,
            bindings,
            primary_material_names=primary_material_names,
        )
    if not bindings:
        raise LookupError(f"{source_entry.asset.name} contains no usable material textures")

    texture_files = {}
    missing = []
    failed = []
    logical_paths = {
        binding.get(role)
        for binding in bindings.values()
        for role in ("d", "n", "m", "a")
        if binding.get(role)
    }
    logical_paths.update(
        logical_path
        for binding in bindings.values()
        for logical_path in material_import.supported_auxiliary_paths(binding)
    )
    for logical_path in sorted(logical_paths, key=str.casefold):
        texture = _texture_entry(logical_path, binding_archive)
        if texture is None:
            missing.append(logical_path)
            continue
        try:
            texture_files[logical_path.casefold()] = _extract_texture_to_cache(
                texture, extracted_directory
            )
        except Exception as error:
            logger.warning("Could not import texture %s: %s", logical_path, error)
            failed.append(logical_path)

    if logical_paths and not texture_files:
        raise LookupError(
            f"No referenced textures from {source_entry.asset.name} could be extracted"
        )
    assigned = material_import.assign_materials(
        skeletal_mesh,
        bindings,
        texture_files,
        source_entry.asset.name,
        lod_index=0,
    )
    return assigned, len(texture_files), missing, failed, source_entry.asset.name


def _referenced_mmb_entries(source_entry):
    """Resolve a source chain's ordered MMB references against the SDF index."""
    data = source_entry.archive.extract(source_entry.asset)
    if data[:4] != mgraph.MAGIC:
        raise ValueError(f"{source_entry.asset.name} is not a BV2 material source")
    references, _compound_sources = _material_source_references(
        data, source_entry.archive
    )
    with _state.lock:
        indexed_mmbs = list(_state.entries)
    by_path = {}
    for candidate in indexed_mmbs:
        by_path.setdefault(candidate.asset.name.casefold(), []).append(candidate)

    resolved = []
    missing = []
    seen = set()
    for logical_path in references:
        key = logical_path.casefold()
        if key in seen:
            continue
        seen.add(key)
        candidates = _prefer_archive(
            by_path.get(key, ()), source_entry.archive
        )
        if candidates:
            resolved.append(candidates[0])
        else:
            missing.append(logical_path)
    return resolved, missing


def _selected_mmb_entries(targets, choices):
    """Filter resolved MMBs using choices captured by the import dialog."""
    if len(choices) == 0:
        return targets
    selected = {
        (choice.cache_key, choice.asset_path.casefold())
        for choice in choices
        if choice.selected
    }
    return [
        target for target in targets
        if (target.cache_key, target.asset.name.casefold()) in selected
    ]


def _process_mmb_entry(
    operator,
    context,
    entry,
    *,
    import_lod0,
    load_as_asset,
    import_materials,
    extracted_directory,
    material_source_path=None,
):
    """Extract, parse, and optionally import one indexed MMB."""
    settings = context.scene.SWOMT
    temporary_directory = None
    try:
        if load_as_asset:
            mmb_path, extracted = _extract_to_cache(
                entry, extracted_directory=extracted_directory
            )
        else:
            temporary_directory = tempfile.TemporaryDirectory(
                prefix="afop_sdf_import_"
            )
            temporary_path = os.path.join(
                temporary_directory.name,
                *_safe_asset_parts(entry.asset.name),
            )
            mmb_path, extracted = _extract_to_cache(
                entry, destination=temporary_path
            )

        sidecar = _matching_sidecar(entry) if load_as_asset else None
        if sidecar is not None:
            mcloth_path = os.path.splitext(mmb_path)[0] + ".mcloth"
            try:
                _extract_to_cache(sidecar, destination=mcloth_path)
            except Exception as error:
                logger.warning(
                    "Could not extract paired mcloth for %s: %s",
                    entry.asset.name,
                    error,
                )
                operator.report(
                    {"WARNING"},
                    f"MMB loaded, but paired mcloth extraction failed: {error}",
                )

        try:
            with open(mmb_path, "rb") as stream:
                probe = SkeletalMeshAsset()
                probe.parse(stream)
            probe.name = os.path.splitext(os.path.basename(mmb_path))[0]
        except Exception as error:
            raise ValueError(
                f"The extracted MMB could not be parsed: {entry.asset.name}: {error}"
            ) from error

        if not import_lod0:
            settings.AssetPath = mmb_path
            if addon_state.asset is None:
                raise ValueError(f"The extracted MMB could not be loaded: {entry.asset.name}")
            return {"FINISHED"}, extracted, None

        from .io import _import_all_lods

        if load_as_asset:
            settings.AssetPath = mmb_path
            if addon_state.asset is None:
                raise ValueError(f"The extracted MMB could not be loaded: {entry.asset.name}")
            imported_asset = addon_state.asset
            result = _import_all_lods(context, 0)
        else:
            imported_asset = probe
            result = _import_all_lods(
                context,
                0,
                skeletal_mesh=probe,
                asset_path=mmb_path,
            )

        material_summary = None
        if import_materials and "FINISHED" in result:
            try:
                material_summary = _import_materials_for_entry(
                    entry,
                    imported_asset,
                    extracted_directory,
                    material_source_path=material_source_path,
                )
            except Exception as error:
                logger.warning(
                    "MMB imported, but materials could not be imported for %s: %s",
                    entry.asset.name,
                    error,
                )
                operator.report(
                    {"WARNING"}, f"Mesh imported without materials: {error}"
                )
        return result, extracted, material_summary
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()


class SDFMMBImportChoice(bpy.types.PropertyGroup):
    """One resolved MMB shown in the multi-import selection dialog."""

    asset_path: bpy.props.StringProperty(options={"SKIP_SAVE"})
    archive_label: bpy.props.StringProperty(options={"SKIP_SAVE"})
    cache_key: bpy.props.StringProperty(options={"HIDDEN", "SKIP_SAVE"})
    selected: bpy.props.BoolProperty(default=True, options={"SKIP_SAVE"})


class SDFMMBChoiceList(bpy.types.UIList):
    """Scrollable checkbox list for graph/compound MMB references."""

    bl_idname = "SWOMT_UL_sdf_mmb_choices"

    def draw_item(
        self, context, layout, data, item, icon, active_data, active_propname, index
    ):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            row.prop(item, "selected", text="")
            row.label(text=item.asset_path, icon="MESH_DATA")
            archive_column = row.column(align=True)
            archive_column.alignment = "RIGHT"
            archive_column.label(text=item.archive_label)
        else:
            layout.label(text="", icon="MESH_DATA")


class SDFAssetList(bpy.types.UIList):
    """Searchable list of mesh and material-source paths in SDF archives."""

    bl_idname = "SWOMT_UL_sdf_assets"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            icons = {
                _ASSET_MMB: "MESH_DATA",
                _ASSET_MGRAPH: "MATERIAL",
                _ASSET_MCOMPOUND: "NODETREE",
            }
            row.label(text=item.asset_path, icon=icons.get(item.asset_type, "FILE"))
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
    bl_description = (
        "Delete cached MMB, mcloth, material-graph, compound-node, shader, and texture indexes; "
        "extracted asset files are preserved"
    )

    def execute(self, context):
        from ..settings import blender_sdf_index_cache_directory

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
    """Decrypt and index supported assets in the selected AFOP SDF archives."""

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
    """Import a selected MMB or the MMBs referenced by a material source."""

    bl_idname = "object.import_sdf_mmb"
    bl_label = "Import Selected SDF Asset"

    import_lod0: bpy.props.BoolProperty(default=True, options={"HIDDEN"})
    material_source: bpy.props.EnumProperty(
        name="Material Source",
        description=(
            "Choose which matching mgraphobject or mcompoundnode "
            "supplies the materials"
        ),
        items=_material_source_enum_items,
        default=0,
        options={"SKIP_SAVE"},
    )
    mmb_choices: bpy.props.CollectionProperty(
        type=SDFMMBImportChoice,
        options={"SKIP_SAVE"},
    )
    mmb_choice_index: bpy.props.IntProperty(
        default=0,
        options={"SKIP_SAVE"},
    )

    @classmethod
    def description(cls, context, properties):
        if not properties.import_lod0:
            return "Extract the selected MMB and load it in Asset Path"
        settings = getattr(context.scene, "SWOMT", None)
        if (
            settings is not None
            and 0 <= settings.sdf_asset_index < len(settings.sdf_assets)
            and settings.sdf_assets[settings.sdf_asset_index].asset_type != _ASSET_MMB
        ):
            return "Choose and import LOD0 from MMBs referenced by this source"
        return "Import LOD0 from the selected MMB"

    @classmethod
    def poll(cls, context):
        settings = getattr(context.scene, "SWOMT", None)
        return (
            settings is not None
            and 0 <= settings.sdf_asset_index < len(settings.sdf_assets)
        )

    def invoke(self, context, _event):
        global _material_source_dialog_items
        settings = context.scene.SWOMT
        if not self.import_lod0:
            return self.execute(context)
        item = settings.sdf_assets[settings.sdf_asset_index]
        asset_type, entry = _search_entry_for_id(item.entry_id)
        if entry is None:
            return self.execute(context)
        if asset_type != _ASSET_MMB:
            try:
                targets, _missing = _referenced_mmb_entries(entry)
            except Exception as error:
                logger.debug(
                    "Could not enumerate referenced MMBs for %s: %s",
                    item.asset_path,
                    error,
                )
                return self.execute(context)
            if len(targets) <= 1:
                return self.execute(context)
            self.mmb_choices.clear()
            for target in targets:
                choice = self.mmb_choices.add()
                choice.name = target.asset.name
                choice.asset_path = target.asset.name
                choice.archive_label = target.archive_label
                choice.cache_key = target.cache_key
                choice.selected = True
            self.mmb_choice_index = 0
            return context.window_manager.invoke_props_dialog(self, width=1000)
        if not settings.sdf_import_materials:
            return self.execute(context)
        try:
            options = _material_source_options(entry)
        except Exception as error:
            logger.debug("Could not enumerate material sources for %s: %s", item.asset_path, error)
            return self.execute(context)
        if len(options) <= 1:
            return self.execute(context)

        _material_source_dialog_items = [
            (
                _MATERIAL_SOURCE_AUTO,
                "Automatic (Recommended)",
                "Use the highest-ranked graph, then the compound fallback",
                0,
            )
        ]
        for index, (kind, candidate, _data, richness, _rank, _affinity) in enumerate(
            options, start=1
        ):
            _material_source_dialog_items.append((
                candidate.asset.name,
                f"[{kind}] {candidate.asset.name}",
                f"{richness} texture constant(s), archive: {candidate.archive_label}",
                index,
            ))
        self.material_source = _MATERIAL_SOURCE_AUTO
        return context.window_manager.invoke_props_dialog(self, width=900)

    def draw(self, _context):
        layout = self.layout
        if len(self.mmb_choices):
            selected_count = sum(choice.selected for choice in self.mmb_choices)
            layout.label(
                text="Multiple MMB files are referenced by this source.",
                icon="MESH_DATA",
            )
            layout.label(text="Select the MMB files to import:")
            layout.template_list(
                "SWOMT_UL_sdf_mmb_choices",
                "",
                self,
                "mmb_choices",
                self,
                "mmb_choice_index",
                rows=min(12, max(4, len(self.mmb_choices))),
            )
            layout.label(
                text=f"{selected_count} of {len(self.mmb_choices)} selected",
                icon="CHECKMARK" if selected_count else "ERROR",
            )
            return
        layout.label(text="Multiple material sources match this MMB.", icon="MATERIAL")
        layout.label(text="Choose the texture/material variant to import:")
        layout.prop(self, "material_source", text="")

    def execute(self, context):
        settings = context.scene.SWOMT
        item = settings.sdf_assets[settings.sdf_asset_index]
        asset_type, selected_entry = _search_entry_for_id(item.entry_id)
        if selected_entry is None:
            self.report(
                {"ERROR"},
                "The SDF index changed; reload the archives and try again.",
            )
            return {"CANCELLED"}
        if asset_type != _ASSET_MMB and not self.import_lod0:
            self.report(
                {"ERROR"},
                "Load Selected applies only to MMB files; use Import Referenced MMBs.",
            )
            return {"CANCELLED"}

        source_entry = None
        missing_references = []
        if asset_type == _ASSET_MMB:
            targets = [selected_entry]
            material_source_path = self.material_source
        else:
            source_entry = selected_entry
            try:
                targets, missing_references = _referenced_mmb_entries(source_entry)
            except Exception as error:
                logger.exception(
                    "Could not read MMB references from %s: %s",
                    source_entry.asset.name,
                    error,
                )
                self.report({"ERROR"}, f"Could not read referenced MMBs: {error}")
                return {"CANCELLED"}
            if not targets:
                detail = (
                    f"; {len(missing_references)} reference(s) were not indexed"
                    if missing_references else ""
                )
                self.report(
                    {"ERROR"},
                    f"{source_entry.asset.name} has no importable MMB references{detail}.",
                )
                return {"CANCELLED"}
            targets = _selected_mmb_entries(targets, self.mmb_choices)
            if len(self.mmb_choices) and not targets:
                self.report({"ERROR"}, "Select at least one MMB file to import.")
                return {"CANCELLED"}
            material_source_path = source_entry.asset.name

        load_as_asset = not self.import_lod0 or settings.sdf_load_as_asset
        successes = []
        failures = []
        material_summaries = []
        for target in targets:
            try:
                result, extracted, material_summary = _process_mmb_entry(
                    self,
                    context,
                    target,
                    import_lod0=self.import_lod0,
                    load_as_asset=load_as_asset,
                    import_materials=settings.sdf_import_materials,
                    extracted_directory=settings.sdf_extracted_directory,
                    material_source_path=material_source_path,
                )
                if "FINISHED" not in result:
                    failures.append(target.asset.name)
                    continue
                successes.append((target, extracted))
                if material_summary is not None:
                    material_summaries.append(material_summary)
            except Exception as error:
                logger.exception(
                    "Could not import indexed MMB %s: %s", target.asset.name, error
                )
                failures.append(target.asset.name)
                if asset_type == _ASSET_MMB:
                    self.report({"ERROR"}, f"MMB import failed: {error}")
                    return {"CANCELLED"}
                self.report(
                    {"WARNING"}, f"Could not import {target.asset.name}: {error}"
                )

        if not successes:
            self.report({"ERROR"}, "No referenced MMB files could be imported.")
            return {"CANCELLED"}

        if asset_type == _ASSET_MMB:
            target, extracted = successes[0]
            if not self.import_lod0:
                action = "Extracted and loaded" if extracted else "Loaded cached"
                self.report({"INFO"}, f"{action} {target.asset.name}")
                return {"FINISHED"}
            action = "Imported and loaded" if load_as_asset else "Imported"
            if not material_summaries:
                self.report({"INFO"}, f"{action} LOD0 from {target.asset.name}")
            else:
                assigned, texture_count, missing, failed, source_name = (
                    material_summaries[0]
                )
                self.report(
                    {"INFO"},
                    f"{action} LOD0 with {assigned} material(s) and "
                    f"{texture_count} texture(s) from {source_name}",
                )
                if missing or failed:
                    self.report(
                        {"WARNING"},
                        f"{len(missing)} referenced texture(s) were not indexed and "
                        f"{len(failed)} could not be converted.",
                    )
            return {"FINISHED"}

        assigned_total = sum(summary[0] for summary in material_summaries)
        texture_total = sum(summary[1] for summary in material_summaries)
        message = (
            f"Imported {len(successes)} referenced MMB file"
            f"{'s' if len(successes) != 1 else ''} from {source_entry.asset.name}"
        )
        if material_summaries:
            message += (
                f" with {assigned_total} material assignment"
                f"{'s' if assigned_total != 1 else ''} and {texture_total} texture"
                f"{'s' if texture_total != 1 else ''}"
            )
        self.report({"INFO"}, message)
        if missing_references or failures:
            self.report(
                {"WARNING"},
                f"{len(missing_references)} referenced MMB path(s) were not indexed and "
                f"{len(failures)} indexed MMB(s) could not be imported.",
            )
        return {"FINISHED"}


CLASSES = (
    SDFMMBImportChoice,
    SDFMMBChoiceList,
    SDFAssetList,
    BrowseSDFDirectory,
    BrowseSDFExtractedDirectory,
    ClearSDFIndexCache,
    IndexSDFArchives,
    ImportSDFMMB,
)

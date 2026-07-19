"""Standalone AFoP SDF material-corpus audit.

This command uses the add-on's pure-Python SDF readers but does not import or
launch Blender. Run ``--help`` for paths and cache options.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys
from pathlib import Path


TOOLS_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY = TOOLS_DIRECTORY.parent
for path in (str(REPOSITORY), str(TOOLS_DIRECTORY)):
    if path not in sys.path:
        sys.path.insert(0, path)

import material_audit
import oodle_helper
import sdf_toc
from sdf_reader import SdfArchive


INDEX_CACHE_VERSION = 4
TARGET_SUFFIXES = (
    ".mmb", ".mcloth", ".mgraphobject", ".mcompoundnode", ".mshader", ".dds",
)
AUDIT_SOURCE_SUFFIXES = (".mgraphobject", ".mcompoundnode")
OODLE_NAMES = (
    oodle_helper.OODLE_DLL_NAME,
    "oo2core_9_win64.dll",
    "oo2core_8_win64.dll",
    "oo2core_7_win64.dll",
    "oo2core_win64.dll",
)


class PrimitiveUnpickler(pickle.Unpickler):
    """Reject global/class lookups in index caches."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(
            f"SDF index cache requested forbidden global {module}.{name}"
        )


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--game-directory", required=True, type=Path,
        help="AFoP folder containing one or more .sdftoc files",
    )
    parser.add_argument(
        "--output-directory", required=True, type=Path,
        help="Directory that will receive the JSON and CSV reports",
    )
    parser.add_argument(
        "--cache-directory", type=Path,
        help="Optional add-on-compatible targeted SDF index-cache directory",
    )
    parser.add_argument(
        "--oodle", type=Path,
        help="Explicit supported Oodle Windows DLL path",
    )
    parser.add_argument(
        "--download-oodle", action="store_true",
        help="Download the validated Oodle runtime into the repository if absent",
    )
    parser.add_argument(
        "--no-rebuild", action="store_true",
        help="Require a valid cache entry for every .sdftoc instead of rebuilding",
    )
    return parser.parse_args()


def _discover_archives(root):
    return sorted(
        (
            path for path in root.rglob("*")
            if path.is_file() and path.name.casefold().endswith(".sdftoc")
        ),
        key=lambda path: str(path).casefold(),
    )


def _archive_label(root, toc_path):
    relative = toc_path.relative_to(root).as_posix()
    return root.name if "/" not in relative else relative.split("/", 1)[0]


def _archive_cache_key(toc_path):
    stat = toc_path.stat()
    identity = f"{toc_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha1(identity.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def _index_cache_path(cache_directory, toc_path):
    identity = os.path.normcase(str(toc_path.resolve()))
    digest = hashlib.sha1(identity.encode("utf-8", "surrogatepass")).hexdigest()
    return cache_directory / f"sdf_mesh_index_{digest}.pickle"


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
    name, asset_hash, dds_index, unknown, slice_records = record
    asset = sdf_toc.Asset(
        name=name, hash=asset_hash, dds_index=dds_index, unk=unknown
    )
    for values in slice_records:
        (
            decompressed_size, compressed_size, is_compressed, is_oodle,
            is_encrypted, offset, index, page_sizes,
        ) = values
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


def _read_cache(cache_directory, toc_path):
    if cache_directory is None:
        return None
    cache_path = _index_cache_path(cache_directory, toc_path)
    stat = toc_path.stat()
    try:
        with cache_path.open("rb") as stream:
            payload = PrimitiveUnpickler(stream).load()
        if not isinstance(payload, dict):
            return None
        if payload.get("version") != INDEX_CACHE_VERSION:
            return None
        if payload.get("toc_path") != os.path.normcase(str(toc_path.resolve())):
            return None
        if (
            payload.get("toc_size") != stat.st_size
            or payload.get("toc_mtime_ns") != stat.st_mtime_ns
        ):
            return None
        records = payload.get("assets")
        dds_block = payload.get("dds_block", b"")
        if not isinstance(records, (tuple, list)) or not isinstance(dds_block, bytes):
            return None
        return [_unpack_asset(record) for record in records], dds_block
    except (
        OSError, EOFError, pickle.PickleError, TypeError, ValueError, AttributeError,
    ):
        return None


def _write_cache(cache_directory, toc_path, assets, dds_block):
    if cache_directory is None:
        return
    cache_directory.mkdir(parents=True, exist_ok=True)
    cache_path = _index_cache_path(cache_directory, toc_path)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    stat = toc_path.stat()
    payload = {
        "version": INDEX_CACHE_VERSION,
        "toc_path": os.path.normcase(str(toc_path.resolve())),
        "toc_size": stat.st_size,
        "toc_mtime_ns": stat.st_mtime_ns,
        "assets": tuple(_pack_asset(asset) for asset in assets),
        "dds_block": bytes(dds_block),
    }
    try:
        with temporary.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, cache_path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _find_oodle(arguments, game_directory):
    if arguments.oodle is not None:
        candidate = arguments.oodle.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Oodle DLL does not exist: {candidate}")
        return candidate
    for directory in (REPOSITORY, TOOLS_DIRECTORY):
        for filename in OODLE_NAMES:
            candidate = directory / filename
            if candidate.is_file():
                return candidate.resolve()
    for filename in OODLE_NAMES:
        matches = list(game_directory.rglob(filename))
        if matches:
            return matches[0].resolve()
    if arguments.download_oodle:
        return Path(oodle_helper.ensure_oodle_dll(str(REPOSITORY))).resolve()
    raise FileNotFoundError(
        "No supported Oodle DLL was found. Pass --oodle PATH or "
        "--download-oodle."
    )


def _load_targeted_assets(archive, toc_path, cache_directory, no_rebuild):
    cached = _read_cache(cache_directory, toc_path)
    if cached is not None:
        assets, dds_block = cached
        archive.dds_block = dds_block
        return assets, True
    if no_rebuild:
        raise FileNotFoundError(f"No current SDF index cache exists for {toc_path}")
    archive.load()
    assets = [
        asset for asset in archive.assets
        if asset.name.casefold().endswith(TARGET_SUFFIXES)
    ]
    _write_cache(cache_directory, toc_path, assets, archive.dds_block)
    archive.assets = []
    return assets, False


def run(arguments):
    game_directory = arguments.game_directory.resolve()
    if game_directory.is_file():
        game_directory = game_directory.parent
    if not game_directory.is_dir():
        raise FileNotFoundError(f"Game directory does not exist: {game_directory}")
    output_directory = arguments.output_directory.resolve()
    cache_directory = (
        arguments.cache_directory.resolve()
        if arguments.cache_directory is not None else None
    )
    toc_paths = _discover_archives(game_directory)
    if not toc_paths:
        raise FileNotFoundError(f"No .sdftoc files were found under {game_directory}")
    oodle_path = _find_oodle(arguments, game_directory)
    print(f"Using Oodle: {oodle_path}")
    print(f"Found {len(toc_paths)} SDF archive(s)")

    source_records = []
    shader_records = []
    mmb_paths = []
    issues = []
    input_signature = set()
    loaded_archives = 0
    for index, toc_path in enumerate(toc_paths, 1):
        label = _archive_label(game_directory, toc_path)
        print(f"[{index}/{len(toc_paths)}] Indexing {label}: {toc_path}")
        try:
            archive = SdfArchive(str(toc_path), str(oodle_path))
            assets, cached = _load_targeted_assets(
                archive, toc_path, cache_directory, arguments.no_rebuild
            )
            cache_key = _archive_cache_key(toc_path)
            input_signature.add(cache_key)
            print(
                f"[{index}/{len(toc_paths)}] "
                f"{'Loaded cache' if cached else 'Rebuilt index'}; "
                f"reading {len(assets):,} targeted entries"
            )
            for asset in assets:
                lower = asset.name.casefold()
                if lower.endswith(".mmb"):
                    mmb_paths.append(asset.name)
                    continue
                if not lower.endswith((*AUDIT_SOURCE_SUFFIXES, ".mshader")):
                    continue
                kind = (
                    "mgraphobject" if lower.endswith(".mgraphobject")
                    else "mcompoundnode" if lower.endswith(".mcompoundnode")
                    else "mshader"
                )
                try:
                    record = {
                        "path": asset.name,
                        "kind": kind,
                        "archive": label,
                        "cache_key": cache_key,
                        "data": archive.extract(asset),
                    }
                    if kind == "mshader":
                        shader_records.append(record)
                    else:
                        source_records.append(record)
                except Exception as error:
                    issues.append({
                        "severity": "warning",
                        "kind": "asset_extraction_error",
                        "source": asset.name,
                        "detail": str(error),
                    })
                    print(f"WARNING: could not extract {asset.name}: {error}")
            loaded_archives += 1
        except Exception as error:
            issues.append({
                "severity": "warning",
                "kind": "archive_index_error",
                "source": str(toc_path),
                "detail": str(error),
            })
            print(f"WARNING: could not audit {toc_path}: {error}")

    if not loaded_archives:
        raise RuntimeError("No SDF archive could be indexed")
    source_records.sort(key=material_audit._record_key)
    shader_records.sort(key=material_audit._record_key)
    mmb_paths.sort(key=str.casefold)
    print(
        f"Building report from {len(source_records):,} material sources, "
        f"{len(shader_records):,} shaders, and {len(mmb_paths):,} MMB paths"
    )
    report = material_audit.build_report(
        source_records,
        shader_records,
        mmb_paths,
        input_signature=input_signature,
    )
    if issues:
        report["issues"][0:0] = issues
        report["summary"]["issues"] += len(issues)
        for issue in issues:
            severity = issue["severity"]
            counts = report["summary"]["issues_by_severity"]
            counts[severity] = counts.get(severity, 0) + 1
    material_audit.write_reports(report, str(output_directory))
    summary = report["summary"]
    print(
        f"Complete: {summary['sources']:,} sources, {summary['shaders']:,} "
        f"shaders, {summary['materials']:,} bindings, and "
        f"{summary['issues']:,} review item(s)"
    )
    print(f"Reports: {output_directory}")


def main():
    run(_arguments())


if __name__ == "__main__":
    main()

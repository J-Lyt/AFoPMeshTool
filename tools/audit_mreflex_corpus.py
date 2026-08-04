"""Audit MReflex/MMB node matching with SDF material-source associations.

Run this script through Blender 5.0 or newer because MMB parsing depends on
``mathutils``. The installed game archives are read-only; reports and optional
targeted index caches are written only to the requested locations.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys


TOOLS_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY = TOOLS_DIRECTORY.parent
for path in (REPOSITORY, TOOLS_DIRECTORY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_material_corpus as sdf_audit
from sdf.reader import SdfArchive


REPORT_JSON = "afop_mreflex_audit.json"
REPORT_MARKDOWN = "afop_mreflex_audit.md"
SOURCE_SUFFIXES = (".mgraphobject", ".mcompoundnode")
RULE_RANK = {
    "longest_same_directory_prefix": 3,
    "unique_same_directory": 2,
    "unique_source_mmb": 1,
}


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--game-directory", required=True, type=Path,
        help="AFoP folder containing one or more .sdftoc files",
    )
    parser.add_argument(
        "--corpus-directory", required=True, type=Path,
        help="Directory whose relative paths mirror the SDF .mreflex paths",
    )
    parser.add_argument(
        "--output-directory", required=True, type=Path,
        help="Directory that will receive JSON and Markdown reports",
    )
    parser.add_argument(
        "--cache-directory", type=Path,
        help="Optional targeted SDF index-cache directory",
    )
    parser.add_argument("--oodle", type=Path, help="Explicit Oodle DLL path")
    parser.add_argument(
        "--download-oodle", action="store_true",
        help="Download the validated Oodle runtime if absent",
    )
    parser.add_argument(
        "--no-rebuild", action="store_true",
        help="Require a current targeted cache for every archive",
    )
    command_line = sys.argv
    if "--" in command_line:
        command_line = command_line[command_line.index("--") + 1:]
    else:
        command_line = command_line[1:]
    return parser.parse_args(command_line)


def _load_addon():
    name = "afop_mreflex_corpus_audit"
    spec = importlib.util.spec_from_file_location(
        name,
        REPOSITORY / "__init__.py",
        submodule_search_locations=[str(REPOSITORY)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the AFoP Mesh Tool package")
    addon = importlib.util.module_from_spec(spec)
    sys.modules[name] = addon
    spec.loader.exec_module(addon)
    return addon


def _normalise(value):
    return str(value or "").replace("\\", "/").lstrip("/")


def _choose_source(records, preferred_archive):
    same_archive = [
        record for record in records
        if record["archive"].casefold() == preferred_archive.casefold()
    ]
    return min(
        same_archive or records,
        key=lambda record: (
            record["path"].casefold(), record["archive"].casefold()
        ),
    )


def _preferred_asset(candidates, reflex_path):
    root = reflex_path.split("/", 1)[0].casefold()
    preferred_archive = "rogue" if root == "blue" else root
    return min(candidates, key=lambda item: (
        item[0].casefold() != preferred_archive,
        item[0].casefold(),
        item[2].name.casefold(),
    ))


def _weighted_bone_indices(asset, data):
    weighted = set()
    warnings = []
    for mesh in asset.meshes:
        if not mesh.lods or mesh.lods[0].vertex_count == 0:
            continue
        slots = list(mesh.mesh_bones.keys())
        if not slots:
            continue
        try:
            raw_mesh = mesh.extract_mesh_file(io.BytesIO(data))
            for vertex_weights in mesh.lods[0].get_bone_weights(raw_mesh):
                for slot, weight in vertex_weights.items():
                    if weight > 0.0 and 0 <= slot < len(slots):
                        weighted.add(slots[slot])
        except Exception as error:
            warnings.append(f"{mesh.name}: {error}")
    return weighted, warnings


def _atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _markdown(report):
    summary = report["summary"]
    lines = [
        "# MReflex corpus node-matching audit",
        "",
        "Generated from the local MReflex corpus and the installed Rogue, "
        "DLC1, DLC2, and DLC3 SDF archives.",
        "",
        "## Summary",
        "",
        "| Category | Files |",
        "| --- | ---: |",
        f"| Full MReflex corpus | {summary['corpus']} |",
        f"| Same-path MMB pair | {summary['same_path_pairs']} |",
        f"| Additional graph-derived MMB pair | {summary['graph_pairs']} |",
        f"| All confirmed pairs | {summary['paired']} |",
        f"| All kind-10 nodes uniquely matched | {summary['categories'].get('complete', 0)} |",
        f"| Some kind-10 nodes unmatched | {summary['categories'].get('partial', 0)} |",
        f"| No kind-10 nodes matched | {summary['categories'].get('zero', 0)} |",
        f"| Valid file with no kind-10 nodes | {summary['categories'].get('node_free', 0)} |",
        f"| No confirmed MMB association | {summary['unassociated']} |",
        "",
        (
            f"Across {summary['node_bearing_pairs']} node-bearing pairs: "
            f"**{summary['matched_nodes']:,} of {summary['nodes']:,} nodes "
            f"matched**, leaving **{summary['unmatched_nodes']:,} unmatched**."
        ),
        "",
        "## Graph-derived association evidence",
        "",
        "| Rule | Pairs | Matched nodes | Total nodes |",
        "| --- | ---: | ---: | ---: |",
    ]
    labels = {
        "longest_same_directory_prefix": "Longest same-directory MMB-stem prefix",
        "unique_same_directory": "Only same-directory MMB in source chain",
        "unique_source_mmb": "Only MMB in source chain",
    }
    for rule in RULE_RANK:
        values = summary["graph_rules"].get(rule, {})
        lines.append(
            f"| {labels[rule]} | {values.get('pairs', 0)} | "
            f"{values.get('matched', 0)} | {values.get('nodes', 0)} |"
        )

    lines.extend(["", "## Graph-derived pairs", ""])
    for row in report["pairs"]:
        if row["association"] == "same_path":
            continue
        status = (
            "error: " + row["error"]
            if row["category"] == "error"
            else f"{row.get('matched', 0)}/{row.get('nodes', 0)} nodes matched"
        )
        lines.append(
            f"- `{row['mreflex']}` → `{row['mmb']}` "
            f"({row['association']}; {status})"
        )

    lines.extend(["", "## Unresolved equal-strength graph conflicts", ""])
    conflicts = report["graph_conflicts"]
    if conflicts:
        for reflex_path, evidence in conflicts.items():
            owners = sorted({item["mmb"] for item in evidence})
            lines.append(f"- `{reflex_path}` → " + "; ".join(
                f"`{owner}`" for owner in owners
            ))
    else:
        lines.append("None.")

    lines.extend(["", "## Files without a confirmed MMB association", ""])
    for path in report["unassociated"]:
        lines.append(f"- `{path}`")
    lines.append("")
    return "\n".join(lines)


def run(arguments):
    addon = _load_addon()
    mgraph = addon.formats.mgraph
    mmb_format = addon.formats.mmb
    mreflex = addon.formats.mreflex

    game_directory = arguments.game_directory.resolve()
    corpus_directory = arguments.corpus_directory.resolve()
    output_directory = arguments.output_directory.resolve()
    cache_directory = (
        arguments.cache_directory.resolve()
        if arguments.cache_directory is not None else None
    )
    if not game_directory.is_dir():
        raise FileNotFoundError(f"Game directory does not exist: {game_directory}")
    if not corpus_directory.is_dir():
        raise FileNotFoundError(
            f"MReflex corpus directory does not exist: {corpus_directory}"
        )
    toc_paths = sdf_audit._discover_archives(game_directory)
    if not toc_paths:
        raise FileNotFoundError("No .sdftoc files were found")
    oodle_path = sdf_audit._find_oodle(arguments, game_directory)
    corpus_files = {
        path.relative_to(corpus_directory).as_posix().casefold(): path
        for path in corpus_directory.rglob("*.mreflex")
    }

    sources = []
    mmb_assets = defaultdict(list)
    archives = []
    for archive_index, toc_path in enumerate(toc_paths, 1):
        label = sdf_audit._archive_label(game_directory, toc_path)
        archive = SdfArchive(str(toc_path), str(oodle_path))
        assets, cached = sdf_audit._load_targeted_assets(
            archive, toc_path, cache_directory, arguments.no_rebuild
        )
        archives.append(archive)
        print(
            f"[{archive_index}/{len(toc_paths)}] {label}: "
            f"{'cache' if cached else 'rebuilt'}; {len(assets):,} entries",
            flush=True,
        )
        for asset in assets:
            path = _normalise(asset.name)
            lower = path.casefold()
            if lower.endswith(".mmb"):
                mmb_assets[lower].append((label, archive, asset))
                continue
            if not lower.endswith(SOURCE_SUFFIXES):
                continue
            try:
                data = archive.extract(asset)
            except Exception as error:
                print(f"WARNING: could not extract {path}: {error}", flush=True)
                continue
            lower_data = data.lower()
            if not any(
                suffix in lower_data
                for suffix in (b".mmb", b".mreflex", b".mcompoundnode")
            ):
                continue
            sources.append({
                "path": path,
                "archive": label,
                "kind": (
                    "compound" if lower.endswith(".mcompoundnode") else "graph"
                ),
                "data": data,
            })

    by_path = defaultdict(list)
    parsed = {}
    for source in sources:
        by_path[source["path"].casefold()].append(source)
        parsed[id(source)] = {
            "mmbs": [
                _normalise(value) for value in mgraph.referenced_meshes(source["data"])
            ],
            "mreflexes": [
                _normalise(value)
                for value in mgraph.referenced_mreflexes(source["data"])
            ],
            "compounds": [
                _normalise(value)
                for value in mgraph.referenced_compounds(source["data"])
            ],
        }

    def source_closure(source):
        records = [source]
        seen = {source["path"].casefold()}
        pending = list(parsed[id(source)]["compounds"])
        while pending:
            path = pending.pop(0)
            key = path.casefold()
            if key in seen:
                continue
            seen.add(key)
            candidates = by_path.get(key, ())
            if not candidates:
                continue
            record = _choose_source(candidates, source["archive"])
            records.append(record)
            pending.extend(parsed[id(record)]["compounds"])
        return records

    proposals = defaultdict(list)
    for source in sources:
        records = source_closure(source)
        mmbs = list(dict.fromkeys(
            path.casefold()
            for record in records for path in parsed[id(record)]["mmbs"]
        ))
        reflexes = list(dict.fromkeys(
            path.casefold()
            for record in records for path in parsed[id(record)]["mreflexes"]
        ))
        if not mmbs or not reflexes:
            continue
        for reflex_path in reflexes:
            if reflex_path not in corpus_files:
                continue
            owner, rule = mgraph.mreflex_owner(reflex_path, mmbs)
            if owner is None or owner not in mmb_assets:
                continue
            proposals[reflex_path].append({
                "mmb": owner,
                "rule": rule,
                "source": source["path"],
                "source_kind": source["kind"],
                "archive": source["archive"],
            })

    graph_associations = {}
    graph_conflicts = {}
    for reflex_path, evidence in proposals.items():
        strongest = max(RULE_RANK[item["rule"]] for item in evidence)
        winning = [
            item for item in evidence if RULE_RANK[item["rule"]] == strongest
        ]
        owners = {item["mmb"] for item in winning}
        if len(owners) == 1:
            graph_associations[reflex_path] = {
                "mmb": next(iter(owners)),
                "rule": winning[0]["rule"],
                "evidence": winning,
            }
        else:
            graph_conflicts[reflex_path] = winning

    associations = {}
    for reflex_path in corpus_files:
        same_path = PurePosixPath(reflex_path).with_suffix(".mmb").as_posix()
        if same_path in mmb_assets:
            associations[reflex_path] = {
                "mmb": same_path,
                "rule": "same_path",
                "evidence": [],
            }
        elif reflex_path in graph_associations:
            associations[reflex_path] = graph_associations[reflex_path]

    rows = []
    for pair_index, (reflex_path, association) in enumerate(
        sorted(associations.items()), 1
    ):
        mmb_path = association["mmb"]
        row = {
            "mreflex": reflex_path,
            "mmb": mmb_path,
            "association": association["rule"],
            "evidence": association["evidence"],
        }
        try:
            _label, archive, asset_record = _preferred_asset(
                mmb_assets[mmb_path], reflex_path
            )
            mmb_data = archive.extract(asset_record)
            skeletal_mesh = mmb_format.SkeletalMeshAsset()
            skeletal_mesh.parse(io.BytesIO(mmb_data))
            reflex_data = corpus_files[reflex_path].read_bytes()
            nodes = mreflex.read_dangle_nodes(reflex_data)
            anchors = mreflex.read_transform_anchors(reflex_data, nodes)
            weighted, weight_warnings = (
                _weighted_bone_indices(skeletal_mesh, mmb_data)
                if nodes else (set(), [])
            )
            bones = []
            for bone_index, bone in enumerate(skeletal_mesh.bones):
                matrix = tuple(
                    float(bone.matrix[row_index][column_index])
                    for row_index in range(4) for column_index in range(4)
                )
                bones.append(mreflex.BoneCandidate(
                    index=bone_index,
                    name=bone.name,
                    parent_index=bone.parent_index,
                    reflex_matrix=mreflex.reflect_x_basis(matrix),
                ))
            matches = (
                mreflex.match_dangle_nodes(
                    nodes, bones, weighted, anchors=anchors
                ) if nodes else {}
            )
            matched = len(matches)
            row.update({
                "nodes": len(nodes),
                "matched": matched,
                "weighted_bones": len(weighted),
                "weight_warnings": weight_warnings,
                "hierarchy_only": sum(
                    match.hierarchy_only for match in matches.values()
                ),
                "family_verified": sum(
                    match.family_verified for match in matches.values()
                ),
                "category": (
                    "node_free" if not nodes
                    else "complete" if matched == len(nodes)
                    else "partial" if matched
                    else "zero"
                ),
                "matches": {
                    str(record_index): match.bone_name
                    for record_index, match in matches.items()
                },
            })
        except Exception as error:
            row.update({"category": "error", "error": str(error)})
        rows.append(row)
        if pair_index % 50 == 0:
            print(
                f"Matched {pair_index}/{len(associations)} pairs", flush=True
            )

    categories = Counter(row["category"] for row in rows)
    graph_rows = [row for row in rows if row["association"] != "same_path"]
    graph_rules = {}
    for rule in RULE_RANK:
        members = [row for row in graph_rows if row["association"] == rule]
        graph_rules[rule] = {
            "pairs": len(members),
            "nodes": sum(row.get("nodes", 0) for row in members),
            "matched": sum(row.get("matched", 0) for row in members),
            "categories": dict(Counter(row["category"] for row in members)),
        }
    unassociated = sorted(set(corpus_files) - set(associations))
    summary = {
        "corpus": len(corpus_files),
        "same_path_pairs": sum(
            row["association"] == "same_path" for row in rows
        ),
        "graph_pairs": len(graph_rows),
        "paired": len(rows),
        "unassociated": len(unassociated),
        "graph_conflicts": sum(
            path in unassociated for path in graph_conflicts
        ),
        "categories": dict(categories),
        "node_bearing_pairs": sum(row.get("nodes", 0) > 0 for row in rows),
        "nodes": sum(row.get("nodes", 0) for row in rows),
        "matched_nodes": sum(row.get("matched", 0) for row in rows),
        "unmatched_nodes": sum(
            row.get("nodes", 0) - row.get("matched", 0) for row in rows
        ),
        "graph_rules": graph_rules,
    }
    report = {
        "schema_version": 1,
        "summary": summary,
        "pairs": rows,
        "graph_conflicts": {
            path: evidence for path, evidence in graph_conflicts.items()
            if path in unassociated
        },
        "unassociated": unassociated,
    }
    _atomic_text(
        output_directory / REPORT_JSON,
        json.dumps(report, indent=2, ensure_ascii=False),
    )
    _atomic_text(output_directory / REPORT_MARKDOWN, _markdown(report))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Reports: {output_directory}", flush=True)
    return report


def main():
    run(_arguments())


if __name__ == "__main__":
    main()

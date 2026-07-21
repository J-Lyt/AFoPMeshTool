# Developer tools

## Standalone material-corpus audit

`audit_material_corpus.py` inventories all `.mgraphobject`, `.mcompoundnode`,
`.mshader`, and referenced `.mmb` metadata directly from AFoP SDF archives.
It is pure Python and does not import or launch Blender.

Run it with Python 3.11 or newer:

```powershell
python tools\audit_material_corpus.py `
    --game-directory 'D:\path\to\AFOP' `
    --output-directory 'D:\path\to\material_audit' `
    --cache-directory 'D:\path\to\sdf_index_cache'
```

On the standard Windows Blender 5.0 installation, the wrapper uses Blender's
bundled CPython executable without launching Blender:

```powershell
& tools\audit_material_corpus.ps1 `
    -GameDirectory 'D:\path\to\AFOP' `
    -OutputDirectory 'D:\path\to\material_audit' `
    -CacheDirectory 'D:\path\to\sdf_index_cache'
```

The index cache is optional and is compatible with the add-on's targeted SDF
cache format. Pass `--oodle D:\path\oo2core_9_win64.dll` when a supported DLL
cannot be found in the repository or game directory. `--download-oodle` uses
the add-on's validated downloader, and `--no-rebuild` makes missing/stale cache
entries an error instead of decrypting the archive index again.

Blender's bundled Python executable may be used when no system Python is
installed; this still runs as an ordinary Python process and never imports
`bpy`.

## Material regression harness

`material_regression.py` validates material binding and Blender node creation
without extracted game data. It generates temporary triangle meshes and PNG
textures, then covers:

- direct shader-pin and compound-forwarded texture bindings;
- authoritative empty shader schemas (no legacy role fallback);
- `_CLOTH_SIM` material exclusion;
- character skin, hair, wildlife skin, Medusa, natural rock, terrain, moss,
  constants, and wildlife-gear material profiles;
- alpha, packed channel, bioluminescence, projection, and profile-audit wiring.
- Banshee pattern manifests, palettes, controls, and wildlife node updates.

Run it from the repository root with Blender 5.0 or newer:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe' `
    --background --factory-startup --python-exit-code 1 `
    --python tools\material_regression.py
```

The runner prints one line per check and raises after reporting every failure.
`--python-exit-code 1` makes a failure visible to shell scripts and CI jobs.
Keep the runner developer-only: do not add it to `updater.CODE_FILES` or the
add-on registration/UI.

## Mesh pipeline smoke test

`mesh_pipeline_regression.py` exercises the complete Blender-facing MMB path
against one extracted asset: load, import every LOD0 mesh, export to a temporary
directory, and parse the result again. It never modifies the source asset.

Run it from the repository root with Blender 5.0 or newer:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe' `
    --background --factory-startup --python-exit-code 1 `
    --python tools\mesh_pipeline_regression.py -- `
    --mmb 'D:\path\asset.mmb'
```

Use a small, non-cloth asset for a quick structural check. Cloth assets remain
subject to the separate byte-level and in-game validation described in the
reverse-engineering notes. Keep this runner developer-only and out of the
add-on updater manifest and UI.

## Real-import material snapshots

`material_snapshot.py` records the material result of an approved `.blend`
import. Baselines contain no geometry or image pixels: they retain mesh/object
names, material slots, logical texture references, `afop_*` metadata, node
settings, unlinked defaults, and node connections.

Capture a visually approved import:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe' `
    --background 'D:\path\approved_asset.blend' --python-exit-code 1 `
    --python tools\material_snapshot.py -- `
    --output tools\material_baselines\approved_asset.json
```

Compare a new import with its baseline:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe' `
    --background 'D:\path\new_asset.blend' --python-exit-code 1 `
    --python tools\material_snapshot.py -- `
    --baseline tools\material_baselines\approved_asset.json `
    --source-name approved_asset.blend
```

The source label must remain stable when the new `.blend` has a different
filename. A mismatch prints a unified JSON diff and exits with an error.
Baselines must only be replaced after the changed Blender result has been
visually reviewed and accepted.

For a directory whose `.blend` filenames match the baseline filenames, run the
whole comparison batch with:

```powershell
& tools\material_snapshot_batch.ps1 `
    -Mode compare `
    -BlendDirectory 'D:\path\approved_blends'
```

Use `-Mode capture` only when creating or deliberately replacing visually
approved baselines.

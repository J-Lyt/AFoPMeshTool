Based on the work by [AlexP0](https://github.com/AlexP0) for [SWOutlawsMeshTool](https://github.com/AlexP0/SWOutlawsMeshTool)

**Installation**
1. Click the green '<> Code' button and then 'Download ZIP'
2. In Blender, Go to 'Edit > Preferences'
3. Go to Add-ons
4. Click the little downward arrow top-right corner
5. Install from Disk...
6. Select the .zip file you downloaded.

**Usage**
1. Open the **AFoP Mesh Tool** panel in Blender's Scene Properties tab.
2. Enter an `.mmb` file in **Asset Path**, or select one with the folder button.
   The asset loads automatically when the path changes.
3. Use the buttons under **Import** to import a LOD for every mesh, or expand an
   individual mesh and click **Import** beside the LOD you need.
4. Make the required changes to the imported meshes.
5. Set **Export Path** if needed. It defaults to the loaded asset's folder.
6. Export an individual LOD from its mesh row, or use **Export All LODs**. By
   default, exports are written as `*_MOD.mmb` with a paired `.mcloth` when one
   exists; enable **Overwrite existing file** only when you intend to replace
   the selected output.

**Loading directly from the game SDF archives**

You can save the game folder and extraction destination under the add-on's
**Default Game Directory** and **Default Extracted Files** preferences. The
extraction folder defaults to Blender's user data-files location.

1. Expand **Load from Game Files** in the AFoP Mesh Tool panel.
2. Select the AFOP game folder containing `sdf.sdftoc` and the `.sdfdata` files.
3. The archives load automatically; use **Reload SDF Archives** if they need refreshing.
4. Enable **Import Materials and Textures** before **Import Selected** to read the
   matching `.mgraphobject` or `.mcompoundnode`, extract its referenced textures,
   and assign Blender materials to the imported LOD0 render mesh parts.
5. Type in the search field to filter the indexed `.mmb` paths.
6. Select an asset. Use **Load Selected** to extract and make it the current
   Asset Path without importing, or click **Import Selected** to import LOD0.
   The **Load as Asset** checkbox beneath Import Selected is enabled by default;
   uncheck it to import through a temporary file without replacing the current asset.

The add-on first looks for an Oodle runtime (`oodle-data-shared.dll` or
`oo2core_*.dll`) in its own folder. If one is not installed, it downloads the
latest Windows x64 `oodle-data-shared.dll` from the
[OodleUE release](https://github.com/WorkingRobot/OodleUE/releases/latest) and
installs only that DLL beside the add-on.

Extracted `.mmb`, `.mcloth`, and imported `.dds` texture files are stored in the configured
**Extracted Files** directory. Archive indices are cached and reused
automatically; use **Reload SDF Archives** after clearing the cache or when the
game files change.

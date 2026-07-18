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
4. Choose which asset types appear in search results. **MMB** is enabled by
   default; **MGraph** and **MCompoundNode** can be enabled independently.
5. Search and select an asset. An MMB can be loaded or imported directly;
   importing a graph or compound imports all its indexed MMB references,
   including references provided by linked compound nodes.
6. Enable **Import Materials and Textures** to extract referenced textures and
   apply materials. A selected graph or compound is used as the material source.
   **Load as Asset** keeps the selected MMB, or the last referenced MMB, loaded.

The add-on first looks for an Oodle runtime (`oodle-data-shared.dll` or
`oo2core_*.dll`) in its own folder. If one is not installed, it downloads the
latest Windows x64 `oodle-data-shared.dll` from the
[OodleUE release](https://github.com/WorkingRobot/OodleUE/releases/latest) and
installs only that DLL beside the add-on.

Extracted `.mmb`, `.mcloth`, and imported `.dds` texture files are stored in the configured
**Extracted Files** directory. Archive indices are cached and reused
automatically; use **Reload SDF Archives** after clearing the cache or when the
game files change.

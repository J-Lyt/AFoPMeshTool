# AFoP Mesh Tool

A Blender 5.0 add-on for importing and exporting meshes from
*Avatar: Frontiers of Pandora*.

Based on the work by [AlexP0](https://github.com/AlexP0) for [SWOutlawsMeshTool](https://github.com/AlexP0/SWOutlawsMeshTool)

## Installation

1. Click the green '<> Code' button and then 'Download ZIP'
2. In Blender, Go to 'Edit > Preferences'
3. Go to 'Add-ons'
4. Click the downward arrow in the top-right corner
5. Install from Disk...
6. Select the downloaded .zip file.

## Importing and exporting

1. Open **AFoP Mesh Tool** in Blender's Scene Properties tab.
2. Select an `.mmb` file in **Asset Path**.
3. Use the buttons under **Import** to import a LOD for every mesh, or expand an
   individual mesh and click **Import** beside the LOD you need.
4. Edit the imported meshes.
5. Set **Export Path** if required. It defaults to the source asset's folder.
6. Export an individual LOD or use **Export All LODs**.

Exports are written as `*_MOD.mmb` by default, with a paired `.mcloth` when one
exists. Enable **Overwrite existing file** only when you intend to replace the
imported `.mmb` file.

## Loading assets from the game files

The add-on can search and extract assets directly from the game's SDF archives.
Default game and extraction folders can be set in the add-on preferences.

1. Expand **Load from Game Files**.
2. Select the AFoP game root folder. e.g. `"...\Ubisoft\AFOP"`
3. Wait for the archive index to load.
4. Select which asset types to search. **MMB** is enabled by default;
   **MGraph** and **MCompoundNode** are optional.
5. Select which archives are include when searching. **Rogue**, **DLC1**, **DLC2**, and
   **DLC3** are all enabled by default.
6. Search for and select an asset, then load or import it.

Broad searches initially show up to 500 matches. Use the **Show All** button beside the
notice to display every match for the current search.

Selecting an MGraph or MCompoundNode will import its referenced MMB files. 

Enable **Import Materials and Textures** to extract the associated textures and build
Blender materials. When several material sources are available (MGraphs/MCompoundNode), the importer
allows the source to be selected.

When a Banshee MMB is loaded, a **Banshee Pattern** menu is shown. Select
one of the color patterns and click **Apply Pattern** to update the imported 
body, head, and body-linked weakpoint materials. Use **Remove Pattern** to restore 
their originally imported pattern settings.

When a selected MGraph directly references Banshee color and control files,
that pattern is applied automatically during import.

## Extracted files and archive cache

Extracted `.mmb`, `.mcloth`, and `.dds` files are stored under **Extracted
Files**. The archive index is cached and reused automatically. Reload the SDF
archives after clearing the cache or changing the installed game data.

Reading the archives requires an Oodle runtime. The add-on uses a compatible
DLL installed beside it or downloads the latest Windows x64 `oodle-data-shared.dll` from the
[OodleUE release](https://github.com/WorkingRobot/OodleUE/releases/latest)
when one is not available.

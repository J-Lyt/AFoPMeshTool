param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('compare', 'capture')]
    [string]$Mode,

    [Parameter(Mandatory = $true)]
    [string]$BlendDirectory,

    [string]$BlenderPath = 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe',

    [string]$BaselineDirectory = (Join-Path $PSScriptRoot 'material_baselines')
)

$ErrorActionPreference = 'Stop'
$resolvedBlender = (Resolve-Path -LiteralPath $BlenderPath).Path
$resolvedBlendDirectory = (Resolve-Path -LiteralPath $BlendDirectory).Path
$snapshotScript = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'material_snapshot.py')).Path

if ($Mode -eq 'capture' -and -not (Test-Path -LiteralPath $BaselineDirectory)) {
    New-Item -ItemType Directory -Path $BaselineDirectory | Out-Null
}
$resolvedBaselineDirectory = (Resolve-Path -LiteralPath $BaselineDirectory).Path
$blendFiles = @(Get-ChildItem -LiteralPath $resolvedBlendDirectory -File -Filter '*.blend')
if ($blendFiles.Count -eq 0) {
    throw "No .blend files were found in $resolvedBlendDirectory"
}

foreach ($blendFile in $blendFiles) {
    $baseline = Join-Path $resolvedBaselineDirectory ($blendFile.BaseName + '.json')
    if ($Mode -eq 'compare' -and -not (Test-Path -LiteralPath $baseline)) {
        throw "No baseline exists for $($blendFile.Name): $baseline"
    }
    $snapshotArgument = if ($Mode -eq 'capture') { '--output' } else { '--baseline' }
    & $resolvedBlender `
        --background $blendFile.FullName `
        --python-exit-code 1 `
        --python $snapshotScript `
        -- $snapshotArgument $baseline
    if ($LASTEXITCODE -ne 0) {
        throw "Material snapshot $Mode failed for $($blendFile.FullName)"
    }
}

Write-Output "Material snapshot batch $Mode passed for $($blendFiles.Count) file(s)."

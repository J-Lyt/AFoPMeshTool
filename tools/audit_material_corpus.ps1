param(
    [Parameter(Mandatory = $true)]
    [string]$GameDirectory,

    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,

    [string]$CacheDirectory = '',
    [string]$OodlePath = '',
    [switch]$DownloadOodle,
    [switch]$NoRebuild,

    [string]$PythonPath = (
        'C:\Program Files\Blender Foundation\Blender 5.0\5.0\python\bin\python.exe'
    )
)

$ErrorActionPreference = 'Stop'
$resolvedPython = (Resolve-Path -LiteralPath $PythonPath).Path
$resolvedGame = (Resolve-Path -LiteralPath $GameDirectory).Path
$auditScript = (Resolve-Path -LiteralPath (
    Join-Path $PSScriptRoot 'audit_material_corpus.py'
)).Path
$arguments = @(
    $auditScript,
    '--game-directory', $resolvedGame,
    '--output-directory', [System.IO.Path]::GetFullPath($OutputDirectory)
)
if ($CacheDirectory) {
    $arguments += @(
        '--cache-directory', [System.IO.Path]::GetFullPath($CacheDirectory)
    )
}
if ($OodlePath) {
    $arguments += @('--oodle', (Resolve-Path -LiteralPath $OodlePath).Path)
}
if ($DownloadOodle) {
    $arguments += '--download-oodle'
}
if ($NoRebuild) {
    $arguments += '--no-rebuild'
}

& $resolvedPython @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Material corpus audit failed with exit code $LASTEXITCODE"
}

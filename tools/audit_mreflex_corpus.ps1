param(
    [Parameter(Mandatory = $true)]
    [string]$GameDirectory,
    [Parameter(Mandatory = $true)]
    [string]$CorpusDirectory,
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,
    [string]$CacheDirectory,
    [string]$OodlePath,
    [switch]$NoRebuild
)

$python = 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Blender 5.0 was not found at $python"
}

$arguments = @(
    '--background',
    '--factory-startup',
    '--python-exit-code', '1',
    '--python', (Join-Path $PSScriptRoot 'audit_mreflex_corpus.py'),
    '--',
    '--game-directory', $GameDirectory,
    '--corpus-directory', $CorpusDirectory,
    '--output-directory', $OutputDirectory
)
if ($CacheDirectory) {
    $arguments += @('--cache-directory', $CacheDirectory)
}
if ($OodlePath) {
    $arguments += @('--oodle', $OodlePath)
}
if ($NoRebuild) {
    $arguments += '--no-rebuild'
}

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "MReflex corpus audit failed with exit code $LASTEXITCODE"
}

param(
    [string]$ExternalFactorGlob = "",
    [string]$AcceptedSummaryCsv = "",
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8002
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$WorkspaceRoot = Split-Path -Parent $RepoRoot

if (-not $ExternalFactorGlob) {
    $ExternalFactorGlob = Join-Path $WorkspaceRoot "microcap_alpha\outputs\alpha101_external_accepted317_20260618\microcap400_alphaprobe_weekly_*.csv"
}
if (-not $AcceptedSummaryCsv) {
    $AcceptedSummaryCsv = Join-Path $WorkspaceRoot "microcap_alpha\outputs\alphaprobe_deepseek_stage2_from_chains_v1\snapshot_iter354_accepted317_20260618\accepted_summary.csv"
}

$env:ALPHA101_EXTERNAL_FACTOR_GLOB = $ExternalFactorGlob
$env:ALPHA101_ACCEPTED_SUMMARY_CSV = $AcceptedSummaryCsv

Write-Host "ALPHA101_EXTERNAL_FACTOR_GLOB=$env:ALPHA101_EXTERNAL_FACTOR_GLOB"
Write-Host "ALPHA101_ACCEPTED_SUMMARY_CSV=$env:ALPHA101_ACCEPTED_SUMMARY_CSV"
Write-Host "Open http://$HostAddress`:$Port"

python -m uvicorn alpha101.research.server:app --host $HostAddress --port $Port

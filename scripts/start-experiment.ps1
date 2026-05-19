param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Topic,

    [string]$BaseBranch = "main",

    [string]$BranchPrefix = "exp",

    [switch]$NoTag,

    [switch]$AllowDirty,

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $output = & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        $joined = $Arguments -join " "
        throw "git $joined failed.`n$output"
    }
    return $output
}

function Test-GitRefExists {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Ref
    )

    & git show-ref --verify --quiet $Ref
    return ($LASTEXITCODE -eq 0)
}

function Convert-ToSlug {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    $slug = $Value.ToLowerInvariant()
    $slug = [regex]::Replace($slug, "[^a-z0-9]+", "-")
    $slug = $slug.Trim("-")
    if ([string]::IsNullOrWhiteSpace($slug)) {
        throw "Topic must contain at least one ASCII letter or digit."
    }
    return $slug
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $display = "git " + ($Arguments -join " ")
    if ($DryRun) {
        Write-Host "[dry-run] $display"
        return
    }

    Invoke-Git -Arguments $Arguments | Out-Null
    Write-Host $display
}

Invoke-Git -Arguments @("rev-parse", "--is-inside-work-tree") | Out-Null

$currentBranch = (Invoke-Git -Arguments @("branch", "--show-current") | Out-String).Trim()
if ($currentBranch -ne $BaseBranch) {
    throw "Current branch is '$currentBranch'. Switch to '$BaseBranch' before starting a new experiment."
}

if (-not $AllowDirty) {
    $statusOutput = Invoke-Git -Arguments @("status", "--porcelain")
    if (($statusOutput | Out-String).Trim()) {
        throw "Working tree is not clean. Commit or stash changes first, or rerun with -AllowDirty."
    }
}

$slug = Convert-ToSlug -Value $Topic
$branchName = "$BranchPrefix/$slug"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$tagName = "baseline/$timestamp-$slug"

if (Test-GitRefExists -Ref "refs/heads/$branchName") {
    throw "Branch '$branchName' already exists."
}

if (-not $NoTag -and (Test-GitRefExists -Ref "refs/tags/$tagName")) {
    throw "Tag '$tagName' already exists."
}

$baseCommit = (Invoke-Git -Arguments @("rev-parse", "--short", "HEAD") | Out-String).Trim()
$tagMessage = "Baseline on $BaseBranch at $baseCommit before $branchName"

if (-not $NoTag) {
    Invoke-Step -Arguments @("tag", "-a", $tagName, "-m", $tagMessage)
}

Invoke-Step -Arguments @("switch", "-c", $branchName)

Write-Host ""
Write-Host "Ready."
Write-Host "Base branch : $BaseBranch"
Write-Host "New branch  : $branchName"
if (-not $NoTag) {
    Write-Host "Baseline tag: $tagName"
    Write-Host "Compare with: git diff $tagName..$branchName"
}

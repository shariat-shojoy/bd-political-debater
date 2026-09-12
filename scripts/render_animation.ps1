# ============================================================
# Render SadTalker videos for a saved debate
# ============================================================
# Activates the .venv-sadtalker virtualenv and runs inference on each
# turn (avatar_image, audio_wav) pair, producing per-turn MP4s.
#
# The MP4s land in: data\animations\<debate_stem>\turn_NN_agent.mp4
# The Streamlit stage will auto-detect them and overlay on the speaking
# character (replacing the SVG cartoon).
#
# Usage:
#   .\scripts\render_animation.ps1 -DebatePath download\debate_1975_coup.json
#   .\scripts\render_animation.ps1 -DebatePath download\debate_1975_coup.json -Enhancer gfpgan
# ============================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$DebatePath,

    [Parameter(Mandatory=$false)]
    [string]$AudioDir,

    [Parameter(Mandatory=$false)]
    [string]$OutDir,

    [Parameter(Mandatory=$false)]
    [ValidateSet("gfpgan","restoreformer","none")]
    [string]$Enhancer = "gfpgan",

    [Parameter(Mandatory=$false)]
    [ValidateSet("full","crop","extfull")]
    [string]$Preprocess = "full",

    [Parameter(Mandatory=$false)]
    [switch]$Still
)

$ErrorActionPreference = "Stop"
$PROJECT_ROOT = Resolve-Path "$PSScriptRoot\.."
Set-Location $PROJECT_ROOT

$venvPath = "$PROJECT_ROOT\.venv-sadtalker"
$sadTalkerDir = "$PROJECT_ROOT\SadTalker"
$stPython = "$venvPath\Scripts\python.exe"
$inferScript = "$sadTalkerDir\inference.py"

# ---------- Sanity checks ----------
if (-not (Test-Path $venvPath)) {
    Write-Host "FAIL .venv-sadtalker not found. Run .\scripts\install_sadtalker.ps1 first." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $sadTalkerDir)) {
    Write-Host "FAIL SadTalker/ directory not found. Run .\scripts\install_sadtalker.ps1 first." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $inferScript)) {
    Write-Host "FAIL inference.py not found at $inferScript. SadTalker structure may have changed." -ForegroundColor Red
    exit 1
}

# Resolve paths
$debatePath = (Resolve-Path $DebatePath).Path
if (-not $AudioDir) {
    $debateStem = [System.IO.Path]::GetFileNameWithoutExtension($debatePath)
    $debateParent = Split-Path $debatePath -Parent
    $audioDir = Join-Path $debateParent ($debateStem + "_audio")
} else {
    $audioDir = (Resolve-Path $AudioDir).Path
}
if (-not $OutDir) {
    $debateStem = [System.IO.Path]::GetFileNameWithoutExtension($debatePath)
    $debateParent = Split-Path $debatePath -Parent
    $outDir = Join-Path $debateParent ($debateStem + "_anim")
}

if (-not (Test-Path $audioDir)) {
    Write-Host "FAIL Audio directory not found: $audioDir" -ForegroundColor Red
    Write-Host "  Run audio synthesis first:" -ForegroundColor Yellow
    Write-Host "    python app\tts\synth.py --debate $debatePath --out $audioDir --provider mms" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

# ---------- Read debate JSON + iterate turns ----------
$debate = Get-Content $debatePath -Raw | ConvertFrom-Json
$turns = $debate.turns

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " Rendering SadTalker videos" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Debate:    $debatePath"
Write-Host "  Audio:     $audioDir"
Write-Host "  Out:       $outDir"
Write-Host "  Turns:     $($turns.Count)"
Write-Host "  Enhancer:  $Enhancer"
Write-Host "  Preprocess:$Preprocess"
Write-Host "  Still:     $Still"
Write-Host ""

$success = 0
$failed = @()

foreach ($turn in $turns) {
    $idx = $turn.turn_index
    $agent = $turn.agent
    $audioFile = Join-Path $audioDir ("turn_{0:D2}_{1}.wav" -f $idx, $agent)
    $videoFile = Join-Path $outDir ("turn_{0:D2}_{1}.mp4" -f $idx, $agent)
    $imageFile = Join-Path $PROJECT_ROOT ("assets\{0}_avatar.png" -f $agent)

    if (-not (Test-Path $audioFile)) {
        Write-Host "  [SKIP] turn $($idx + 1) ($agent): audio not found at $audioFile" -ForegroundColor Yellow
        $failed += "turn $idx ($agent): audio missing"
        continue
    }
    if (-not (Test-Path $imageFile)) {
        Write-Host "  [SKIP] turn $($idx + 1) ($agent): avatar image not found at $imageFile" -ForegroundColor Yellow
        $failed += "turn $idx ($agent): avatar missing"
        continue
    }

    Write-Host "  [RENDER] turn $($idx + 1) / $($turns.Count) ($agent)... " -NoNewline -ForegroundColor Cyan

    $args = @(
        $inferScript,
        "--driven_audio", $audioFile,
        "--source_image", $imageFile,
        "--result_dir", $outDir,
        "--enhancer", $Enhancer,
        "--preprocess", $Preprocess
    )
    if ($Still) { $args += "--still" }

    $t0 = Get-Date
    Push-Location $sadTalkerDir
    try {
        $proc = Start-Process -FilePath $stPython -ArgumentList $args -NoNewWindow -Wait -PassThru -RedirectStandardOutput "$outDir\render_$idx.log" -RedirectStandardError "$outDir\render_$idx.err"
        $elapsed = (Get-Date) - $t0
        if ($proc.ExitCode -eq 0) {
            # SadTalker writes a file like <result_dir><timestamp>.mp4 - find and rename
            $latestMp4 = Get-ChildItem $outDir -Filter "*.mp4" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($latestMp4 -and $latestMp4.Name -ne (Split-Path $videoFile -Leaf)) {
                if (Test-Path $videoFile) { Remove-Item $videoFile -Force }
                Rename-Item $latestMp4.FullName -NewName (Split-Path $videoFile -Leaf)
            }
            Write-Host "OK done in $($elapsed.TotalSeconds)s" -ForegroundColor Green
            $success++
        } else {
            Write-Host "FAIL exit code $($proc.ExitCode)" -ForegroundColor Red
            Write-Host "    log: $outDir\render_$idx.log" -ForegroundColor Gray
            Write-Host "    err: $outDir\render_$idx.err" -ForegroundColor Gray
            $failed += "turn $idx ($agent): exit code $($proc.ExitCode)"
        }
    } catch {
        Write-Host "FAIL exception: $_" -ForegroundColor Red
        $failed += "turn $idx ($agent): exception $_"
    } finally {
        Pop-Location
    }
}

# ---------- Summary ----------
Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " Rendering complete" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Success: $success / $($turns.Count)"
if ($failed.Count -gt 0) {
    Write-Host "  Failed:  $($failed.Count)" -ForegroundColor Red
    foreach ($f in $failed) { Write-Host "    - $f" -ForegroundColor Red }
}
Write-Host "  Output:  $outDir"
Write-Host ""
Write-Host "Next: open the Streamlit UI and open the live stage - MP4s will auto-overlay." -ForegroundColor Cyan
Write-Host ""

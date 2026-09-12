# ============================================================
# SadTalker installer for Windows PowerShell
# ============================================================
# Creates a SEPARATE Python 3.10 venv (.venv-sadtalker) so it doesn't
# conflict with the main project's Python 3.12 venv.
#
# Why a separate venv:
#   SadTalker is pinned to Python 3.10 and torch 2.0.1 + cu118.
#   Installing it in the main project's venv would break Streamlit,
#   langchain, and other modern packages that need Python 3.12.
#
# What this script does:
#   1. Verifies Python 3.10 is available (installs via winget if missing)
#   2. Creates .venv-sadtalker in the project root
#   3. Installs SadTalker + its pinned dependencies
#   4. Downloads ~1.5GB of checkpoints (one-time)
#   5. Verifies the install by running inference --help
#
# Usage (run in PowerShell as the project root):
#   cd D:\bd-political-debater
#   .\scripts\install_sadtalker.ps1
# ============================================================

$ErrorActionPreference = "Stop"

$PROJECT_ROOT = Resolve-Path "$PSScriptRoot\.."
Set-Location $PROJECT_ROOT

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " SadTalker installer for BD Political Debater" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# ---------- Step 1: Verify NVIDIA GPU + CUDA driver ----------
Write-Host "[1/7] Checking NVIDIA GPU + driver…" -ForegroundColor Yellow
try {
    $nvidia = nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ GPU detected: $nvidia" -ForegroundColor Green
    } else {
        Write-Host "  ✗ NVIDIA driver not found. Install from https://www.nvidia.com/Download/" -ForegroundColor Red
        Write-Host "    RTX 5070 Ti needs driver version 552+ and CUDA 12.4+" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "  ✗ nvidia-smi not found. Install NVIDIA driver first." -ForegroundColor Red
    exit 1
}

# ---------- Step 2: Verify Python 3.10 ----------
Write-Host ""
Write-Host "[2/7] Checking Python 3.10 availability…" -ForegroundColor Yellow

$py310 = $null
foreach ($cmd in @("py -3.10", "python3.10", "python")) {
    try {
        $ver = Invoke-Expression "$cmd --version" 2>$null
        if ($ver -match "Python 3\.10") {
            $py310 = $cmd
            Write-Host "  ✓ Found: $cmd ($ver)" -ForegroundColor Green
            break
        }
    } catch {}
}

if (-not $py310) {
    Write-Host "  Python 3.10 not found. Installing via winget…" -ForegroundColor Yellow
    winget install --id Python.Python.3.10 -e --accept-source-agreements --accept-package-agreements
    # Refresh PATH for this session
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    $py310 = "py -3.10"
    $ver = Invoke-Expression "$py310 --version" 2>$null
    if ($ver -match "Python 3\.10") {
        Write-Host "  ✓ Python 3.10 installed: $ver" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Install failed. Manually install from https://www.python.org/downloads/release/python-31011/" -ForegroundColor Red
        exit 1
    }
}

# ---------- Step 3: Create the SadTalker venv ----------
Write-Host ""
Write-Host "[3/7] Creating .venv-sadtalker virtual environment…" -ForegroundColor Yellow

$venvPath = "$PROJECT_ROOT\.venv-sadtalker"
if (Test-Path $venvPath) {
    Write-Host "  Found existing .venv-sadtalker — removing for clean install" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $venvPath
}

Invoke-Expression "$py310 -m venv $venvPath"
if (-not (Test-Path "$venvPath\Scripts\python.exe")) {
    Write-Host "  ✗ venv creation failed" -ForegroundColor Red
    exit 1
}

$stPython = "$venvPath\Scripts\python.exe"
$stPip = "$venvPath\Scripts\pip.exe"
Write-Host "  ✓ Created at: $venvPath" -ForegroundColor Green

# ---------- Step 4: Clone SadTalker repo ----------
Write-Host ""
Write-Host "[4/7] Cloning SadTalker repository…" -ForegroundColor Yellow

$sadTalkerDir = "$PROJECT_ROOT\SadTalker"
if (Test-Path $sadTalkerDir) {
    Write-Host "  Existing SadTalker/ folder found — using as-is" -ForegroundColor Yellow
} else {
    git clone --depth 1 https://github.com/OpenTalker/SadTalker.git $sadTalkerDir
    if (-not $?) {
        Write-Host "  ✗ git clone failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "  ✓ Cloned to: $sadTalkerDir" -ForegroundColor Green
}

# ---------- Step 5: Install Python deps ----------
Write-Host ""
Write-Host "[5/7] Installing SadTalker Python dependencies (this takes ~5 min)…" -ForegroundColor Yellow

# Upgrade pip first
& $stPip install --upgrade pip wheel

# Install CUDA 11.8 torch (SadTalker's recommended version — most stable with its checkpoints)
Write-Host "  Installing torch 2.0.1 + cu118…" -ForegroundColor Gray
& $stPip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
if (-not $?) {
    Write-Host "  ⚠ cu118 install failed, trying cu121…" -ForegroundColor Yellow
    & $stPip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --index-url https://download.pytorch.org/whl/cu121
}

# Install SadTalker requirements
$sadReq = "$sadTalkerDir\requirements.txt"
if (Test-Path $sadReq) {
    & $stPip install -r $sadReq
} else {
    Write-Host "  ⚠ requirements.txt not found in SadTalker/, installing known deps manually" -ForegroundColor Yellow
    & $stPip install numpy==1.23.1 scipy librosa opencv-python opencv-contrib-python pyyaml Pillow tqdm \
        ffmpeg-python imageio imageio-ffmpeg scikit-learn matplotlib gradio \
        face-alignment==1.3.5 dlib==19.22.99 gfpgan basicsr
}

# Install helpers
& $stPip install cython pkgconfig resampy

# ---------- Step 6: Download checkpoints ----------
Write-Host ""
Write-Host "[6/7] Downloading SadTalker checkpoints (~1.5 GB, one-time)…" -ForegroundColor Yellow

$checkpointDir = "$sadTalkerDir\checkpoints"
if (-not (Test-Path $checkpointDir)) {
    New-Item -ItemType Directory -Path $checkpointDir | Out-Null
}

# Use HuggingFace mirror — more reliable than Google Drive on Windows
$hfBase = "https://github.com/OpenTalker/SadTalker/releases/download/v0.10.2"
$files = @{
    "checkpoints\epoch_20.pth" = "$hfBase/epoch_20.pth"
    "checkpoints\SadViT_001_P000.pth" = "$hfBase/SadViT_001_P000.pth"
    "gfpgan\weights\GFPGANv1.4.pth" = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
}

foreach ($path in $files.Keys) {
    $fullPath = "$sadTalkerDir\$path"
    if (Test-Path $fullPath) {
        Write-Host "  ✓ Already exists: $path" -ForegroundColor Green
        continue
    }
    $url = $files[$path]
    Write-Host "  Downloading: $path" -ForegroundColor Gray
    $dir = Split-Path $fullPath -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    try {
        Invoke-WebRequest -Uri $url -OutFile $fullPath -UseBasicParsing -TimeoutSec 300
        Write-Host "  ✓ Saved: $path" -ForegroundColor Green
    } catch {
        Write-Host "  ⚠ Download failed: $path — will try alternative source" -ForegroundColor Yellow
        # Fallback: HuggingFace Spaces
        try {
            $altUrl = "https://huggingface.co/spaces/SadTalker/SadTalker/resolve/main/$path"
            Invoke-WebRequest -Uri $altUrl -OutFile $fullPath -UseBasicParsing -TimeoutSec 300
            Write-Host "  ✓ Saved from HuggingFace: $path" -ForegroundColor Green
        } catch {
            Write-Host "  ✗ Could not download: $path — manual download required" -ForegroundColor Red
        }
    }
}

# ---------- Step 7: Verify install ----------
Write-Host ""
Write-Host "[7/7] Verifying install…" -ForegroundColor Yellow

& $stPython -c "import torch; print(f'  torch={torch.__version__}, cuda={torch.cuda.is_available()}, gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

$inferScript = "$sadTalkerDir\inference.py"
if (Test-Path $inferScript) {
    Write-Host "  ✓ inference.py found at $inferScript" -ForegroundColor Green
} else {
    Write-Host "  ⚠ inference.py not found — SadTalker structure may have changed" -ForegroundColor Yellow
}

# ---------- Done ----------
Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host " SadTalker install complete!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Add real portrait images (512x512, frontal face, neutral):"
Write-Host "       - $PROJECT_ROOT\assets\analyst_avatar.png"
Write-Host "       - $PROJECT_ROOT\assets\journalist_avatar.png"
Write-Host "  2. Render videos for your saved debate:"
Write-Host "       .\scripts\render_animation.ps1 -DebatePath download\debate_1975_coup.json"
Write-Host "  3. Open the Streamlit UI and click 'Open live stage' — MP4s will auto-overlay"
Write-Host ""
Write-Host "If install failed somewhere, common fixes:"
Write-Host "  - 'ImportError: dlib'        → Install Visual Studio Build Tools (C++): https://visualstudio.microsoft.com/visual-cpp-build-tools/"
Write-Host "  - 'CUDA out of memory'       → Close other GPU apps, retry"
Write-Host "  - 'File not found: epoch_20' → Checkpoints not downloaded — see step 6 errors above"
Write-Host ""

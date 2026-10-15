[CmdletBinding()]
param(
    [string]$ComfyCommit = 'b0f4b7b294ce482a2e071d9d762c133d38c7aa07',
    [string]$GGUFCommit = '373048b8403a7820620065210a691263d4da0a61',
    [string]$GGUFVersion = '0.19.0',
    [string]$KJCommit = 'dcfcb5d',
    [string]$IPAdapterCommit = 'b188a6cb39b512a9c6da7235b880af42c78ccd0d',
    [string]$ControlNetAuxCommit = '59b1fc411ede8623b2997855b8018f0b3b6cf49f',
    [string]$AnimaIPAdapterCommit = '6b77cd0c367d76402174ace2be50d3cb6aa77855'
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Comfy = Join-Path $Root 'ComfyUI'
$Venv = Join-Path $Comfy '.venv'
$Python = Join-Path $Venv 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $Comfy)) {
    Write-Host '[1/5] Cloning ComfyUI...'
    git clone https://github.com/Comfy-Org/ComfyUI.git $Comfy
    if ($LASTEXITCODE -ne 0) { throw "ComfyUI clone failed with exit code $LASTEXITCODE" }
} elseif (-not (Test-Path -LiteralPath (Join-Path $Comfy '.git'))) {
    throw "The ComfyUI path exists but is not a Git checkout: $Comfy"
} else {
    Write-Host "[1/5] Reusing ComfyUI checkout at $Comfy"
}

if ($ComfyCommit) {
    git -C $Comfy fetch origin $ComfyCommit
    if ($LASTEXITCODE -ne 0) { throw "ComfyUI fetch failed with exit code $LASTEXITCODE" }
    git -C $Comfy checkout --detach $ComfyCommit
    if ($LASTEXITCODE -ne 0) { throw "ComfyUI checkout failed with exit code $LASTEXITCODE" }
}

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host '[2/5] Creating Python 3.12 virtual environment...'
    py -3.12 -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "Python venv creation failed with exit code $LASTEXITCODE" }
} else {
    Write-Host '[2/5] Reusing existing virtual environment'
}

Write-Host '[3/5] Installing pinned CUDA 12.8 PyTorch and ComfyUI requirements...'
& $Python -m pip install --upgrade pip wheel
if ($LASTEXITCODE -ne 0) { throw "pip installation failed with exit code $LASTEXITCODE" }
& $Python -m pip install `
    'torch==2.11.0+cu128' `
    'torchvision==0.26.0+cu128' `
    'torchaudio==2.11.0+cu128' `
    --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed with exit code $LASTEXITCODE" }
& $Python -m pip install -r (Join-Path $Comfy 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed with exit code $LASTEXITCODE" }

Write-Host '[4/5] Installing the project node packs for model loading, face reference and pose control...'
$NodesDir = Join-Path $Comfy 'custom_nodes'
New-Item -ItemType Directory -Force -Path $NodesDir | Out-Null
$KJ = Join-Path $NodesDir 'ComfyUI-KJNodes'
if (-not (Test-Path -LiteralPath $KJ)) {
    git clone https://github.com/kijai/ComfyUI-KJNodes.git $KJ
    if ($LASTEXITCODE -ne 0) { throw "KJNodes clone failed with exit code $LASTEXITCODE" }
} else {
    Write-Host '  ComfyUI-KJNodes already exists - reusing checkout.'
}
if ($KJCommit) {
    git -C $KJ checkout --detach $KJCommit
    if ($LASTEXITCODE -ne 0) { throw "KJNodes checkout failed with exit code $LASTEXITCODE" }
}
$KJRequirements = Join-Path $KJ 'requirements.txt'
if (Test-Path -LiteralPath $KJRequirements) {
    & $Python -m pip install -r $KJRequirements
    if ($LASTEXITCODE -ne 0) { throw "KJNodes dependency installation failed with exit code $LASTEXITCODE" }
}

$GGUF = Join-Path $NodesDir 'ComfyUI-GGUF'
if (-not (Test-Path -LiteralPath $GGUF)) {
    git clone https://github.com/leejet/ComfyUI-GGUF.git $GGUF
    if ($LASTEXITCODE -ne 0) { throw "ComfyUI-GGUF clone failed with exit code $LASTEXITCODE" }
} else {
    Write-Host '  ComfyUI-GGUF already exists - reusing checkout.'
}
if ($GGUFCommit) {
    git -C $GGUF checkout --detach $GGUFCommit
    if ($LASTEXITCODE -ne 0) { throw "ComfyUI-GGUF checkout failed with exit code $LASTEXITCODE" }
}
$GGUFRequirements = Join-Path $GGUF 'requirements.txt'
if (Test-Path -LiteralPath $GGUFRequirements) {
    & $Python -m pip install -r $GGUFRequirements
    if ($LASTEXITCODE -ne 0) { throw "ComfyUI-GGUF dependency installation failed with exit code $LASTEXITCODE" }
}
& $Python -m pip install "gguf==$GGUFVersion"
if ($LASTEXITCODE -ne 0) { throw "GGUF Python package pin installation failed with exit code $LASTEXITCODE" }

$IPAdapter = Join-Path $NodesDir 'ComfyUI_IPAdapter_plus'
if (-not (Test-Path -LiteralPath $IPAdapter)) {
    git clone https://github.com/comfyorg/comfyui-ipadapter.git $IPAdapter
    if ($LASTEXITCODE -ne 0) { throw "IPAdapter clone failed with exit code $LASTEXITCODE" }
} else {
    Write-Host '  ComfyUI_IPAdapter_plus already exists - reusing checkout.'
}
if ($IPAdapterCommit) {
    git -C $IPAdapter checkout --detach $IPAdapterCommit
    if ($LASTEXITCODE -ne 0) { throw "IPAdapter checkout failed with exit code $LASTEXITCODE" }
}

$AnimaIPAdapter = Join-Path $NodesDir 'ComfyUI-Anima_IP-Adapter'
if (-not (Test-Path -LiteralPath $AnimaIPAdapter)) {
    git clone https://github.com/LuciferTC9527/ComfyUI-Anima_IP-Adapter.git $AnimaIPAdapter
    if ($LASTEXITCODE -ne 0) { throw "Anima IPAdapter clone failed with exit code $LASTEXITCODE" }
} else {
    Write-Host '  ComfyUI-Anima_IP-Adapter already exists - reusing checkout.'
}
if ($AnimaIPAdapterCommit) {
    git -C $AnimaIPAdapter checkout --detach $AnimaIPAdapterCommit
    if ($LASTEXITCODE -ne 0) { throw "Anima IPAdapter checkout failed with exit code $LASTEXITCODE" }
}
$AnimaIPAdapterRequirements = Join-Path $AnimaIPAdapter 'requirements.txt'
if (Test-Path -LiteralPath $AnimaIPAdapterRequirements) {
    & $Python -m pip install -r $AnimaIPAdapterRequirements
    if ($LASTEXITCODE -ne 0) { throw "Anima IPAdapter dependency installation failed with exit code $LASTEXITCODE" }
}

$ControlNetAux = Join-Path $NodesDir 'comfyui_controlnet_aux'
if (-not (Test-Path -LiteralPath $ControlNetAux)) {
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git $ControlNetAux
    if ($LASTEXITCODE -ne 0) { throw "ControlNet Aux clone failed with exit code $LASTEXITCODE" }
} else {
    Write-Host '  comfyui_controlnet_aux already exists - reusing checkout.'
}
if ($ControlNetAuxCommit) {
    git -C $ControlNetAux checkout --detach $ControlNetAuxCommit
    if ($LASTEXITCODE -ne 0) { throw "ControlNet Aux checkout failed with exit code $LASTEXITCODE" }
}
$ControlNetAuxRequirements = Join-Path $ControlNetAux 'requirements.txt'
if (Test-Path -LiteralPath $ControlNetAuxRequirements) {
    & $Python -m pip install -r $ControlNetAuxRequirements
    if ($LASTEXITCODE -ne 0) { throw "ControlNet Aux dependency installation failed with exit code $LASTEXITCODE" }
}

$ProjectApiSource = Join-Path $Root 'project_nodes\comfyui_shim.py'
$ProjectApiTarget = Join-Path $NodesDir 'project_workflow_api.py'
if (-not (Test-Path -LiteralPath $ProjectApiSource -PathType Leaf)) {
    throw "Project workflow API shim is missing at $ProjectApiSource"
}
Copy-Item -LiteralPath $ProjectApiSource -Destination $ProjectApiTarget -Force
Write-Host '  Installed the project workflow catalog API custom-node shim.'

Write-Host '[5/5] Creating isolated model and runtime directories...'
foreach ($dir in @('diffusion_models', 'text_encoders', 'vae', 'checkpoints', 'loras', 'upscale_models', 'ipadapter', 'clip_vision', 'controlnet', 'siglip2')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Comfy "models\$dir") | Out-Null
}
foreach ($dir in @('input', 'output', '.codex-runtime')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Comfy $dir) | Out-Null
}

& $Python -c "import sys,torch; print('Python',sys.version); print('Torch',torch.__version__); print('CUDA',torch.version.cuda); print('CUDA available',torch.cuda.is_available()); print('CUDA devices',torch.cuda.device_count()); [print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
if ($LASTEXITCODE -ne 0) { throw "CUDA runtime probe failed with exit code $LASTEXITCODE" }

Write-Host ''
Write-Host 'Setup complete. Next:'
Write-Host '  .\scripts\download_models.ps1 -Profile all'
Write-Host '  .\scripts\manage_comfyui.ps1 start'

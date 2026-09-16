[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory)]
    [ValidateSet('start', 'stop', 'status')]
    [string]$Action,
    [int]$Port = 8191,
    [string]$BindHost = '0.0.0.0'
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Comfy = Join-Path $Root 'ComfyUI'
$Python = Join-Path $Comfy '.venv\Scripts\python.exe'
$Launcher = Join-Path $Root 'scripts\launch_comfyui.py'
$Main = Join-Path $Comfy 'main.py'
$Runtime = Join-Path $Comfy '.codex-runtime'
$PidFile = Join-Path $Runtime 'comfyui.pid'
$Stdout = Join-Path $Runtime 'comfyui.stdout.log'
$Stderr = Join-Path $Runtime 'comfyui.stderr.log'

function Get-Listeners {
    @(Get-NetTCPConnection -LocalAddress $BindHost -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique)
}

function Get-VerifiedProcess([int]$ProcessId) {
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    if (-not $proc) { return $null }
    # The listening socket belongs to ComfyUI's main.py child, not to the
    # project launcher. Verify the exact project entry point and port before
    # allowing status/stop to claim or terminate the listener.
    if ([string]$proc.CommandLine -notlike "*$Main*") { return $null }
    if ([string]$proc.CommandLine -notlike "*--port $Port*") { return $null }
    return $proc
}

function Show-Status {
    $listeners = Get-Listeners
    [pscustomobject]@{
        action = 'status'
        host = $BindHost
        port = $Port
        listener_pids = @($listeners)
        verified_listener_pids = @($listeners | Where-Object { Get-VerifiedProcess $_ } | ForEach-Object { [int]$_ })
        pid_file = $PidFile
    } | ConvertTo-Json -Depth 4
}

if ($Action -eq 'status') {
    Show-Status
    exit 0
}

if ($Action -eq 'start') {
    if (Get-Listeners) { throw "Port $Port is already in use; refusing to start a second project server." }
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Project venv missing at $Python; run setup_comfyui.ps1 first." }
    if (-not $PSCmdlet.ShouldProcess("$BindHost`:$Port", 'Start project ComfyUI')) { exit 0 }
    New-Item -ItemType Directory -Force -Path $Runtime | Out-Null
    $arguments = @(
        ('"' + $Launcher + '"'),
        '--host', $BindHost,
        '--port', [string]$Port,
        '--',
        '--cpu-vae',
        '--vram-headroom', '1',
        '--reserve-vram', '1',
        '--async-offload', '2'
    )
    $proc = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru
    @{ pid = $proc.Id; root = $Root; host = $BindHost; port = $Port; started_at = [DateTime]::UtcNow.ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath $PidFile -Encoding UTF8
    $deadline = [DateTime]::UtcNow.AddSeconds(45)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (Get-Listeners) { Show-Status; exit 0 }
        $proc.Refresh()
        if ($proc.HasExited) { throw "ComfyUI failed to start; inspect $Stderr" }
        Start-Sleep -Milliseconds 500
    }
    throw "ComfyUI start timed out; inspect $Stdout and $Stderr"
}

if ($Action -eq 'stop') {
    $listeners = Get-Listeners
    if (-not $listeners) {
        if (Test-Path -LiteralPath $PidFile) { Remove-Item -LiteralPath $PidFile -Force }
        Show-Status
        exit 0
    }
    $verified = @($listeners | ForEach-Object { Get-VerifiedProcess $_ }) | Where-Object { $_ }
    if ($verified.Count -ne $listeners.Count) { throw 'Stop refused: port listener is not this project ComfyUI process.' }
    foreach ($proc in $verified) {
        if ($PSCmdlet.ShouldProcess("PID $($proc.ProcessId)", 'Stop project ComfyUI')) { Stop-Process -Id $proc.ProcessId -Force }
    }
    if (Test-Path -LiteralPath $PidFile) { Remove-Item -LiteralPath $PidFile -Force }
    Show-Status
}

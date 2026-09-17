[CmdletBinding()]
param(
    [string]$ServerUri = 'http://127.0.0.1:8191',
    [string]$WorkflowName = 'anima-turbo-v1-1',
    [string]$WorkflowPath,
    [string]$Prompt,
    [string]$NegativePrompt,
    [long]$Seed,
    [int]$Width,
    [int]$Height,
    [int]$BatchSize,
    [int]$Steps,
    [double]$Cfg,
    [double]$Denoise,
    [string]$SamplerName,
    [string]$Scheduler,
    [string]$FilenamePrefix,
    [string]$InputImage,
    [string]$MaskImage,
    [ValidateSet('red', 'green', 'blue', 'alpha')]
    [string]$MaskChannel,
    [int]$GrowMaskBy,
    [string]$FaceImage,
    [string]$ControlImage,
    [double]$ControlStrength,
    [double]$ControlStartPercent,
    [double]$ControlEndPercent,
    [double]$IPAdapterWeight,
    [double]$LoraConceptStrength,
    [double]$LoraRapiStrength,
    [string]$ClientId = 'project-workflow-api-local',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

if ($PSBoundParameters.ContainsKey('WorkflowPath')) {
    if (-not (Test-Path -LiteralPath $WorkflowPath -PathType Leaf)) {
        throw "Workflow file not found: $WorkflowPath"
    }
    $workflow = Get-Content -LiteralPath $WorkflowPath -Raw | ConvertFrom-Json
    if ($DryRun) {
        $workflow | ConvertTo-Json -Depth 50
        exit 0
    }
    $body = @{ prompt = $workflow; client_id = $ClientId } | ConvertTo-Json -Depth 50
    $response = Invoke-RestMethod -Method Post -Uri ($ServerUri.TrimEnd('/') + '/prompt') -ContentType 'application/json' -Body $body
    $response | ConvertTo-Json -Depth 10
    exit 0
}

$bodyObject = [ordered]@{
    workflow = $WorkflowName
    client_id = $ClientId
}

$parameterNames = @(
    'Prompt', 'NegativePrompt', 'Seed', 'Width', 'Height', 'BatchSize', 'Steps',
    'Cfg', 'Denoise', 'SamplerName', 'Scheduler', 'FilenamePrefix',
    'InputImage', 'MaskImage', 'MaskChannel', 'GrowMaskBy', 'FaceImage',
    'ControlImage', 'ControlStrength', 'ControlStartPercent', 'ControlEndPercent', 'IPAdapterWeight'
)
$parameterMap = @{
    Prompt = 'prompt'
    NegativePrompt = 'negative_prompt'
    Seed = 'seed'
    Width = 'width'
    Height = 'height'
    BatchSize = 'batch_size'
    Steps = 'steps'
    Cfg = 'cfg'
    Denoise = 'denoise'
    SamplerName = 'sampler_name'
    Scheduler = 'scheduler'
    FilenamePrefix = 'filename_prefix'
    InputImage = 'input_image'
    MaskImage = 'mask_image'
    MaskChannel = 'mask_channel'
    GrowMaskBy = 'grow_mask_by'
    FaceImage = 'face_image'
    ControlImage = 'control_image'
    ControlStrength = 'control_strength'
    ControlStartPercent = 'control_start_percent'
    ControlEndPercent = 'control_end_percent'
    IPAdapterWeight = 'ipadapter_weight'
}
$parameters = [ordered]@{}
foreach ($name in $parameterNames) {
    if ($PSBoundParameters.ContainsKey($name)) {
        $parameters[$parameterMap[$name]] = $PSBoundParameters[$name]
    }
}

$loras = [ordered]@{}
if ($PSBoundParameters.ContainsKey('LoraConceptStrength')) { $loras['concept'] = $LoraConceptStrength }
if ($PSBoundParameters.ContainsKey('LoraRapiStrength')) { $loras['rapi'] = $LoraRapiStrength }
if ($loras.Count -gt 0) { $parameters['loras'] = $loras }
if ($parameters.Count -gt 0) { $bodyObject['parameters'] = $parameters }
if ($DryRun) { $bodyObject['dry_run'] = $true }

$body = $bodyObject | ConvertTo-Json -Depth 50
$response = Invoke-RestMethod -Method Post -Uri ($ServerUri.TrimEnd('/') + '/project/prompt') -ContentType 'application/json' -Body $body
$response | ConvertTo-Json -Depth 50

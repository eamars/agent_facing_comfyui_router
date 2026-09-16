[CmdletBinding()]
param(
    [string]$BaseUrl = 'http://127.0.0.1:8191',
    [string]$FixturePath = (Join-Path $PSScriptRoot '..\workflows\agent_route_simulation_10.json')
)

$ErrorActionPreference = 'Stop'
$BaseUrl = $BaseUrl.TrimEnd('/')
$fixture = Get-Content -LiteralPath $FixturePath -Raw | ConvertFrom-Json

$results = foreach ($sample in $fixture.prompts) {
    # The fixture's agent_route_query is the simulated agent extraction from
    # user_request. Only this workflow-level query crosses the route API;
    # sample.prompt is deliberately never sent to /project/generate.
    $routeBody = $sample.agent_route_query | ConvertTo-Json -Depth 8
    $response = Invoke-RestMethod "$BaseUrl/project/resolve" `
        -Method Post -ContentType 'application/json' -Body $routeBody
    $top = @($response.matches)[0]
    [pscustomobject]@{
        id = $sample.id
        user_request = $sample.user_request
        agent_route_query = $sample.agent_route_query
        selected_workflow = $response.selected_workflow
        selection_status = $response.selection.status
        expected_workflow = $sample.expected_workflow
        expected_match = ($response.selected_workflow -eq $sample.expected_workflow)
        prompt_sent_to_generation = $false
    }
}

[pscustomobject]@{
    fixture = $FixturePath
    generation_requested = $false
    sample_count = @($results).Count
    expected_matches = @($results | Where-Object { $_.expected_match }).Count
    results = $results
} | ConvertTo-Json -Depth 12

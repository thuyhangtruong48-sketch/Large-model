$ProjectDir = Split-Path -Parent $PSScriptRoot

function Test-Port($Port) {
    try {
        return (Test-NetConnection -ComputerName "127.0.0.1" -Port $Port -WarningAction SilentlyContinue).TcpTestSucceeded
    } catch {
        return $false
    }
}

Write-Host "Docker containers" -ForegroundColor Cyan
docker ps --format "table {{.Names}}`t{{.Image}}`t{{.Status}}`t{{.Ports}}"

Write-Host ""
Write-Host "Ports" -ForegroundColor Cyan
Write-Host "  Flask:         $(Test-Port 5001)"
Write-Host "  Ollama:        $(Test-Port 11434)"
Write-Host "  Milvus:        $(Test-Port 19530)"
Write-Host "  Elasticsearch: $(Test-Port 9200)"
Write-Host "  Neo4j:         $(Test-Port 7687)"

Write-Host ""
Write-Host "Flask status" -ForegroundColor Cyan
try {
    Invoke-RestMethod "http://127.0.0.1:5001/status" | ConvertTo-Json -Depth 5
} catch {
    Write-Warning $_.Exception.Message
}

Write-Host ""
Write-Host "Flask process" -ForegroundColor Cyan
Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
    Where-Object { $_.CommandLine -like "*flaskapi.py*" } |
    Select-Object ProcessId, ParentProcessId, CommandLine

Write-Host ""
Write-Host "Recent Flask log" -ForegroundColor Cyan
Get-Content -Tail 60 (Join-Path $ProjectDir "flask_run.log") -ErrorAction SilentlyContinue


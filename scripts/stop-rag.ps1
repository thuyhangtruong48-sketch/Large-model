param(
    [switch]$All
)

$ProjectDir = Split-Path -Parent $PSScriptRoot
$ComposeFile = Join-Path $ProjectDir "docker\docker-compose.rag.yml"
$ComposeEnv = Join-Path $ProjectDir "docker\.env.rag"
$Neo4jHomes = @(
    "D:\software\neo4j\neo4j-community-5.26.0",
    "D:\lihao\damoxing\neo4j-community-5.26.0-windows\neo4j-community-5.26.0"
)

function Stop-PortProcess($Port, $Name) {
    $connections = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    foreach ($connection in $connections) {
        $pidToStop = $connection.OwningProcess
        if ($pidToStop -and $pidToStop -ne 0) {
            Write-Host "Stopping $Name process on port $Port (PID $pidToStop)" -ForegroundColor Cyan
            Stop-Process -Id $pidToStop -Force -ErrorAction SilentlyContinue
        }
    }
}

Write-Host "Stopping Flask" -ForegroundColor Cyan
Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
    Where-Object { $_.CommandLine -like "*flaskapi.py*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

Write-Host "Stopping Docker services" -ForegroundColor Cyan
docker stop milvus-standalone milvus-etcd milvus-minio es8 2>$null

Write-Host "Docker compose fallback" -ForegroundColor Cyan
docker compose --env-file $ComposeEnv -f $ComposeFile stop 2>$null

if ($All) {
    Write-Host "Stopping Ollama" -ForegroundColor Cyan
    Stop-Process -Name ollama -Force -ErrorAction SilentlyContinue

    Write-Host "Stopping Neo4j" -ForegroundColor Cyan
    foreach ($neo4jHome in $Neo4jHomes) {
        $neo4jBat = Join-Path $neo4jHome "bin\neo4j.bat"
        if (Test-Path $neo4jBat) {
            & $neo4jBat stop 2>$null
        }
    }
    Stop-PortProcess 7687 "Neo4j"

    Write-Host "Done. Flask, Docker RAG services, Ollama, and Neo4j were stopped." -ForegroundColor Green
} else {
    Write-Host "Done. Ollama and Neo4j are left running intentionally. Use -All to stop them too." -ForegroundColor Green
}

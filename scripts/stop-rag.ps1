$ProjectDir = Split-Path -Parent $PSScriptRoot
$ComposeFile = Join-Path $ProjectDir "docker\docker-compose.rag.yml"
$ComposeEnv = Join-Path $ProjectDir "docker\.env.rag"

Write-Host "Stopping Flask" -ForegroundColor Cyan
Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
    Where-Object { $_.CommandLine -like "*flaskapi.py*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

Write-Host "Stopping Docker services" -ForegroundColor Cyan
docker stop milvus-standalone milvus-etcd milvus-minio es8 2>$null

Write-Host "Docker compose fallback" -ForegroundColor Cyan
docker compose --env-file $ComposeEnv -f $ComposeFile stop 2>$null

Write-Host "Done. Ollama and Neo4j are left running intentionally." -ForegroundColor Green


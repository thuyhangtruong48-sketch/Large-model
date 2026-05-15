param(
    [switch]$NoFlask
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot
$ComposeFile = Join-Path $ProjectDir "docker\docker-compose.rag.yml"
$ComposeEnv = Join-Path $ProjectDir "docker\.env.rag"
$FlaskLog = Join-Path $ProjectDir "flask_run.log"
$FlaskErr = Join-Path $ProjectDir "flask_run.err.log"
$PythonExe = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$DockerDesktop = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
$Neo4jHomes = @(
    "D:\software\neo4j\neo4j-community-5.26.0",
    "D:\lihao\damoxing\neo4j-community-5.26.0-windows\neo4j-community-5.26.0"
)

function Write-Step($Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Test-Port($Port) {
    try {
        return (Test-NetConnection -ComputerName "127.0.0.1" -Port $Port -WarningAction SilentlyContinue).TcpTestSucceeded
    } catch {
        return $false
    }
}

function Wait-Port($Port, $Name, $Seconds = 120) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-Port $Port) {
            Write-Host "$Name ready on port $Port" -ForegroundColor Green
            return $true
        }
        Start-Sleep -Seconds 3
    }
    Write-Warning "$Name is not ready on port $Port after ${Seconds}s"
    return $false
}

function Wait-Docker() {
    $deadline = (Get-Date).AddSeconds(180)
    while ((Get-Date) -lt $deadline) {
        docker ps *> $null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Docker Engine ready" -ForegroundColor Green
            return
        }
        Start-Sleep -Seconds 5
    }
    throw "Docker Engine did not become ready. Please open Docker Desktop and retry."
}

function Ensure-DockerDesktop() {
    Write-Step "Checking Docker Desktop"
    docker ps *> $null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Docker Engine already running" -ForegroundColor Green
        return
    }

    if (-not (Test-Path $DockerDesktop)) {
        throw "Docker Desktop executable not found: $DockerDesktop"
    }
    Start-Process -FilePath $DockerDesktop -WindowStyle Hidden
    Wait-Docker
}

function Ensure-Ollama() {
    Write-Step "Checking Ollama"
    if (Test-Port 11434) {
        Write-Host "Ollama already running" -ForegroundColor Green
    } else {
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
        Wait-Port 11434 "Ollama" 60 | Out-Null
    }
    ollama list
}

function Ensure-DockerServices() {
    Write-Step "Starting Docker services"
    $existingMilvus = @("milvus-etcd", "milvus-minio", "milvus-standalone") | Where-Object {
        docker container inspect $_ *> $null
        $LASTEXITCODE -eq 0
    }

    if ($existingMilvus.Count -eq 3) {
        docker start milvus-etcd milvus-minio milvus-standalone | Out-Host
    } else {
        docker compose --env-file $ComposeEnv -f $ComposeFile up -d milvus-etcd milvus-minio milvus-standalone
    }

    docker container inspect es8 *> $null
    if ($LASTEXITCODE -eq 0) {
        docker start es8 | Out-Host
    } else {
        docker compose --env-file $ComposeEnv -f $ComposeFile up -d es8
    }

    Wait-Port 19530 "Milvus" 120 | Out-Null
    Wait-Port 9200 "Elasticsearch" 120 | Out-Null
}

function Ensure-Neo4j() {
    Write-Step "Checking Neo4j"
    if (Test-Port 7687) {
        Write-Host "Neo4j already running" -ForegroundColor Green
        return
    }

    $home = $Neo4jHomes | Where-Object { Test-Path (Join-Path $_ "bin\neo4j.bat") } | Select-Object -First 1
    if (-not $home) {
        Write-Warning "Neo4j installation not found. Skipping Neo4j startup."
        return
    }

    $log = Join-Path $home "logs\codex-neo4j-start.log"
    $err = Join-Path $home "logs\codex-neo4j-start.err.log"
    Start-Process -FilePath (Join-Path $home "bin\neo4j.bat") -ArgumentList "console" -WorkingDirectory $home -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden
    Wait-Port 7687 "Neo4j" 120 | Out-Null
}

function Ensure-Flask() {
    if ($NoFlask) {
        return
    }

    Write-Step "Starting Flask app"
    if (-not (Test-Path $PythonExe)) {
        throw "Virtualenv Python not found: $PythonExe"
    }

    Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
        Where-Object { $_.CommandLine -like "*flaskapi.py*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

    Remove-Item $FlaskLog, $FlaskErr -ErrorAction SilentlyContinue
    Start-Process -FilePath $PythonExe -ArgumentList "flaskapi.py" -WorkingDirectory $ProjectDir -RedirectStandardOutput $FlaskLog -RedirectStandardError $FlaskErr -WindowStyle Hidden
    Wait-Port 5001 "Flask" 120 | Out-Null
}

Ensure-DockerDesktop
Ensure-Ollama
Ensure-DockerServices
Ensure-Neo4j
Ensure-Flask

Write-Step "RAG environment status"
docker ps --format "table {{.Names}}`t{{.Status}}`t{{.Ports}}"
Write-Host ""
Write-Host "Ports:" -ForegroundColor Cyan
Write-Host "  Flask:         $(Test-Port 5001)"
Write-Host "  Ollama:        $(Test-Port 11434)"
Write-Host "  Milvus:        $(Test-Port 19530)"
Write-Host "  Elasticsearch: $(Test-Port 9200)"
Write-Host "  Neo4j:         $(Test-Port 7687)"
Write-Host ""
Write-Host "Open http://127.0.0.1:5001/" -ForegroundColor Green


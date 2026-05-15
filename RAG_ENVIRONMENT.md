# RAG Environment

This project now keeps its startup entry points under the project directory:

- `docker/docker-compose.rag.yml`: Docker services for Milvus and Elasticsearch.
- `docker/.env.rag`: image versions and Docker data root.
- `scripts/start-rag.ps1`: one-command startup for Docker Desktop, Ollama, Milvus, Elasticsearch, Neo4j, and Flask.
- `scripts/status-rag.ps1`: status check.
- `scripts/stop-rag.ps1`: stop Flask and Docker services.

## Start

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-rag.ps1
```

## Status

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\status-rag.ps1
```

## Stop

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop-rag.ps1
```

## Notes

- Flask is started with `.venv\Scripts\python.exe flaskapi.py`.
- Existing Docker containers named `milvus-etcd`, `milvus-minio`, `milvus-standalone`, and `es8` are reused to preserve current data.
- Milvus data remains in `D:/milvus/volumes`, configured by `docker/.env.rag`.
- Neo4j currently runs from the local Windows installation because the graph database already lives there. Migrating Neo4j into Docker should be done with a dump/load step, not by blindly moving files.


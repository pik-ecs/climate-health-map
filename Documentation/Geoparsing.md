```bash
export UV_NO_SOURCES_PACKAGE="nacsos_data openalex_ingest"

uv run --extra extract --prerelease=allow \
    healthmap geoparser \
    --batch-size=1000 \
    --loglevel="INFO" \
    --config="../nacsos-support/.config/remote.env"
```
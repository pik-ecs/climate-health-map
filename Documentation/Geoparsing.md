```bash
uv run --extra extract --no-sources-package nacsos_data --prerelease=allow spacy download en_core_web_trf

uv run --extra extract --no-sources-package nacsos_data --prerelease=allow \
    healthmap geoparser \
    --batch-size=1000 \
    --loglevel="INFO" \
    --config="../nacsos-support/.config/remote.env"
```
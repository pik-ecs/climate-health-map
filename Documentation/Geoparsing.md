```bash
export UV_NO_SOURCES_PACKAGE="nacsos_data openalex_ingest"

uv run --extra extract --prerelease=allow \
    healthmap geoparser \
    --batch-size=1000 \
    --loglevel="INFO" \
    --config="../nacsos-support/.config/remote.env"
    
    
    

uv run --extra extract --no-sources-package nacsos_data --prerelease=allow \
    healthmap geoparser \
    --batch-size=1000 \
    --loglevel="INFO" \
    --no-only-incl \
    --published-after 2024 \
    --show-count \
    --config="../living-cdr-map/conf/secret.env"

```

Additional input:
* https://github.com/mcallaghan/climate-health-impacts/blob/main/3_match_place_grid.ipynb
* https://gitlab.pik-potsdam.de/mcc-apsis/climate-trend-attribution
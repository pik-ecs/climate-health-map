# Climate & health map update pipeline
This repository contains all the tooling to update the climate & health map.
This is somewhat redundant to the [Lancet Countdown project](https://github.com/destiny-evidence/nacsos2eppi/tree/main/projects/lancet) for now.

Data available on zenodo: https://doi.org/10.5281/zenodo.19456817

## Documentation
* [All-in-one update flow](./Documentation/MapUpdate.md)
* [Export human annotations](./Documentation/Annotations.md)
* [Running import query](./Documentation/ImportQuery.md)
* [Classification workflow](./Documentation/Classification.md)
* [Extracting geolocations](./Documentation/Geoparsing.md)
* [Exporting data in various formats](./Documentation/Export.md)
* [Gap-filling missing meta-data](./Documentation/DataHydration.md)

## Installation
```bash
# basic dependencies
uv sync
# full dependencies
uv sync --extra classify --extra extract --extra notebook

# Formatting and basic static check
uv run ruff check
uv run ruff format
uv run mypy
```

### Windows installation
```shell
$env:GIT_SSH_COMMAND = "C:/PROGRA~1/Git/usr/bin/ssh.exe"
uv sync --no-sources-package="nacsos_data" --no-sources-package="openalex_ingest"
```
Bundle source:
```bash
 tar -czv --exclude='*/__pycache__' -f source.tar.gz README.md pyproject.toml uv.lock src Documentation
```


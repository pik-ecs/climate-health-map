# Climate & health map update pipeline
This repository contains all the tooling to update the climate & health map.
This is somewhat redundant to the [Lancet Countdown project](https://github.com/destiny-evidence/nacsos2eppi/tree/main/projects/lancet) for now.

Data available on zenodo: https://doi.org/10.5281/zenodo.19456817

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

## Create DB user
```sql
CREATE USER username WITH PASSWORD '???';
GRANT CONNECT ON DATABASE nacsos_core TO username;
GRANT USAGE ON SCHEMA public TO username;

GRANT SELECT ON TABLE project TO username;
GRANT SELECT ON TABLE item TO username;
GRANT SELECT ON TABLE academic_item TO username;
GRANT SELECT ON TABLE academic_item_variant TO username;
GRANT SELECT ON TABLE bot_annotation_metadata TO username;
GRANT SELECT ON TABLE bot_annotation TO username;
GRANT SELECT ON TABLE assignment TO username;
GRANT SELECT ON TABLE assignment_scope TO username;
GRANT SELECT ON TABLE annotation_scheme TO username;
GRANT SELECT ON TABLE annotation TO username;
GRANT SELECT ON TABLE import TO username;
GRANT SELECT ON TABLE import_revision TO username;
GRANT SELECT ON TABLE m2m_import_item TO username;
GRANT SELECT ON TABLE enhancement TO username;
```

## Documentation
* [Export human annotations](./Documentation/Annotations.md)
* [Classification workflow](./Documentation/Classification.md)

Bundle source:
```bash
 tar -czv --exclude='*/__pycache__' -f source.tar.gz README.md pyproject.toml uv.lock src Documentation
```

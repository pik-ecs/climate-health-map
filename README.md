# Climate & health map update pipeline
This repository contains all the tooling to update the climate & health map.
This is somewhat redundant to the [Lancet Countdown project](https://github.com/destiny-evidence/nacsos2eppi/tree/main/projects/lancet) for now.

## Installation
```bash
# basic dependencies
uv sync
# full dependencies
uv sync --extra classify --extra extract --extra notebook

# Formatting and basic static check
uv run ruff check
uv run ruff format
```

## Documentation
* [Export human annotations](./Documentation/Annotations.md)
* [Classification workflow](./Documentation/Classification.md)

# Human annotations

--> TODO: refine notes <--

* screening/abstract coding on NACSOS
* two schemes (major category and impacts)
* `climate_health_map.data.annotations.scopes` clearly defines which labels are safe to use (needs updating sometimes)
* human annotations used to train and eval ML model
* pipeline default expects export at `data/exports/annotations.csv` 
* originally in https://github.com/destiny-evidence/nacsos2eppi/tree/main/projects/lancet

```bash
# if necessary, tunnel port to database
ssh -N -L 5433:localhost:5432 se164

# check options
uv run healthmap export-annotations --help

# fetch latest annotations
uv run healthmap export-annotations --config=config/secret.env --target=data/exports/annotations_20260213.csv --loglevel=INFO --no-overwrite
uv run healthmap export-annotations --config=config/secret.env --target=data/exports/annotations_20260213.csv --loglevel=INFO --overwrite
```

# Human annotations

--> TODO: refine notes <--

* screening/abstract coding on NACSOS
* two schemes (major category and impacts)
* `climate_health_map.data.annotations.scopes` clearly defines which labels are safe to use (needs updating sometimes)
* human annotations used to train and eval ML model
* pipeline default expects export at `data/exports/annotations.csv` 

```bash
# if necessary, tunnel port to database
ssh -N -L 5433:localhost:5432 se164

# check options
uv run export_annotations --help

# fetch latest annotations
uv run export_annotations --config=config/secret.env --target=data/exports/annotations.csv --loglevel=INFO --no-overwrite
```

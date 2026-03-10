```bash
ssh -N -L 5433:localhost:5432 se164


uv run healthmap export classifications --config config/secret.env --target data/exports/classifications.csv
uv run healthmap export affiliations  --config config/secret.env --target data/exports/affiliations.csv --all-authors
uv run healthmap export places --config config/secret.env --target data/exports/places.csv

# or just 
uv run healthmap export all --config config/secret.env --target data/exports/2026/ --all-authors --filetype csv

# prepare scatterplot
uv run --extra scatter healthmap scatterplot reduce-topic-scores --source "data/exports/2026/classifications.csv" --target "data/exports/2026/scatterplot.csv" --model-path "data/trained/scatterplot_topicmodel_new.pkl"

uv run healthmap export lithub --source data/exports/2026 --target data/exports/lithub/ --year-start 1990 --year-end 2025
```
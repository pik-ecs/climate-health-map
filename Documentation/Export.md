```bash
ssh -N -L 5433:localhost:5432 se164


uv run healthmap export classifications --config config/secret.env --target data/exports/2026/classifications.csv
uv run healthmap export affiliations  --config config/secret.env --target data/exports/2026/affiliations.csv --all-authors
uv run healthmap export places --config config/secret.env --target data/exports/2026/places.csv
uv run healthmap export items --config config/secret.env --target data/exports/2026/items.csv --label-filter "rel_major|1"

# or just 
uv run healthmap export all --config config/secret.env --target data/exports/2026/ --all-authors --filetype csv


# Prepare files for literature hub
uv run healthmap export lithub --source data/exports/2026 --target data/exports/lithub/ --year-start 1990 --year-end 2025

# Prepare excel exports for Lancet Countdown
uv run healthmap export lancet excel --source data/exports/2026/ --target data/exports/lancet

uv run healthmap export lancet excel-regional --source data/exports/2026/ --target data/exports/lancet --year-start 2006
```

# Scatterplot preparation
```bash
# local
uv run --extra scatter healthmap scatterplot reduce-topic-scores --source "data/exports/2026/classifications.csv" --target "data/exports/2026/scatterplot.csv" --model-path "data/trained/scatterplot_topicmodel_new.pkl"
uv run --extra scatter healthmap scatterplot reduce-topic-scores --source "data/exports/2026/classifications.csv" --target "data/exports/2026/scatterplot.csv" --model-path "data/trained/scatterplot_topicmodel_new.pkl"
# place keywords
uv run healthmap scatterplot topic-keywords --source-topics data/exports/2026/classifications.csv --source-scatter data/exports/2026/scatterplot.csv --target data/exports/2026/keywords.csv
uv run healthmap scatterplot text-keywords --source-scatter data/exports/2026/scatterplot.csv --source-items data/exports/2026/items.csv --target data/exports/2026/kws.csv --limit 10000 --n-clusters 15 --n-clusters 5 --n-clusters 3

# cluster
export NUMBA_NUM_THREADS=40
export OMP_NUM_THREADS=40
export MKL_NUM_THREADS=40
export NUMEXPR_NUM_THREADS=40
export VECLIB_MAXIMUM_THREADS=40
export OPENBLAS_NUM_THREADS=40
uv run --extra scatter healthmap scatterplot reduce-topic-scores --source "../data/exports/2026/classifications.csv" --target "../data/exports/2026/scatterplot.csv" --model-path "../data/trained/scatterplot_topicmodel_new.pkl" --n-jobs 40
scp foote:/data/rd5/ecs/Data/LancetCountdown/LivingPipeline/data/trained/scatterplot_topicmodel_new.pkl data/trained
scp foote:/data/rd5/ecs/Data/LancetCountdown/LivingPipeline/data/exports/2026/scatterplot.csv data/exports/2026
```

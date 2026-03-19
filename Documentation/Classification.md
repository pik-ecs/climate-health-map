# Instructions for classification

Classification has three steps
1) Tuning (run all sorts of models for all labels repeatedly with hyper-parameter tuning); this writes info files with the best model parameters for each model/label/repeat combo.
2) Training (using best model config from above per label and train+store a model)
3) Application (point to a file, and this will apply trained classifiers)

Original classification:
https://github.com/AnneIsARealProgrammerNow/ClimateHealth_Wellcome/blob/v0.1/active_learning_with_evaluation.ipynb

## Cluster environment
When running any of the sl
```bash

```

# Hyper-parameter tuning
```bash
ssh foote

cd /data/rd5/ecs/Data/LancetCountdown/LivingPipeline/
git clone git@gitlab.pik-potsdam.de:mcc-apsis/living-evidence-maps/climate-health-map.git
cd climate-health-map

# make python and uv available
module load anaconda/2025
curl -LsSf https://astral.sh/uv/install.sh | sh
cp ~/.local/bin/uv* .

python -m venv venv
source venv/bin/activate
pip install uv
export UV_NO_SOURCES_PACKAGE="nacsos_data openalex_ingest"
# ignore the warning, we want a separate environment!
uv sync --extra classify --prerelease=allow 
#source .venv/bin/activate
uv run --extra classify --prerelease=allow \
    healthmap classification slurm-tuning-scripts \
    --training-data="../data/exports/annotations_20260213.csv" \
    --target-dir="../data/tuning" \
    --models-path="../data/offline_models" \
    --venv-path=".venv/" \
    --log-path="../data/logs/tuning/" \
    --slurm-user="timrepke@pik-potsdam.de" \
    --num-repeats=3 \
    --train-proportion=0.66 \
    --min-minor-class=20 \
    --max-vocab=10000 \
    --max-ngram=4 \
    --slurm-hours=3 \
    --tuning-trials-trans=25 \
    --tuning-trials-trad=500 \
    --ensure-models-offline \
    --loglevel="DEBUG" \
    --on-exists="IGNORE"
```

## Tuning quality summary (optional)
TODO: elaborate what this produces and how to interpret outputs
```bash
rsync -avh --progress -e ssh foote:/data/rd5/ecs/Data/LancetCountdown/LivingPipeline/data/tuning data/

uv run healthmap classification tuning-quality --source=data/tuning --target=data/quality/tuning
```

# Training
TODO: elaborate on what this does, requires, and produces

```bash
export UV_NO_SOURCES_PACKAGE="nacsos_data openalex_ingest"

# prepare job scripts
uv run --extra classify --prerelease=allow \
    healthmap classification slurm-train-scripts \
    --training-data="../data/exports/annotations_20260213.csv" \
    --tuning-dir="../data/tuning" \
    --target-dir="../data/trained" \
    --models-path="../data/offline_models" \
    --venv-path=".venv/" \
    --log-path="../data/logs/train/" \
    --slurm-user="timrepke@pik-potsdam.de" \
    --n-folds=10 \
    --min-n-majority=20 \
    --random-seed=4243 \
    --slurm-hours=5 \
    --ensure-models-offline \
    --loglevel="DEBUG" \
    --on-exists="IGNORE"

# submit jobs
sbatch train-trans.slurm
sbatch train-trad.slurm
```

## k-fold evaluation (optional)
```bash
# download to local dir (optional)
rsync -avh --progress --include='stats.json' --include='*/' --exclude='*' -e ssh foote:/data/rd5/ecs/Data/LancetCountdown/LivingPipeline/data/trained data/
```

# Classification
TODO: describe

```bash
export UV_NO_SOURCES_PACKAGE="nacsos_data openalex_ingest"

# Export records that have no major relevance classification
uv run healthmap export items --config config/secret.env --target data/exports/items.csv --label-missing --label-filter "rel_major|1" 

# Transfer to HPC
scp data/exports/items_20260304.csv foote:/data/rd5/ecs/Data/LancetCountdown/LivingPipeline/climate-health-map/data/exports

# need to replace {group}
uv run --extra classify --prerelease=allow \
    healthmap classification classify \
    --source="../data/exports/items.csv" \
    --target="../data/enhanced/labels_{group}.csv" \
    --group="{group}" \
    --cache-dir="../data/offline_models" \
    --models-dir="../data/trained" \
    --on-exists="IGNORE" \
    --on-missing-model="IGNORE" \
    --loglevel="INFO"

# SLURM
uv run --extra classify --prerelease=allow \
    healthmap classification slurm-classify-scripts \
    --source="../data/exports/items.csv" \
    --target="../data/enhanced/" \
    --cache-dir="../data/offline_models" \
    --models-dir="../data/trained" \
    --on-exists="IGNORE" \
    --on-missing-model="IGNORE" \
    --venv-path=".venv/" \
    --log-path="../data/logs/classify/" \
    --slurm-user="timrepke@pik-potsdam.de" \
    --slurm-hours=12 \
    --loglevel="INFO" 
```

# Persisting predictions to NACSOS
TODO: push classifications upstream to platform

```bash
export UV_NO_SOURCES_PACKAGE="nacsos_data openalex_ingest"

rsync -avh --progress -e ssh foote:/data/rd5/ecs/Data/LancetCountdown/LivingPipeline/data/enhanced data/
uv run healthmap ingest classifications-dir \
    --source="data/enhanced/" \
    --on-conflict="IGNORE" \
    --batch-size=1000 \
    --loglevel="INFO" \
    --config="config/secret.env"  
```

# Legacy classifiers
```bash
uv run --extra classify healthmap classification classify-legacy \
   --source '/data/rd5/ecs/Data/LancetCountdown/LivingPipeline/data/exports/items.csv' \
   --target '../data/enhanced/incl.csv' \
   --models-dir '../data/trained/' \
   --cache-dir '../data/offline_models/' \
   --run-major-incl --run-major-categories
```


# Additional classifiers
We also have our [policy instruments sector](https://huggingface.co/evidence-for-climate-solutions/climatebert-policyinstruments-sector) classifier on huggingface.
This is loaded separately and applied.

# Topic models
```bash
uv run healthmap topicmodel --source data/exports/2026/items.csv --target data/enhanced/topicmodel.csv --offline-models-path data/offline_models

uv run healthmap ingest classifications-file \
    --source data/enhanced/topicmodel.csv \ 
    --on-conflict="IGNORE" \
    --batch-size=1000 \
    --loglevel="INFO" \
    --config="config/secret.env"  
```

# Some SLURM tips
TODO: elaborate

```bash
# Check progress
squeue --me -t all
squeue --job [jobid]
less data/logs/[jobid]_[array].err
tail -f data/logs/[jobid]_[array].err

queue -t all -p gpu --format "%.18i %.9P %.8j %.8u %.8T %.12M %.14l %.10D %.20R %.20p %.15r %.20V"
queue -t all -p standard --format "%.18i %.9P %.8j %.8u %.8T %.12M %.14l %.10D %.20R %.20p %.15r %.20V"
queue --me -t all --format "%.18i %.9P %.8j %.8u %.8T %.12M %.14l %.10D %.20R %.20p %.15r %.20V"

# measure resource utilisation and efficiency fo jobs
sacct -u timrepke
seff <Jobid>

# kill all or one
scancel --me
scancel [jobid]
```


```bash
uv sync --no-sources-package nacsos_data --extra extract --extra classify
uv run  --no-sources-package nacsos_data nacsos ...
```

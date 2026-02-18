# Instructions for classification

Classification has three steps
1) Tuning (run all sorts of models for all labels repeatedly with hyper-parameter tuning); this writes info files with the best model parameters for each model/label/repeat combo.
2) Training (using best model config from above per label and train+store a model)
3) Application (point to a file, and this will apply trained classifiers)



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
# ignore the warning, we want a separate environment!
uv sync --no-sources --extra classify --prerelease=allow 
source .venv/bin/activate
uv run --no-sources --extra classify --prerelease=allow healthmap slurm-tune-scripts \
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

# Classifier quality summary (optional)
TODO: elaborate what this produces and how to interpret outputs
```bash
rsync -avh --progress -e ssh foote:/data/rd5/ecs/Data/LancetCountdown/LivingPipeline/data/tuning data/

uv run healthmap tuning-quality --source=data/tuning --target=quality/tuning
```

# Training
TODO: elaborate on what this does, requires, and produces
```bash
uv run --no-sources --extra classify --prerelease=allow healthmap train \
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
```

# Classification
TODO: NACSOS missing classification exporter
TODO: run classification

# SLURM
TODO: notes on slurm

# Persisting predictions to NACSOS
TODO: push classifications upstream to platform



# Below is deprecated!!!
# Below is deprecated!!!
# Below is deprecated!!!
# Below is deprecated!!!
# Below is deprecated!!!
# Below is deprecated!!!
# Below is deprecated!!!
```bash

export PYTHONPATH=.:$PYTHONPATH && python S02_Classify/classify.py --help


export PYTHONPATH=.:$PYTHONPATH && python S02_Classify/classify.py \                                                                                     :(
    --training-data data/00_TrainingData/original_converted/annotations.csv \
    --predict-data data/03_Predictions/dataset.csv \
    --target-dir data/03_Predictions/ \
    --column Agroforestry \
    --model TINYBERT

```

## Set up on cluster

```bash
# copy data to cluster
scp data/dataset_nacsos.csv foote:/p/tmp/timrepke/living-cdr-map/data/
scp -r data/00_TrainingData/ foote:/p/tmp/timrepke/living-cdr-map/data/
scp -r data/03_Predictions/ foote:/p/tmp/timrepke/living-cdr-map/data/
# or
rsync -avh --progress -e ssh  data/00_TrainingData/ foote:/p/tmp/timrepke/living-cdr-map/data/00_TrainingData/ 

# copy data from cluster
scp -r foote:/p/tmp/timrepke/living-cdr-map/data/03_Predictions data/
scp -r foote:/p/tmp/timrepke/living-cdr-map/data/00_TrainingData data/

# sync data from cluster
rsync -avh --progress -e ssh  foote:/p/tmp/timrepke/living-cdr-map/data/03_Predictions data/
# sync data to VM or rechner 
rsync -avh --progress -e ssh  data/03_Predictions rept@10.10.13.45:/home/rept/workspace/living-cdr-map/data/
rsync -avh --progress -e ssh  data/03_Predictions rept@10.10.12.41:/home/rept/workspace/living-cdr-map/data/
```

```bash
ssh user@hpc
# ---
cd /p/tmp/[username]
git clone git@gitlab.pik-potsdam.de:mcc-apsis/living-evidence-maps/cdr-map/living-cdr-map.git
module load anaconda/2025
cd living-cdr-map
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:.

python S02_Classify/classify_slurm.py \
    --training-data "data/00_TrainingData/original_converted/annotations.csv" \
    --predict-data "data/dataset_nacsos.csv" \
    --target-dir "data/03_Predictions" \
    --models-path "data/models" \
    --venv-path "venv" \
    --log-path "data/logs" \
    --on-exists "SKIP" \
    --slurm-user "your.address@pik-potsdam.de"
 # OR THIS: --training-data "data/00_TrainingData/original_converted/annotations_extended.csv" \ 

# Submit
sbatch S02_Classify/classify-gpu.slurm
sbatch S02_Classify/classify-cpu.slurm

# Check progress
squeue --me -t all
squeue --job [jobid]
less data/logs/[jobid]_[array].err
tail -f data/logs/[jobid]_[array].err

# kill all or one
scancel --me
scancel [jobid]
```

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

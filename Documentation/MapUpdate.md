# Step 1) Update import
Either in the UI or [via a script](./ImportQuery.md)
Make sure to filter for xpac and exclude sources.

Download update data. Note: update the date!
```bash
# forward port if needed
ssh -N -J ts01 -L 5000:localhost:5432 se164

# download items that are not classified for relevance
# ADJUST DATE!
uv run healthmap export items --config config/secret.env --target data/exports/items_20260929.csv --label-missing --label-filter "rel_major|1" 
```

# Step 2) Classify
Classification is currently using the PIK HPC.

```bash
# Transfer data
scp data/exports/items_20260304.csv foote:/data/rd5/ecs/Data/LancetCountdown/LivingPipeline/data/exports

# [Log in to cluster]

# go to workdir
cd /data/rd5/ecs/Data/LancetCountdown/LivingPipeline/climate-health-map

# prepare slurm script
# ADJUST THE DATE! Also: check classify.slurm parameters before submitting
uv run --extra classify --prerelease=allow \
    healthmap classification slurm-classify-scripts \
    --source="../data/exports/items_20260929.csv" \
    --target="../data/enhanced/20260929" \
    --cache-dir="../data/offline_models" \
    --models-dir="../data/trained" \
    --on-exists="SKIP" \
    --on-missing-model="IGNORE" \
    --venv-path=".venv/" \
    --log-path="../data/logs/classify/" \
    --slurm-user="timrepke@pik-potsdam.de" \
    --slurm-hours=12 \
    --loglevel="INFO"

# Revert dependencies
uv pip install "transformers<4.50.0"

# Submit
sbatch classify.slurm
# Monitor
squeue --me -t all -p gpu --format "%.18i %.10q %.9P %.8j %.8u %.5T %.12M %.14l %.10D %.20R %.20p %.15r %.20V"
# See log (adjust job and array ID)
tail -f ../data/logs/classify/2338983_2.err

# Transfer data to local
rsync -avh --progress -e ssh foote:/data/rd5/ecs/Data/LancetCountdown/LivingPipeline/data/enhanced data/ 
```

## Alternative: Using legacy classifiers for primary inclusion and major category 
Run on apsis-rechner
```bash
# Transfer data first
rsync -avh --progress -e ssh data srv-mcc-apsis:/home/rept/workspace/climate-health-map/

cd /home/rept/workspace/climate-health-map

# ADJUST DATE!
uv run --with "scikit-learn~=1.5.2" --extra notebook --extra classify healthmap classification classify-legacy \
   --source 'data/exports/items_20260929.csv' \
   --target 'data/enhanced/20260929/legacy.csv' \
   --models-dir 'data/trained/' \
   --cache-dir 'data/offline_models/' \
   --run-major-incl --run-major-categories

# Transfer data to local
scp srv-mcc-apsis:/home/rept/workspace/climate-health-map/data/enhanced/20260929/legacy.csv data/enhanced/20260929/

# Ingest to database (this overwrites the cat and rel_major enhancements if they already exist)
# ADJUST DATE!
uv run healthmap ingest classifications-file \
    --source="data/enhanced/20260929/legacy.csv" \
    --on-conflict="IGNORE" \
    --batch-size=1000 \
    --loglevel="INFO" \
    --config="config/secret.env"
```

# Step 3) Ingest to NACSOS
Run locally
```bash
# Get data from server
rsync -avh --progress -e ssh foote:/data/rd5/ecs/Data/LancetCountdown/LivingPipeline/data/enhanced data/

# Load into database
# ADJUST DATE!
export UV_NO_SOURCES_PACKAGE="nacsos_data openalex_ingest"
uv run healthmap ingest classifications-dir \
    --source="data/enhanced/20260929" \
    --on-conflict="IGNORE" \
    --batch-size=1000 \
    --loglevel="INFO" \
    --config="config/secret.env"  
```

# Step 4) Topic models
This can run locally or on apsis-rechner
```bash
# Apply topic model
# ADJUST DATE!
uv run healthmap topicmodel --source data/exports/items_20260929.csv --target data/enhanced/20260929/topicmodel.csv --offline-models-path data/offline_models

# Load into database
# ADJUST DATE!
uv run healthmap ingest classifications-file \
    --source="data/enhanced/20260929/topicmodel.csv" \
    --on-conflict="IGNORE" \
    --batch-size=1000 \
    --loglevel="INFO" \
    --config="config/secret.env"
```

# Step 5) Additional classifiers
We also have our [policy instruments sector](https://huggingface.co/evidence-for-climate-solutions/climatebert-policyinstruments-sector) classifier on huggingface.
This is loaded separately and applied.

# Step 6) mordecai3
Run on apsis-rechner

If required, get es-geonames back up again
```bash
# check status
sudo docker ps
# if not in the list, continue:

# go to dir
cd /srv/mordecai3/es-geonames
# check for update
git pull origin master
# start container
sudo docker compose up
# populate index 
source venv/bin/activate 
./create_index.sh 
```

Run mordecai 
* this directly writes to the database
* is applied to items that have rel_major|1=true enhancement and no mordecai enhancement
```bash
export UV_NO_SOURCES_PACKAGE="nacsos_data openalex_ingest"
cd /home/rept/workspace/climate-health-map
 uv run --python 3.13 --extra notebook --extra extract --prerelease=allow \
    healthmap geoparser \
    --batch-size=1000 \
    --loglevel="INFO" \
    --config="config/secret.env"
```
If you are in a time pinch, you can apply this to _all_ (new) records before the relevance classifications are imported
by adding `--no-only-incl --created-after="2026-09-30"` (set date to just before your import was).

# Step 7) Consolidation raw data files
```bash
# Export all data
uv run healthmap export all --config config/secret.env --target data/exports/20260929/ --all-authors --filetype csv

# Update the group share
rsync -avh --progress -e ssh data foote:/data/rd5/ecs/Data/LancetCountdown/LivingPipeline/
```

# Step 8) Semantic landscape (scatterplot)

## Option 1) Run locally or on apsis-rechner -> assumed method
See `ensure_offline_nltk` in `src/climate_health_map/shared/text.py` to see which NLTK sources may need to be downloaded if you run into errors.
```bash
# Compute positions
#uv run --extra scatter healthmap scatterplot reduce-text --source "data/exports/20260929/classifications.csv" --target "data/exports/20260929/scatterplot-text.csv"   --model-path "data/trained/scatterplot_topicmodel_text.pkl"
uv run --python 3.14 --extra scatter healthmap scatterplot reduce-topic-scores --source "data/exports/20260929/classifications.csv" --target "data/exports/20260929/scatterplot-topics.csv" --model-path "data/trained/scatterplot_topicmodel_topics.pkl"

# place keywords
uv run healthmap scatterplot topic-keywords --source-topics data/exports/20260929/classifications.csv --source-scatter data/exports/20260929/scatterplot-topics.csv --target data/exports/20260929/keywords-topics.csv
uv run healthmap scatterplot text-keywords  --source-items data/exports/20260929/items.csv            --source-scatter data/exports/20260929/scatterplot-topics.csv --target data/exports/20260929/keywords-text.csv --limit 10000 --n-clusters 15 --n-clusters 5 --n-clusters 3

rsync -avh --progress -e ssh srv-mcc-apsis:/home/rept/workspace/climate-health-map/data/exports/20260929 data/exports/ 
```

## Option 2) Run on HPC
```bash
# important to constrain CPU use, otherwise you lock the entire node!
export NUMBA_NUM_THREADS=40
export OMP_NUM_THREADS=40
export MKL_NUM_THREADS=40
export NUMEXPR_NUM_THREADS=40
export VECLIB_MAXIMUM_THREADS=40
export OPENBLAS_NUM_THREADS=40
uv run --extra scatter healthmap scatterplot reduce-topic-scores --source "../data/exports/20260929/classifications.csv" --target "../data/exports/20260929/scatterplot.csv" --model-path "../data/trained/scatterplot_topicmodel_new.pkl" --n-jobs 40

scp foote:/data/rd5/ecs/Data/LancetCountdown/LivingPipeline/data/trained/scatterplot_topicmodel_new.pkl data/trained
scp foote:/data/rd5/ecs/Data/LancetCountdown/LivingPipeline/data/exports/20260929/scatterplot.csv data/exports/20260929
```

# Step 9) Prepare lithub export
Warning, this needs a lot of RAM (10–15GB). Make sure you have enough free memory of swap.
```bash
# Prepare lithub (most) assets
# ADJUST DATE!
uv run healthmap export lithub --source data/exports/20260929 --target data/exports/lithub/20260929 --year-start 1990 --year-end 2026
```

# Step 10) Update online version

```bash
# [log in to se164]

cd /data/lithub/.data

# Backup old data
sudo -u lithub mv healthmap_2026 healthmap_2026.20260309
sudo -u lithub mkdir healthmap_2026
sudo chown -R timrepke healthmap_2026

# [from local]
rsync -avh --progress -e ssh data/exports/lithub/20260929/ se164:/data/lithub/.data/healthmap_2026

# [on se164]
sudo chown -R lithub:lithub healthmap_2026
sudo systemctl restart lithub
```
Check online version is loading properly (first load is slow, no caching yet)
https://climateliterature.org/#/project/healthmap_2026

```bash
# Update the group share
rsync -avh --progress -e ssh data foote:/data/rd5/ecs/Data/LancetCountdown/LivingPipeline/
```
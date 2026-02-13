import logging
import os
from pathlib import Path
from typing import Annotated, Any

import typer

from climate_health_map.data import LABELS
from climate_health_map.shared.types import OnConflict
from .util import prepare_nltk

logger = logging.getLogger('precompute ranks')

CPU_MODELS = [
    'SVM',
    'REG',
    'SGD',
    'LGBM',
]
TRANSFORMERS = {
    'CLIMATEBERT': 'climatebert/distilroberta-base-climate-f',
    'SCIBERT': 'allenai/scibert_scivocab_uncased',
    'TINYBERT': 'prajjwal1/bert-tiny',
}
GPU_MODELS = list(TRANSFORMERS.keys())


def ensure_offline_models(model_data_path: Path) -> None:
    from huggingface_hub import snapshot_download

    for model in GPU_MODELS:
        name = TRANSFORMERS[model]

        logger.info(f'Downloading model: {model} ({name}) so it is available offline in {model_data_path}')
        snapshot_download(
            repo_id=name,
            repo_type='model',
            cache_dir=model_data_path,
            force_download=False,
        )


def sbatch(
    slurm_params: dict[str, Any],
    script_params: dict[str, Any],
    n_repeats: int,
    models: list[str],
    venv_path: Path,
    models_path: Path,
):
    # TODO: load dataset and check available columns
    array = [
        f'"{label.name}____{model}____{repeat}"'
        for group in LABELS.values()  # TODO: use get_filtered_labels
        for label in group.labels
        for model in models
        for repeat in range(n_repeats)
        if (
            not (Path(script_params['target-dir']) / f'{model}-{label.parent}-{label.name}-{repeat}-pred.csv').exists()
            or script_params['on-exists'] != OnConflict.SKIP.value
        )
    ]
    logger.info(f'Number of array jobs: {len(array)}')

    # For information on array jobs, see: https://hpcdocs.hpc.arizona.edu/running_jobs/batch_jobs/array_jobs/
    batch = f"""#!/bin/bash

#SBATCH --mail-type=END,FAIL  # 'NONE', 'BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL'
#SBATCH --array=1-{len(array)}
"""
    for k, v in slurm_params.items():
        if v is None:
            batch += f'#SBATCH --{k}\n'
        elif type(v) is list:
            batch += f'#SBATCH --{k}={",".join([str(vi) for vi in v])}\n'
        else:
            batch += f'#SBATCH --{k}={v}\n'

    batch += f"""

# Set this to exit the script when an error occurs
set -e
# Set this to print commands before executing
set -o xtrace

# Set up python environment
module load anaconda/2025
module load cuda
source "{venv_path}/bin/activate"

# Python env vars
export PYTHONPATH=$PYTHONPATH:{os.getcwd()}
export PYTHONUNBUFFERED=1

# Environment variables for script
export OPENBLAS_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OFFLINE_MODEL_PATH={models_path}

echo "Using python from $(which python)"
echo "Python version is $(python --version)"

PARAMS=(
   {'\n   '.join(array)}
)

job=$(($SLURM_ARRAY_TASK_ID - 1))
param=${{PARAMS[$job]}}
label=$(echo $param | awk -F'____' '{{print $1}}')
model=$(echo $param | awk -F'____' '{{print $2}}')
repeat=$(echo $param | awk -F'____' '{{print $3}}')

echo "array_task_id" $SLURM_ARRAY_TASK_ID " --> job" $job
echo $label
echo $model
echo $repeat

python S02_Classify/02_classify.py single \\
"""
    for k, v in script_params.items():
        if v is None:
            batch += f'       --{k} \\\n'
        elif type(v) is list:
            for vi in v:
                batch += f'       --{k} "{vi}" \\\n'
        else:
            batch += f'       --{k} "{v}" \\\n'

    return batch[:-3]


def create_sbatch_files(
    training_data: Annotated[Path, typer.Option(help='')],
    predict_data: Annotated[Path, typer.Option(help='')],
    target_dir: Annotated[Path, typer.Option(help='')],
    models_path: Annotated[Path, typer.Option(help='')],
    venv_path: Annotated[Path, typer.Option(help='')],
    log_path: Annotated[Path, typer.Option(help='')],
    slurm_user: Annotated[str, typer.Option(help='email address to notify when done')],
    on_exists: Annotated[OnConflict, typer.Option(help='')] = OnConflict.IGNORE,
    num_repeats: int = 3,
    train_proportion: float = 0.85,
    max_vocab: int = 7500,
    max_ngram: int = 1,
    min_df: int = 3,
    random_state: int | None = None,
    slurm_hours: int = 2,
    tuning_trials: int | None = None,
    ensure_models_offline: bool = True,
):
    # Ensure directories are ready
    venv_path = venv_path.absolute().resolve()
    log_path = log_path.absolute().resolve()
    models_path = models_path.absolute().resolve()
    log_path.mkdir(parents=True, exist_ok=True)

    # Ensure models are downloaded
    if ensure_models_offline:
        logger.info('Making sure all models are available offline!')
        ensure_offline_models(model_data_path=models_path)
        prepare_nltk()

    # Prepare some variables to use in the batch file

    sbatch_args = {
        'time': f'{slurm_hours:0>2}:00:00',
        'nodes': '1',
        'mem': '8G',
        'mail-user': slurm_user,
        'output': f'{log_path}/%A_%a.out',
        'error': f'{log_path}/%A_%a.err',
        'chdir': os.getcwd(),
    }
    script_args = {
        'training-data': training_data,
        'predict-data': predict_data,
        'target-dir': target_dir,
        'train-proportion': train_proportion,
        'column': '${label}',
        'repeat': '${repeat}',
        'model': '${model}',
        'on-exists': on_exists.value,
    }
    if random_state is not None:
        script_args['random-state'] = random_state
    if tuning_trials is not None:
        script_args['n-tuning-trials'] = tuning_trials

    sbatch_gpu = sbatch(
        slurm_params=sbatch_args
        | {
            'gres': 'gpu:1',  # number of GPUs
            'partition': 'gpu',
            'qos': 'gpumedium',  # or gpumedium (has MaxJobsPU=None but half priority, see `$ sacctmgr show qos`)
            'cpus-per-task': 5,
            'oversubscribe': None,  # use non-utilized GPUs on busy nodes
        },
        script_params=script_args
        | {
            'n-tuning-jobs': 1,
        },
        n_repeats=num_repeats,
        models=GPU_MODELS,
        venv_path=venv_path,
        models_path=models_path,
    )

    sbatch_cpu = sbatch(
        slurm_params=sbatch_args
        | {
            'cpus-per-task': 12,
            'partition': 'standard',
            'qos': 'short',
        },
        script_params=script_args
        | {
            'max-vocab': max_vocab,
            'max-ngram': max_ngram,
            'min-df': min_df,
            'n-tuning-jobs': 5,
        },
        n_repeats=num_repeats,
        models=CPU_MODELS,
        venv_path=venv_path,
        models_path=models_path,
    )

    with open('S02_Classify/classify-gpu.slurm', 'w') as slurm_file:
        slurm_file.write(sbatch_gpu)
    with open('S02_Classify/classify-cpu.slurm', 'w') as slurm_file:
        slurm_file.write(sbatch_cpu)

    logger.info('Run the following to get things on GPU going: sbatch S02_Classify/classify-gpu.slurm')
    logger.info('Run the following to get things on CPU going: sbatch S02_Classify/classify-cpu.slurm')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', level=logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('filelock').setLevel(logging.WARNING)
    logger = logging.getLogger('classify')
    typer.run(create_sbatch_files)

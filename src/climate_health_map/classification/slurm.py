import logging
import os
from pathlib import Path
from typing import Annotated, Any

import typer

from climate_health_map.shared import get_logger
from climate_health_map.data import Group, get_filtered_labels
from climate_health_map.shared.types import OnConflict
from .util import ensure_offline_nltk, ensure_offline_transformers, MODELS_TRANS, MODELS_TRAD

logger = logging.getLogger('slurm-prep')
app = typer.Typer(no_args_is_help=True)


def _compile_sbatch_script(
    slurm_params: dict[str, Any],
    script_params: dict[str, Any],
    venv_path: Path,
    models_path: Path,
    array: list[str],
    params: list[str],
    command: str,
):
    awk_params = '\n'.join(
        [f"{param}=$(echo $param | awk -F'____' '{{print ${pi}}}')" for pi, param in enumerate(params, start=1)],
    )
    env_vars = '\n'.join([f'echo ${param}' for param in params])
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
source "{venv_path.resolve()}/bin/activate"

# Python env vars
export PYTHONPATH=$PYTHONPATH:{os.getcwd()}
export PYTHONUNBUFFERED=1

# Environment variables for script
export OPENBLAS_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OFFLINE_MODEL_PATH={models_path.resolve()}
export NLTK_DATA={models_path.resolve()}/nltk_data

echo "Using python from $(which python)"
echo "Python version is $(python --version)"

PARAMS=(
   {'\n   '.join(array)}
)

job=$(($SLURM_ARRAY_TASK_ID - 1))
param=${{PARAMS[$job]}}
{awk_params}

echo "array_task_id" $SLURM_ARRAY_TASK_ID " --> job" $job
{env_vars}

uv run --no-sources --extra classify --prerelease=allow healthmap {command} \\
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


def _compile_tuning_sbatch_script(
    slurm_params: dict[str, Any],
    script_params: dict[str, Any],
    schema: dict[str, Group],
    n_repeats: int,
    models: list[str],
    venv_path: Path,
    models_path: Path,
):
    array = [
        f'"{label.column}____{model}____{repeat}"'
        for group in schema.values()
        for label in group.labels
        for model in models
        for repeat in range(n_repeats)
        if (
            not (Path(script_params['target-dir']) / f'{model}-{label.parent}-{label.name}-{repeat}-pred.csv').exists()
            or script_params['on-exists'] != OnConflict.SKIP.value
        )
    ]
    logger.info(f'Number of array jobs: {len(array)}')
    return _compile_sbatch_script(
        slurm_params=slurm_params,
        script_params=script_params,
        venv_path=venv_path,
        models_path=models_path,
        array=array,
        params=['label', 'model', 'repeat'],
        command='tune',
    )


def _ensure_directories(venv_path: Path, log_path: Path, training_data: Path, target_dir: Path, models_path: Path | None = None) -> None:
    for info, path in [
        ('virtual environment', venv_path),
        ('training data', training_data),
    ]:
        path = path.absolute().resolve()
        if not path.exists():
            raise FileNotFoundError(f'Path for {info} does not exist at {path}')
        logger.info(f'Will use {info} at {path}')

    for info, path in [
        ('logging directory', log_path),
        ('model directory', models_path),
        ('target directory', target_dir),
    ]:
        if path is None:
            continue
        path = path.absolute().resolve()
        logger.info(f'Will use {info} at {path}')
        path.mkdir(parents=True, exist_ok=True)


@app.command('slurm-tune-scripts', help='Write slurm sbatch script for properly submitting job arrays')
def prepare_tuning_slurm(
    training_data: Annotated[Path, typer.Option(help='')],
    target_dir: Annotated[Path, typer.Option(help='')],
    models_path: Annotated[Path, typer.Option(help='')],
    venv_path: Annotated[Path, typer.Option(help='')],
    log_path: Annotated[Path, typer.Option(help='')],
    slurm_user: Annotated[str, typer.Option(help='email address to notify when done')],
    num_repeats: Annotated[int, typer.Option(help='')] = 3,
    train_proportion: Annotated[float, typer.Option(help='')] = 0.85,
    max_vocab: Annotated[int, typer.Option(help='')] = 7500,
    max_ngram: Annotated[int, typer.Option(help='')] = 1,
    min_df: Annotated[int, typer.Option(help='')] = 3,
    max_df: Annotated[float, typer.Option(help='')] = 0.8,
    min_minor_class: Annotated[int, typer.Option(help='')] = 20,
    random_state: Annotated[int | None, typer.Option(help='')] = None,
    slurm_hours: Annotated[int, typer.Option(help='')] = 2,
    tuning_trials_trad: Annotated[int | None, typer.Option(help='')] = None,
    tuning_trials_trans: Annotated[int | None, typer.Option(help='')] = None,
    on_exists: Annotated[OnConflict, typer.Option(help='')] = OnConflict.SKIP.value,
    ensure_models_offline: Annotated[bool, typer.Option(help='')] = True,
    loglevel: Annotated[str, typer.Option(help='Verbosity of logger')] = 'INFO',
):
    logger.info('Ensuring that all paths and files are in place...')
    _ensure_directories(venv_path=venv_path, log_path=log_path, models_path=models_path, target_dir=target_dir, training_data=training_data)

    if ensure_models_offline:
        logger.info('Making sure all models are available offline!')
        ensure_offline_transformers(model_data_path=models_path, logger=logger)

        logger.info('Making sure NLTK is available offline!')
        ensure_offline_nltk(target_dir=models_path / 'nltk_data', logger=logger)

    logger.info('Preparing basic script parameters...')
    sbatch_args = {
        'time': f'{slurm_hours:0>2}:00:00',
        'nodes': '1',
        'mem': '12G',
        'mail-user': slurm_user,
        'output': f'{log_path.resolve()}/%A_%a.out',
        'error': f'{log_path.resolve()}/%A_%a.err',
        'chdir': os.getcwd(),
    }
    script_args = {
        'training-data': training_data.resolve(),
        'target-dir': target_dir.resolve(),
        'train-proportion': train_proportion,
        'column': '${label}',
        'repeat': '${repeat}',
        'model': '${model}',
        'on-exists': on_exists.value,
        'loglevel': loglevel,
    }
    if random_state is not None:
        script_args['random-state'] = random_state

    logger.info(f'Filtering labels/schema based on available columns in the dataset at {training_data}')
    schema = get_filtered_labels(dataset_path=training_data, min_minor_class=min_minor_class)

    logger.info('Compiling sbatch script for transformer model tuning...')

    if tuning_trials_trans is not None:
        script_args['n-tuning-trials'] = tuning_trials_trans
    sbatch_trans = _compile_tuning_sbatch_script(
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
        models=MODELS_TRANS,
        venv_path=venv_path,
        models_path=models_path,
        schema=schema,
    )

    fn_slurm_trans = 'tune-trans.slurm'
    logger.info(f'Writing transformer tuning script as `{fn_slurm_trans}`')
    with open(fn_slurm_trans, 'w') as slurm_file:
        slurm_file.write(sbatch_trans)
    # Make sure the tuning trials are reset
    script_args.pop('n-tuning-trials', None)

    logger.info('Compiling sbatch script for traditional model tuning...')
    if tuning_trials_trad is not None:
        script_args['n-tuning-trials'] = tuning_trials_trad
    sbatch_trad = _compile_tuning_sbatch_script(
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
            'max-df': max_df,
            'n-tuning-jobs': 5,
        },
        n_repeats=num_repeats,
        models=MODELS_TRAD,
        venv_path=venv_path,
        models_path=models_path,
        schema=schema,
    )

    fn_slurm_trad = 'tune-trad.slurm'
    logger.info(f'Writing transformer tuning script as `{fn_slurm_trad}`')
    with open(fn_slurm_trad, 'w') as slurm_file:
        slurm_file.write(sbatch_trad)

    logger.info(f'Run the following to tune transformer models: sbatch {fn_slurm_trans}')
    logger.info(f'Run the following to tune traditional models: sbatch {fn_slurm_trad}')


if __name__ == '__main__':
    logger = get_logger('slurm-prep', run_log_init=True, loglevel='DEBUG')
    app()

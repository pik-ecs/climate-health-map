import json
import logging
import re
from pathlib import Path
from typing import Annotated

import typer
import pandas as pd

from climate_health_map.shared.types import OnConflict

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger('classify')


def predict_all(
    training_data: Annotated[Path, typer.Option(help='Path to csv file with training data')],
    target_dir: Annotated[Path, typer.Option(help='Path to output new directory')],
    reference_dir: Annotated[Path, typer.Option(help='Path to old output directory to sample params from')],
    min_f1: Annotated[float, typer.Option(help='Minimum F1 score to load')] = 0.6,
    on_exists: Annotated[OnConflict, typer.Option(help='')] = OnConflict.IGNORE,
):
    logger.info(f'Reading model parameters from {reference_dir}')
    data = []
    for fn in reference_dir.glob('predictions/*.json'):
        model, parent, label, repeat = re.fullmatch(r'(.+?)_(.+?)_(.+?)_(\d)\.json', fn.name).groups()
        repeat = int(repeat)
        info = json.load(open(fn))

        data.append(
            {
                'f1': info['f1'],
                'precision': info['precision'],
                'recall': info['recall'],
                'model': model,
                'parent': parent,
                'label': label,
                'pl': f'{parent}->{label}',
                'repeat': repeat,
                'fn': fn,
            },
        )
    df_res = pd.DataFrame(data)
    logger.info(f'Found {df_res.shape} setups')

    best_models = df_res.loc[df_res.groupby('label')['f1'].idxmax()]
    for _, row in best_models.iterrows():
        logger.info(f'Loading scores from best model ({row["model"]}) for "{row["label"]}" (F1: {row["f1"]})')
        if min_f1 is not None and row['f1'] < min_f1:
            logger.warning(f'  -> Not including column "{row["label"]}"')
            continue

        logger.info(f'Using following settings for {row["label"]}: {row.to_dict()}')
        try:
            # TODO
            pass
            # train(
            #     training_data=training_data,
            #     target_dir=target_dir,
            #     model=[row['model']],
            #     model_params=row['fn'],
            #     column=row['label'],
            #     on_exists=on_exists,
            #     n_tuning_trials=0,
            # )
        except Exception as e:
            logger.error(f'Issue training for {row["label"]}  -> {e}')
            logger.exception(e)
            logger.warning('  -> Ignoring and continuing.')

import json
from pathlib import Path
from typing import Annotated

import typer
import pandas as pd

from climate_health_map.shared.env import get_logger

pd.options.display.max_columns = 650
pd.options.display.max_rows = 200
pd.options.display.width = 100000


def main(
    source: Annotated[Path, typer.Option(help='Path to folder containing training stats')],
    target: Annotated[Path, typer.Option(help='Path to folder to write quality summary to')],
):
    logger = get_logger('classify-train', loglevel='DEBUG', run_log_init=True)
    target.mkdir(parents=True, exist_ok=True)
    folds = []
    for file in source.glob('*/*.json'):
        logger.debug(f'Reading stats from {file}')
        obj = json.load(open(file))
        params = obj.pop('params')
        tuning = obj.pop('tuning_quality')
        obj_folds = obj.pop('folds')
        base = obj | {f'tuning_{k}': v for k, v in tuning.items()} | {'params': params}
        for fi, fold in enumerate(obj_folds):
            folds.append(fold | base | {'fold': fi})
    df = pd.DataFrame(folds)
    columns = df['column'].unique()
    logger.info(f'Found {len(df)} folds for {len(columns)} columns: {list(columns)}')

    df.to_csv(target / 'complete.csv', index=False)
    df.groupby('column').describe().to_csv(target / 'summary.csv')


if __name__ == '__main__':
    typer.run(main)

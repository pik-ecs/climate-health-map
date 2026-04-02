import json
from pathlib import Path
from typing import Annotated

import typer
import pandas as pd

from climate_health_map.shared.env import get_logger
from climate_health_map.data.labels import LABELS

pd.options.display.max_columns = 650
pd.options.display.max_rows = 200
pd.options.display.width = 100000


def pretty_table(
    source_annotations: Annotated[Path, typer.Option(help='Path to folder containing training stats')] = 'data/exports/annotations_20260213.csv',
    source_quality: Annotated[Path, typer.Option(help='Path to folder containing training stats')] = 'data/quality/training/summary.csv',
    source_tuning: Annotated[Path, typer.Option(help='Path to folder containing training stats')] = 'data/quality/tuning/column_best.csv',
    target: Annotated[Path, typer.Option(help='Path to folder to write quality summary to')] = 'data/quality/summary.xlsx',
):
    logger = get_logger('classify-train', loglevel='INFO', run_log_init=True)
    df_anno = pd.read_csv(source_annotations)
    df_qual = pd.read_csv(source_quality, header=[0, 1], index_col=0)
    df_tune = pd.read_csv(source_tuning, header=[0, 1], index_col=0)

    rows = {}
    for key in ['attr', 'cat', 'driver', 'event', 'expose', 'health', 'rel_impacts', 'rel_major', 'type']:
        group = LABELS[key]
        logger.info(f'Preparing data for {key}: {group.name}')
        for label in group.labels:
            if label.column not in df_anno.columns or label.column not in df_qual.index:
                logger.warning(f'skip {label.column}')
                continue

            rows[f'{group.name}:\n{label.name}\n(yes: {(df_anno[label.column] == 1).sum():,}, no: {(df_anno[label.column] == 0).sum():,})'] = {
                'Classifier': df_tune.loc[label.column, ('model', 'Unnamed: 1_level_1')],
                'Precision': f'{df_qual.loc[label.column, ("precision_test", "mean")]:.0%} (σ={df_qual.loc[label.column, ("precision_test", "std")]:.0%})',
                'Recall': f'{df_qual.loc[label.column, ("recall_test", "mean")]:.0%} (σ={df_qual.loc[label.column, ("recall_test", "std")]:.0%})',
                'F1-score': f'{df_qual.loc[label.column, ("f1_test", "mean")]:.0%} (σ={df_qual.loc[label.column, ("f1_test", "std")]:.0%})',
            }
    pd.DataFrame(rows).T.to_excel(target)


def main(
    source: Annotated[Path, typer.Option(help='Path to folder containing training stats')],
    target: Annotated[Path, typer.Option(help='Path to folder to write quality summary to')],
) -> None:
    logger = get_logger('classify-train', loglevel='DEBUG', run_log_init=True)
    target.mkdir(parents=True, exist_ok=True)
    folds = []
    for file in source.glob('*/stats.json'):
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
    typer.run(pretty_table)

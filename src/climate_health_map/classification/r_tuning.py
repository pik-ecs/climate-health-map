from pathlib import Path
from typing import Annotated

import typer
from sklearn.model_selection import train_test_split

from climate_health_map.classification.c_iterator import it_models
from climate_health_map.classification.util import get_prediction_stats
from climate_health_map.data.dataset import Dataset
from climate_health_map.shared.encoder import json_dumps
from climate_health_map.shared.types import OnConflict
from climate_health_map.data.labels import LABELS, LABELS_LOOKUP
from climate_health_map.shared.env import get_logger


def tune(
    training_data: Annotated[Path, typer.Option(help='Path to csv file with training data')],
    target_dir: Annotated[Path, typer.Option(help='Path to output directory')],
    column: Annotated[str, typer.Option(help='Label/classifier to train (column name)')],
    model: Annotated[list[str] | None, typer.Option(help='Selection of models to tune on this data')] = None,
    train_proportion: Annotated[float, typer.Option(help='Proportion of data to be used for training')] = 0.8,
    random_state: Annotated[int, typer.Option(help='Seed for random processes')] = 43,
    repeat: Annotated[int, typer.Option(help='It is assumed that this method is called several times; this is to indicate which iteration this is.')] = 1,
    n_tuning_trials: Annotated[int | None, typer.Option(help='Number of optima hyper-parameter tuning trials')] = None,
    n_tuning_jobs: Annotated[int | None, typer.Option(help='Number of parallel optima hyper-parameter tuning jobs')] = None,
    max_vocab: Annotated[int, typer.Option(help='Maximum vocab size (only for sparse representations)')] = 7500,
    max_ngram: Annotated[int, typer.Option(help='n-gram range (only for sparse representations)')] = 1,
    min_df: Annotated[int, typer.Option(help='Minimum document frequency (only for sparse representations)')] = 3,
    max_df: Annotated[float, typer.Option(help='Minimum document frequency (only for sparse representations)')] = 0.8,
    min_minor_class: Annotated[int, typer.Option(help='Minimum number of samples for the under-represented class')] = 20,
    on_exists: Annotated[OnConflict, typer.Option(help='How to react when the expected output file already exists in `target_dir`')] = OnConflict.IGNORE,
    on_uneligible: Annotated[OnConflict, typer.Option(help='How to react when the column has not enough data to tune a model')] = OnConflict.IGNORE,
    loglevel: Annotated[str, typer.Option(help='Verbosity of logger')] = 'INFO',
) -> None:
    logger = get_logger('classify-train', loglevel=loglevel, run_log_init=True)
    dataset = Dataset(dataset_path=training_data, logger=logger)

    target_dir.mkdir(parents=True, exist_ok=True)

    label = LABELS_LOOKUP[column]
    logger.info(f'Label "{column}" is in group "{label.parent}": {LABELS[label.parent]}')

    if not dataset.is_label_eligible(label=label, min_minor_class=min_minor_class):
        if on_uneligible == OnConflict.BREAK:
            raise AssertionError(f'Label "{column}" is not eligible for training.')

        logger.warning(f'Label "{column}" is not eligible for training; ending process without error (on_uneligible={on_uneligible}).')
        return None

    mask = dataset.get_mask(column=column)
    data = dataset.get_simplified_df(column=column, mask=mask)

    logger.info(f'Prepared labeled dataset with {data.shape} records, of which {data["label"].sum()} are >=1')

    idxs = list(data.index)
    train_idxs, test_idxs = train_test_split(
        idxs,
        train_size=train_proportion,
        stratify=data['label'],
        random_state=random_state * repeat,
    )

    for model_name, classifier in it_models(
        dataset=data,
        models=model,
        min_df=min_df,
        max_ngram=max_ngram,
        max_vocab=max_vocab,
        n_tuning_jobs=n_tuning_jobs,
        n_tuning_trials=n_tuning_trials,
        logger=logger,
    ):
        info_file = target_dir / f'{model_name}_{label.parent}_{column}_{repeat}.json'
        test_file = target_dir / f'{model_name}_{label.parent}_{column}_{repeat}_val.csv'

        if info_file.exists() and on_exists == OnConflict.BREAK:
            raise FileExistsError(f'File {info_file} already exists')
        if info_file.exists() and on_exists == OnConflict.SKIP:
            logger.warning(f'File {info_file} already exists; skipping.')
            continue

        logger.info(f'Initializing and training ranker {model_name} for column {column} (run: {repeat})')
        classifier.train(idxs=train_idxs)

        logger.info(f'Making predictions for "{column}" on all annotated data')
        y_pred = classifier.predict()
        logger.debug(f'Testing ranker {model_name} for column "{column}" on {1 - train_proportion:.0%} test set')
        ds, stats = get_prediction_stats(dataset=data, y_pred=y_pred, test_idxs=test_idxs, train_idxs=train_idxs)
        logger.debug(f'Stats for {model_name} ({column}): {stats}')

        logger.info(f'Writing info to {info_file}')
        json_dumps(
            info_file,
            {
                'params': classifier.get_params(),
                'is_tuned': True,
                'repeat': repeat,
                'random_state': random_state * repeat,
                'train_proportion': train_proportion,
                'column': column,
                'label_name': label.name,
                'label_parent': label.parent,
                'model': model_name,
            }
            | stats,
            indent=2,
        )

        logger.info(f'Writing test predictions to {test_file}')
        ds.to_csv(test_file)
        del ds

        logger.info(f'Finished with model {model_name} for column {column} (run: {repeat})')

    logger.info('Done with all!')
    return None

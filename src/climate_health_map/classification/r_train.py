from pathlib import Path
from typing import Annotated

import typer
from sklearn.model_selection import StratifiedKFold

from climate_health_map.classification.c_iterator import get_model
from climate_health_map.classification.util import get_prediction_stats, read_tuning_info
from climate_health_map.data.dataset import Dataset, downsampling_mask
from climate_health_map.shared.encoder import json_dump
from climate_health_map.shared.types import OnConflict
from climate_health_map.data.labels import LABELS, LABELS_LOOKUP
from climate_health_map.shared.env import get_logger


def train(
    training_data: Annotated[Path, typer.Option(help='Path to csv file with training data')],
    tuning_dir: Annotated[Path, typer.Option(help='Path to directory containing all the tuning outputs')],
    output_dir: Annotated[Path, typer.Option(help='Path to output directory')],
    column: Annotated[str, typer.Option(help='Label/classifier to train (column name)')],
    random_seed: Annotated[int | None, typer.Option(help='Random seed (optional)')] = None,
    n_folds: Annotated[int, typer.Option(help='Number of folds in k-fold validation')] = 10,
    min_n_majority: Annotated[int, typer.Option(help='Minimum number of majority class to keep when downsampling')] = 20,
    on_exists: Annotated[OnConflict, typer.Option(help='How to react when the expected output file already exists in `target_dir`')] = OnConflict.IGNORE,
    loglevel: Annotated[str, typer.Option(help='Verbosity of logger')] = 'INFO',
) -> None:
    logger = get_logger('classify-train', loglevel=loglevel, run_log_init=True)

    target_dir = (output_dir / column).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Will write trained model to {target_dir}')
    stats_file = target_dir / 'stats.json'
    model_target = target_dir / 'model'

    if stats_file.exists():
        logger.warning(f'Output directory {target_dir} already exists.')
        if on_exists == OnConflict.BREAK:
            raise FileExistsError(f'Output directory {target_dir} already exists.')
        if on_exists == OnConflict.SKIP:
            return

    df_tuning = read_tuning_info(tuning_dir)
    if column not in df_tuning.index:
        raise KeyError(f'No tuning data available for column "{column}" in tuning directory {tuning_dir}')
    best = df_tuning.loc[column].sort_values(by=['f1_test']).iloc[0]
    info = best['params']
    model_params = info['hyperparams']
    downsampling = model_params.pop('downsampling', None)
    vectoriser_info = info['vectoriser'] if 'vectoriser' in info and info['vectoriser'] is not None else {}
    logger.info(
        f'When tuning, F1 {best["f1_test"]} ({best["f1_train"]} on training data) '
        f'for model "{best["model"]}" ({best["repeat"]}) with {downsampling} downsampling',
    )

    # Prepare return info
    stats = {
        'label_name': best['label_name'],
        'label_parent': best['label_parent'],
        'model': best['model'],
        'column': column,
        'params': info,
        'random_state': random_seed,
        'tuning_quality': {
            'precision_test': best['precision_test'],
            'recall_test': best['recall_test'],
            'f1_test': best['f1_test'],
            'precision_train': best['precision_train'],
            'recall_train': best['recall_train'],
            'f1_train': best['f1_train'],
        },
        'folds': [],
    }

    label = LABELS_LOOKUP[column]
    logger.info(f'Label "{column}" is in group "{label.parent}": {LABELS[label.parent]}')

    logger.info(f'Loading training data from {training_data}')
    dataset = Dataset(dataset_path=training_data, logger=logger)
    df = dataset.get_simplified_df(column=column)
    mask_column = dataset.get_mask(column, ensure_text=True)
    y = df[mask_column]['label']

    folding = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    for i, (train_idxs, test_idxs) in enumerate(folding.split(None, y)):
        logger.info(f'Executing evaluation for fold {i + 1}/{n_folds}')

        # Prepare downsampled training data indexes
        mask_sampling = downsampling_mask(y=y[train_idxs], sampling=downsampling, min_n_majority=min_n_majority)
        train_idxs_sampled = train_idxs[mask_sampling]

        logger.info('Preparing model...')
        classifier = get_model(
            dataset=df,
            model=best['model'],
            max_df=vectoriser_info.get('max_df', 0.8),
            min_df=vectoriser_info.get('min_df', 4),
            max_vocab=vectoriser_info.get('max_features', 10000),
            max_ngram=vectoriser_info.get('ngram_range', [0, 3])[1],
            model_params=model_params,
            logger=logger.getChild(f'fold-{i}'),
        )

        logger.info('Training model...')
        classifier.train(train_idxs_sampled)

        logger.info(f'Evaluating model at fold {i + 1}/{n_folds}')
        y_pred = classifier.predict()
        _ds, fold_stats = get_prediction_stats(dataset=df, y_pred=y_pred, test_idxs=test_idxs, train_idxs=train_idxs)
        stats['folds'].append(fold_stats)
        logger.debug(f'Stats for fold {i + 1}/{n_folds}: {fold_stats}')

        # Clean up training fold
        del classifier
        del _ds

    logger.info(f'Finished all {n_folds} evaluation folds.')

    # Final training with full dataset
    logger.info('Preparing final model...')
    classifier = get_model(
        dataset=df,
        model=best['model'],
        max_df=vectoriser_info.get('max_df', 0.8),
        min_df=vectoriser_info.get('min_df', 4),
        max_vocab=vectoriser_info.get('max_features', 10000),
        max_ngram=vectoriser_info.get('ngram_range', [0, 3])[1],
        model_params=model_params,
        logger=logger.getChild('total'),
    )

    logger.info('Training final model on full dataset...')
    classifier.train(df.index)

    logger.info(f'Write trained model to {model_target}')
    classifier.store(target=model_target)

    logger.info(f'Writing k-fold statistics to {stats_file}')
    json_dump(stats_file, stats)

    logger.info('All done!')

from joblib import load
from pathlib import Path
from typing import Annotated

import typer
from tqdm import tqdm
from sklearn.pipeline import Pipeline

from climate_health_map.shared import read_any_pd
from climate_health_map.shared.text import text_from_table
from climate_health_map.shared.types import OnConflict


def classify(
    source: Annotated[Path, typer.Option(help='Path to source data to classify')],
    target: Annotated[Path, typer.Option(help='Path to write classifications to')],
    models_dir: Annotated[Path, typer.Option(help='Path to trained models (not the huggingface `OFFLINE_MODEL_PATH`!)')],
    on_exists: Annotated[OnConflict, typer.Option(help='How to behave when the target file already exists')] = OnConflict.IGNORE,
    run_major_incl: Annotated[bool, typer.Option(help='Run C&H inclusion classifier')] = True,
    run_major_categories: Annotated[bool, typer.Option(help='Run major category classifier (mitigation, adaptation, impacts)')] = True,
    run_impacts: Annotated[bool, typer.Option(help='Run impacts classifier')] = False,
    batch_size: Annotated[int, typer.Option(help='Processing batch size')] = 1000,
    loglevel: Annotated[str, typer.Option(help='Verbosity of logger')] = 'INFO',
) -> None:
    from climate_health_map import get_logger

    if run_impacts:
        raise NotImplementedError('Missing classifiers')

    logger = get_logger('classify', loglevel=loglevel, run_log_init=True)
    if target.exists() and on_exists == OnConflict.BREAK:
        raise FileExistsError(f'Output already exists at {target}')
    if target.exists() and on_exists == OnConflict.SKIP:
        logger.warning(f'Output already exists at {target}')
        return

    logger.info(f'Reading data from {source.resolve()}')
    df_source = read_any_pd(source)

    logger.info(f'Loading models from {models_dir}')
    clf_major_rel: Pipeline | None = load(models_dir / 'binary_climate_health.pkl') if run_major_incl else None
    clf_major_cat: Pipeline | None = load(models_dir / 'multi_climate_health.pkl') if run_major_categories else None
    _clf_impacts: Pipeline | None = load(models_dir / 'multi_impacts.pkl') if run_impacts else None

    logger.info(f'Classifying {df_source.shape[0]:,} items...')
    progress = tqdm(total=df_source.shape[0] // batch_size)
    for idx_begin, idx_end in range(0, len(df_source), batch_size):
        batch = df_source.iloc[idx_begin:idx_end]
        texts = text_from_table(df=batch)
        predictions = {}
        if run_major_incl:
            progress.set_postfix_str('Applying major inclusion classifier...')
            y_pred = clf_major_rel.predict_proba(texts)
            predictions['rel_major|0'] = y_pred[:, 0]
            predictions['rel_major|1'] = y_pred[:, 1]

        if run_major_categories:
            progress.set_postfix_str('Applying major categories classifier...')
            y_pred = clf_major_cat.predict_proba(texts)
            predictions['cat|0'] = y_pred[:, 0]
            predictions['cat|1'] = y_pred[:, 1]
            predictions['cat|2'] = y_pred[:, 2]

        if run_impacts:
            # TODO
            raise NotImplementedError('')
        progress.update(idx_end - idx_begin)
    progress.close()

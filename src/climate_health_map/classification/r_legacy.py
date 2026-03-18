import json
import logging
import os

from joblib import load
from pathlib import Path
from typing import Annotated

import typer
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline

from climate_health_map.shared import read_any_pd, write_any_df
from climate_health_map.shared.text import text_from_table
from climate_health_map.shared.types import OnConflict

LABEL_MAPPING = {
    'impacts_attr': {  # TODO: double check the numbers match
        'LABEL_0': 'attr|0',
        'LABEL_1': 'attr|1',
        'LABEL_2': 'attr|2',
        'LABEL_3': 'attr|3',
        'LABEL_4': 'attr|4',
    },
    'impacts_driver': {  # TODO: double check the numbers match
        'LABEL_0': 'driver|0',
        'LABEL_1': 'driver|1',
        'LABEL_2': 'driver|2',
        'LABEL_3': 'driver|3',
        'LABEL_4': 'driver|4',
        'LABEL_5': 'driver|5',
        'LABEL_6': 'driver|6',
    },
    'impacts_event': {  # TODO: double check the numbers match
        'LABEL_0': 'event|0',
        'LABEL_1': 'event|1',
        'LABEL_2': 'event|2',
        'LABEL_3': 'event|3',
        'LABEL_4': 'event|4',
        'LABEL_5': 'event|5',
        'LABEL_6': 'event|6',
    },
    'impacts_rel': {  # no label map
        '0': 'rel_impacts|1',  # FIXME
    },
}


def classify_transformer(data: pd.Series, label_map: dict[str, str], model_dir: Path, cache_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    import torch
    from transformers import TextClassificationPipeline, AutoTokenizer, AutoModelForSequenceClassification

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info('Loading model...')
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, ignore_mismatched_sizes=True)

    with open(model_dir / 'config.json', 'r') as f:
        info = json.load(f)

    logger.info('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(info['_name_or_path'], model_max_length=512, cache_dir=cache_dir)  # type: ignore[no-untyped-call]

    logger.info('Constructing pipe')
    pipe = TextClassificationPipeline(  # type: ignore[no-untyped-call]
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        top_k=None,
        function_to_apply='softmax',
        device=device,
    )

    logger.info('Constructing label->column lookup')
    logger.warning(f'Label mismatch: {set(model.config.label2id) - set(label_map)}  (OK when empty; shows labels in HF model missing in configured labels)')
    logger.warning(f'Label mismatch: {set(label_map) - set(model.config.label2id)}  (OK when empty; shows configured labels missing in HF model -> BAD!)')

    logger.info('Classifying in batches')
    y_pred: list[list[dict[str, str | float]]] = pipe(data.tolist(), batch_size=32)  # type:ignore[assignment]
    return pd.DataFrame(
        [
            {'item_id': idx} | {label_map[lab['label']]: lab['score'] for lab in pred if lab['label'] in label_map}  # type: ignore[index]
            for idx, pred in zip(data.index, y_pred, strict=True)
        ],
    ).set_index('item_id')


def classify(
    source: Annotated[Path, typer.Option(help='Path to source data to classify')],
    target: Annotated[Path, typer.Option(help='Path to write classifications to')],
    models_dir: Annotated[Path, typer.Option(help='Path to trained models (not the huggingface `OFFLINE_MODEL_PATH`!)')],
    cache_dir: Annotated[Path | None, typer.Option(help='Optional parameter to overwrite `OFFLINE_MODEL_PATH`')] = None,
    on_exists: Annotated[OnConflict, typer.Option(help='How to behave when the target file already exists')] = OnConflict.IGNORE,
    run_major_incl: Annotated[bool, typer.Option(help='Run C&H inclusion classifier')] = True,
    run_major_categories: Annotated[bool, typer.Option(help='Run major category classifier (mitigation, adaptation, impacts)')] = True,
    run_impacts: Annotated[bool, typer.Option(help='Run impacts classifier')] = False,
    batch_size: Annotated[int, typer.Option(help='Processing batch size')] = 1000,
    loglevel: Annotated[str, typer.Option(help='Verbosity of logger')] = 'INFO',
) -> None:
    from climate_health_map import get_logger

    logger = get_logger('classify', loglevel=loglevel, run_log_init=True)
    if target.exists() and on_exists == OnConflict.BREAK:
        raise FileExistsError(f'Output already exists at {target}')
    if target.exists() and on_exists == OnConflict.SKIP:
        logger.warning(f'Output already exists at {target}')
        return
    models_dir = models_dir.resolve()
    if cache_dir is None:
        cache_dir_ = os.getenv('OFFLINE_MODEL_PATH')
        if not cache_dir_:
            raise RuntimeError('Offline model path not set')
        cache_dir = Path(cache_dir_)

    logger.info(f'Reading data from {source.resolve()}')
    df_source = read_any_pd(source)

    logger.info(f'Loading models from {models_dir}')
    clf_major_rel: Pipeline | None = load(models_dir / 'binary_climate_health.pkl') if run_major_incl else None
    clf_major_cat: Pipeline | None = load(models_dir / 'multi_climate_health.pkl') if run_major_categories else None

    logger.info(f'Classifying {df_source.shape[0]:,} items...')
    progress = tqdm(total=df_source.shape[0] // batch_size)
    df_predictions = pd.DataFrame()
    for idx_begin in range(0, len(df_source), batch_size):
        batch = df_source.iloc[idx_begin : idx_begin + batch_size]
        predictions = {'item_id': batch['item_id'].tolist()}
        texts = text_from_table(df=batch)
        if run_major_incl:
            progress.set_postfix_str('Applying major inclusion classifier...')
            y_pred = clf_major_rel.predict_proba(texts)  # type: ignore[union-attr]
            predictions['rel_major|0'] = y_pred[:, 0]
            predictions['rel_major|1'] = y_pred[:, 1]

        if run_major_categories:
            progress.set_postfix_str('Applying major categories classifier...')
            y_pred = clf_major_cat.predict_proba(texts)  # type: ignore[union-attr]
            # old classifier: AMI
            # new scheme: MAI
            predictions['cat|0'] = y_pred[:, 1]
            predictions['cat|1'] = y_pred[:, 0]
            predictions['cat|2'] = y_pred[:, 2]

        df_predictions = pd.concat([df_predictions, pd.DataFrame(predictions).set_index('item_id')])
        progress.update(len(batch))
    progress.close()

    if run_impacts:
        for key, mapping in LABEL_MAPPING.items():
            logger.info(f'Applying {key} classifier...')
            df_pred = classify_transformer(text_from_table(df=df_source), mapping, model_dir=models_dir / key, cache_dir=cache_dir, logger=logger)
            df_predictions = df_predictions.merge(df_pred, how='outer')

    write_any_df(df_predictions, target=target, index=True)

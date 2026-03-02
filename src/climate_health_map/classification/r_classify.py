import json
import logging
import os
from pathlib import Path
from typing import Annotated

import typer
import numpy as np
import pandas as pd

from climate_health_map.shared import read_any_pd, write_any_df
from climate_health_map.shared.types import OnConflict
from climate_health_map.data.labels import Label, LABELS, HfGroup


def classify_huggingface(data: pd.DataFrame, group: HfGroup, cache_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    import torch
    from transformers import TextClassificationPipeline, AutoTokenizer, AutoModelForSequenceClassification

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info('Loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(group.token_model, model_max_length=512, cache_dir=cache_dir)  # type: ignore[no-untyped-call]
    logger.info('Loading model')
    model = AutoModelForSequenceClassification.from_pretrained(group.model)
    logger.info('Constructing pipe')
    pipe = TextClassificationPipeline(  # type: ignore[no-untyped-call]
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        top_k=None,
        function_to_apply=group.normalisation,
        device=device,
    )

    logger.info('Constructing label->column lookup')
    label_map = {label.hf_name: label.column for label in group.labels}
    logger.warning(f'Label mismatch: {set(model.config.label2id) - set(label_map)}  (OK when empty; shows labels in HF model missing in configured labels)')
    logger.warning(f'Label mismatch: {set(label_map) - set(model.config.label2id)}  (OK when empty; shows configured labels missing in HF model -> BAD!)')

    logger.info('Classifying in batches')
    y_pred: list[list[dict[str, str | float]]] = pipe(data['text'].tolist(), batch_size=32)  # type:ignore[assignment]
    return pd.DataFrame(
        [
            {'item_id': idx} | {label_map[lab['label']]: lab['score'] for lab in pred if lab['label'] in label_map}
            for idx, pred in zip(data.index, y_pred, strict=True)
        ]
    ).set_index('item_id')


def classify_local(texts: list[str], label: Label, models_dir: Path, on_missing_model: OnConflict, logger: logging.Logger) -> np.ndarray | None:
    logger.info('Assuming this is a local model, checking...')
    model_dir = models_dir / f'{label.column}/model'
    stats_file = model_dir / f'{label.column}/stats.json'
    if not stats_file.exists() or not model_dir.exists():
        logger.warning(f'Could not find the expected stats file at {stats_file}')
        if on_missing_model == OnConflict.BREAK:
            logger.warning(f'... escalating to parent.')
            raise FileNotFoundError(f'Could not find a model for "{label.column}" in {models_dir}')
        else:
            logger.warning(f'... but ignoring it.')
            return None

    logger.info(f'Reading stats file of pre-trained model from {stats_file}')
    with open(stats_file) as f_in:
        info = json.load(f_in)

    logger.info('Importing model dependencies...')
    from .c_transformer import MODELS_TRANS
    from .c_traditional import MODELS_TRAD

    models = MODELS_TRANS | MODELS_TRAD
    if info['model'] not in models:
        raise KeyError(f'The model `{info["model"]}` is not known!')

    logger.info(f'Loading model for "{info['model']} from {models_dir}')
    Model = models[info['model']]
    classifier = Model.load(model_dir)  # type:ignore[union-attr]

    logger.info(f'Predicting on {len(texts):,} texts...')
    return classifier.predict(texts=texts)


def classify(
    source: Annotated[Path, typer.Option(help='Path to source data to classify')],
    target: Annotated[Path, typer.Option(help='Path to write classifications to')],
    group: Annotated[str, typer.Option(help='Label/classifier to predict (group name)')],
    models_dir: Annotated[Path, typer.Option(help='Path to trained models (not the huggingface `OFFLINE_MODEL_PATH`!)')],
    cache_dir: Annotated[Path | None, typer.Option(help='Optional to override OFFLINE_MODEL_PATH')] = None,
    on_exists: Annotated[OnConflict, typer.Option(help='How to behave when the target file already exists')] = OnConflict.IGNORE,
    on_missing_model: Annotated[OnConflict, typer.Option(help="How to behave when we don't have a model for this column")] = OnConflict.IGNORE,
    loglevel: Annotated[str, typer.Option(help='Verbosity of logger')] = 'INFO',
) -> None:
    from climate_health_map import get_logger

    logger = get_logger('classify', loglevel=loglevel, run_log_init=True)
    label_group = LABELS[group]
    logger.info(f'Found requested label group: {label_group}')

    if target.exists() and on_exists == OnConflict.IGNORE:
        raise FileExistsError(f'Output already exists at {target}')
    if target.exists() and on_exists == OnConflict.SKIP:
        logger.warning(f'Output already exists at {target}')
        return
    if cache_dir is None:
        cd = os.getenv('OFFLINE_MODEL_PATH')
        if cd is None:
            raise AssertionError('You should either set the `OFFLINE_MODEL_PATH` environment variable or the `--cache-dir` argument')
        cache_dir = Path(cd)

    logger.info(f'Reading data from {source.resolve()}')
    df_source = read_any_pd(source)

    if 'item_id' in df_source.columns:
        logger.info('Setting `item_id` column as index')
        df_source.set_index('item_id', inplace=True, drop=False)
    if 'text' not in df_source.columns:
        logger.info('Adding virtual text column...')
        df_source['text'] = df_source.apply(lambda row: f'{row["title"] or ""} {row["abstract"] or ""}', axis=1)

    if type(label_group) is HfGroup:
        logger.info('Going to use huggingface model for classification.')
        df_res = classify_huggingface(df_source, group=label_group, cache_dir=cache_dir, logger=logger)
    else:
        logger.info('Will iterate internal models for classification.')
        results = {'item_id': df_source.index}
        texts = df_source['text'].to_list()
        for label in label_group.labels:
            logger.info(f'Classifying {label.column} ({label.name})')
            results[label.column] = classify_local(texts=texts, label=label, models_dir=models_dir, on_missing_model=on_missing_model, logger=logger)
        logger.info('Packaging classifications into dataframe...')
        df_res = pd.DataFrame.from_dict(results).set_index('item_id')

    logger.info(f'Constructed predictions dataframe of shape {df_res.shape}')

    logger.info(f'Writing classifications to {target.resolve()}')
    write_any_df(df_res.reset_index(), target=target, kwargs={'index': False})
    logger.info(f'All done.')


if __name__ == '__main__':
    typer.run(classify)

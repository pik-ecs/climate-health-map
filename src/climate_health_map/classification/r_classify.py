import json
import logging
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
import pandas as pd

from climate_health_map import get_logger
from climate_health_map.shared.types import OnConflict
from climate_health_map.data.labels import Label, LABELS, HfGroup
from .c_transformer import MODELS_TRANS
from .c_traditional import MODELS_TRAD


def classify_huggingface(data: pd.DataFrame, group: HfGroup, logger: logging.Logger) -> pd.DataFrame:
    from transformers import TextClassificationPipeline, AutoTokenizer, AutoModelForSequenceClassification

    logger.info('Loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(group.token_model, model_max_length=512)
    logger.info('Loading model')
    model = AutoModelForSequenceClassification.from_pretrained(group.model)
    logger.info('Constructing pipe')
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, truncation=True, top_k=None, function_to_apply='softmax')
    # TODO: model to gpu
    _res = pipe('Carbon pricing is thought of as an economically efficient policy to reduce GHG emissions')  # FIXME
    # TODO: result to cpu
    # TODO: construct dataframe with id index from input and column from group


def classify_local(texts: list[str], label: Label, models_dir: Path, on_missing_model: OnConflict, logger: logging.Logger) -> np.ndarray | None:
    logger.info('Assuming this is a local model, checking...')
    model_dir = models_dir / f'{label.column}/model'
    stats_file = model_dir / f'{label.column}/stats.json'
    if not stats_file.exists() or not model_dir.exists():
        if on_missing_model == OnConflict.BREAK:
            raise FileNotFoundError(f'Could not find a model for "{label.column}" in {models_dir}')
        else:
            return None

    with open(stats_file) as f_in:
        info = json.load(f_in)

    models = MODELS_TRANS | MODELS_TRAD
    if info['model'] not in models:
        raise KeyError(f'The model `{info["model"]}` is not known!')

    Model = models[info['model']]
    classifier = Model.load(model_dir)

    return classifier.predict(texts=texts)


def classify(
    source: Annotated[Path, typer.Option(help='Path to source data to classify')],
    target: Annotated[Path, typer.Option(help='Path to write classifications to')],
    group: Annotated[str, typer.Option(help='Label/classifier to predict (group name)')],
    models_dir: Annotated[Path, typer.Option(help='Path to trained models (not the huggingface `OFFLINE_MODEL_PATH`!)')],
    on_exists: Annotated[OnConflict, typer.Option(help='How to behave when the target file already exists')] = OnConflict.IGNORE,
    on_missing_model: Annotated[OnConflict, typer.Option(help="How to behave when we don't have a model for this column")] = OnConflict.IGNORE,
    loglevel: Annotated[str, typer.Option(help='Verbosity of logger')] = 'INFO',
):
    logger = get_logger('classify', loglevel=loglevel, run_log_init=True)
    label_group = LABELS[group]
    logger.info(f'Found requested label group: {label_group}')

    if target.exists() and on_exists == OnConflict.IGNORE:
        raise FileExistsError(f'Output already exists at {target}')
    if target.exists() and on_exists == OnConflict.SKIP:
        logger.warning(f'Output already exists at {target}')
        return

    df_source = pd.read_csv(source)
    if 'item_id' in df_source.columns:
        df_source.set_index('item_id', inplace=True)

    if type(label_group) is HfGroup:
        logger.info('Going to use huggingface model for classification.')
        df_res = classify_huggingface(None, group=label_group, logger=logger)
    else:
        logger.info('Will iterate internal models for classification.')
        results = {'item_id': df_source.index}
        texts = df_source['text'].to_list()
        for label in label_group.labels:
            results[label.column] = classify_local(texts=texts, label=label, models_dir=models_dir, on_missing_model=on_missing_model, logger=logger)
        df_res = pd.DataFrame.from_dict(results).set_index('item_id')

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix == '.csv':
        df_res.to_csv(target)
    elif target.suffix == '.feather':
        df_res.to_feather(target)
    else:
        raise ValueError(f'Unsupported output file type: {target.suffix}')
    logger.info(f'Wrote to {target}')


if __name__ == '__main__':
    typer.run(classify)

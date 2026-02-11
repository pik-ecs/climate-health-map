import json
import logging
import re
from pathlib import Path
from typing import Annotated, Generator, Any

import typer
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from S02_Classify.labels import FileConflict, COLUMN_LABELS, COLUMN_GROUPS

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger('classify')

app = typer.Typer()


def it_models(
    dataset: pd.DataFrame,
    models: list[str] | None = None,
    n_tuning_trials: int | None = None,
    n_tuning_jobs: int = 1,
    max_vocab: int = 7500,
    max_ngram: int = 1,
    min_df: int = 3,
    model_params: dict[str, Any] | None = None,
) -> Generator[tuple[str, Any], None, None]:
    if models is None:
        models = [
            'TINYBERT',
            'CLIMATEBERT',
        ]
    try:
        import torch

        memory = torch.cuda.get_device_properties(0).total_memory
        if memory > 24000000000:
            memory_factor = 2.0
        elif memory > 12000000000:  # 12 GB VRAM
            memory_factor = 1.2
        else:
            memory_factor = 1.0
    except:  # noqa: E722
        memory_factor = 1.0

    def cbert():
        from S02_Classify.transformer import TransRanker

        return TransRanker(
            dataset=dataset,
            model_params=model_params,
            min_batch_size=2,
            max_batch_size=int(6 * memory_factor),
            models=['climatebert/distilroberta-base-climate-f'],
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 20,
            test_split=0.1,
        )

    def sbert():
        from S02_Classify.transformer import TransRanker

        return TransRanker(
            dataset=dataset,
            model_params=model_params,
            min_batch_size=2,
            max_batch_size=int(6 * memory_factor),
            models=['allenai/scibert_scivocab_uncased'],
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 20,
            test_split=0.1,
        )

    def tbert():
        from S02_Classify.transformer import TransRanker

        return TransRanker(
            dataset=dataset,
            model_params=model_params,
            min_batch_size=2,
            max_batch_size=int(10 * memory_factor),
            models=['prajjwal1/bert-tiny'],
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 20,
            test_split=0.1,
        )

    def reg():
        from S02_Classify.basics import RegressionRanker

        return RegressionRanker(
            dataset=dataset,
            model_params=model_params,
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 50,
            n_jobs=n_tuning_jobs,
            max_features=max_vocab,
            ngram_range=(1, max_ngram),
            min_df=min_df,
        )

    def svm():
        from S02_Classify.basics import SVMRanker

        return SVMRanker(
            dataset=dataset,
            model_params=model_params,
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 50,
            n_jobs=n_tuning_jobs,
            max_features=max_vocab,
            ngram_range=(1, max_ngram),
            min_df=min_df,
        )

    def lgbm():
        from S02_Classify.basics import LightGBMRanker

        return LightGBMRanker(
            dataset=dataset,
            model_params=model_params,
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 50,
            n_jobs=n_tuning_jobs,
            max_features=max_vocab,
            ngram_range=(1, max_ngram),
            min_df=min_df,
        )

    def sgd():
        from S02_Classify.basics import SGDRanker

        return SGDRanker(
            dataset=dataset,
            model_params=model_params,
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 50,
            n_jobs=n_tuning_jobs,
            max_features=max_vocab,
            ngram_range=(1, max_ngram),
            min_df=min_df,
        )

    CONFIGS = {
        'CLIMATEBERT': cbert,
        'SCIBERT': sbert,
        'TINYBERT': tbert,
        # TODO: Maybe add 'malteos/scincl', 'distilbert-base',
        'REG': reg,
        'SVM': svm,
        'LGBM': lgbm,
        'SGD': sgd,
    }

    logger.info(f'Will iterate models: {models}')
    for model in models:
        if model in CONFIGS:
            yield model, CONFIGS[model]()


@app.command('apply')
def predict_all(
    training_data: Annotated[Path, typer.Option(help='Path to csv file with training data')],
    target_dir: Annotated[Path, typer.Option(help='Path to output new directory')],
    reference_dir: Annotated[Path, typer.Option(help='Path to old output directory to sample params from')],
    min_f1: Annotated[float, typer.Option(help='Minimum F1 score to load')] = 0.6,
    on_exists: Annotated[FileConflict, typer.Option(help='')] = FileConflict.IGNORE,
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
            train(
                training_data=training_data,
                target_dir=target_dir,
                model=[row['model']],
                model_params=row['fn'],
                column=row['label'],
                on_exists=on_exists,
                n_tuning_trials=0,
            )
        except Exception as e:
            logger.error(f'Issue training for {row["label"]}  -> {e}')
            logger.exception(e)
            logger.warning(f'  -> Ignoring and continuing.')


@app.command('single')
def train(
    training_data: Annotated[Path, typer.Option(help='Path to csv file with training data')],
    target_dir: Annotated[Path, typer.Option(help='Path to output directory')],
    column: Annotated[str, typer.Option(help='Label/classifier to train (column name)')],
    model: Annotated[list[str] | None, typer.Option(help='')] = None,
    model_params: Annotated[Path, typer.Option(help='To skip tuning, you can point to a file of a previous training cycle')] = None,
    train_proportion: Annotated[float, typer.Option(help='')] = 0.8,
    random_state: Annotated[int, typer.Option(help='')] = 43,
    repeat: Annotated[int, typer.Option(help='')] = 1,
    n_tuning_trials: Annotated[int | None, typer.Option(help='')] = None,
    n_tuning_jobs: Annotated[int | None, typer.Option(help='')] = None,
    max_vocab: Annotated[int, typer.Option(help='')] = 7500,
    max_ngram: Annotated[int, typer.Option(help='')] = 1,
    min_df: Annotated[int, typer.Option(help='')] = 3,
    on_exists: Annotated[FileConflict, typer.Option(help='')] = FileConflict.IGNORE,
):
    logger.info(f'Loading training data from {training_data}')
    df = pd.read_csv(training_data).replace({np.nan: None}).rename(columns={'Unnamed: 0': 'id'}).set_index('id', drop=False)
    logger.info(f'Loaded {df.shape} records')

    fn_source = (target_dir / 'dataset.csv').resolve()
    logger.info(f'Assuming data to classify in {fn_source}')
    if not fn_source.is_file():
        raise FileNotFoundError(fn_source)

    label = COLUMN_LABELS[column]
    parent = label.parent
    logger.info(f'Label "{column}" is in group "{parent}" -> {COLUMN_GROUPS[parent]}')

    # ,title,abstract,author_keywords,document_type,year,assessment|0,assessment|1,assessment|2,assessment|3,
    # context|0,context|1,context|2,context|3,context|4,context|5,
    # method|7,method|3,method|4,method|9,method|0,method|1,method|2,method|8,method|5,method|6,relevance|1,
    # technology|0,technology|1,technology|2,technology|3,technology|4,technology|5,technology|6,technology|7,technology|8,technology|9,
    # technology|16,technology|10,technology|11,technology|?,technology|13,technology|14,technology|15,technology|17,
    # technology|17-ids,technology|17-doi,technology|18,technology|18-ids,technology|17-relevance,technology|18-relevance
    if column == 'relevance|1':
        mask = df['relevance|1'].notna()
    elif column == 'technology|17':
        mask = df['technology|17-relevance'] >= 1
    elif column == 'technology|18':
        mask = df['technology|18-relevance'] >= 1
    else:
        # Must be relevant, and have at least one true in the label group and at least one relevant in the technology group
        cols_prim = [c for c in COLUMN_GROUPS[parent] if c in df.columns]
        cols_sec = [c for c in COLUMN_GROUPS['technology'] if c in df.columns]
        logger.debug(f'Column masks on {cols_prim} and {cols_sec}')
        mask = (df['relevance|1'] == 1) & df[cols_prim].any(axis=1) & df[cols_sec].any(axis=1)

    logger.info(f'Filtered data down to {mask.sum():,} records for column {column}')

    dataset = pd.DataFrame(
        [
            {
                'id': row['id'],
                'text': f'{row["title"] or ""} {row["abstract"] or ""}',
                'label': int(row[column] >= 1) if row[column] is not None else 0,
            }
            for _, row in df[mask].iterrows()
        ],
    ).set_index('id')

    logger.info(f'Prepared labeled dataset with {dataset.shape} records, of which {dataset["label"].sum()} are >=1')

    model_params_ = None
    if model_params:
        logger.info(f'Loading model parameters from {model_params.resolve()}')
        with open(model_params, 'r') as f:
            model_params_ = json.load(f)
        n_tuning_trials = 0

    idxs = list(dataset.index)
    train_idxs, test_idxs = train_test_split(
        idxs,
        train_size=train_proportion,
        stratify=dataset['label'],
        random_state=random_state * repeat,
    )

    predictions_dir = (target_dir / 'predictions').resolve()
    fn_source = (target_dir / 'dataset.csv').resolve()

    for model_name, classifier in it_models(
        dataset=dataset,
        models=model,
        min_df=min_df,
        max_ngram=max_ngram,
        max_vocab=max_vocab,
        n_tuning_jobs=n_tuning_jobs,
        n_tuning_trials=n_tuning_trials,
        model_params=model_params_['params']['hyperparams'],
    ):
        info_file = predictions_dir / f'{model_name}_{parent}_{column}_{repeat}.json'
        test_file = predictions_dir / f'{model_name}_{parent}_{column}_{repeat}_val.csv'
        pred_file = predictions_dir / f'{model_name}_{parent}_{column}_{repeat}_pred.csv'

        if on_exists == FileConflict.BREAK and pred_file.exists():
            raise FileExistsError(f'File {pred_file} already exists')
        if on_exists == FileConflict.SKIP and pred_file.exists():
            logger.warning(f'File {pred_file} already exists; skipping.')
            continue

        pred_file.parent.mkdir(exist_ok=True, parents=True)

        if model_params_ is None:
            logger.info(f'Initializing and training ranker {model_name} for column {column} (run: {repeat})')
            classifier.train(idxs=train_idxs)

            logger.info(f'Making predictions for "{column}" on all annotated data')
            y_pred = classifier.predict()
            ds = dataset.copy().drop(columns=['text'])
            ds['score'] = y_pred

            y_test_true = ds.loc[test_idxs, 'label']
            y_test_pred = ds.loc[test_idxs, 'score'] > 0.5

            logger.debug(f'Testing ranker {model_name} for column "{column}" on {1 - train_proportion:.0%} test set')
            prec = precision_score(y_test_true, y_test_pred, zero_division=0)
            rec = recall_score(y_test_true, y_test_pred, zero_division=0)
            f1 = f1_score(y_test_true, y_test_pred, zero_division=0)
            logger.debug(f'Recall is {rec:.2%}, precision is {prec:.2%} (F1={f1:.2%}) with {model_name} for column {column}')

            logger.info(f'Writing test predictions to {test_file}')
            ds.to_csv(test_file)
            del ds

            logger.info(f'Writing info to {info_file}')
            with open(info_file, 'w') as f_info:
                json.dump(
                    {
                        'params': classifier.get_params(),
                        'is_tuned': True,
                        'recall': rec,
                        'precision': prec,
                        'f1': f1,
                        'repeat': repeat,
                        'random_state': random_state,
                    },
                    f_info,
                    indent=2,
                )
        else:
            logger.info(f'Initializing and training ranker {model_name} for column {column} (run: {repeat}) on all data with existing model params')
            classifier.train(idxs=idxs)
            logger.info(f'Writing info to {info_file}')
            with open(info_file, 'w') as f_info:
                json.dump(
                    model_params_
                    | {
                        'is_tuned': False,
                        'param_source': str(model_params.resolve()),
                    },
                    f_info,
                    indent=2,
                )

        logger.info(f'Loading new data from {fn_source}')
        df_new = pd.read_csv(fn_source)
        df_new['text'] = df_new.apply(lambda row: f'{row["title"] or ""} {row["abstract"] or ""}', axis=1)
        logger.info(f'Predicting on new data with {df_new.shape} records')
        y_pred = classifier.predict(data=df_new)
        df_new['score'] = y_pred
        logger.info(f'Writing predictions to {pred_file}')
        df_new[['item_id', 'openalex_id', 'doi', 'wos_id', 'scopus_id', 'publication_year', 'score']].to_csv(pred_file, index=False)
        logger.info(f'Done with classifier {model_name}')

    logger.info('Done with all!')


if __name__ == '__main__':
    app()

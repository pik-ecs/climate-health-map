import re
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

MODELS_TRAD = {
    'SVM',
    'REG',
    'SGD',
    'LGBM',
    'NB',
    'ISOFOREST',
    'RANDOMFOREST',
}
MODELS_TRANS = {
    'CLIMATEBERT': 'climatebert/distilroberta-base-climate-f',
    'SCIBERT': 'allenai/scibert_scivocab_uncased',
    'TINYBERT': 'prajjwal1/bert-tiny',
}


def ensure_offline_transformers(model_data_path: Path, logger: logging.Logger) -> None:
    from huggingface_hub import snapshot_download

    for key, name in MODELS_TRANS.items():
        logger.info(f'Downloading model: {key} ({name}) so it is available offline in {model_data_path}')
        snapshot_download(
            repo_id=name,
            repo_type='model',
            cache_dir=model_data_path,
            force_download=False,
        )


def ensure_offline_nltk(logger: logging.Logger, target_dir: Path | None = None) -> None:
    logger.debug('Loading NLTK data...')
    from nltk import download
    import ssl

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    download('stopwords', download_dir=str(target_dir))
    download('punkt', download_dir=str(target_dir))
    download('punkt_tab', download_dir=str(target_dir))
    download('wordnet', download_dir=str(target_dir))
    download('averaged_perceptron_tagger_eng', download_dir=str(target_dir))


def text_utils():
    from nltk import WordNetLemmatizer, pos_tag, wordpunct_tokenize, sent_tokenize, word_tokenize
    from nltk.corpus import stopwords as sw
    from nltk.corpus import wordnet as wn

    lemmatizer = WordNetLemmatizer()
    stopwords = sw.words('english')
    NOALPH = re.compile(r'[^A-Za-z]+')

    def lemmatize(token, tag):
        tag = {'N': wn.NOUN, 'V': wn.VERB, 'R': wn.ADV, 'J': wn.ADJ}.get(tag[0], wn.NOUN)
        return lemmatizer.lemmatize(token, tag)

    def process_text_aggressive(text: str):
        return ' '.join(
            [
                lemmatize(tok, tag)
                for sentence in sent_tokenize(text)
                for tok, tag in pos_tag(wordpunct_tokenize(sentence))
                if tok not in stopwords and len(NOALPH.sub('', tok)) >= 3
            ],
        )

    def process_text_light(text: str):
        return ' '.join([tok for tok in word_tokenize(text) if tok not in stopwords])

    return lemmatize, process_text_aggressive, process_text_light


def get_prediction_stats(
    dataset: pd.DataFrame,
    y_pred: np.ndarray,
    test_idxs: list[int],
    train_idxs: list[int],
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, float]]:
    ds = dataset.copy().drop(columns=['text'])
    ds['score'] = y_pred

    y_test_true = ds.loc[test_idxs, 'label']
    y_test_pred = ds.loc[test_idxs, 'score'] > threshold
    y_train_true = ds.loc[train_idxs, 'label']
    y_train_pred = ds.loc[train_idxs, 'score'] > threshold

    stats = {
        'precision_test': precision_score(y_test_true, y_test_pred, zero_division=0),
        'recall_test': recall_score(y_test_true, y_test_pred, zero_division=0),
        'f1_test': f1_score(y_test_true, y_test_pred, zero_division=0),
        'precision_train': precision_score(y_train_true, y_train_pred, zero_division=0),
        'recall_train': recall_score(y_train_true, y_train_pred, zero_division=0),
        'f1_train': f1_score(y_train_true, y_train_pred, zero_division=0),
        'n_train': len(train_idxs),
        'n_test': len(test_idxs),
        'train_pos': y_train_true.sum(),
        'train_neg': len(train_idxs) - y_train_true.sum(),
        'test_pos': y_test_true.sum(),
        'test_neg': len(test_idxs) - y_test_true.sum(),
        'decision_threshold': threshold,
    }

    return ds, stats


def read_tuning_info(tuning_dir: Path):
    infos = []
    for file in tuning_dir.glob('*.json'):
        with open(file) as f:
            infos.append(json.load(f))
    return pd.DataFrame.from_records(infos)


def get_best_infos(tuning_dir: Path, metric: str = 'f1'):
    return read_tuning_info(tuning_dir).sort_values(by=[f'{metric}_test'], ascending=False).groupby('column').first()


def get_best_info(tuning_dir: Path, column: str, metric: str = 'f1'):
    df_tuning = get_best_infos(tuning_dir=tuning_dir, metric=metric)
    if column not in df_tuning.index:
        raise KeyError(f'No tuning data available for column "{column}" in tuning directory {tuning_dir}')
    best = df_tuning.loc[column]
    return best.to_dict(), best['params'], best['params']['hyperparams']


def columns_with_tuning_info(tuning_dir: Path) -> set[str]:
    tuned_columns = set()
    for file in tuning_dir.glob('*.json'):
        with open(file) as f:
            info = json.load(f)
            tuned_columns.add(info['column'])
    return tuned_columns

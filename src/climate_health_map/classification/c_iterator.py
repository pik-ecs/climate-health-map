import logging
from typing import Generator, Any, Callable, Union
import pandas as pd

from sklearn.base import ClassifierMixin
from .c_traditional import (
    NaiveBayesClassifier,
    RegressionClassifier,
    SVMClassifier,
    RandomForestClassifier,
    IsolationForestClassifier,
    SGDClassifier,
    LightGBMClassifier,
)
from .c_transformer import MODELS_TRANS, TransformerClassifier


def it_models(  # noqa: C901
    dataset: pd.DataFrame,
    models: list[str] | None = None,
    n_tuning_trials: int | None = None,
    n_tuning_jobs: int = 1,
    max_vocab: int = 7500,
    max_ngram: int = 1,
    min_df: int = 3,
    max_df: float = 0.8,
    model_params: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> Generator[tuple[str, Union['TransformerClassifier', 'ClassifierMixin']], None, None]:
    if logger is None:
        logger = logging.getLogger('classify-iterator')

    models = models or ['TINYBERT', 'CLIMATEBERT']
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

    CONFIGS: dict[str, Callable[[], Union['TransformerClassifier', 'ClassifierMixin']]] = {
        'CLIMATEBERT': lambda: TransformerClassifier(
            dataset=dataset,
            model_params=model_params,
            min_batch_size=2,
            max_batch_size=int(6 * memory_factor),
            models=[MODELS_TRANS['CLIMATEBERT']],
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 20,
            test_split=0.1,
        ),
        'SCIBERT': lambda: TransformerClassifier(
            dataset=dataset,
            model_params=model_params,
            min_batch_size=2,
            max_batch_size=int(6 * memory_factor),
            models=[MODELS_TRANS['SCIBERT']],
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 20,
            test_split=0.1,
        ),
        'TINYBERT': lambda: TransformerClassifier(
            dataset=dataset,
            model_params=model_params,
            min_batch_size=2,
            max_batch_size=int(10 * memory_factor),
            models=[MODELS_TRANS['TINYBERT']],
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 20,
            test_split=0.1,
        ),
        # TODO: Maybe add 'malteos/scincl', 'distilbert-base',
        'REG': lambda: RegressionClassifier(
            dataset=dataset,
            model_params=model_params,
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 50,
            n_jobs=n_tuning_jobs,
            max_features=max_vocab,
            ngram_range=(1, max_ngram),
            min_df=min_df,
            max_df=max_df,
        ),
        'SVM': lambda: SVMClassifier(
            dataset=dataset,
            model_params=model_params,
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 50,
            n_jobs=n_tuning_jobs,
            max_features=max_vocab,
            ngram_range=(1, max_ngram),
            min_df=min_df,
            max_df=max_df,
        ),
        'LGBM': lambda: LightGBMClassifier(
            dataset=dataset,
            model_params=model_params,
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 50,
            n_jobs=n_tuning_jobs,
            max_features=max_vocab,
            ngram_range=(1, max_ngram),
            min_df=min_df,
            max_df=max_df,
        ),
        'SGD': lambda: SGDClassifier(
            dataset=dataset,
            model_params=model_params,
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 50,
            n_jobs=n_tuning_jobs,
            max_features=max_vocab,
            ngram_range=(1, max_ngram),
            min_df=min_df,
            max_df=max_df,
        ),
        'NB': lambda: NaiveBayesClassifier(
            dataset=dataset,
            model_params=model_params,
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 1,
            n_jobs=n_tuning_jobs,
            max_features=max_vocab,
            ngram_range=(1, max_ngram),
            min_df=min_df,
            max_df=max_df,
        ),
        'ISOFOREST': lambda: IsolationForestClassifier(
            dataset=dataset,
            model_params=model_params,
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 50,
            n_jobs=n_tuning_jobs,
            max_features=max_vocab,
            ngram_range=(1, max_ngram),
            min_df=min_df,
            max_df=max_df,
        ),
        'RANDOMFOREST': lambda: RandomForestClassifier(
            dataset=dataset,
            model_params=model_params,
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 50,
            n_jobs=n_tuning_jobs,
            max_features=max_vocab,
            ngram_range=(1, max_ngram),
            min_df=min_df,
            max_df=max_df,
        ),
    }

    logger.info(f'Will iterate models: {models}')
    for model in models:
        if model in CONFIGS:
            yield model, CONFIGS[model]()
        else:
            logger.warning(f'Model key "{model}" is not known.')


def get_model(
    dataset: pd.DataFrame,
    model: str,
    n_tuning_trials: int | None = None,
    n_tuning_jobs: int = 1,
    max_vocab: int = 7500,
    max_ngram: int = 1,
    min_df: int = 3,
    max_df: float = 0.8,
    model_params: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> Union['TransformerClassifier', 'ClassifierMixin']:
    for _, classifier in it_models(
        dataset=dataset,
        models=[model],
        n_tuning_trials=n_tuning_trials,
        n_tuning_jobs=n_tuning_jobs,
        max_vocab=max_vocab,
        max_ngram=max_ngram,
        min_df=min_df,
        max_df=max_df,
        model_params=model_params,
        logger=logger,
    ):
        return classifier
    raise KeyError(f'No model for key "{model}"')

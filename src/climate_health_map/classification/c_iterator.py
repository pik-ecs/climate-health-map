import logging
from typing import Generator, Any, TYPE_CHECKING, Callable, Union
import pandas as pd

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin
    from .c_transformer import TransformerClassifier


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
        from .c_transformer import TransformerClassifier

        return TransformerClassifier(
            dataset=dataset,
            model_params=model_params,
            min_batch_size=2,
            max_batch_size=int(6 * memory_factor),
            models=['climatebert/distilroberta-base-climate-f'],
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 20,
            test_split=0.1,
        )

    def sbert():
        from .c_transformer import TransformerClassifier

        return TransformerClassifier(
            dataset=dataset,
            model_params=model_params,
            min_batch_size=2,
            max_batch_size=int(6 * memory_factor),
            models=['allenai/scibert_scivocab_uncased'],
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 20,
            test_split=0.1,
        )

    def tbert():
        from .c_transformer import TransformerClassifier

        return TransformerClassifier(
            dataset=dataset,
            model_params=model_params,
            min_batch_size=2,
            max_batch_size=int(10 * memory_factor),
            models=['prajjwal1/bert-tiny'],
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 20,
            test_split=0.1,
        )

    def reg():
        from .c_traditional import RegressionClassifier

        return RegressionClassifier(
            dataset=dataset,
            model_params=model_params,
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 50,
            n_jobs=n_tuning_jobs,
            max_features=max_vocab,
            ngram_range=(1, max_ngram),
            min_df=min_df,
            max_df=max_df,
        )

    def svm():
        from .c_traditional import SVMClassifier

        return SVMClassifier(
            dataset=dataset,
            model_params=model_params,
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 50,
            n_jobs=n_tuning_jobs,
            max_features=max_vocab,
            ngram_range=(1, max_ngram),
            min_df=min_df,
            max_df=max_df,
        )

    def lgbm():
        from .c_traditional import LightGBMClassifier

        return LightGBMClassifier(
            dataset=dataset,
            model_params=model_params,
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 50,
            n_jobs=n_tuning_jobs,
            max_features=max_vocab,
            ngram_range=(1, max_ngram),
            min_df=min_df,
            max_df=max_df,
        )

    def sgd():
        from .c_traditional import SGDClassifier

        return SGDClassifier(
            dataset=dataset,
            model_params=model_params,
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 50,
            n_jobs=n_tuning_jobs,
            max_features=max_vocab,
            ngram_range=(1, max_ngram),
            min_df=min_df,
            max_df=max_df,
        )

    def forest():
        from .c_traditional import IsolationForestClassifier

        return IsolationForestClassifier(
            dataset=dataset,
            model_params=model_params,
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 50,
            n_jobs=n_tuning_jobs,
            max_features=max_vocab,
            ngram_range=(1, max_ngram),
            min_df=min_df,
            max_df=max_df,
        )

    def rforest():
        from .c_traditional import RandomForestClassifier

        return RandomForestClassifier(
            dataset=dataset,
            model_params=model_params,
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 50,
            n_jobs=n_tuning_jobs,
            max_features=max_vocab,
            ngram_range=(1, max_ngram),
            min_df=min_df,
            max_df=max_df,
        )

    def nb():
        from .c_traditional import NaiveBayesClassifier

        return NaiveBayesClassifier(
            dataset=dataset,
            model_params=model_params,
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 1,
            n_jobs=n_tuning_jobs,
            max_features=max_vocab,
            ngram_range=(1, max_ngram),
            min_df=min_df,
            max_df=max_df,
        )

    CONFIGS: dict[str, Callable[[], Union['TransformerClassifier', 'ClassifierMixin']]] = {
        'CLIMATEBERT': cbert,
        'SCIBERT': sbert,
        'TINYBERT': tbert,
        # TODO: Maybe add 'malteos/scincl', 'distilbert-base',
        'REG': reg,
        'SVM': svm,
        'LGBM': lgbm,
        'SGD': sgd,
        'NB': nb,
        'ISOFOREST': forest,
        'RANDOMFOREST': rforest,
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

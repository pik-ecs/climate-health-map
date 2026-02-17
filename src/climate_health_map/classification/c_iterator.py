import logging
from typing import Generator, Any
import pandas as pd


def it_models(  # noqa: C901
    dataset: pd.DataFrame,
    models: list[str] | None = None,
    n_tuning_trials: int | None = None,
    n_tuning_jobs: int = 1,
    max_vocab: int = 7500,
    max_ngram: int = 1,
    min_df: int = 3,
    model_params: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> Generator[tuple[str, Any], None, None]:
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
        from .c_transformer import TransRanker

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
        from .c_transformer import TransRanker

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
        from .c_transformer import TransRanker

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
        from .c_traditional import RegressionClassifier

        return RegressionClassifier(
            dataset=dataset,
            model_params=model_params,
            tuning_trials=n_tuning_trials if n_tuning_trials is not None else 50,
            n_jobs=n_tuning_jobs,
            max_features=max_vocab,
            ngram_range=(1, max_ngram),
            min_df=min_df,
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
        'NB': nb,
        'ISOFOREST': forest,
        'RANDOMFOREST': rforest,
    }

    logger.info(f'Will iterate models: {models}')
    for model in models:
        if model in CONFIGS:
            yield model, CONFIGS[model]()

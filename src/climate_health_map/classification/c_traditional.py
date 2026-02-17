import os
import logging
import warnings
from typing import Any, Type
from abc import abstractmethod, ABC

import optuna
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.base import ClassifierMixin
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer

from climate_health_map.data.dataset import downsampling_mask
from .util import text_utils

logger = logging.getLogger('classify-traditional')
logging.getLogger('LightGBM').setLevel(logging.ERROR)

# Stop optuna from logging all trial results
# optuna.logging.set_verbosity(optuna.logging.WARNING)

# Capturing some sklearn warnings to clear up logs
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore')

# Not everything is caught when parallelising, this helps...
os.environ['PYTHONWARNINGS'] = 'ignore'

lemmatize, process_text_aggressive, process_text_light = text_utils()


class _SimpleClassification(ABC):
    def __init__(
        self,
        BaseModel: Type[ClassifierMixin],
        model_params: dict[str, Any],
        dataset: pd.DataFrame,
        tuning_trials: int = 35,
        scoring: str | None = None,
        random_seed: int | None = None,
        n_jobs: int = 5,
        max_features: int = 75000,
        ngram_range: tuple[int, int] = (1, 3),
        min_df: int | float = 3,
        max_df: int | float = 0.8,
        **kwargs: dict[str, Any],
    ):
        self.dataset = dataset
        self.model_params = model_params
        self.final_params = {}
        self.BaseModel = BaseModel
        self.scoring = scoring
        self.tuning_trials = tuning_trials
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.model = None

        stripped_texts = [process_text_aggressive(txt) for txt in tqdm(dataset['text'], desc='tokenising')]
        self.vectorizer = TfidfVectorizer(
            # See https://github.com/AnneIsARealProgrammerNow/ClimateHealth_Wellcome/blob/v0.1/active_learning_with_evaluation.ipynb
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            strip_accents='unicode',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
        )
        self.scaler = StandardScaler(with_mean=False)
        vectors = self.vectorizer.fit_transform(stripped_texts)
        self.vectors = self.scaler.fit_transform(vectors)

    @property
    @classmethod
    @abstractmethod
    def name(cls) -> str:
        raise NotImplementedError

    @abstractmethod
    def _hp_space(self, trial: optuna.Trial) -> dict[str, Any]:
        raise NotImplementedError()

    def _tune(self, trial: optuna.Trial, x: np.ndarray, y: np.ndarray) -> float:
        self.model_params = self.model_params | self._hp_space(trial)
        seed = None
        if self.random_seed is not None:
            seed = self.random_seed * trial.number
        sampling = self.model_params.get('downsampling', 0)
        mask = downsampling_mask(y, sampling=sampling)
        model_params = {k: v for k, v in self.model_params.items() if k != 'downsampling'}
        logger.debug(f'Downsampling from {y.shape[0]:,} ({y.sum():,} incl) to {mask.sum():,} ({y[mask].sum():,} incl)')
        logger.debug(f'Preparing tuning trial with model_params: {model_params}')
        model = self.BaseModel(**model_params)
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
        score = cross_val_score(model, x[mask], y[mask], cv=cv, scoring=self.scoring)
        logger.debug(f'Using {y[mask].sum()}/{mask.sum()} inclusion for tuning trial {trial.number} | scores: {score} | sampling: {sampling} | {model_params}')
        mean = score.mean()
        return 0 if np.isnan(mean) else mean

    def train(self, idxs: list[int] | None = None) -> None:
        if not idxs:
            idxs = self.dataset.index
        mask = self.dataset.index.isin(idxs)
        x = self.vectors[mask]
        y = self.dataset.loc[idxs]['label'].to_numpy()

        logger.debug(f'Fitting on {y.shape[0]:,} samples ({y.sum():,} of which included)')

        if self.tuning_trials > 0:
            logger.debug(f'Running hyper-parameter tuning ({self.tuning_trials} trials)')
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: self._tune(trial, x, y), n_trials=self.tuning_trials, n_jobs=self.n_jobs)
            logger.debug(f'Hyper-parameter-tuning for {self.name} done with best score {study.best_value}')
            self.final_params = self.model_params | study.best_params
        else:
            logger.debug('Not running hyper-parameter tuning, using provided model params.')
            self.final_params = self.model_params

        mask = downsampling_mask(y, sampling=self.final_params.get('downsampling', 0))
        model_params = {k: v for k, v in self.final_params.items() if k != 'downsampling'}
        self.model = self.BaseModel(**model_params)
        self.model.fit(x[mask], y[mask])

    def predict(self, idxs: list[int] | None = None, data: pd.DataFrame | None = None) -> np.ndarray:
        needs_vectors = data is not None
        data = self.dataset if data is None else data

        if not idxs:
            idxs = data.index

        if len(idxs) == 0:
            return np.array([])

        if needs_vectors:
            stripped_texts = [process_text_aggressive(txt) for txt in tqdm(data.loc[idxs]['text'], desc='tokenising')]
            vectors = self.vectorizer.transform(stripped_texts)
            vectors = self.scaler.fit_transform(vectors)
        else:
            vectors = self.vectors[self.dataset.index.isin(idxs)]

        y_true = data.loc[idxs]['label'].to_numpy() if 'label' in data.columns else None
        logger.debug(f'Predicting on {len(idxs):,} samples ({y_true.sum() if y_true is not None else "??"} of which should be included)')
        if hasattr(self.model, 'predict_proba'):
            y_preds = self.model.predict_proba(vectors)
        else:
            y_preds = self.model.predict(vectors)

        logger.debug(f'  > Predictions found {(y_preds > 0.5).sum():,} to be included')
        if len(y_preds.shape) == 1:
            return y_preds
        return y_preds[:, 1]

    def get_params(self) -> dict[str, Any]:
        return {
            'vectoriser': {
                'ngram_range': self.vectorizer.ngram_range,
                'max_features': self.vectorizer.max_features,
                'min_df': self.vectorizer.min_df,
            },
            'model': self.name,
            'hyperparams': {k: getattr(self.model, k) if hasattr(self.model, k) else v for k, v in self.final_params.items()},
        }


class SVMClassifier(_SimpleClassification):
    name = 'svm'

    def __init__(
        self,
        dataset: pd.DataFrame,
        tuning_trials: int = 35,
        model_params: dict[str, Any] | None = None,
        random_seed: int | None = None,
        scoring: str | None = None,
        n_jobs: int = 5,
        max_features: int = 75000,
        ngram_range: tuple[int, int] = (1, 3),
        min_df: int | float = 3,
        max_df: float = 0.8,
        **kwargs: dict[str, Any],
    ):
        from sklearn.svm import SVC

        super().__init__(
            BaseModel=SVC,
            model_params={'kernel': 'linear', 'class_weight': 'balanced', 'degree': 3, 'gamma': 'auto', 'probability': True, 'C': 1.0, 'max_iter': 1000}
            | (model_params or {}),
            tuning_trials=tuning_trials,
            scoring=scoring or 'f1',
            dataset=dataset,
            random_seed=random_seed,
            n_jobs=n_jobs,
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            **kwargs,
        )

    def _hp_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            'C': trial.suggest_float('C', low=0.001, high=100, log=True),
            'gamma': trial.suggest_float('gamma', 0.001, 1.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),  # , 'poly', 'sigmoid'
            'downsampling': trial.suggest_float('downsampling', low=0.0, high=0.95),
        }


class SGDClassifier(_SimpleClassification):
    name = 'sgd'

    def __init__(
        self,
        dataset: pd.DataFrame,
        tuning_trials: int = 35,
        model_params: dict[str, Any] | None = None,
        random_seed: int | None = None,
        scoring: str | None = None,
        n_jobs: int = 5,
        max_features: int = 75000,
        ngram_range: tuple[int, int] = (1, 3),
        min_df: int | float = 3,
        max_df: float = 0.8,
        **kwargs: dict[str, Any],
    ):
        from sklearn.linear_model import SGDClassifier

        super().__init__(
            BaseModel=SGDClassifier,
            model_params={'class_weight': 'balanced', 'loss': 'log_loss', 'max_iter': 1000} | (model_params or {}),
            tuning_trials=tuning_trials,
            scoring=scoring or 'f1',
            dataset=dataset,
            random_seed=random_seed,
            n_jobs=n_jobs,
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            **kwargs,
        )

    def _hp_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            'alpha': trial.suggest_float('alpha', 0.0001, 100000, log=True),
            'downsampling': trial.suggest_float('downsampling', low=0.0, high=0.95),
        }


class RegressionClassifier(_SimpleClassification):
    name = 'logreg'

    def __init__(
        self,
        dataset: pd.DataFrame,
        tuning_trials: int = 35,
        model_params: dict[str, Any] | None = None,
        random_seed: int | None = None,
        scoring: str | None = None,
        n_jobs: int = 5,
        max_features: int = 75000,
        ngram_range: tuple[int, int] = (1, 3),
        min_df: int | float = 3,
        max_df: float = 0.8,
        **kwargs: dict[str, Any],
    ):
        from sklearn.linear_model import LogisticRegression

        super().__init__(
            BaseModel=LogisticRegression,
            model_params={
                'class_weight': 'balanced',
                'tol': 0.0001,
                'C': 1.0,
                'solver': 'lbfgs',
                'max_iter': 100,
            }
            | (model_params or {}),
            tuning_trials=tuning_trials,
            scoring=scoring or 'f1',
            dataset=dataset,
            random_seed=random_seed,
            n_jobs=n_jobs,
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            **kwargs,
        )

    def _hp_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            'C': trial.suggest_float('C', low=0.01, high=10, log=True),
            'solver': trial.suggest_categorical('solver', ['saga', 'liblinear', 'lbfgs']),
            'downsampling': trial.suggest_float('downsampling', low=0.0, high=0.95),
        }


class RandomForestClassifier(_SimpleClassification):
    name = 'randforest'

    def __init__(
        self,
        dataset: pd.DataFrame,
        tuning_trials: int = 35,
        model_params: dict[str, Any] | None = None,
        random_seed: int | None = None,
        scoring: str | None = None,
        n_jobs: int = 5,
        max_features: int = 75000,
        ngram_range: tuple[int, int] = (1, 3),
        min_df: int | float = 3,
        max_df: float = 0.8,
        **kwargs: dict[str, Any],
    ):
        from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier_

        super().__init__(
            BaseModel=RandomForestClassifier_,
            model_params={
                'n_estimators': 1000,
                'verbose': 0,
                'random_state': random_seed,
                'max_features': 'sqrt',
                'min_samples_split': 2,
                'max_depth': None,
            }
            | (model_params or {}),
            tuning_trials=tuning_trials,
            scoring=scoring or 'f1',
            dataset=dataset,
            random_seed=random_seed,
            n_jobs=n_jobs,
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            **kwargs,
        )

    def _hp_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            'n_estimators': trial.suggest_int('n_estimators', low=100, high=5000),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'downsampling': trial.suggest_float('downsampling', low=0.0, high=0.95),
        }


class IsolationForestClassifier(_SimpleClassification):
    name = 'isoforest'

    def __init__(
        self,
        dataset: pd.DataFrame,
        tuning_trials: int = 35,
        model_params: dict[str, Any] | None = None,
        random_seed: int | None = None,
        scoring: str | None = None,
        n_jobs: int = 5,
        max_features: int = 75000,
        ngram_range: tuple[int, int] = (1, 3),
        min_df: int | float = 3,
        max_df: float = 0.8,
        **kwargs: dict[str, Any],
    ):
        from sklearn.ensemble import IsolationForest

        super().__init__(
            BaseModel=IsolationForest,
            model_params={
                'n_estimators': 100,
                'max_samples': 'auto',
                'contamination': 'auto',
                'max_features': 1.0,
                'bootstrap': False,
                'n_jobs': None,
                'random_state': None,
                'verbose': 0,
                'warm_start': False,
            }
            | (model_params or {}),
            tuning_trials=tuning_trials,
            scoring=scoring or 'f1',
            dataset=dataset,
            random_seed=random_seed,
            n_jobs=n_jobs,
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            **kwargs,
        )

    def _hp_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            'n_estimators': trial.suggest_int('n_estimators', low=20, high=250),
            'max_features': trial.suggest_float('max_features', low=0.2, high=1.0),
            'downsampling': trial.suggest_float('downsampling', low=0.0, high=0.95),
        }


class NaiveBayesClassifier(_SimpleClassification):
    name = 'naivebayes'

    def __init__(
        self,
        dataset: pd.DataFrame,
        tuning_trials: int = 35,
        model_params: dict[str, Any] | None = None,
        random_seed: int | None = None,
        scoring: str | None = None,
        n_jobs: int = 5,
        max_features: int = 75000,
        ngram_range: tuple[int, int] = (1, 3),
        min_df: int | float = 3,
        max_df: float = 0.8,
        **kwargs: dict[str, Any],
    ):
        from sklearn.naive_bayes import MultinomialNB

        super().__init__(
            BaseModel=MultinomialNB,
            model_params={
                'force_alpha': True,
                'alpha': 1.0,
                'fit_prior': True,
            }
            | (model_params or {}),
            tuning_trials=tuning_trials,
            scoring=scoring or 'f1',
            dataset=dataset,
            random_seed=random_seed,
            n_jobs=n_jobs,
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            **kwargs,
        )

    def _hp_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            'downsampling': trial.suggest_float('downsampling', low=0.0, high=0.95),
            'alpha': trial.suggest_float('alpha', low=0.0, high=1.0),
            'fit_prior': trial.suggest_categorical('fit_prior', [True, False]),
        }


class LightGBMClassifier(_SimpleClassification):
    name = 'lightgbm'

    def __init__(
        self,
        dataset: pd.DataFrame,
        tuning_trials: int = 35,
        model_params: dict[str, Any] | None = None,
        random_seed: int | None = None,
        scoring: str | None = None,
        n_jobs: int = 5,
        max_features: int = 75000,
        ngram_range: tuple[int, int] = (1, 3),
        min_df: int | float = 3,
        max_df: float = 0.8,
        **kwargs: dict[str, Any],
    ):
        from lightgbm import LGBMClassifier

        super().__init__(
            BaseModel=LGBMClassifier,
            model_params={
                'objective': 'binary',  # (not multiclass))
                'learning_rate': 0.1,
                'n_estimators': 100,  # Number of boosting rounds
                'num_leaves': 31,  # Number of leaves in each tree
                'random_state': random_seed,  # For reproducibility
                'verbose': -1,
                **(model_params or {}),
            },
            tuning_trials=tuning_trials,
            scoring=scoring or 'f1',
            dataset=dataset,
            random_seed=random_seed,
            n_jobs=n_jobs,
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            **kwargs,
        )

    def _hp_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            # Controls step size in boosting
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            # Number of boosting rounds (discrete float, behaves like int)
            'n_estimators': trial.suggest_int('n_estimators', 50, 500, log=True),
            # Number of leaves in each tree (higher = more complex)
            'num_leaves': trial.suggest_int('num_leaves', 10, 50, log=True),
            # Depth of trees (-1 means no limit)
            'max_depth': trial.suggest_int('max_depth', -1, 20),
            # Minimum data points in a leaf
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 20, log=True),
            # Fraction of samples used in each boosting iteration
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            # Fraction of features used per tree
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            # L1 regularization
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            # L2 regularization
            'reg_lambda': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'downsampling': trial.suggest_float('downsampling', low=0.0, high=0.95),
        }

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .env import essentials, get_logger
from .config import Settings, load_settings


def ensure_directories(logger: logging.Logger, **paths: tuple[Path | None, bool] | Path | None) -> None:
    """Batch-test that all necessary paths exist.

    * Directory paths -> this method will make sure it exists.
    * tuple[Path, bool] -> if bool is true, check if file/path exists and throw exception otherwise.
    """
    path: Path | None
    assert_exist: bool
    for info, entry in paths.items():
        path, assert_exist = entry if type(entry) is tuple else (entry, False)  # type:ignore [assignment]
        if path is not None:
            path = path.absolute().resolve()
        if assert_exist and (path is None or not path.exists()):
            raise FileNotFoundError(f'Path for {info} does not exist at {path}')
        elif not assert_exist:
            if path is None:
                continue
            path.mkdir(parents=True, exist_ok=True)
        logger.info(f'Will use {info} at {path}')


def read_any_pd(source: Path, index_column: str | None = None, **kwargs: Any) -> pd.DataFrame:
    kwargs = kwargs or {}
    if source.suffix == '.csv':
        kwargs_ = {k: v for k, v in kwargs.items() if k in pd.read_csv.__annotations__}
        df = pd.read_csv(source, **kwargs_)
    elif source.suffix == '.feather' or source.suffix == '.arrow':
        kwargs_ = {k: v for k, v in kwargs.items() if k in pd.read_feather.__annotations__}
        df = pd.read_feather(source, **kwargs_)
    elif source.suffix == '.parquet':
        kwargs_ = {k: v for k, v in kwargs.items() if k in pd.read_parquet.__annotations__}
        df = pd.read_parquet(source, **kwargs_)
    else:
        raise ValueError(f'Unsupported file type: {source.suffix}')

    if index_column is not None:
        return df.set_index(index_column, drop=True)
    return df


def write_any_df(df: pd.DataFrame, target: Path, **kwargs: Any) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {'index': False} | (kwargs or {})
    if target.suffix == '.csv':
        kwargs_ = {k: v for k, v in kwargs.items() if k in pd.DataFrame.to_csv.__annotations__}
        df.to_csv(target, **kwargs_)
    elif target.suffix == '.feather' or target.suffix == '.arrow':
        kwargs_ = {k: v for k, v in kwargs.items() if k in pd.DataFrame.to_feather.__annotations__}
        df.to_feather(target, **kwargs_)
    elif target.suffix == '.parquet':
        kwargs_ = {k: v for k, v in kwargs.items() if k in pd.DataFrame.to_parquet.__annotations__}
        df.to_parquet(target, **kwargs_)
    else:
        raise ValueError(f'Unsupported file type: "{target.suffix}"')


def estimate_pd_memory_needs(source: Path) -> float:
    size_factor = {
        '.csv': 5,
        '.feather': 20,
        '.arrow': 20,
        '.parquet': 20,
    }
    if source.suffix not in size_factor:
        raise ValueError(f'Unsupported file type: {source.suffix}')
    source_size_gb = source.stat().st_size / 1024 / 1024 / 1024
    return source_size_gb * size_factor[source.suffix]


__all__ = [
    'essentials',
    'get_logger',
    'Settings',
    'load_settings',
    'ensure_directories',
    'read_any_pd',
    'write_any_df',
]

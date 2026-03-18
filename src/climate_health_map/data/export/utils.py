import logging
from typing import Any, Literal
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import sqlalchemy as sa

from climate_health_map.shared import essentials, read_any_pd
from climate_health_map.shared.types import OnConflict
from climate_health_map.data.labels import LABELS, Collection, Group, AggTopic, AggAggTopic
from climate_health_map.data.keywords import search_keywords, REVIEW_KEYWORDS, EVALUATION_KEYWORDS, MENTAL_HEALTH_TERMS, search_regexes
from climate_health_map.topics import rescale_topic_scores as rescale_topic_scores_func


class ExportContext:
    def __init__(
        self,
        config: Path,
        target: Path,
        params: dict[str, Any] | None = None,
        batch_size: int = 1000,
        project_id: str | None = None,
        import_ids: list[str] | None = None,
        on_exists: OnConflict = OnConflict.IGNORE,
        max_file_size: int | None = None,
        loglevel: str = 'INFO',
    ):
        self.logger, self.settings, self.db_engine = essentials(config=config, loglevel=loglevel, logger_name='export', run_log_init=True)
        self.db_engine.engine.echo = self.logger.level == logging.DEBUG

        if target.exists() and on_exists == OnConflict.SKIP:
            self.logger.warning(f'Target file already exists (ending silently): {target.resolve()}')
            return
        if target.exists() and on_exists == OnConflict.BREAK:
            raise FileExistsError(f'Target file {target.resolve()} already exists')
        self.logger.info(f'Going to write result to {target.resolve()}')
        self.target = target
        self.params = params or {}
        self.batch_size = batch_size
        if import_ids is None and self.params.get('import_ids') is None:
            self.params['import_ids'] = self.settings.IMPORTS
        if project_id is None and self.params.get('project_id') is None:
            self.params['project_id'] = self.settings.PROJECT_ID
        self.query: sa.TextClause | None = None
        self.max_file_size = max_file_size

    def __enter__(self) -> 'ExportContext':
        # return the instance so it can be used inside the with-block
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.query is None:
            raise RuntimeError('Query must be defined')

        with self.db_engine.session() as session:
            self.logger.info('Running query...')
            rslt = session.execute(self.query.execution_options(yield_per=self.batch_size), self.params)

            columns: list[str] | None = None
            # TODO: handle the case for `max_file_size`
            # TODO: handle arbitrary file type like `write_any_pd` (and possibly push batch/chunk logic upstream)

            n_rows = 0
            for batch in tqdm(rslt.mappings().partitions(self.batch_size)):
                sub_df = pd.DataFrame(batch).replace({np.nan: None})
                n_rows += sub_df.shape[0]
                if columns is None:
                    columns = sub_df.columns
                    sub_df.to_csv(self.target, index=False)
                else:
                    sub_df.to_csv(self.target, index=False, header=False, columns=columns, mode='a')

            self.logger.info(f'Wrote {n_rows:,} results to {self.target}')

        # return False to propagate exceptions
        return False  # type:ignore [return-value]


def replace_human_annotations(df: pd.DataFrame, source: Path, logger: logging.Logger) -> pd.DataFrame:
    logger.info('Override predictions with human annotations')

    # Load human annotations
    df_human = read_any_pd(source, index_column='item_id')
    for group in LABELS.values():
        if group.collection not in {Collection.MAJOR, Collection.IMPACTS, Collection.EXTERNAL}:
            continue
        for label in group.labels:
            # Make sure all values are within range and not exactly 0 or 1
            df[label.column] = df[label.column].clip(lower=0.01, upper=0.99)

            for val in [0, 1]:
                mask = df_human[label.column] == val
                item_ids = df_human[mask]['item_id'].tolist()
                df.loc[df.index.isin(item_ids), label.column] = val

            logger.debug(f' > Human {label.column}==1: {(df[label.column] == 1).sum():,} | {label.column}==0: {(df[label.column] == 0).sum():,}')

    return df


def read_export(
    source_items: Path,
    source_classifications: Path,
    source_annotations: Path | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
    filter_rel: bool = True,
    filter_mai: bool = True,
    rescale_topic_scores: bool = False,
    include_keyword_columns: bool = False,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    logger = logger or logging.getLogger('reader')

    df_items = read_any_pd(source_items, index_column='item_id')
    logger.info(f'Loaded items table: {df_items.shape}')

    mask_items = np.ones(df_items.shape[0], dtype=bool)
    if year_start is not None:
        mask_items &= df_items['publication_year'].fillna(0) >= year_start
    if year_end is not None:
        mask_items &= df_items['publication_year'].fillna(0) <= year_end
    logger.info(f'Keeping {mask_items.sum():,} after PY filtering')

    df_classifications = read_any_pd(source_classifications, index_column='item_id')
    logger.info(f'Loaded classifications table: {df_classifications.shape}')

    mask_classifications = np.ones(df_classifications.shape[0], dtype=bool)
    if filter_rel:
        mask_classifications &= df_classifications['rel_major|1'] > 0.5
        logger.info(f'Keeping {mask_classifications.sum():,} after relevance filtering')
    if filter_mai:
        mask_classifications &= (df_classifications[['cat|0', 'cat|1', 'cat|2']] > 0.5).any(axis=1)
        logger.info(f'Keeping {mask_classifications.sum():,} after mitigation/adaptation/impacts filtering')

    df_items = df_items[mask_items]
    df_classifications = df_classifications[mask_classifications]

    if source_annotations is not None:
        logger.info('Replacing scores with human annotations where available')
        df_classifications = replace_human_annotations(df_classifications, source=source_annotations, logger=logger)

    if rescale_topic_scores:
        df_classifications = rescale_topic_scores_func(df=df_classifications, logger=logger)
        logger.info('Rescaled topic scores')

    df = df_items.join(df_classifications, how='inner').copy()
    logger.info(f'Joined tables into shape {df.shape}')

    if include_keyword_columns:
        logger.info('Applying keyword columns...')
        df['keywords|0'] = (search_regexes(df['title'], regexes=REVIEW_KEYWORDS) | search_regexes(df['abstract'], regexes=REVIEW_KEYWORDS)).astype(int)
        df['keywords|1'] = (search_regexes(df['title'], regexes=EVALUATION_KEYWORDS) | search_regexes(df['abstract'], regexes=EVALUATION_KEYWORDS)).astype(int)
        df['keywords|2'] = (search_keywords(df['title'], terms=MENTAL_HEALTH_TERMS) | search_keywords(df['abstract'], terms=MENTAL_HEALTH_TERMS)).astype(int)

    return df


def label_group_counts(
    df: pd.DataFrame,
    group: Group | AggTopic | AggAggTopic,
    geography_filter: Literal['affiliation', 'location', 'region_affiliation', 'region_location'] | None = None,
    count_primary_class: bool = False,
    threshold: float = 0.5,
) -> pd.Series:
    table_index = pd.RangeIndex(start=df['Publication year'].min(), stop=df['Publication year'].max() + 1, step=1, name='Publication year')

    columns = [label.column for label in group.labels if label.column in df.columns]
    mask = (df[columns].notna() & (df[columns] > threshold)).any(axis=1)
    primary = df[mask][columns].fillna(0).idxmax(axis=1)

    data = {}
    totals = {}
    for label in group.labels:
        if label.column not in df.columns:
            continue

        extra_mask = (primary == label.column) if count_primary_class else df[label.column] > threshold
        if group.collection == Collection.IMPACTS:
            extra_mask &= df['incl_impacts']
        if geography_filter == 'location':
            extra_mask &= df['incl_location']
        if geography_filter == 'affiliation':
            extra_mask &= df['incl_affiliation']
        if geography_filter == 'region_location':
            extra_mask &= df['incl_region_location']
        if geography_filter == 'region_affiliation':
            extra_mask &= df['incl_region_affiliation']

        totals[(group.name, label.name)] = df[mask & extra_mask]['item_id'].nunique()
        data[(group.name, label.name)] = df[mask & extra_mask].groupby('Publication_year')['item_id'].nunique()

    return pd.Series(data, index=table_index).fillna(0).astype(int)

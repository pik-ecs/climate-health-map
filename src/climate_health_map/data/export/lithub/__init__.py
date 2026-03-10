from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from climate_health_map.scatterplot import project_topic_names
from climate_health_map.shared import read_any_pd, get_logger
from climate_health_map.data.geographies import load_country_infos
from .info import info
from .writers import write_base_info, write_sqlite, write_keywords, write_geographies


def _flatten_country_groups(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {'item_id': item_id}
            | {
                f'{prefix}_{column}|{grouping}': count
                for column in [
                    'Group (Lancet 2026)',
                    'Group (WHO 2026)',
                    'Group (HDI 2026)',
                    'Region (IPCC AR6, 6)',
                    'Region (IPCC AR6, 10)',
                    'Region (WorldBank 2026)',
                    'Income group (WorldBank 2026)',
                    'Lending category (WorldBank 2026)',
                    'Continent (Name)',
                ]
                for grouping, count in group[group[column].notna()][column].value_counts().items()
            }
            for item_id, group in df.groupby('item_id')
        ],
    ).set_index('item_id')


def prepare_lithub_export(
    source: Annotated[Path, typer.Option(help='Source directory')],
    target: Annotated[Path, typer.Option(help='Target directory')],
    year_start: Annotated[int, typer.Option(help='Start year (incl)')] = 1990,
    year_end: Annotated[int, typer.Option(help='End year (incl)')] = 2025,
    loglevel: Annotated[str, typer.Option(help='Verbosity of logger')] = 'INFO',
):
    target.rmdir()
    target.mkdir(parents=True, exist_ok=True)

    logger = get_logger(loglevel=loglevel, logger_name='lithub', run_log_init=True)
    df_countries = load_country_infos()
    logger.info(f'Loaded country infos: {df_countries.shape}')

    df_items = read_any_pd(source / 'items.csv', index_column='item_id')#, nrows=10000
    logger.info(f'Loaded items table: {df_items.shape}')

    df_items = df_items[(df_items['publication_year'].fillna(0) >= year_start) & (df_items['publication_year'].fillna(0) <= year_end)]
    logger.info(f'Keeping {len(df_items):,} after PY filtering')

    df_classifications = read_any_pd(source / 'classifications.csv', index_column='item_id')
    logger.info(f'Loaded classifications table: {df_classifications.shape}')

    df_scatterplot = read_any_pd(source / 'scatterplot.csv', index_column='item_id')
    logger.info(f'Loaded scatterplot table: {df_scatterplot.shape}')

    df = df_items.join(df_classifications)
    logger.info(f'Joined tables: {df.shape}')

    df = df[df['rel_major|1'] > 0.5]
    logger.info(f'Keeping {len(df):,} after relevance filtering')
    df = df[(df[['cat|0', 'cat|1', 'cat|2']] > 0.5).any(axis=1)].copy()
    df['idx'] = np.arange(len(df))
    logger.info(f'Keeping {len(df):,} after mitigation/adaptation/impacts filtering')

    # df = rescale_projection(df=df, logger=logger)
    # df = rescale_topic_scores(df=df, logger=logger)
    # df, cols = rename_columns(df=df, logger=logger)
    # df = replace_human_annotations(df, logger=logger)

    logger.info('Reading affiliation data...')
    df_affiliations = read_any_pd(source / 'affiliations.csv', keep_default_na=False).merge(df_countries, left_on='iso2', right_on='iso2', how='left')
    df_affiliations_flat = _flatten_country_groups(prefix='Affiliation', df=df_affiliations)
    logger.info(f'Loaded affiliations table: {df_affiliations.shape}; flattened: {df_affiliations_flat.shape}')

    logger.info('Reading mordecai data...')
    df_places = read_any_pd(source / 'places.csv', keep_default_na=False).merge(df_countries, left_on='country_code3', right_on='iso3', how='left')
    df_places_flat = _flatten_country_groups(prefix='Location', df=df_places)
    logger.info(f'Loaded places table: {df_places.shape}; flattened: {df_places_flat.shape}')

    df = df.merge(df_affiliations_flat, how='left').merge(df_places_flat, how='left')
    logger.info(f'Joined output table: {df.shape}')

    # TODO: topics_openalex.csv

    # Drop duplicate columns
    df = df.loc[:, ~df.columns.duplicated()].copy()

    write_sqlite(df, target=target / info.db_filename, logger=logger)
    # write_keywords(df, target=target / info.keywords_filename, logger=logger)
    write_base_info(df, target=target / info.arrow_filename, logger=logger)

    write_geographies(
        df=df.join(df_places),
        target_min=target / info.slim_geo_filename,
        target_full=target / info.full_geo_filename,
        logger=logger,
        chunk_size=2000,
    )

    df_keywords = project_topic_names(df_topic_scores=df, logger=logger)
    if df_keywords is not None:
        write_keywords(df_keywords, target / 'keywords.arrow')

    # TODO: write dataset info file after replacing PY start/end and num records

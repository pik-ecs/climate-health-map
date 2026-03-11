import logging
from pathlib import Path
from typing import Annotated

import typer
import numpy as np
import pandas as pd

from climate_health_map.data.labels import Collection, LABELS
from climate_health_map.data.geographies import load_country_infos, flatten_country_groups
from climate_health_map.shared import read_any_pd, get_logger
from climate_health_map.topics import rescale_topic_scores
from climate_health_map.scatterplot import rescale_projection

from .info import info, filter_labels
from .writers import write_base_info, write_sqlite, write_keywords, write_geographies
from ...keywords import search_keywords, REVIEW_KEYWORDS, EVALUATION_KEYWORDS, MENTAL_HEALTH_TERMS


def _replace_human_annotations(df: pd.DataFrame, source: Path, logger: logging.Logger) -> pd.DataFrame:
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


def prepare_lithub_export(
    source: Annotated[Path, typer.Option(help='Source directory')],
    target: Annotated[Path, typer.Option(help='Target directory')],
    source_annotations: Annotated[Path | None, typer.Option(help='path to human annotations')] = None,
    year_start: Annotated[int, typer.Option(help='Start year (incl)')] = 1990,
    year_end: Annotated[int, typer.Option(help='End year (incl)')] = 2025,
    skip_sqlite: Annotated[bool, typer.Option(help='Skip sqlite export for speedy info update')] = False,
    loglevel: Annotated[str, typer.Option(help='Verbosity of logger')] = 'INFO',
):
    target.mkdir(parents=True, exist_ok=True)

    logger = get_logger(loglevel=loglevel, logger_name='lithub', run_log_init=True)
    df_countries = load_country_infos()
    logger.info(f'Loaded country infos: {df_countries.shape}')

    df_items = read_any_pd(source / 'items.csv', index_column='item_id')
    logger.info(f'Loaded items table: {df_items.shape}')

    df_items = df_items[
        df_items['publication_year'].notna() & (df_items['publication_year'].fillna(0) >= year_start) & (df_items['publication_year'].fillna(0) <= year_end)
    ]
    logger.info(f'Keeping {len(df_items):,} after PY filtering')

    df_classifications = read_any_pd(source / 'classifications.csv', index_column='item_id')
    logger.info(f'Loaded classifications table: {df_classifications.shape}')

    df_classifications = df_classifications[df_classifications['rel_major|1'] > 0.5]
    logger.info(f'Keeping {len(df_classifications):,} after relevance filtering')
    df_classifications = df_classifications[(df_classifications[['cat|0', 'cat|1', 'cat|2']] > 0.5).any(axis=1)].copy()
    logger.info(f'Keeping {len(df_classifications):,} after mitigation/adaptation/impacts filtering')

    if source_annotations is not None:
        logger.info('Replacing scores with human annotations where available')
        df_classifications = _replace_human_annotations(df_classifications, source=source_annotations, logger=logger)

    df_classifications = rescale_topic_scores(df=df_classifications, logger=logger)
    logger.info('Rescaled topic scores')

    df = df_items.join(df_classifications, how='inner')
    df['idx'] = np.arange(len(df))  # set a continuous index now that we are done filtering/joining
    logger.info(f'Joined tables: {df.shape}')

    # give python the option to free some memory
    del df_items
    del df_classifications

    logger.info('Reading affiliation data...')
    df_affiliations = read_any_pd(source / 'affiliations.csv', keep_default_na=False).merge(df_countries, left_on='iso2', right_on='iso2', how='left')
    logger.info('Flattening affiliation data...')
    df_affiliations_flat = flatten_country_groups(prefix='Affiliation', df=df_affiliations)
    logger.info(f'Loaded affiliations table: {df_affiliations.shape}; flattened: {df_affiliations_flat.shape}')

    logger.info('Reading mordecai data...')
    df_places = read_any_pd(source / 'places.csv', keep_default_na=False).merge(df_countries, left_on='country_code3', right_on='iso3', how='left')
    logger.info('Flattening mordecai data...')
    df_places_flat = flatten_country_groups(prefix='Location', df=df_places)
    logger.info(f'Loaded places table: {df_places.shape}; flattened: {df_places_flat.shape}')

    df = df.join(df_affiliations_flat, how='left').join(df_places_flat, how='left')
    logger.info(f'Joined output table: {df.shape}')

    if not skip_sqlite:
        logger.info('Applying keyword columns...')
        df['keywords|0'] = (search_keywords(df['title'], terms=REVIEW_KEYWORDS) | search_keywords(df['abstract'], terms=REVIEW_KEYWORDS)).astype(int)
        df['keywords|1'] = (search_keywords(df['title'], terms=EVALUATION_KEYWORDS) | search_keywords(df['abstract'], terms=EVALUATION_KEYWORDS)).astype(int)
        df['keywords|2'] = (search_keywords(df['title'], terms=MENTAL_HEALTH_TERMS) | search_keywords(df['abstract'], terms=MENTAL_HEALTH_TERMS)).astype(int)

        logger.info('Writing sqlite...')
        write_sqlite(df, target=target / info.db_filename, logger=logger)

    write_geographies(
        df=df.join(df_places),
        target_min=target / info.slim_geo_filename,
        target_full=target / info.full_geo_filename,
        logger=logger,
        chunk_size=2000,
    )

    df_scatterplot = read_any_pd(source / 'scatterplot.csv', index_column='item_id')
    logger.info(f'Loaded scatterplot table: {df_scatterplot.shape}')
    df_scatterplot = rescale_projection(df=df_scatterplot, logger=logger)
    logger.info('Rescaled x/y values of scatterplot')

    df_keywords = read_any_pd(source / 'keywords.csv')
    logger.info(f'Loaded keywords table: {df_keywords.shape}')
    df_keywords = rescale_projection(df_keywords, logger=logger)
    logger.info('Rescaled x/y values of keywords')

    write_keywords(df_keywords, target=target / info.keywords_filename, logger=logger)
    write_base_info(df.join(df_scatterplot), target=target / info.arrow_filename, logger=logger)

    info.start_year = year_start
    info.end_year = year_end
    info_ = filter_labels(df=df, info_=info)
    # info.total = df.shape[0]
    with open(target / 'info.json', 'w') as fp_info:
        fp_info.write(info_.model_dump_json(indent=2, exclude_none=True))

    logger.info(f'Export finished, now available at {target.resolve()}')

    # TODO: topics_openalex.csv

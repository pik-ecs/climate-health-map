from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
import numpy as np

from climate_health_map.data.geographies import load_country_infos, flatten_country_groups, read_places_export
from climate_health_map.shared import read_any_pd, get_logger
from climate_health_map.scatterplot import rescale_projection

from .info import info, filter_labels
from .writers import write_base_info, write_sqlite, write_keywords, write_geographies
from .._utils import read_export


def prepare_lithub_export(
    source: Annotated[Path, typer.Option(help='Source directory')],
    target: Annotated[Path, typer.Option(help='Target directory')],
    source_annotations: Annotated[Path | None, typer.Option(help='path to human annotations')] = None,
    year_start: Annotated[int, typer.Option(help='Start year (incl)')] = 1990,
    year_end: Annotated[int, typer.Option(help='End year (incl)')] = 2025,
    skip_sqlite: Annotated[bool, typer.Option(help='Skip sqlite export for speedy info update')] = False,
    skip_geo: Annotated[bool, typer.Option(help='Skip geography stuff for speedy info update')] = False,
    loglevel: Annotated[str, typer.Option(help='Verbosity of logger')] = 'INFO',
):
    target.mkdir(parents=True, exist_ok=True)

    logger = get_logger(loglevel=loglevel, logger_name='lithub', run_log_init=True)
    df_countries = load_country_infos()
    logger.info(f'Loaded country infos: {df_countries.shape}')

    df = read_export(
        source_items=source / 'items.csv',
        source_annotations=source_annotations,
        source_classifications=source / 'classifications.csv',
        logger=logger,
        filter_mai=True,
        filter_rel=True,
        year_end=year_end,
        year_start=year_start,
        rescale_topic_scores=True,
        include_keyword_columns=True,
    )
    df['idx'] = np.arange(len(df))  # set a continuous index now that we are done filtering/joining
    logger.info(f'Joined tables: {df.shape}')

    logger.info('Reading affiliation data...')
    df_affiliations = read_any_pd(source / 'affiliations.csv', keep_default_na=False).merge(df_countries, left_on='iso2', right_on='iso2', how='left')
    logger.info('Flattening affiliation data...')
    df_affiliations_flat = flatten_country_groups(prefix='Affiliation', df=df_affiliations)
    logger.info(f'Loaded affiliations table: {df_affiliations.shape}; flattened: {df_affiliations_flat.shape}')

    logger.info('Reading mordecai data...')
    df_places = read_places_export(source=source / 'places.csv', df_countries=df_countries)
    logger.info('Flattening mordecai data...')
    df_places_flat = flatten_country_groups(prefix='Location', df=df_places)
    logger.info(f'Loaded places table: {df_places.shape}; flattened: {df_places_flat.shape}')

    df = df.join(df_affiliations_flat, how='left').join(df_places_flat, how='left')
    logger.info(f'Joined output table: {df.shape}')

    if not skip_sqlite:
        logger.info('Writing sqlite...')
        write_sqlite(df, target=target / info.db_filename, logger=logger)

    if not skip_geo:
        write_geographies(
            df=df_places.join(df[['idx']], how='left'),
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
    if (source / 'kws.csv').exists():
        df_keywords = pd.concat([df_keywords, read_any_pd(source / 'kws.csv')])
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

import logging
from pathlib import Path

import pandas as pd
from climate_health_map.data.geographies import load_country_infos, flatten_country_groups, read_places_export, get_naming_mask, fix_geographies, FEATURE_LOOKUP
from climate_health_map.shared import read_any_pd
from climate_health_map.data.labels import LABELS
from climate_health_map.data.export.utils import read_export


def read_base_data(
    source: Path,
    source_annotations: Path | None = None,
    year_start: int | None = 1990,
    year_end: int | None = 2025,
    filter_mai: bool = True,
    filter_rel: bool = True,
    prefilter_affiliations: bool = True,
    prefilter_locations: bool = True,
    fix_locations: bool = True,
    include_topic_groups: bool = True,
    exclude_columns: set[str] | None = None,
    region_iso3: set[str] | None = None,
    merge_taiwan_china: bool = True,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[tuple[str, str, str]]], pd.DataFrame, pd.DataFrame, dict[str, list[tuple[str, str, str]]]]:
    logger = logger or logging.getLogger('loader')
    exclude_columns = exclude_columns or set()

    df_countries = load_country_infos()
    logger.info(f'Loaded country infos: {df_countries.shape}')

    df = read_export(
        source_items=source / 'items.csv',
        source_annotations=source_annotations,
        source_classifications=source / 'classifications.csv',
        logger=logger,
        filter_mai=filter_mai,
        filter_rel=filter_rel,
        year_end=year_end,
        year_start=year_start,
        rescale_topic_scores=True,
        include_keyword_columns=True,
    )
    logger.info(f'Base-table: {df.shape}')
    df.drop(columns=exclude_columns, inplace=True)
    logger.info(f'Base-table after dropping exclude columns: {df.shape}')

    logger.info('Dropping some scores for specific columns to improve count logics...')
    for label in LABELS['sector'].labels:
        # Drop scores of sector columns if mitigation < 0.5
        df.loc[df['cat|0'] <= 0.5, label.column] = pd.NA

    df = df[df.columns[df.any()]]
    logger.info(f'Base-table after dropping empty columns: {df.shape}')

    logger.info('Reading affiliation data...')
    df_affiliations = read_any_pd(source / 'affiliations.csv', keep_default_na=False)
    if merge_taiwan_china:
        logger.info('Replacing Taiwanese affiliations with China')
        df_affiliations.loc[df_affiliations['iso2'] == 'TW', 'iso2'] = 'CN'
    df_affiliations = df_affiliations.merge(df_countries, left_on='iso2', right_on='iso2', how='left').set_index('item_id')
    logger.info(f'Affiliation shape: {df_affiliations.shape} (unique: {df_affiliations.index.nunique():,})')
    if prefilter_affiliations:
        df_affiliations = df_affiliations.loc[df_affiliations.index.intersection(df.index)]
        logger.info(f'Filtered affiliation shape: {df_affiliations.shape} (unique: {df_affiliations.index.nunique():,})')

    logger.info('Flattening affiliation data...')
    df_affiliations_flat, affiliation_groups = flatten_country_groups(prefix='Affiliation', df=df_affiliations)
    logger.info(f'Flattened affiliations table: {df_affiliations_flat.shape} (unique: {df_affiliations_flat.index.nunique():,})')

    logger.info('Reading mordecai data...')
    df_locations = read_places_export(source=source / 'places.csv', df_countries=df_countries, merge_taiwan_china=merge_taiwan_china).set_index('item_id')
    logger.info(f'locations shape: {df_locations.shape} (unique: {df_locations.index.nunique():,})')
    if prefilter_locations:
        df_locations = df_locations.loc[df_locations.index.intersection(df.index)]
        logger.info(f'Filtered locations shape: {df_locations.shape} (unique: {df_locations.index.nunique():,})')
    if fix_locations:
        logger.info(f'Dropping locations w/o geoname info (shape before: {df_locations.shape})...')
        df_locations = df_locations[df_locations['geonameid'].notna()]
        logger.info(f'Preparing filter mask (shape before: {df_locations.shape})...')
        mask_search_names = get_naming_mask(place_df=df_locations)
        logger.info(f'Fixing geographies and adding mask ({mask_search_names.sum():,} / {df_locations.shape[0]:,})...')
        df_locations = fix_geographies(place_df=df_locations[mask_search_names])
        logger.info(f'Adding feature column (current shape: {df_locations.shape})...')
        df_locations['feature'] = df_locations.apply(lambda row: FEATURE_LOOKUP.get(f'{row["feature_class"]}.{row["feature_code"] or ""}'), axis='columns')
        logger.info(f'Fixed locations shape: {df_locations.shape} (unique: {df_locations.index.nunique():,})')

    logger.info('Flattening mordecai data...')
    df_locations_flat, location_groups = flatten_country_groups(prefix='Location', df=df_locations)
    logger.info(f'Flattened locations table: {df_locations_flat.shape} (unique: {df_locations_flat.index.nunique():,})')

    if include_topic_groups:
        for grouping in ['topic-agg', 'topic-agg-agg']:
            agg_topics = LABELS[grouping]
            for group in agg_topics.labels:
                df[group.column] = (df[[col for col in group.topics if col not in exclude_columns]] > 0.5).any(axis=1).astype(int)
        logger.info(f'Base-table shape after adding aggregated topic columns: {df.shape}')

    if 'item_id' in df_locations.columns:
        df_locations.set_index('item_id', inplace=True)
    if 'item_id' in df_affiliations.columns:
        df_affiliations.set_index('item_id', inplace=True)

    major_rename = {label.column: label.name for label in LABELS['cat'].labels}
    df['Climate category'] = df.rename(columns=major_rename)[major_rename.values()].idxmax(axis=1)
    df['incl_major'] = (df['rel_major|1'] > 0.5) & (df[['cat|0', 'cat|1', 'cat|2']] > 0.5).any(axis=1)
    df['incl_impacts'] = (df['cat|2'] > 0.5) & (df['rel_impacts|1'] > 0.5)
    df['incl_location'] = df.index.isin(df_locations.index)
    df['incl_affiliation'] = df.index.isin(df_affiliations.index)
    df['publication_year'] = df['publication_year'].astype('Int32')

    if region_iso3:
        logger.info(f'locations/affiliations tables before region filter on ISO3: {df_locations.shape} / {df_affiliations.shape}')
        df_locations = df_locations[df_locations['iso3'].isin(region_iso3)]
        df_affiliations = df_affiliations[df_affiliations['iso3'].isin(region_iso3)]
        logger.info(f'locations/affiliations tables after region filter on ISO3: {df_locations.shape} / {df_affiliations.shape}')
        df['incl_region_location'] = df.index.isin(df_locations.index)
        df['incl_region_affiliation'] = df.index.isin(df_affiliations.index)

    df = df.rename(columns={'publication_year': 'Publication year'}).reset_index().set_index('item_id', drop=False)
    df_locations = (
        df_locations.reset_index()
        .set_index('item_id', drop=False)
        .join(df[['Publication year', 'Climate category', 'incl_major', 'incl_impacts', 'incl_location', 'incl_affiliation']])
    )
    df_affiliations = (
        df_affiliations.reset_index()
        .set_index('item_id', drop=False)
        .join(df[['Publication year', 'Climate category', 'incl_major', 'incl_impacts', 'incl_location', 'incl_affiliation']])
    )

    # Return copies to clear memory and fragmentation
    return df.copy(), df_locations.copy(), df_locations_flat.copy(), location_groups, df_affiliations.copy(), df_affiliations_flat.copy(), affiliation_groups

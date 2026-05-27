import logging
from pathlib import Path
from typing import Annotated, Literal
from datetime import datetime

import typer
import pandas as pd
import cartopy.io.shapereader as shpreader
import geopandas

from climate_health_map.data.geographies import merge_grid_info, load_grid_data, load_annual_population, LOCATION_FEATURE_CODES, load_country_infos
from climate_health_map.shared import get_logger
from climate_health_map.data.labels import LABELS, Collection
from .loaders import read_base_data
from .regional_groups import region_groups as indicator_regions

exclude_columns = {
    'topic-4-0|8',
    'topic-4-0|18',
    'topic-4-0|38',
    'topic-agg-4|0',
    'topic-agg-agg|4',
    # 'driver|0',  # CO2rise
    # 'driver|2',  # Seasonal Change
    # 'driver|4',  # Sea-level rise
    # 'health|10',  # Metabolic Disorders
}
default_label_groups = {
    # 'rel_major',
    # 'rel_impacts',
    'cat',
    'driver',
    'event',
    'health',
    'expose',
    'attr',
    'sector',
    'keywords',
    'topic-agg-agg',
    # 'type',
    # 'cont',
    # 'gender_outcome',
    # 'notes_major',
    # 'notes_impacts',
    # 'topic',
}


def annual_counts(
    df: pd.DataFrame,
    label_groups: set[str],
    region: str,
    geography_filter: Literal['affiliation', 'location', 'region_affiliation', 'region_location'] | None = None,
    count_primary_class: bool = False,
    threshold: float = 0.5,
    group_by: str | list[str] | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    logger = logger if logger is not None else logging.getLogger('counting')

    py_range = list(range(df['Publication year'].min(), df['Publication year'].max() + 1))

    if group_by is None:
        group_by = 'Publication year'
        agg_cols = [group_by, 'item_id']
        table_index = pd.MultiIndex.from_product([['Full dataset'], py_range], names=['Group', 'Publication year'])
    elif type(group_by) is str:
        group_by = [group_by, 'Publication year']
        agg_cols = group_by + ['item_id']
        table_index = pd.MultiIndex.from_product([df[col].unique() for col in group_by if col != 'Publication year'] + [py_range], names=group_by)
    else:
        raise NotImplementedError()

    data = {
        ('General', 'Number of studies on climate & health'): df.loc[df['incl_major'], agg_cols].groupby(group_by)['item_id'].nunique(),
        ('General', 'Number of studies mentioning location'): df.loc[df['incl_location'] & df['incl_major'], agg_cols].groupby(group_by)['item_id'].nunique(),
        ('General', 'Number of studies with affiliation info'): df.loc[df['incl_affiliation'] & df['incl_major'], agg_cols]
        .groupby(group_by)['item_id']
        .nunique(),
        ('General', f'Number of studies mentioning location in {region}'): df.loc[df['incl_region_location'] & df['incl_major'], agg_cols]
        .groupby(group_by)['item_id']
        .nunique(),
        ('General', f'Number of studies with affiliation in {region}'): df.loc[df['incl_region_affiliation'] & df['incl_major'], agg_cols]
        .groupby(group_by)['item_id']
        .nunique(),
        ('General', 'Number of studies on impacts'): df.loc[df['incl_major'] & df['incl_impacts'], agg_cols].groupby(group_by)['item_id'].nunique(),
        ('General', 'Number of studies on impacts and location'): df.loc[df['incl_major'] & df['incl_impacts'] & df['incl_location'], agg_cols]
        .groupby(group_by)['item_id']
        .nunique(),
        ('General', f'Number of studies on impacts in {region}'): df.loc[df['incl_major'] & df['incl_impacts'] & df['incl_region_location'], agg_cols]
        .groupby(group_by)['item_id']
        .nunique(),
        ('General', f'Number of studies on impacts in {region} with attributable impact'): df.loc[
            df['incl_major'] & df['incl_impacts'] & df['incl_location'] & df['incl_region_location'] & df['grid_attributable'], agg_cols
        ]
        .groupby(group_by)['item_id']
        .nunique(),
    }

    totals = {
        ('General', 'Number of studies on climate & health'): df.loc[df['incl_major'], 'item_id'].nunique(),
        ('General', 'Number of studies mentioning location'): df.loc[df['incl_major'] & df['incl_location'], 'item_id'].nunique(),
        ('General', 'Number of studies with affiliation info'): df.loc[df['incl_major'] & df['incl_affiliation'], 'item_id'].nunique(),
        ('General', f'Number of studies mentioning location in {region}'): df.loc[df['incl_major'] & df['incl_region_location'], 'item_id'].nunique(),
        ('General', f'Number of studies with affiliation in {region}'): df.loc[df['incl_major'] & df['incl_region_affiliation'], 'item_id'].nunique(),
        ('General', 'Number of studies on impacts'): df.loc[df['incl_major'] & df['incl_impacts'], 'item_id'].nunique(),
        ('General', 'Number of studies on impacts and location'): df.loc[df['incl_major'] & df['incl_impacts'] & df['incl_location'], 'item_id'].nunique(),
        ('General', f'Number of studies on impacts in {region}'): df.loc[
            df['incl_major'] & df['incl_impacts'] & df['incl_region_location'], 'item_id'
        ].nunique(),
        ('General', f'Number of studies on impacts in {region} with attributable impact'): df.loc[
            df['incl_major'] & df['incl_impacts'] & df['incl_location'] & df['incl_region_location'] & df['grid_attributable'], 'item_id'
        ].nunique(),
    }

    column_groups = ['General']
    for group_ in label_groups:
        group = LABELS[group_]
        column_groups.append(group.name)
        columns = [label.column for label in group.labels if label.column in df.columns and label.column not in exclude_columns]
        mask = df[columns].notna().any(axis=1) & (df[columns] > threshold).any(axis=1)
        df_filtered = df[mask]
        primary = df_filtered[columns].idxmax(axis=1)
        for label in group.labels:
            if label.column not in df_filtered.columns or label.column in exclude_columns:
                continue
            # logger.info(f'{group_} -> {label} // data: {len(data):,}, totals: {len(totals):,}')

            extra_mask = (primary == label.column) if count_primary_class else df_filtered[label.column] > threshold
            if group.collection == Collection.IMPACTS:
                extra_mask &= df_filtered['incl_impacts'] & df_filtered['grid_attributable']
            if geography_filter == 'location':
                extra_mask &= df_filtered['incl_location']
            if geography_filter == 'affiliation':
                extra_mask &= df_filtered['incl_affiliation']
            if geography_filter == 'region_location':
                extra_mask &= df['incl_location'] & df_filtered['incl_region_location']
            if geography_filter == 'region_affiliation':
                extra_mask &= df_filtered['incl_region_affiliation']

            data[(group.name, label.name)] = df_filtered.loc[extra_mask, agg_cols].groupby(group_by)['item_id'].nunique()
            totals[(group.name, label.name)] = df_filtered.loc[extra_mask, 'item_id'].nunique()
            # logger.info(f'{data[(group.name, label.name)].shape}')
        logger.info(f'Done with {group_} // data: {len(data):,}, totals: {len(totals):,}')

    table = pd.DataFrame(data, index=table_index).fillna(0).astype(int)
    table = pd.concat([table, table.groupby(level=0).sum().set_index(pd.Index(['Total'] * len(table.groupby(level=0)), name='Publication year'), append=True)])
    if len(table.groupby(level=0)) > 1:
        table.loc[('All groups', 'Total')] = pd.Series(totals)

    return table[column_groups]  # table.sort_index(axis=1)[column_groups]


def prepare_lancet_excel_export(
    source: Annotated[Path, typer.Option(help='Source directory')],
    target: Annotated[Path, typer.Option(help='Target directory')],
    source_annotations: Annotated[Path | None, typer.Option(help='path to human annotations')] = None,
    year_start: Annotated[int, typer.Option(help='Start year (incl)')] = 1990,
    year_end: Annotated[int, typer.Option(help='End year (incl)')] = 2025,
    threshold: Annotated[float, typer.Option(help='Threshold')] = 0.5,
    loglevel: Annotated[str, typer.Option(help='Verbosity of logger')] = 'INFO',
) -> None:
    logger_ = get_logger(loglevel=loglevel, logger_name='lancet-excel', run_log_init=True)
    target.mkdir(parents=True, exist_ok=True)
    df_grid = load_grid_data()
    df_population = load_annual_population()
    df_countries = load_country_infos()

    shpfilename = shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries')
    shapes = geopandas.read_file(shpfilename, encoding='utf-8')[['ISO_A2', 'ISO_A3', 'geometry']]

    # ['featurecla', 'scalerank', 'LABELRANK', 'SOVEREIGNT', 'SOV_A3', 'ADM0_DIF', 'LEVEL', 'TYPE', 'TLC', 'ADMIN', 'ADM0_A3', 'GEOU_DIF', 'GEOUNIT', 'GU_A3',
    #  'SU_DIF', 'SUBUNIT', 'SU_A3', 'BRK_DIFF', 'NAME', 'NAME_LONG', 'BRK_A3', 'BRK_NAME', 'BRK_GROUP', 'ABBREV', 'POSTAL', 'FORMAL_EN', 'FORMAL_FR',
    #  'NAME_CIAWF', 'NOTE_ADM0', 'NOTE_BRK', 'NAME_SORT', 'NAME_ALT', 'MAPCOLOR7', 'MAPCOLOR8', 'MAPCOLOR9', 'MAPCOLOR13', 'POP_EST', 'POP_RANK', 'POP_YEAR',
    #  'GDP_MD', 'GDP_YEAR', 'ECONOMY', 'INCOME_GRP', 'FIPS_10', 'ISO_A2', 'ISO_A2_EH', 'ISO_A3', 'ISO_A3_EH', 'ISO_N3', 'ISO_N3_EH', 'UN_A3', 'WB_A2', 'WB_A3',
    #  'WOE_ID', 'WOE_ID_EH', 'WOE_NOTE', 'ADM0_ISO', 'ADM0_DIFF', 'ADM0_TLC', 'ADM0_A3_US', 'ADM0_A3_FR', 'ADM0_A3_RU', 'ADM0_A3_ES', 'ADM0_A3_CN', 'ADM0_A3_TW',
    #  'ADM0_A3_IN', 'ADM0_A3_NP', 'ADM0_A3_PK', 'ADM0_A3_DE', 'ADM0_A3_GB', 'ADM0_A3_BR', 'ADM0_A3_IL', 'ADM0_A3_PS', 'ADM0_A3_SA', 'ADM0_A3_EG', 'ADM0_A3_MA',
    #  'ADM0_A3_PT', 'ADM0_A3_AR', 'ADM0_A3_JP', 'ADM0_A3_KO', 'ADM0_A3_VN', 'ADM0_A3_TR', 'ADM0_A3_ID', 'ADM0_A3_PL', 'ADM0_A3_GR', 'ADM0_A3_IT', 'ADM0_A3_NL',
    #  'ADM0_A3_SE', 'ADM0_A3_BD', 'ADM0_A3_UA', 'ADM0_A3_UN', 'ADM0_A3_WB', 'CONTINENT', 'REGION_UN', 'SUBREGION', 'REGION_WB', 'NAME_LEN', 'LONG_LEN',
    #  'ABBREV_LEN', 'TINY', 'HOMEPART', 'MIN_ZOOM', 'MIN_LABEL', 'MAX_LABEL', 'LABEL_X', 'LABEL_Y', 'NE_ID', 'WIKIDATAID', 'NAME_AR', 'NAME_BN', 'NAME_DE',
    #  'NAME_EN', 'NAME_ES', 'NAME_FA', 'NAME_FR', 'NAME_EL', 'NAME_HE', 'NAME_HI', 'NAME_HU', 'NAME_ID', 'NAME_IT', 'NAME_JA', 'NAME_KO', 'NAME_NL', 'NAME_PL',
    #  'NAME_PT', 'NAME_RU', 'NAME_SV', 'NAME_TR', 'NAME_UK', 'NAME_UR', 'NAME_VI', 'NAME_ZH', 'NAME_ZHT', 'FCLASS_ISO', 'TLC_DIFF', 'FCLASS_TLC', 'FCLASS_US',
    #  'FCLASS_FR', 'FCLASS_RU', 'FCLASS_ES', 'FCLASS_CN', 'FCLASS_TW', 'FCLASS_IN', 'FCLASS_NP', 'FCLASS_PK', 'FCLASS_DE', 'FCLASS_GB', 'FCLASS_BR', 'FCLASS_IL',
    #  'FCLASS_PS', 'FCLASS_SA', 'FCLASS_EG', 'FCLASS_MA', 'FCLASS_PT', 'FCLASS_AR', 'FCLASS_JP', 'FCLASS_KO', 'FCLASS_VN', 'FCLASS_TR', 'FCLASS_ID', 'FCLASS_PL',
    #  'FCLASS_GR', 'FCLASS_IT', 'FCLASS_NL', 'FCLASS_SE', 'FCLASS_BD', 'FCLASS_UA', 'geometry']

    for region, region_config in indicator_regions.items():
        logger = logger_.getChild(region)
        logger.info(f'Processing {region}...')

        df, df_locations, df_locations_flat, location_groups, df_affiliations, df_affiliations_flat, affiliation_groups = read_base_data(
            source=source,
            source_annotations=source_annotations,
            year_start=year_start,
            year_end=year_end,
            filter_mai=True,
            filter_rel=True,
            prefilter_affiliations=True,
            prefilter_locations=True,
            fix_locations=True,
            region_iso3=region_config['Countries'],
            logger=logger_.getChild('reader'),
        )
        df = (
            df.rename(columns={'publication_year': 'Publication year'})
            .reset_index(drop='item_id' in df.columns)
            .set_index(
                'item_id',
                drop='item_id' not in df.columns,
            )
        )

        if region == 'Europe':
            shapes_ = geopandas.read_file('data/shapes_2026/NUTS_RG_10M_2024_4326_with_UK_2021', encoding='utf-8').rename(
                columns={'CNTR_CODE': 'ISO_A2'},
            ).replace({'ISO_A2': {'UK': 'GB'}})
            shapes_ = shapes_.merge(df_countries, left_on='ISO_A2', right_on='iso2').rename(columns={'iso3': 'ISO_A3'})
        else:
            shapes_ = shapes.drop(columns=['ISO_A2', 'WB_A3', 'WB_REGION', 'WB_STATUS', 'NAM_0', 'NAM_1', 'ADM1CD_c', 'GEOM_SRCE'], errors='ignore')

        df, df_locations, grid_counts = merge_grid_info(
            df=df,
            df_locations=df_locations,
            df_grid=df_grid,
            df_population=df_population,
            shapes=shapes_,
            location_codes_abstracted=LOCATION_FEATURE_CODES[0],
            location_codes_direct=LOCATION_FEATURE_CODES[3] | LOCATION_FEATURE_CODES[2] | LOCATION_FEATURE_CODES[1],
            logger=logger_.getChild('merge'),
            reference_year=2024,  # We propbably want to use the latest population count for normalisation
            resolution=2.5,
        )
        df_locations = df_locations.set_index('item_id', drop=False)

        for count_primary_class, postfix in [
            (False, 'multi'),
            (True, 'primary'),
        ]:
            with pd.ExcelWriter(target / f'{datetime.today().strftime("%Y%m%d")}_LancetCountdown_{region}_{postfix}.xlsx', engine='xlsxwriter') as writer:
                for geo_filter, sheet_name in [
                    (None, 'Total counts'),
                    ('affiliation', 'Counts w. affiliation'),
                    ('location', 'Counts w. location'),
                    ('region_affiliation', f'{region} affiliation'),
                    ('region_location', f'{region} location'),
                ]:
                    logger.info(f'Working on {region} {postfix} {sheet_name}...')
                    subframe = annual_counts(
                        df=df,
                        label_groups=default_label_groups,
                        count_primary_class=count_primary_class,
                        geography_filter=geo_filter,  # type: ignore[arg-type]
                        threshold=threshold,
                        region=region,
                        logger=logger.getChild('counter'),
                    )
                    subframe.droplevel(level=0).to_excel(writer, sheet_name=sheet_name, startrow=2, startcol=1, header=True, index=True)

                for grouping, name in region_config['Groups'].items():
                    for sheet_name, join_df in [
                        (f'{name} (by location)', df_locations),
                        (f'{name} (by affiliation)', df_affiliations),
                    ]:
                        logger.info(f'Working on {region} {postfix} {sheet_name}...')
                        subframe = annual_counts(
                            df=join_df[[grouping]].join(df, how='inner'),
                            label_groups=default_label_groups,
                            count_primary_class=count_primary_class,
                            group_by=grouping,
                            threshold=threshold,
                            region=region,
                            logger=logger.getChild('counter'),
                        )
                        subframe.to_excel(writer, sheet_name=sheet_name, startrow=2, startcol=1, header=True, index=True)


if __name__ == '__main__':
    typer.run(prepare_lancet_excel_export)

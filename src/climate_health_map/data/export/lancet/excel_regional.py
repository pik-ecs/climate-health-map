from pathlib import Path
from typing import Annotated, Literal
from datetime import datetime

import typer
import pandas as pd

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
    group_by: str | None = None,
) -> pd.DataFrame:
    py_range = list(range(df['Publication year'].min(), df['Publication year'].max() + 1))

    if group_by is None:
        group_by = 'Publication year'
        table_index = pd.MultiIndex.from_product([['Full dataset'], py_range], names=['Group', 'Publication year'])
    elif type(group_by) is str:
        group_by = [group_by, 'Publication year']
        table_index = pd.MultiIndex.from_product([df[col].unique() for col in group_by if col != 'Publication year'] + [py_range], names=group_by)
    else:
        raise NotImplementedError()

    data = {
        ('General', 'Number of studies on climate & health'): df.groupby(group_by)['item_id'].nunique(),
        ('General', 'Number of studies mentioning location'): df[df['incl_location']].groupby(group_by)['item_id'].nunique(),
        ('General', 'Number of studies with affiliation info'): df[df['incl_affiliation']].groupby(group_by)['item_id'].nunique(),
        ('General', f'Number of studies mentioning location in {region}'): df[df['incl_region_location']].groupby(group_by)['item_id'].nunique(),
        ('General', f'Number of studies with affiliation in {region}'): df[df['incl_region_affiliation']].groupby(group_by)['item_id'].nunique(),
        ('General', 'Number of studies on impacts'): df[df['incl_impacts']].groupby(group_by)['item_id'].nunique(),
        ('General', 'Number of studies on impacts and location'): df[df['incl_impacts'] & df['incl_location']].groupby(group_by)['item_id'].nunique(),
        ('General', 'Number of studies on impacts in region'): df[df['incl_impacts'] & df['incl_region_location']].groupby(group_by)['item_id'].nunique(),
    }
    totals = {
        ('General', 'Number of studies on climate & health'): df['item_id'].nunique(),
        ('General', 'Number of studies mentioning location'): df[df['incl_location']]['item_id'].nunique(),
        ('General', 'Number of studies with affiliation info'): df[df['incl_affiliation']]['item_id'].nunique(),
        ('General', f'Number of studies mentioning location in {region}'): df[df['incl_region_location']]['item_id'].nunique(),
        ('General', f'Number of studies with affiliation in {region}'): df[df['incl_region_affiliation']]['item_id'].nunique(),
        ('General', 'Number of studies on impacts'): df[df['incl_impacts']]['item_id'].nunique(),
        ('General', 'Number of studies on impacts and location'): df[df['incl_impacts'] & df['incl_location']]['item_id'].nunique(),
        ('General', 'Number of studies on impacts in region'): df[df['incl_impacts'] & df['incl_region_location']]['item_id'].nunique(),
    }
    column_groups = ['General']
    for group_ in label_groups:
        group = LABELS[group_]
        column_groups.append(group.name)
        columns = [label.column for label in group.labels if label.column in df.columns and label.column not in exclude_columns]
        mask = (df[columns].notna() & (df[columns] > threshold)).any(axis=1)
        primary = df[mask][columns].fillna(0).idxmax(axis=1)
        for label in group.labels:
            if label.column not in df.columns or label.column in exclude_columns:
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

            data[(group.name, label.name)] = df[mask & extra_mask].groupby(group_by)['item_id'].nunique()
            totals[(group.name, label.name)] = df[mask & extra_mask]['item_id'].nunique()

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
):
    logger_ = get_logger(loglevel=loglevel, logger_name='lancet-excel', run_log_init=True)
    target.mkdir(parents=True, exist_ok=True)

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
            logger=logger.getChild('reader'),
        )
        df = df.rename(columns={'publication_year': 'Publication year'}).reset_index().set_index('item_id', drop=False)

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
                    subframe = annual_counts(
                        df=df,
                        label_groups=default_label_groups,
                        count_primary_class=count_primary_class,
                        geography_filter=geo_filter,
                        threshold=threshold,
                        region=region,
                    )
                    subframe.droplevel(level=0).to_excel(writer, sheet_name=sheet_name, startrow=2, startcol=1, header=True, index=True)

                for grouping, name in region_config['Groups'].items():
                    for sheet_name, join_df in [
                        (f'{name} (by location)', df_locations),
                        (f'{name} (by affiliation)', df_affiliations),
                    ]:
                        subframe = annual_counts(
                            df=join_df[[grouping]].join(df, how='inner'),
                            label_groups=default_label_groups,
                            count_primary_class=count_primary_class,
                            group_by=grouping,
                            threshold=threshold,
                            region=region,
                        )
                        subframe.to_excel(writer, sheet_name=sheet_name, startrow=2, startcol=1, header=True, index=True)


if __name__ == '__main__':
    typer.run(prepare_lancet_excel_export)

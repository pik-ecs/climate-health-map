import shutil
from pathlib import Path
from typing import Annotated, Literal, Any
from datetime import datetime
import logging

import numpy as np
import typer
import pandas as pd
from openpyxl.styles import Font, Alignment
from openpyxl.worksheet.formula import DataTableFormula, ArrayFormula
from openpyxl.worksheet.worksheet import Worksheet

from climate_health_map.data.geographies import load_annual_population, load_grid_data, merge_grid_info, LOCATION_FEATURE_CODES
from climate_health_map.shared import get_logger
from climate_health_map.data.labels import LABELS, Collection
from .loaders import read_base_data

pd.options.display.max_columns = 650
pd.options.display.max_rows = 300
pd.options.display.width = 50000
pd.options.display.max_colwidth = 50000

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
impact_groups = {
    # 'rel_major',
    # 'rel_impacts',
    # 'cat',
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
region_groups = {
    'Group (Lancet 2026)': 'Lancet',
    'Group (WHO 2026)': 'WHO',
    'Group (HDI 2026)': 'HDI',
    'Name (Lancet 2026)': 'Countries',
    # 'Region (IPCC AR6, 6)',
    # 'Region (IPCC AR6, 10)',
    # 'Region (WorldBank 2026)',
    # 'Income group (WorldBank 2026)',
    # 'Lending category (WorldBank 2026)',
    # 'Continent (Name)',
}


def annual_counts(
    df: pd.DataFrame,
    label_groups: set[str],
    geography_filter: Literal['affiliation', 'location'] | None = None,
    count_primary_class: bool = False,
    threshold: float = 0.5,
    group_by: str | list[str] | None = None,
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
    }
    totals = {
        ('General', 'Number of studies on climate & health'): df['item_id'].nunique(),
        ('General', 'Number of studies mentioning location'): df[df['incl_location']]['item_id'].nunique(),
        ('General', 'Number of studies with affiliation info'): df[df['incl_affiliation']]['item_id'].nunique(),
    }
    column_groups = ['General']
    for group_ in label_groups:
        group = LABELS[group_]
        column_groups.append(group.name)
        columns = [label.column for label in group.labels if label.column in df.columns and label.column not in exclude_columns]
        mask = (df[columns].notna() & (df[columns] > threshold)).any(axis=1)
        primary = df[columns].fillna(0).idxmax(axis=1)
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

            data[(group.name, label.name)] = df[mask & extra_mask].groupby(group_by)['item_id'].nunique()
            totals[(group.name, label.name)] = df[mask & extra_mask]['item_id'].nunique()

    table = pd.DataFrame(data, index=table_index).fillna(0).astype(int)
    table = pd.concat([table, table.groupby(level=0).sum().set_index(pd.Index(['Total'] * len(table.groupby(level=0)), name='Publication year'), append=True)])

    if len(table.groupby(level=0)) > 1:
        table.loc[('All groups', 'Total'),:] = pd.Series(totals)

    return table[column_groups]  # table.sort_index(axis=1)[column_groups]


def write_with_fmt(sheet: Worksheet, row: int, col: int, data: bool | float | str | DataTableFormula | ArrayFormula, fmt: dict[str, Any]):
    """
    Equivalent to xlsxwriter's worksheet.write(row, col, value, format)
    Note: openpyxl uses 1-based indexing, but this function converts
    0-based (xlsxwriter style) to 1-based automatically.
    """
    cell = sheet.cell(row=row + 1, column=col + 1, value=data)

    for field, value in fmt.items():
        setattr(cell, field, value)


def summary_tables(df: pd.DataFrame, writer: pd.ExcelWriter, sheet_name: str, col_offset: int, row_offset: int) -> None:
    fmt = {
        'font': Font(bold=True),
        'alignment': Alignment(horizontal='left', vertical='top', wrap_text=False),
        # 'border': Border(
        #     top=Side(border_style='thin', color='000000'),
        #     left=Side(border_style='thin', color='000000'),
        #     right=Side(border_style='thin', color='000000'),
        #     bottom=Side(border_style='thin', color='000000'),
        # ),
    }

    if sheet_name not in writer.sheets:
        writer.book.add_worksheet(sheet_name)
    worksheet = writer.sheets[sheet_name]

    write_with_fmt(sheet=worksheet, row=row_offset, col=col_offset, data='Totals', fmt=fmt)
    totals = pd.concat(
        [
            pd.DataFrame({
                (grp, f'{df_.index.min()}–{df_.index.max()}'): df_.sum(axis=0),
                (grp, '2006–2015'): df_.loc[2006:2015].sum(axis=0),
                (grp, '2016–2025'): df_.loc[2016:2025].sum(axis=0),
                (grp, '2006–2025'): df_.loc[2006:2025].sum(axis=0),
                (grp, 'CAGR 2006–2015'): (pow(df_.loc[2015] / df_.loc[2006], 1 / 10) - 1) * 100,
                (grp, 'CAGR 2016–2025'): (pow(df_.loc[2025] / df_.loc[2016], 1 / 10) - 1) * 100,
                (grp, f'CAGR {df_.index.min()}–{df_.index.max()}'): (pow(df_.iloc[-1] / df_.iloc[0], 1 / df_.shape[0]) - 1) * 100,
            })
            for grp, df_ in (
            df.drop(index='Total', level=1, errors='ignore')
            .drop('', level=1, errors='ignore')
            .drop(np.nan, level=0, errors='ignore')
            .rename(int, level=1)
            .reset_index(level=0)
            .groupby(df.index.names[0])[list(set(df.columns) - {df.index.names[0]})]
        )], axis=1
    ).T

    # No grouping, drop group level from index
    if len(totals.groupby(level=0)) < 2:
        totals = totals.droplevel(level=0)

    totals.sort_index(axis=1).to_excel(writer, sheet_name=sheet_name, startrow=row_offset + 1, startcol=col_offset, header=True, index=True, inf_rep='', float_format='%.2f')


def write_workbook(
    target: Path,
    df: pd.DataFrame,
    df_locations: pd.DataFrame,
    df_affiliations: pd.DataFrame,
    df_grid: pd.DataFrame,
    df_population: pd.DataFrame,
    mask: pd.Series,
    label_groups: set[str],
    count_primary_class: bool,
    properties: dict[str, str],
    threshold: float = 0.5,
) -> None:
    with pd.ExcelWriter(target, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        logging.debug('Computing impact filtering')
        df_imp = pd.DataFrame(
            {
                'Relevant records': df[df['incl_major']].groupby('Publication year')['item_id'].nunique(),
                'Relevant with affiliation information': df[df['incl_affiliation'] & df['incl_major']].groupby('Publication year')['item_id'].nunique(),
                'Relevant with location mention': df[df['incl_location'] & df['incl_major']].groupby('Publication year')['item_id'].nunique(),
                'Relevant records and climate category impacts': df[df['incl_major'] & (df['cat|2'] > 0.5)].groupby('Publication year')['item_id'].nunique(),
                'Relevant impacts research': df[df['incl_impacts'] & df['incl_major']].groupby('Publication year')['item_id'].nunique(),
                'Relevant impacts research with location': df[df['incl_location'] & df['incl_major'] & df['incl_impacts']]
                .groupby('Publication year')['item_id']
                .nunique(),
                'Records on impacts with location in attributable cell': df[
                    df['incl_location'] & df['incl_major'] & df['incl_impacts'] & df['grid_attributable']
                    ]
                .groupby('Publication year')['item_id']
                .nunique(),
                'Records on impacts with location in non-attributable cell': df[
                    df['incl_location'] & df['incl_major'] & df['incl_impacts'] & ~df['grid_attributable']
                    ]
                .groupby('Publication year')['item_id']
                .nunique(),
            },
        )
        pd.concat([df_imp, df_imp.sum().rename('Total').to_frame().T]).fillna(0).to_excel(writer, sheet_name='Publication filtering')

        logging.debug(f'df.shape={df.shape} | df[mask].shape={df[mask].shape}')
        df = df[mask]

        # write_with_style(df=subframe, writer=writer, sheet_name='Total counts', startrow=2, startcol=1, header=True, index=True)
        logging.debug('Preparing full subframe...')
        subframe_full = annual_counts(df=df, label_groups=label_groups, count_primary_class=count_primary_class, threshold=threshold)
        subframe_full.droplevel(level=0).to_excel(writer, sheet_name='Total counts', startrow=2, startcol=1, header=True, index=True)
        summary_tables(df=subframe_full, writer=writer, sheet_name='Total counts', col_offset=subframe_full.shape[1] + 5, row_offset=2)

        logging.debug('Preparing annual location-based counts...')
        subframe_loc = annual_counts(
            df=df,
            label_groups=label_groups,
            count_primary_class=count_primary_class,
            geography_filter='location',
            threshold=threshold,
        )
        subframe_loc.droplevel(level=0).to_excel(writer, sheet_name='Totals (with location)', startrow=2, startcol=1, header=True, index=True)
        summary_tables(df=subframe_loc, writer=writer, sheet_name='Totals (with location)', col_offset=subframe_loc.shape[1] + 5, row_offset=2)

        logging.debug('Preparing annual affiliation-based counts....')
        subframe_aff = annual_counts(
            df=df,
            label_groups=label_groups,
            count_primary_class=count_primary_class,
            geography_filter='affiliation',
            threshold=threshold,
        )
        subframe_aff.droplevel(level=0).to_excel(writer, sheet_name='Totals (with affiliation)', startrow=2, startcol=1, header=True, index=True)
        summary_tables(df=subframe_aff, writer=writer, sheet_name='Totals (with affiliation)', col_offset=subframe_aff.shape[1] + 5, row_offset=2)

        for grouping, name in region_groups.items():
            logging.debug(f'Preparing annual affiliation-based counts grouped by {grouping} ({name})...')
            for sheet_name, join_df in [
                (f'{name} (by location)', df_locations),
                (f'{name} (by affiliation)', df_affiliations),
            ]:
                subframe = annual_counts(
                    df=join_df.set_index('item_id', drop=False)[[grouping]].join(df, how='inner'),
                    label_groups=label_groups,
                    count_primary_class=count_primary_class,
                    group_by=grouping,
                    threshold=threshold,
                )
                subframe.to_excel(writer, sheet_name=sheet_name, startrow=2, startcol=1, header=True, index=True)
                summary_tables(df=subframe, writer=writer, sheet_name=sheet_name, col_offset=subframe.shape[1] + 5, row_offset=2)

        logging.debug('Preparing reference tables...')
        df_population.pivot(
            index='iso3',
            columns='year',
            values='Population',
        ).to_excel(writer, sheet_name='Population data', startrow=1, startcol=1, header=True, index=True)
        df_grid.to_excel(writer, sheet_name='Impacts attribution', startrow=1, startcol=1, header=True, index=False)


def prepare_lancet_excel_export(
    source: Annotated[Path, typer.Option(help='Source directory')],
    target: Annotated[Path, typer.Option(help='Target directory')],
    shapefile: Annotated[Path, typer.Option(help='Shape file')] = Path('data/shapes_2026/WB_GAD_ADM0_complete.shp'),
    source_annotations: Annotated[Path | None, typer.Option(help='path to human annotations')] = None,
    year_start: Annotated[int, typer.Option(help='Start year (incl)')] = 1990,
    year_end: Annotated[int, typer.Option(help='End year (incl)')] = 2025,
    threshold: Annotated[float, typer.Option(help='Threshold')] = 0.5,
    loglevel: Annotated[str, typer.Option(help='Verbosity of logger')] = 'DEBUG',
) -> None:
    import geopandas as gpd

    logger = get_logger(loglevel=loglevel, logger_name='lancet-excel', run_log_init=True)
    target.mkdir(parents=True, exist_ok=True)

    df, df_locations, _df_locations_flat, location_groups, df_affiliations, _df_affiliations_flat, affiliation_groups = read_base_data(
        source=source,
        source_annotations=source_annotations,
        year_start=year_start,
        year_end=year_end,
        filter_mai=True,
        filter_rel=True,
        prefilter_affiliations=True,
        prefilter_locations=True,
        fix_locations=True,
        logger=logger.getChild('reader'),
    )

    df_grid = load_grid_data()
    df_population = load_annual_population()
    df_population = df_population[(df_population['year'] >= df['Publication year'].min()) & (df_population['year'] <= df['Publication year'].max())]

    # Load geoshapes
    shapes = gpd.read_file(shapefile, encoding='utf-8')

    # Match grid info to locations and items
    # This is a bit more complex, because some locations need to be projected to a shape (e.g. country) and then back to grid cells
    df, df_locations, grid_counts = merge_grid_info(
        df=df,
        df_locations=df_locations,
        df_grid=df_grid,
        df_population=df_population,
        shapes=shapes,
        location_codes_abstracted=LOCATION_FEATURE_CODES[0],
        location_codes_direct=LOCATION_FEATURE_CODES[3] | LOCATION_FEATURE_CODES[2] | LOCATION_FEATURE_CODES[1],
        logger=logger,
        reference_year=2024,  # We probably want to use the latest population count for normalisation
        resolution=2.5,
    )

    base_properties = {
        'title': 'Lancet Countdown ',
        'author': 'Tim Repke',
        'company': 'Potsdam Institute for Climate Impact Research (PIK)',
        'comments': 'Last updated February 2026 with data from OpenAlex',
        # "status": "Quo",
        # "manager": "Name",
        # "category": "Example spreadsheets",
        # "subject": "With document properties",
        # "keywords": "Sample, Example, Properties",
    }

    logger.info('Writing major category spreadsheet with multi-class classification')
    fn_target = target / f'{datetime.today().strftime("%Y%m%d")}_LancetCountdown_Indicator531_multi_Global.xlsx'
    shutil.copy(Path('data/exports/lancet/Global/Datasheets_base/LancetCountdown_Indicator531_multi_Global.xlsx'), fn_target)
    write_workbook(
        target=fn_target,
        df=df,
        df_affiliations=df_affiliations,
        df_locations=df_locations,
        df_population=df_population,
        df_grid=df_grid,
        label_groups={'cat'},
        count_primary_class=False,
        threshold=threshold,
        mask=df['item_id'].notna(),
        properties=base_properties | {'title': 'Lancet Countdown Indicator 5.3.1 (2026)'},
    )

    logger.info('Writing major category spreadsheet with single-class classification')
    fn_target = target / f'{datetime.today().strftime("%Y%m%d")}_LancetCountdown_Indicator531_primary_Global.xlsx'
    shutil.copyfile(Path('data/exports/lancet/Global/Datasheets_base/LancetCountdown_Indicator531_primary_Global.xlsx'), fn_target)
    write_workbook(
        target=fn_target,
        df=df,
        df_affiliations=df_affiliations,
        df_locations=df_locations,
        df_population=df_population,
        df_grid=df_grid,
        label_groups={'cat'},
        count_primary_class=True,
        threshold=threshold,
        mask=df['item_id'].notna(),
        properties=base_properties | {'title': 'Lancet Countdown Indicator 5.3.1 (2026)'},
    )

    logger.info('Writing impacts spreadsheet with multi-class classification')
    fn_target = target / f'{datetime.today().strftime("%Y%m%d")}_LancetCountdown_Indicator532_multi_Global.xlsx'
    shutil.copy(Path('data/exports/lancet/Global/Datasheets_base/LancetCountdown_Indicator532_multi_Global.xlsx'), fn_target)
    write_workbook(
        target=target / f'{datetime.today().strftime("%Y%m%d")}_LancetCountdown_Indicator532_multi_Global.xlsx',
        df=df,
        df_affiliations=df_affiliations,
        df_locations=df_locations,
        df_population=df_population,
        df_grid=df_grid,
        label_groups=impact_groups,
        count_primary_class=False,
        threshold=threshold,
        mask=df['incl_impacts'] & df['incl_major'] & df['grid_attributable'] & df['incl_location'],
        properties=base_properties | {'title': 'Lancet Countdown Indicator 5.3.2 (2026)'},
    )

    logger.info('Writing impacts spreadsheet with single-class classification')
    fn_target = target / f'{datetime.today().strftime("%Y%m%d")}_LancetCountdown_Indicator532_primary_Global.xlsx'
    shutil.copy(Path('data/exports/lancet/Global/Datasheets_base/LancetCountdown_Indicator532_primary_Global.xlsx'), fn_target)
    write_workbook(
        target=target / f'{datetime.today().strftime("%Y%m%d")}_LancetCountdown_Indicator532_primary_Global.xlsx',
        df=df,
        df_affiliations=df_affiliations,
        df_locations=df_locations,
        df_population=df_population,
        df_grid=df_grid,
        label_groups=impact_groups,
        count_primary_class=True,
        threshold=threshold,
        mask=df['incl_impacts'] & df['incl_major'] & df['grid_attributable'] & df['incl_location'],
        properties=base_properties | {'title': 'Lancet Countdown Indicator 5.3.2 (2026)'},
    )


if __name__ == '__main__':
    typer.run(prepare_lancet_excel_export)

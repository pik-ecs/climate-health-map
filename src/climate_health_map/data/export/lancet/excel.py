from pathlib import Path
from typing import Annotated, Literal
from datetime import datetime

import numpy as np
import typer
import pandas as pd

from climate_health_map.shared import get_logger
from climate_health_map.data.labels import LABELS, Collection
from .loaders import read_base_data

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
        table.loc[('All groups', 'Total')] = pd.Series(totals)

    return table[column_groups]  # table.sort_index(axis=1)[column_groups]


def summary_tables(df: pd.DataFrame, writer: pd.ExcelWriter, sheet_name: str, col_offset: int, row_offset: int) -> None:
    fmt = writer.book.add_format({'bold': True, 'align': 'left'})
    if sheet_name not in writer.sheets:
        writer.book.add_worksheet(sheet_name)
    worksheet = writer.sheets[sheet_name]

    worksheet.write(row_offset, col_offset, 'Totals', fmt)
    totals = df.xs('Total', level=1)
    totals.index.name = None  # FIXME: get rid of empty row
    totals.to_excel(writer, sheet_name=sheet_name, startrow=row_offset + 1, startcol=col_offset, header=True, index=True)
    row_offset += totals.shape[0] + 5  # caption, two-level column header, one empty row

    worksheet.write(row_offset, col_offset, 'Compound annual growth rate (2021–2025) ', fmt)
    cagr = (
        ((pow((df.xs(2025, level=1) / df.xs(2021, level=1)), (1 / 5)) - 1) * 100).drop(index=[np.nan], errors='ignore').replace({np.inf: pd.NA}).fillna(pd.NA)
    )
    cagr.index.name = None  # FIXME: get rid of empty row
    cagr.to_excel(writer, sheet_name=sheet_name, startrow=row_offset + 1, startcol=col_offset, header=True, index=True)
    row_offset += cagr.shape[0] + 5  # caption, two-level column header, one empty row

    worksheet.write(row_offset, col_offset, 'Compound annual growth rate (2016–2025) ', fmt)
    cagr = (
        ((pow((df.xs(2025, level=1) / df.xs(2016, level=1)), (1 / 5)) - 1) * 100).drop(index=[np.nan], errors='ignore').replace({np.inf: pd.NA}).fillna(pd.NA)
    )
    cagr.index.name = None  # FIXME: get rid of empty row
    cagr.to_excel(writer, sheet_name=sheet_name, startrow=row_offset + 1, startcol=col_offset, header=True, index=True)


def write_workbook(
    target: Path,
    df: pd.DataFrame,
    df_locations: pd.DataFrame,
    df_affiliations: pd.DataFrame,
    label_groups: set[str],
    count_primary_class: bool,
    threshold: float = 0.5,
) -> None:
    with pd.ExcelWriter(target, engine='xlsxwriter') as writer:
        # write_with_style(df=subframe, writer=writer, sheet_name='Total counts', startrow=2, startcol=1, header=True, index=True)
        subframe_full = annual_counts(df=df, label_groups=label_groups, count_primary_class=count_primary_class, threshold=threshold)
        subframe_full.droplevel(level=0).to_excel(writer, sheet_name='Total counts', startrow=2, startcol=1, header=True, index=True)
        summary_tables(df=subframe_full, writer=writer, sheet_name='Total counts', col_offset=subframe_full.shape[1] + 5, row_offset=2)

        subframe_loc = annual_counts(
            df=df,
            label_groups=label_groups,
            count_primary_class=count_primary_class,
            geography_filter='location',
            threshold=threshold,
        )
        subframe_loc.droplevel(level=0).to_excel(writer, sheet_name='Totals (with location)', startrow=2, startcol=1, header=True, index=True)
        summary_tables(df=subframe_loc, writer=writer, sheet_name='Totals (with location)', col_offset=subframe_loc.shape[1] + 5, row_offset=2)

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
            for sheet_name, join_df in [
                (f'{name} (by location)', df_locations),
                (f'{name} (by affiliation)', df_affiliations),
            ]:
                subframe = annual_counts(
                    df=join_df[[grouping]].join(df, how='inner'),
                    label_groups=label_groups,
                    count_primary_class=count_primary_class,
                    group_by=grouping,
                    threshold=threshold,
                )
                subframe.to_excel(writer, sheet_name=sheet_name, startrow=2, startcol=1, header=True, index=True)
                summary_tables(df=subframe, writer=writer, sheet_name=sheet_name, col_offset=subframe.shape[1] + 5, row_offset=2)


def prepare_lancet_excel_export(
    source: Annotated[Path, typer.Option(help='Source directory')],
    target: Annotated[Path, typer.Option(help='Target directory')],
    source_annotations: Annotated[Path | None, typer.Option(help='path to human annotations')] = None,
    year_start: Annotated[int, typer.Option(help='Start year (incl)')] = 1990,
    year_end: Annotated[int, typer.Option(help='End year (incl)')] = 2025,
    threshold: Annotated[float, typer.Option(help='Threshold')] = 0.5,
    loglevel: Annotated[str, typer.Option(help='Verbosity of logger')] = 'INFO',
) -> None:
    logger = get_logger(loglevel=loglevel, logger_name='lancet-excel', run_log_init=True)
    target.mkdir(parents=True, exist_ok=True)
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
        logger=logger.getChild('reader'),
    )
    df = df.rename(columns={'publication_year': 'Publication year'}).reset_index().set_index('item_id', drop=False)

    logger.info('Writing major category spreadsheet with multi-class classification')
    write_workbook(
        target=target / f'{datetime.today().strftime("%Y%m%d")}_LancetCountdown_Indicator531_multi_Global.xlsx',
        df=df,
        df_affiliations=df_affiliations,
        df_locations=df_locations,
        label_groups={'cat'},
        count_primary_class=False,
        threshold=threshold,
    )

    logger.info('Writing major category spreadsheet with single-class classification')
    write_workbook(
        target=target / f'{datetime.today().strftime("%Y%m%d")}_LancetCountdown_Indicator531_primary_Global.xlsx',
        df=df,
        df_affiliations=df_affiliations,
        df_locations=df_locations,
        label_groups={'cat'},
        count_primary_class=True,
        threshold=threshold,
    )

    logger.info('Writing impacts spreadsheet with multi-class classification')
    write_workbook(
        target=target / f'{datetime.today().strftime("%Y%m%d")}_LancetCountdown_Indicator532_multi_Global.xlsx',
        df=df[df['incl_impacts']].copy(),
        df_affiliations=df_affiliations,
        df_locations=df_locations,
        label_groups=impact_groups,
        count_primary_class=False,
        threshold=threshold,
    )

    logger.info('Writing impacts spreadsheet with single-class classification')
    write_workbook(
        target=target / f'{datetime.today().strftime("%Y%m%d")}_LancetCountdown_Indicator532_primary_Global.xlsx',
        df=df[df['incl_impacts']].copy(),
        df_affiliations=df_affiliations,
        df_locations=df_locations,
        label_groups=impact_groups,
        count_primary_class=True,
        threshold=threshold,
    )


def write_df_with_style(
    df: pd.DataFrame,
    writer: pd.ExcelWriter,
    sheet_name: str,
    header: bool = True,
    index: bool = True,
    startrow: int = 2,
    startcol: int = 2,
) -> None:
    df.to_excel(writer, sheet_name=sheet_name, startrow=startrow, startcol=startcol, header=header, index=index)

    workbook = writer.book
    worksheet = writer.sheets[sheet_name]

    header_fmt = workbook.add_format({'bold': True, 'border': 1, 'align': 'center'})  # 'bg_color': '#D9D9D9',
    cell_fmt = workbook.add_format({'border': 1, 'align': 'right'})

    vertical_offset = 0
    if index:
        idx_name = str(df.index.name) if df.index.name else ''
        width = max(df.index.astype(str).str.len().max(), len(idx_name)) + 2
        # Set column width and basic format
        worksheet.set_column(first_col=startcol, last_col=startcol, width=width, cell_format=cell_fmt)
        # Overwrite specific index cells with header format
        # worksheet.write(row=0, col=0, token=idx_name, cell_format=header_fmt)
        worksheet.write(0, 0, idx_name, header_fmt)
        vertical_offset = 1

    for i, col in enumerate(df.columns):
        # Calculate width: max(header length, longest cell in column)
        col_name = str(col)
        width = max(df[col].astype(str).str.len().max(), len(col_name)) + 2

        # Apply width and borders to the entire column
        col_idx = i + vertical_offset
        worksheet.set_column(col_idx, col_idx, width, cell_fmt)

        # Re-write header with bold style
        worksheet.write(0, col_idx, col_name, header_fmt)


if __name__ == '__main__':
    typer.run(prepare_lancet_excel_export)

# from pathlib import Path
# from typing import Annotated
#
# import pandas as pd
# import typer
# import numpy as np
#
# from climate_health_map.data.geographies import load_country_infos, flatten_country_groups, read_places_export
# from climate_health_map.shared import read_any_pd, get_logger
# from .._utils import read_export
#
#
# def compile_counts(df_base: pd.DataFrame) -> pd.DataFrame:
#     return (
#         pd.DataFrame(
#             {
#                 ('General', 'Number of studies on climate & health'): df_base[mask_rel].groupby('publication_year')['openalex_id'].nunique(),
#                 ('General', 'Number of studies with location'): df_base[df_base['incl_py'] & df_base['incl_loc']]
#                 .groupby('publication_year')['openalex_id']
#                 .nunique(),
#                 ('General', 'Number of studies mentioning mental health'): df_base[mask_imp & df_base['mental health']]
#                 .groupby('publication_year')['openalex_id']
#                 .nunique(),
#                 ('General', 'Number of mentioning indigeneous communities'): df_base[mask_imp & df_base['indigenous communities']]
#                 .groupby('publication_year')['openalex_id']
#                 .nunique(),
#                 # ('General', 'Number of relevant studies with location'): df_base[df_base['incl_py'] & df_base['incl_ch'] & df_base['incl_loc']].groupby('publication_year')['openalex_id'].nunique(),
#                 ('Impacts', 'Number of studies on impacts'): df_base[mask_imp].groupby('publication_year')['openalex_id'].nunique(),
#                 ('Impacts', 'Number of impacts studies mentioning location in area with attributable climate trends'): df_base[df_base['incl']]
#                 .groupby('publication_year')['openalex_id']
#                 .nunique(),
#                 ('Impacts', 'Number of impacts studies with location'): df_base[mask_imp & df_base['incl_loc']]
#                 .groupby('publication_year')['openalex_id']
#                 .nunique(),
#                 **{
#                     ('Relevance', name): df_base[df_base['incl_py'] & (df_base[column] > 0.5)].groupby('publication_year')['openalex_id'].nunique()
#                     for column, name in MAJOR_LABELS['Relevance'].items()
#                 },
#                 **{
#                     ('Topic area', name): df_base[mask_rel & (df_base[column] > 0.5)].groupby('publication_year')['openalex_id'].nunique()
#                     for column, name in MAJOR_LABELS['Category'].items()
#                 },
#                 **{
#                     ('Topic area (with location)', name): df_base[mask_rel & df_base['incl_loc'] & (df_base[column] > 0.5)]
#                     .groupby('publication_year')['openalex_id']
#                     .nunique()
#                     for column, name in MAJOR_LABELS['Category'].items()
#                 },
#                 **{
#                     (label, name): df_base[mask & (df_base[column] > 0.5)].groupby('publication_year')['openalex_id'].nunique()
#                     for label, columns in LABELS.items()
#                     for column, name in columns.items()
#                 },
#                 **{
#                     ('Mental health', term): df_base[mask_rel & df_base[term]].groupby('publication_year')['openalex_id'].nunique()
#                     for term in mental_health_terms
#                 },
#                 **{
#                     ('Indigineous communities', term): df_base[mask_rel & df_base[term]].groupby('publication_year')['openalex_id'].nunique()
#                     for term in indigenous_community_terms
#                 },
#             }
#         )
#         .fillna(0)
#         .astype(int)
#         .reset_index()
#         .rename({'publication_year': 'Publication year'}, axis=1)
#         .set_index('Publication year')
#     )
#
#
# COLS_531 = [
#     'Number of studies on climate & health',
#     'Mitigation',
#     'Adaptation',
#     'Impacts',
#     'Number of studies mentioning mental health',
#     'Number of studies mentioning indigeneous communities',
# ]
# a = 0
#
#
# def grouped_counts(group_col, group_name, outcols, sheet_name=None):
#     # 'Country Name to use'
#     # 'LC Grouping'
#     # 'WHO Region'
#     # 'HDI Group (2023-24)'
#
#     accu = []
#     for group in df_base[group_col].unique():
#         gmask = df_base[group_col] == group
#
#         tmp = (
#             pd.DataFrame(
#                 {
#                     ('General', 'Number of studies on climate & health'): df_base[gmask & mask_rel].groupby('publication_year')['openalex_id'].nunique(),
#                     ('General', 'Number of studies with location'): df_base[gmask & df_base['incl_py'] & df_base['incl_loc']]
#                     .groupby('publication_year')['openalex_id']
#                     .nunique(),
#                     ('General', 'Number of relevant studies with location'): df_base[gmask & df_base['incl_py'] & df_base['incl_ch'] & df_base['incl_loc']]
#                     .groupby('publication_year')['openalex_id']
#                     .nunique(),
#                     ('General', 'Number of studies on impacts'): df_base[gmask & mask_imp].groupby('publication_year')['openalex_id'].nunique(),
#                     ('General', 'Number of impacts studies mentioning location in area with attributable climate trends'): df_base[gmask & df_base['incl']]
#                     .groupby('publication_year')['openalex_id']
#                     .nunique(),
#                     ('General', 'Number of impacts studies with location'): df_base[gmask & mask_imp & df_base['incl_loc']]
#                     .groupby('publication_year')['openalex_id']
#                     .nunique(),
#                     ('General', 'Number of studies mentioning mental health'): df_base[gmask & mask_imp & df_base['mental health']]
#                     .groupby('publication_year')['openalex_id']
#                     .nunique(),
#                     ('General', 'Number of mentioning indigeneous communities'): df_base[gmask & mask_imp & df_base['indigenous communities']]
#                     .groupby('publication_year')['openalex_id']
#                     .nunique(),
#                     **{
#                         ('Relevance', name): df_base[gmask & df_base['incl_py'] & (df_base[column] > 0.5)].groupby('publication_year')['openalex_id'].nunique()
#                         for column, name in MAJOR_LABELS['Relevance'].items()
#                     },
#                     **{
#                         ('Topic area', name): df_base[gmask & mask_rel & (df_base[column] > 0.5)].groupby('publication_year')['openalex_id'].nunique()
#                         for column, name in MAJOR_LABELS['Category'].items()
#                     },
#                     **{
#                         ('Topic area (with location)', name): df_base[gmask & mask_rel & df_base['incl_loc'] & (df_base[column] > 0.5)]
#                         .groupby('publication_year')['openalex_id']
#                         .nunique()
#                         for column, name in MAJOR_LABELS['Category'].items()
#                     },
#                     **{
#                         (label, name): df_base[gmask & mask & (df_base[column] > 0.5)].groupby('publication_year')['openalex_id'].nunique()
#                         for label, columns in LABELS.items()
#                         for column, name in columns.items()
#                     },
#                     **{
#                         ('Mental health', term): df_base[gmask & mask_rel & df_base[term]].groupby('publication_year')['openalex_id'].nunique()
#                         for term in mental_health_terms
#                     },
#                     **{
#                         ('Indigineous communities', term): df_base[gmask & mask_rel & df_base[term]].groupby('publication_year')['openalex_id'].nunique()
#                         for term in indigenous_community_terms
#                     },
#                 }
#             )
#             .fillna(0)
#             .astype(int)
#         )
#         tmp[group_name] = group
#         accu.append(tmp)
#
#     tab = pd.concat(accu).reset_index().rename(columns={'publication_year': 'Publication year'}).set_index([group_name, 'Publication year'])
#
#     trimtab = tab.drop(columns=['Topic area (with location)'], level=0).T.reset_index().drop('level_0', axis=1, level=0).set_index('level_1').T.reset_index()
#     display(trimtab)
#     if sheet_name is not None:
#         cols = ['Publication year', group_name] + outcols
#         # cols = outcols
#         srow = 5
#
#         # Main data
#         trimtab[cols].to_excel(writer, sheet_name=sheet_name, startrow=3, startcol=0, index=False)
#
#         # Summary totals
#         trimtab[cols].groupby(group_name).sum().drop(columns='Publication year').to_excel(writer, sheet_name=sheet_name, startrow=srow, startcol=10)
#         srow += trimtab[cols].groupby(group_name).sum().shape[0] + 1
#
#         (
#             df_counts.sum()
#             .to_frame()
#             .T.drop(columns=['Topic area'], level=0)
#             .T.reset_index()
#             .drop('level_0', axis=1)
#             .set_index('level_1')
#             .T.rename(index={0: 'Total'})
#         )[outcols].to_excel(writer, sheet_name=sheet_name, startrow=srow, startcol=10, header=False)
#         srow += 3
#
#         # Summary totals 1990–1999
#         t0 = trimtab[cols][(trimtab['Publication year'] >= 1990) & (trimtab['Publication year'] <= 1999)].groupby(group_name).sum().groupby(group_name).sum()
#         display(t0.drop(columns='Publication year'))
#         t0.drop(columns='Publication year').to_excel(writer, sheet_name=sheet_name, startrow=srow, startcol=10)
#         srow += trimtab.groupby(group_name).sum().shape[0] + 1
#
#         (
#             df_counts[(df_counts.index >= 1990) & (df_counts.index <= 1999)]
#             .sum()
#             .to_frame()
#             .T.drop(columns=['Topic area'], level=0)
#             .T.reset_index()
#             .drop('level_0', axis=1)
#             .set_index('level_1')
#             .T.rename(index={0: 'Total'})
#         )[outcols].to_excel(writer, sheet_name=sheet_name, startrow=srow, startcol=10, header=False)
#         srow += 3
#
#         # Summary totals 2014–2024
#         t1 = trimtab[cols][(trimtab['Publication year'] >= 2014) & (trimtab['Publication year'] <= 2024)].groupby(group_name).sum().groupby(group_name).sum()
#         display(t1.drop(columns='Publication year'))
#         t1.drop(columns='Publication year').to_excel(writer, sheet_name=sheet_name, startrow=srow, startcol=10)
#         srow += trimtab.groupby(group_name).sum().shape[0] + 1
#
#         (
#             df_counts[(df_counts.index >= 2014) & (df_counts.index <= 2024)]
#             .sum()
#             .to_frame()
#             .T.drop(columns=['Topic area'], level=0)
#             .T.reset_index()
#             .drop('level_0', axis=1)
#             .set_index('level_1')
#             .T.rename(index={0: 'Total'})
#         )[outcols].to_excel(writer, sheet_name=sheet_name, startrow=srow, startcol=10, header=False)
#         srow += 3
#
#         # Summary of growth rates
#         display((t1 / t0 * 100).drop(columns='Publication year'))
#         (t1 / t0 * 100).drop(columns='Publication year').to_excel(writer, sheet_name=sheet_name, startrow=srow, startcol=10)
#
#     # display(tab.groupby(group_name).sum().transpose().reset_index().set_index(['level_0', 'level_1']).style.background_gradient(cmap='Blues', axis=None))
#
#     return tab
#
#
# def prepare_lithub_export(
#     source: Annotated[Path, typer.Option(help='Source directory')],
#     target: Annotated[Path, typer.Option(help='Target file')],
#     source_annotations: Annotated[Path | None, typer.Option(help='path to human annotations')] = None,
#     year_start: Annotated[int, typer.Option(help='Start year (incl)')] = 1990,
#     year_end: Annotated[int, typer.Option(help='End year (incl)')] = 2025,
#     loglevel: Annotated[str, typer.Option(help='Verbosity of logger')] = 'INFO',
# ):
#     logger = get_logger(loglevel=loglevel, logger_name='lithub', run_log_init=True)
#     df_countries = load_country_infos()
#     logger.info(f'Loaded country infos: {df_countries.shape}')
#
#     df = read_export(
#         source_items=source / 'items.csv',
#         source_classifications=source / 'classifications.csv',
#         source_annotations=source_annotations,
#         logger=logger,
#         rescale_topic_scores=True,
#         year_end=year_end,
#         year_start=year_start,
#         filter_mai=True,
#         filter_rel=True,
#     )
#     df['idx'] = np.arange(len(df))  # set a continuous index now that we are done filtering/joining
#     logger.info(f'Joined tables: {df.shape}')
#
#     logger.info('Reading affiliation data...')
#     df_affiliations = read_any_pd(source / 'affiliations.csv', keep_default_na=False).merge(df_countries, left_on='iso2', right_on='iso2', how='left')
#     logger.info('Flattening affiliation data...')
#     df_affiliations_flat = flatten_country_groups(prefix='Affiliation', df=df_affiliations)
#     logger.info(f'Loaded affiliations table: {df_affiliations.shape}; flattened: {df_affiliations_flat.shape}')
#
#     logger.info('Reading mordecai data...')
#     df_places = read_places_export(source=source / 'places.csv', df_countries=df_countries, include_grid=True)
#     logger.info('Flattening mordecai data...')
#     df_places_flat = flatten_country_groups(prefix='Location', df=df_places)
#     logger.info(f'Loaded places table: {df_places.shape}; flattened: {df_places_flat.shape}')
#
#     # df = df.join(df_affiliations_flat, how='left').join(df_places_flat, how='outer')
#     # logger.info(f'Joined output table: {df.shape}')
#     # t2 = df_impacts[['openalex_id', *[c for _, cols in LABELS.items() for c in cols.keys()]]]  # 'relevant|1',
#     # t3 = df_places[['doc_id', 'lat', 'lon', 'LAT', 'LON', 'country_code3', 'name']].rename(columns={'doc_id': 'openalex_id', 'country_code3': 'ISO3'})
#     # t4 = df_trends
#     #
#     # print(t1.shape)
#     # print(t2.shape)
#     # print(t3.shape)
#     # print(t4.shape)
#     # df_base = (
#     #     t1
#     #     .merge(df_mental_health, left_on='openalex_id', right_on='openalex_id')
#     #     .merge(df_indigeneous, left_on='openalex_id', right_on='openalex_id')
#     #     .merge(t2, left_on='openalex_id', right_on='openalex_id', how='outer')
#     #     .merge(t3, left_on='openalex_id', right_on='openalex_id', how='outer')
#     #     .merge(t4, left_on=['LAT', 'LON'], right_on=['LAT', 'LON'], how='outer')
#     #     .merge(countries_lc.reset_index(drop=True), left_on='ISO3', right_on='ISO3', how='outer')
#     # )
#     # df_base['incl_attr'] = df_base['attributable']  # associated region has attributable climate trend
#     # df_base['incl_loc'] = df_base['name'].notna()  # has location mention
#     # df_base['incl_ch'] = df_base['rel|1'] > 0.5  # major category relevance (is on climate & health)
#     # df_base['incl_imp'] = df_base['Category'] == 'Impacts'  # major category with highest score is impacts
#     # df_base['incl_py'] = (df_base['publication_year'] >= 1990) & (df_base['publication_year'] <= 2024)  # published between 1990–2024
#     # df_base['incl'] = df_base['incl_ch'] & df_base['incl_imp'] & df_base['incl_py'] & df_base['incl_loc'] & df_base['attributable']  # all the above
#     #
#     # mask_rel = df_base['incl_py'] & df_base['incl_ch']
#     # mask_imp = mask_rel & df_base['incl_imp']
#     # mask = df_base['incl']
#
#     with pd.ExcelWriter(target) as writer:
#         df_counts = compile_counts(df)
#
#         subframe = df_counts[['General', 'Topic area', 'Topic area (with location)']]
#         subframe.to_excel(writer, sheet_name='Global counts', startrow=3, startcol=0)
#         subframe.sum().to_frame().T.rename(index={0: 'Total'}).to_excel(
#             writer, sheet_name='Global counts', startrow=3 + subframe.shape[0] + 2, startcol=0, header=False
#         )
#
#         pd.DataFrame(
#             {
#                 'Baseline average (1990–1999)': subframe[(subframe.index >= 1990) & (subframe.index <= 1999)].mean(),
#                 'Recent years average (2014-2024)': subframe[(subframe.index >= 2014) & (subframe.index <= 2024)].mean(),
#                 'Absolute change': subframe[(subframe.index >= 2014) & (subframe.index <= 2024)].mean()
#                 - subframe[(subframe.index >= 1990) & (subframe.index <= 1999)].mean(),
#                 'Relative change': subframe[(subframe.index >= 2014) & (subframe.index <= 2024)].mean()
#                 / subframe[(subframe.index >= 1990) & (subframe.index <= 1999)].mean()
#                 * 100,
#             }
#         ).T.to_excel(writer, sheet_name='Global counts', startrow=5, startcol=12)
#
#         df_reg = grouped_counts(
#             'LC Grouping',
#             'LC Grouping',
#             [
#                 'Number of studies on climate & health',
#                 'Mitigation',
#                 'Adaptation',
#                 'Impacts',
#                 'Number of studies mentioning mental health',
#                 'Number of mentioning indigeneous communities',
#             ],
#             sheet_name='LC Grouping',
#         )
#         df_hdi = grouped_counts(
#             'HDI Group (2023-24)',
#             'HDI Group',
#             [
#                 'Number of studies on climate & health',
#                 'Mitigation',
#                 'Adaptation',
#                 'Impacts',
#                 'Number of studies mentioning mental health',
#                 'Number of mentioning indigeneous communities',
#             ],
#             sheet_name='HDI Group',
#         )
#         df_who = grouped_counts(
#             'WHO Region',
#             'WHO Region',
#             [
#                 'Number of studies on climate & health',
#                 'Mitigation',
#                 'Adaptation',
#                 'Impacts',
#                 'Number of studies mentioning mental health',
#                 'Number of mentioning indigeneous communities',
#             ],
#             sheet_name='WHO Region',
#         )
#         df_country = grouped_counts(
#             'Country Name to use',
#             'Country',
#             [
#                 'Number of studies on climate & health',
#                 'Mitigation',
#                 'Adaptation',
#                 'Impacts',
#                 'Number of studies mentioning mental health',
#                 'Number of mentioning indigeneous communities',
#             ],
#             sheet_name='Country',
#         )

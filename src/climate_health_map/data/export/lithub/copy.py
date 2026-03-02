# import pandas as pd
# import logging
#
# from climate_health_map.scatterplot import rescale_projection
# from climate_health_map.topics import rescale_topic_scores
#
# cluster_keywords
#
#
# def replace_human_annotations(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
#     logger.info('Override predictions with human annotations')
#
#     # Load human annotations
#     df_human = pd.read_csv(settings.LABELS)
#     for col_human, col_df in [
#         ('Climate and health major categories|relevant', 'rel|1'),
#         ('climateCategory|Mitigation', 'cat|0'),
#         ('climateCategory|Adaptation', 'cat|1'),
#         ('climateCategory|Impacts', 'cat|2'),
#     ]:
#         df[col_df] = df[col_df].clip(lower=0.01, upper=0.99)
#         for val in [True, False]:
#             mask = df_human[col_human] == val
#             openalex_ids = df_human[mask]['doc_id']
#             df.loc[df['openalex_id'].isin(openalex_ids.tolist()), col_df] = int(val)
#
#         logger.info(f' > Human {col_df}==1: {(df[col_df] == 1).sum():,} | {col_df}==0: {(df[col_df] == 0).sum():,}')
#
#     return df
#
#
# def main(log_level: str = 'INFO'):
#     logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s (%(process)d): %(message)s', level=log_level)
#     logger = logging.getLogger('LC-merge')
#     logging.getLogger('matplotlib').setLevel(logging.WARNING)
#
#     settings.LITHUB_BASE.mkdir(parents=True, exist_ok=True)
#
#     async def _main():
#         with Session(rev_engine) as sess:
#             revisions = sess.execute(select(Revision)).scalars().all()
#         print(pd.DataFrame([rev.__dict__ for rev in revisions]))
#
#         logger.info('Ensure our data tables are complete')
#         await collect_data(logger=logger)
#
#         logger.info('Loading database')
#         df = pq.read_table(settings.DB_FILE).to_pandas().drop_duplicates('item_id')
#
#         df = rescale_projection(df=df, logger=logger)
#         df = rescale_topic_scores(df=df, logger=logger)
#         df_keywords = cluster_keywords(df=df, logger=logger)
#         df, cols = rename_columns(df=df, logger=logger)
#         df = replace_human_annotations(df, logger=logger)
#
#         if df_keywords:
#             write_keywords(df_keywords, settings.LITHUB_BASE / 'keywords.arrow')
#
#         df['idx'] = np.arange(df.shape[0])
#
#         # drop duplicate columns
#         df = df.loc[:, ~df.columns.duplicated()].copy()
#
#         logger.info(f'Write to {settings.DATASET}')
#         df.to_parquet(settings.DATASET, compression='gzip')
#
#         logger.info('Write lithub output')
#         write(
#             df_full=df,
#             df_slim=df[['idx', 'x', 'y', 'publication_year']].copy(),
#             scheme_keys=cols,
#             out_slim=settings.LITHUB_BASE / 'slim.arrow',
#             out_sql=settings.LITHUB_BASE / 'documents.sqlite',
#             extra_scheme={},
#         )
#
#         logger.info('Writing info.toml')
#         with open(settings.LITHUB_BASE / 'info.toml', 'w') as f:
#             toml.dump(info.dict(), f)
#
#         logger.info('Writing revisions database to csv file')
#
#         with Session(rev_engine) as sess:
#             revisions = sess.execute(select(Revision)).mappings().all()
#             pd.DataFrame(revisions).to_csv(settings.REVISIONS_CSV, index=False)
#
#         sync_cloud(logger=logger)
#
#     asyncio.run(_main())

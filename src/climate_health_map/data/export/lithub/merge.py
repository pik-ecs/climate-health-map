# import pandas as pd
# import pyarrow as pa
# import pyarrow.parquet as pq
# import sqlalchemy as sa
# from nacsos_data.db.schemas import AcademicItem
# from sqlalchemy.orm import Session
#
#
# async def read_items_from_ids(ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
#     # Get docs that were added in this import_revision
#     stmt = sa.select(
#         AcademicItem.openalex_id.label('item_id'),
#         AcademicItem.title,
#         AcademicItem.text.label('abstract'),
#         AcademicItem.publication_year.label('PY'),
#         AcademicItem.doi,
#         AcademicItem.authors,
#     ).where(AcademicItem.project_id == settings.PROJECT_ID, AcademicItem.openalex_id.in_(ids))
#
#     async with nacsos_engine.session() as session:
#         rslt = (await session.execute(stmt)).mappings().all()
#
#     df = pd.DataFrame(rslt)
#     df['doi'] = df['doi'].astype('str')
#
#     authors = [
#         {'item_id': row['item_id'], 'author': author['name'], 'affiliation_name': affiliation['name'], 'affiliation_country': affiliation['country']}
#         for _, row in df.iterrows()
#         for author in (row['authors'] or [])
#         for affiliation in (author['affiliations'] or [{'name': '', 'country': ''}])
#     ]
#
#     df['authors'] = ['; '.join(set([author['name'] for author in row['authors']])) if row['authors'] else '' for _, row in df.iterrows()]
#     df['authors'] = df['authors'].astype(str)
#     return df, pd.DataFrame(authors)
#
#
# async def collect_data(logger: logging.Logger):
#     with Session(rev_engine, autoflush=True) as sess:
#         revisions = sess.execute(select(Revision).where(Revision.data_collected != True)).scalars().all()
#         logger.info(f'Found {len(revisions)} revisions for which data is not loaded from NACSOS')
#
#         if len(revisions) == 0:
#             logger.warning(f'No uncollected revisions found')
#             return
#
#         df_topicmodel = pd.read_csv(settings.TM_INFO).rename(columns={'id': 'topic_id'})
#
#         for revision in revisions:
#             logger.info(f'Collecting data for revision {revision.revision_id}')
#             preds_bin = pq.read_table(
#                 settings.CLASSIFICATION_CH_BIN,
#                 filters=[
#                     ('Climate and health major categories|relevant', '>', 0.5),
#                     ('revision_id', '=', revision.revision_id),
#                 ],
#             )
#
#             for i, batch in enumerate(preds_bin.to_batches()):
#                 blogger = logger.getChild(f'rev-{revision.revision_id}|batch-{i}')
#
#                 df_pred_bin = batch.to_pandas()
#                 df_pred_bin['batch_id'] = i
#                 df_pred_bin['revision_id'] = revision.revision_id
#
#                 batch_item_ids = df_pred_bin['item_id'].values
#
#                 blogger.debug('Loading multi-predictions')
#                 df_pred_mul = pq.read_table(settings.CLASSIFICATION_CH_MUL, filters=[('item_id', 'in', batch_item_ids)]).to_pandas().drop_duplicates('item_id')
#
#                 blogger.debug('Loading and normalising topic statistics')
#                 df_dtm = (
#                     pq.read_table(settings.TM_DTM, filters=[('doc_id', 'in', batch_item_ids)])
#                     .to_pandas()
#                     .rename(columns={'doc_id': 'item_id'})
#                     .drop_duplicates(['item_id', 'topic_id'])
#                     .merge(df_topicmodel, left_on='topic_id', right_on='topic_id')
#                 )
#                 df_dtm_wide = df_dtm.pivot(values='score', index='item_id', columns='Topic').reset_index()
#
#                 blogger.debug('Loading place names mentioned in abstract')
#                 df_places = pq.read_table(settings.PLACES, filters=[('doc_id', 'in', batch_item_ids)]).to_pandas().rename(columns={'doc_id': 'item_id'})
#                 df_places['val'] = 11
#                 df_places_wide = (
#                     df_places.drop_duplicates(['item_id', 'country_code3'])
#                     # .query('continent!="AN"')
#                     .pivot(values='val', index='item_id', columns='country_code3')
#                     .reset_index()
#                     .drop(columns=[np.nan], errors='ignore')
#                     .fillna(0)
#                 )
#                 df_places_wide = pd.concat([df_places_wide, pd.DataFrame(columns=list(set(ISO3_NAMES.keys()) - set(df_places_wide.columns)))])
#                 for iso3 in ISO3_NAMES.keys():
#                     df_places_wide[iso3] = df_places_wide[iso3].astype('Int32')
#                 for region, cols in REGIONS_LANCET.items():
#                     df_places_wide[f'lancet_{region}'] = df_places_wide[[col for col in cols if col in df_places_wide.columns]].sum(axis=1).astype('Int32')
#
#                 blogger.debug('Loading 2D projection')
#                 df_layout = (
#                     pq.read_table(settings.EMBEDDINGS, filters=[('id', 'in', batch_item_ids)])
#                     .to_pandas()
#                     .rename(columns={'id': 'item_id'})
#                     .drop_duplicates('item_id')
#                 )
#
#                 blogger.debug('Loading data from NACSOS')
#                 df_items, df_authors = await read_items_from_ids(batch_item_ids)
#                 df_items = df_items.drop_duplicates('item_id')
#                 df_authors['batch_id'] = i
#                 df_authors['revision_id'] = revision.revision_id
#                 df_affiliations = (
#                     pd.DataFrame(
#                         [{'item_id': row['item_id'], row['affiliation_country']: 1} for _, row in df_authors.iterrows()],
#                         columns=['item_id'] + list(ISO2_NAMES.keys()),
#                     )
#                     .groupby('item_id')
#                     .sum()
#                     .reset_index()
#                 )
#
#                 for f in [df_items, df_pred_bin, df_pred_mul, df_places_wide, df_affiliations, df_layout, df_dtm_wide]:
#                     if np.nan in f.columns or None in f.columns:
#                         logger.warning(f.columns)
#
#                 blogger.debug('Merging all information')
#                 df = (
#                     df_items.merge(df_pred_bin, left_on='item_id', right_on='item_id', how='outer')
#                     .merge(df_pred_mul, left_on='item_id', right_on='item_id', how='outer')
#                     .merge(df_places_wide.add_prefix('place_'), left_on='item_id', right_on='place_item_id', how='outer')
#                     .merge(df_affiliations.add_prefix('affil_'), left_on='item_id', right_on='affil_item_id', how='outer')
#                     .merge(df_layout, left_on='item_id', right_on='item_id', how='outer')
#                     .merge(df_dtm_wide.add_prefix('Topic|'), left_on='item_id', right_on='Topic|item_id', how='outer')
#                     .rename(columns={'batch_id_x': 'batch_id'})  # 'revision_id_x': 'revision_id'
#                     .reset_index()
#                 )
#
#                 df.to_parquet(settings.DB_FILE, partition_cols=['revision_id', 'batch_id'], compression='gzip', existing_data_behavior='delete_matching')
#                 df_authors.to_parquet(
#                     settings.AUTHORS_FILE, partition_cols=['revision_id', 'batch_id'], compression='gzip', existing_data_behavior='delete_matching'
#                 )
#
#         logger.debug(f'Marking revision {revision.revision_id} as loaded from NACSOS')
#
#         revision.data_collected = True
#         sess.commit()

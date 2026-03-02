# import math
# from pathlib import Path
# from typing import Any
# import pandas as pd
# import pyarrow as pa
# from sqlalchemy import create_engine, text, types
#
# CHUNK_SIZE = 10000
#
#
# def write(scheme_keys: list[str], df_slim: pd.DataFrame, df_full: pd.DataFrame, out_sql: Path, out_slim: Path, extra_scheme: dict[str, Any]) -> None:
#     print(sorted(scheme_keys))
#     df_slim['x'] = df_slim['x'].astype('float16')
#     df_slim['y'] = df_slim['y'].astype('float16')
#     for key in scheme_keys + ['x', 'y']:
#         df_full[key] = df_full[key].astype('float16')
#     for key in ['publication_year', 'idx']:
#         df_slim[key] = df_slim[key].astype('Int64')
#         df_full[key] = df_full[key].astype('Int64')
#
#     print('Creating sqlite db...')
#     out_sql.unlink(missing_ok=True)
#     sql_schema = {
#         'idx': types.BIGINT,
#         'publication_year': types.INT,
#         'title': types.String,
#         'abstract': types.String,
#         'doi': types.String,
#         'openalex_id': types.String,
#         'authors': types.String,
#         'institutions': types.String,
#         'x': types.FLOAT,
#         'y': types.FLOAT,
#         **dict.fromkeys(scheme_keys, types.FLOAT),
#         **extra_scheme,
#     }
#     print('schema', len(sql_schema))
#     print('schema', sql_schema)
#     engine = create_engine(f'sqlite:///{out_sql}', echo=False)
#     df_full.to_sql(name='documents', con=engine, dtype=sql_schema)
#
#     print('Indexing data in sqlite...')
#     con = engine.connect()
#     # -- Set up search on title, text, authors
#     con.execute(text('CREATE VIRTUAL TABLE search USING fts5(idx, title, abstract, authors);'))
#     # -- Indexing data
#     con.execute(text('INSERT INTO search (idx, title, abstract, authors) SELECT idx, title, abstract, authors FROM documents;'))
#     con.commit()
#
#     rslt = con.execute(text('PRAGMA table_info(documents);')).fetchall()
#     db_cols = set([r[1] for r in rslt])
#     print(f'Wrote sqlite file to {out_sql}')
#
#     print('Checking column alignment...')
#     print(db_cols - set(scheme_keys))
#     print(set(scheme_keys) - db_cols)
#
#     con.close()
#
#     schema = pa.schema([('idx', pa.uint32()), ('x', pa.float16()), ('y', pa.float16()), ('publication_year', pa.uint16())])
#
#     print(f'Writing {out_slim}')
#     with pa.OSFile(str(out_slim), 'wb') as sink:
#         with pa.ipc.new_stream(sink, schema) as writer:
#             n_chunks = int(math.ceil(df_slim.shape[0] / CHUNK_SIZE))
#             for chunk in range(n_chunks):
#                 tmp = df_slim[chunk * CHUNK_SIZE : (chunk + 1) * CHUNK_SIZE]
#                 batch = pa.record_batch(tmp, schema)
#                 writer.write(batch)
#     print(f'Wrote {out_slim}')

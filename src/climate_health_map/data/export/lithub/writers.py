import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
from sqlalchemy import create_engine, text, types

from climate_health_map.data.geographies import get_naming_mask, fix_geographies, FEATURE_LOOKUP
from ...labels import LABELS_LOOKUP

CHUNK_SIZE = 10000


def _write_batched_ipc(df: pd.DataFrame, target: Path, schema: pa.Schema, chunk_size: int = CHUNK_SIZE) -> None:
    fa = pa.Table.from_pandas(df=df[schema.names], schema=schema, preserve_index=False)
    with pa.OSFile(target.as_posix(), 'wb') as sink:
        with pa.ipc.new_file(sink, schema=schema) as writer:
            # for batch in fa.to_batches(chunk_size):
            #     writer.write(batch)
            writer.write_table(table=fa, max_chunksize=chunk_size)


def _write_streamed_ipc(df: pd.DataFrame, target: Path, schema: pa.Schema, chunk_size: int = CHUNK_SIZE) -> None:
    with pa.OSFile(target.as_posix(), 'wb') as sink:
        with pa.ipc.new_stream(sink, schema) as writer:
            for chunk_start in range(0, df.shape[0], chunk_size):
                chunk = df.iloc[chunk_start : chunk_start + chunk_size]
                writer.write(pa.record_batch(chunk, schema))


def ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    df['x'] = df['x'].astype('float16')
    df['y'] = df['y'].astype('float16')
    for key in LABELS_LOOKUP.keys():
        if key in df.columns:
            df[key] = df[key].astype('float16')
    for key in ['publication_year', 'idx']:
        df[key] = df[key].astype('Int64')
    return df


def write_keywords(df: pd.DataFrame, target: Path, chunk_size: int = CHUNK_SIZE, logger: logging.Logger | None = None):
    logger = logger or logging.getLogger('lithub.write')
    logger.info(f'Writing keywords with shape {df.shape} to {target}')
    _write_streamed_ipc(
        df.sample(frac=1).sort_values('level', kind='stable')[['x', 'y', 'level', 'keyword']],
        target=target,
        schema=pa.schema([('x', pa.float16()), ('y', pa.float16()), ('level', pa.uint8()), ('keyword', pa.string())]),
        chunk_size=chunk_size,
    )
    logger.info(f'Wrote {target}')


def write_sqlite(df: pd.DataFrame, target: Path, logger: logging.Logger | None = None) -> None:
    logger = logger or logging.getLogger('lithub.write')
    logger.info('Preparing SQLite schema...')
    label_columns = [key for key in LABELS_LOOKUP.keys() if key in df.columns]
    sql_schema = {
        'idx': types.BIGINT,
        'publication_year': types.INT,
        'title': types.String,
        'abstract': types.String,
        'doi': types.String,
        'openalex_id': types.String,
        'authors': types.String,
        'institutions': types.String,
        'x': types.FLOAT,
        'y': types.FLOAT,
        **dict.fromkeys(label_columns, types.FLOAT),
    }

    logger.info('Deleting (possibly) existing SQLite file...')
    target.unlink(missing_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f'Writing data to {target}')
    engine = create_engine(f'sqlite:///{target}', echo=False)
    df.to_sql(name='documents', con=engine, dtype=sql_schema)

    logger.info('Indexing data in sqlite...')
    with engine.connect() as con:
        # Set up search on title, text, authors
        con.execute(text('CREATE VIRTUAL TABLE search USING fts5(idx, title, abstract, authors);'))
        # Indexing data
        con.execute(
            text(
                'INSERT INTO search (idx, title, abstract, authors) SELECT idx, title, abstract, authors FROM documents;',
            ),
        )
        con.commit()

        rslt = con.execute(text('PRAGMA table_info(documents);')).fetchall()
        db_cols = {r[1] for r in rslt}
        logger.debug(f'Written columns: {db_cols}')

    logger.info(f'Wrote sqlite file to {target}')


def write_base_info(df: pd.DataFrame, target: Path, chunk_size: int = CHUNK_SIZE, logger: logging.Logger | None = None):
    logger.info(f'Writing slim-feather to {target}')
    _write_streamed_ipc(
        df=df[['idx', 'x', 'y', 'publication_year']],
        target=target,
        schema=pa.schema(
            [('idx', pa.uint32()), ('x', pa.float16()), ('y', pa.float16()), ('publication_year', pa.uint16())],
        ),
        chunk_size=chunk_size,
    )
    logger.info('Finished writing slim-feather.')


def write_geographies(df: pd.DataFrame, target_min: Path, target_full: Path, chunk_size: int = CHUNK_SIZE, logger: logging.Logger | None = None):
    logger.info('Preparing filter mask...')
    mask_search_names = get_naming_mask(place_df=df)
    logger.info('Fixing geographies...')
    df = fix_geographies(place_df=df[mask_search_names])

    logger.info('Adding feature column...')
    df['feature'] = df.apply(lambda row: FEATURE_LOOKUP.get(f'{row["feature_class"]}.{row["feature_code"] or ""}'), axis='columns')

    logger.info('Preparing dataframe...')
    df = df.rename(columns={'item_id': 'id', 'feature_class': 'class', 'feature_code': 'code', 'iso_num': 'country_num', 'Name (ISO short)': 'country'}).astype(
        {'lat': 'float16', 'lon': 'float16'},
    )

    logger.info('Writing minimal dataframe...')
    _write_batched_ipc(
        df[['idx', 'country_num']],
        target=target_min,
        schema=pa.schema([pa.field('idx', pa.uint32()), pa.field('country_num', pa.uint16())]),
        chunk_size=chunk_size,
    )

    logger.info('Writing full dataframe...')
    _write_batched_ipc(
        df[['idx', 'geonameid', 'name', 'country', 'country_num', 'lat', 'lon', 'class', 'code', 'feature']],
        target=target_full,
        schema=pa.schema(
            [
                pa.field('idx', pa.uint32()),
                pa.field('geonameid', pa.uint32()),  # ~284KB
                pa.field('country_num', pa.uint16()),
                pa.field('lat', pa.float16()),
                pa.field('lon', pa.float16()),
                pa.field('name', pa.string()),  # ~1.5MB
                # pa.field('class', pa.string())  # ~300KB
                # pa.field('code', pa.string())  # ~1.6MB
                pa.field('feature', pa.uint16()),  # ~100KB
            ],
        ),
        chunk_size=chunk_size,
    )

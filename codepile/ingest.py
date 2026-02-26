"""Consider this to be deprecated and unusable. This is just here for reference"""

import uuid
from pathlib import Path
from typing import Annotated

import typer
import pandas as pd
from nacsos_data.db.schemas import AcademicItem
from nacsos_data.db.schemas.enhancements import Enhancement
from sqlalchemy import select
from sqlalchemy.orm import Session
from tqdm import tqdm

from climate_health_map.shared.env import essentials

app = typer.Typer(help='Data ingestion toolkit for old data')


def _match(openalex_id: str, key: str, project_id: str, session: Session) -> tuple[uuid.UUID | None, uuid.UUID | None]:
    item_id = session.scalar(select(AcademicItem.item_id).where(AcademicItem.openalex_id == openalex_id, AcademicItem.project_id == project_id).limit(1))
    if item_id is None:
        return None, None
    enhancement_id = session.scalar(select(Enhancement.enhancement_id).where(Enhancement.item_id == item_id, Enhancement.key == key).limit(1))
    return item_id, enhancement_id


@app.command('places')
def places(
    source: Annotated[Path, typer.Option(help='source parquet')],
    config: Annotated[Path, typer.Option(help='config file')],
    key: Annotated[str, typer.Option(help='enhancement key to use')] = 'mordecai3',
    loglevel: Annotated[str, typer.Option(help='log level')] = 'INFO',
):
    """
    uv run codepile/ingest.py places --source data/2025/places.parquet --config config/secret.env
    """
    logger, settings, db_engine = essentials(config=config, loglevel=loglevel, logger_name='ingest', run_log_init=True)

    logger.info(f'Reading data from {source}')
    df = pd.read_parquet(source)
    openalex_ids = df['doc_id'].unique()
    logger.info(f'Found {len(df):,} entries for {len(openalex_ids):,} openalex ids')

    missing_ids = set()
    skipped_ids = set()
    n_added = 0
    progress = tqdm(total=df.shape[0])
    with db_engine.session() as session:
        for _, row in df.iterrows():
            payload = row.to_dict()
            openalex_id = payload.pop('doc_id')
            item_id, enhancement_id = _match(openalex_id, key, settings.PROJECT_ID, session)
            if enhancement_id is not None and item_id is not None:
                skipped_ids.add(openalex_id)
            if item_id is None:
                missing_ids.add(openalex_id)
            else:
                n_added += 1
                payload.pop('batch_id', None)
                payload.pop('revision_id', None)
                session.add(
                    Enhancement(
                        enhancement_id=uuid.uuid4(),
                        item_id=item_id,
                        key=key,
                        payload=payload,
                    ),
                )
            progress.set_description(f'missed: {len(missing_ids):,} | skipped: {len(skipped_ids):,} | added: {n_added:,}')
            progress.update()

    logger.info(f'Finished with final score of missed: {len(missing_ids):,} | skipped: {len(skipped_ids):,} | added: {n_added:,}')


def _label_ingest(keys: dict[str, str], id_column: str, source: Path, config: Path, loglevel: str = 'INFO'):
    logger, settings, db_engine = essentials(config=config, loglevel=loglevel, logger_name='ingest', run_log_init=True)

    logger.info(f'Reading data from {source}')
    df = pd.read_parquet(source)
    ids = df[id_column].unique()
    columns = list(keys.keys())
    n_values = df[columns].count().sum()
    logger.info(f'Found {len(df):,} entries for {len(ids):,} ids ({id_column}) with {n_values:,} labels')

    n_missing_ids = 0
    n_skipped_labels = 0
    n_added_labels = 0
    progress = tqdm(total=n_values)
    with db_engine.session() as session:
        for _, row in df.iterrows():
            item_id = session.scalar(
                select(AcademicItem.item_id).where(AcademicItem.openalex_id == row[id_column], AcademicItem.project_id == settings.PROJECT_ID).limit(1),
            )
            if not item_id:
                n_missing_ids += 1
                continue
            for col_old, col_new in keys.items():
                if row[col_old].isna():
                    continue
                enhancement_id = session.scalar(select(Enhancement.enhancement_id).where(Enhancement.item_id == item_id, Enhancement.key == col_new).limit(1))
                if enhancement_id is not None:
                    n_skipped_labels += 1
                    continue

                n_added_labels += 1
                session.add(
                    Enhancement(
                        enhancement_id=uuid.uuid4(),
                        item_id=item_id,
                        key=col_new,
                        payload=row[col_old],
                    ),
                )
            progress.set_description(f'missed IDs: {n_missing_ids:,} | skipped labels: {n_skipped_labels:,} | added labels: {n_added_labels:,}')
            progress.update()

    logger.info(f'Finished with missed IDs: {n_missing_ids:,} | skipped labels: {n_skipped_labels:,} | added labels: {n_added_labels:,}')


@app.command('dataset')
def dataset(
    source: Annotated[Path, typer.Option(help='source parquet')],
    config: Annotated[Path, typer.Option(help='config file')],
    loglevel: Annotated[str, typer.Option(help='log level')] = 'INFO',
):
    # ['index', 'publication_year', 'abstract', 'authors', 'doi', 'openalex_id', 'title', 'rel|1', 'revision_id_x', 'cat|1', 'cat|2', 'cat|0', 'revision_id_y',
    #  'batch_id_y', 'place_item_id', 'place_*',
    #  'cont|0', 'cont|1', 'cont|2', 'cont|3', 'cont|4', 'cont|5', 'cont|6', 'affil_item_id', 'affil_*', 'x', 'y', 'Topic|item_id',
    #  't0-2-3|0', 't0-3-0|0', 't0-1-8|4', 't0-0-5|4', 't0-1-8|0', 't0-1-5|1', 't0-3-1|0', 't0-4-0|7', 't0-1-5|0', 't0-3-1|1', 't0-1-4|4', 't0-0-5|0', 't0-2-0|0',
    #  't0-1-0|0', 't0-1-4|8', 't0-2-1|0', 't0-4-0|0', 't0-0-5|1', 't0-1-9|0', 't0-0-0|1', 't0-0-0|0', 't0-0-1|3', 't0-0-1|2', 't0-2-2|0', 't0-4-0|1', 't0-0-2|1',
    #  't0-1-2|1', 't0-0-2|2', 't0-1-2|0', 't0-1-8|1', 't0-0-3|0', 't0-2-2|1', 't0-1-4|1', 't0-1-1|2', 't0-0-5|3', 't0-1-3|0', 't0-0-2|0', 't0-0-5|2', 't0-1-4|6',
    #  't0-1-4|2', 't0-1-4|3', 't0-1-4|7', 't0-1-6|1', 't0-0-4|1', 't0-4-0|6', 't0-0-6|0', 't0-1-0|1', 't0-1-4|5', 't0-1-7|0', 't0-0-1|1', 't0-0-1|0', 't0-0-1|4',
    #  't0-1-3|3', 't0-1-8|3', 't0-1-3|2', 't0-1-8|2', 't0-4-0|2', 't0-3-1|2', 't0-0-4|0', 't0-4-0|3', 't0-1-1|0', 't0-1-6|0', 't0-4-0|4', 't0-4-0|5', 't0-1-1|1',
    #  't0-1-4|0', 't0-3-1|3', 't0-1-3|1', 't0-3-2|0', 't0-0-0|2', 'revision_id', 'batch_id', 'idx']
    _label_ingest(
        keys={
            'rel|1': 'rel_major|1',
            'cat|0': 'cat|0',
            'cat|1': 'cat|1',
            'cat|2': 'cat|2',
            't0-2-3|0': '',
            't0-3-0|0': '',
            't0-1-8|4': '',
            't0-0-5|4': '',
            't0-1-8|0': '',
            't0-1-5|1': '',
            't0-3-1|0': '',
            't0-4-0|7': '',
            't0-1-5|0': '',
            't0-3-1|1': '',
            't0-1-4|4': '',
            't0-0-5|0': '',
            't0-2-0|0': '',
            't0-1-0|0': '',
            't0-1-4|8': '',
            't0-2-1|0': '',
            't0-4-0|0': '',
            't0-0-5|1': '',
            't0-1-9|0': '',
            't0-0-0|1': '',
            't0-0-0|0': '',
            't0-0-1|3': '',
            't0-0-1|2': '',
            't0-2-2|0': '',
            't0-4-0|1': '',
            't0-0-2|1': '',
            't0-1-2|1': '',
            't0-0-2|2': '',
            't0-1-2|0': '',
            't0-1-8|1': '',
            't0-0-3|0': '',
            't0-2-2|1': '',
            't0-1-4|1': '',
            't0-1-1|2': '',
            't0-0-5|3': '',
            't0-1-3|0': '',
            't0-0-2|0': '',
            't0-0-5|2': '',
            't0-1-4|6': '',
            't0-1-4|2': '',
            't0-1-4|3': '',
            't0-1-4|7': '',
            't0-1-6|1': '',
            't0-0-4|1': '',
            't0-4-0|6': '',
            't0-0-6|0': '',
            't0-1-0|1': '',
            't0-1-4|5': '',
            't0-1-7|0': '',
            't0-0-1|1': '',
            't0-0-1|0': '',
            't0-0-1|4': '',
            't0-1-3|3': '',
            't0-1-8|3': '',
            't0-1-3|2': '',
            't0-1-8|2': '',
            't0-4-0|2': '',
            't0-3-1|2': '',
            't0-0-4|0': '',
            't0-4-0|3': '',
            't0-1-1|0': '',
            't0-1-6|0': '',
            't0-4-0|4': '',
            't0-4-0|5': '',
            't0-1-1|1': '',
            't0-1-4|0': '',
            't0-3-1|3': '',
            't0-1-3|1': '',
            't0-3-2|0': '',
            't0-0-0|2': '',
        }, id_column='openalex_id', source=source, config=config, loglevel=loglevel,
    )

    pass



# pass


if __name__ == '__main__':
    app()

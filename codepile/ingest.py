"""Consider this to be deprecated and unusable. This is just here for reference"""

import uuid
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
import pandas as pd
from nacsos_data.db.schemas import AcademicItem
from nacsos_data.db.schemas.enhancements import Enhancement
from nacsos_data.util import clear_empty
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
      Found 165,040 entries for 37,329 openalex ids
      Finished with final score of missed: 34 | skipped: 0 | added: 37,295

    SELECT e.*
    FROM enhancement e
         JOIN item i ON i.item_id = e.item_id
    WHERE i.project_id = '52f62f17-cefb-4152-bcca-2fcb77541e83'
      AND e.key = 'mordecai3';
    """
    logger, settings, db_engine = essentials(config=config, loglevel=loglevel, logger_name='ingest', run_log_init=True)

    logger.info(f'Reading data from {source}')
    df = pd.read_parquet(source).replace({np.nan: None})
    openalex_ids = df['doc_id'].unique()
    logger.info(f'Found {len(df):,} entries for {len(openalex_ids):,} openalex ids')

    missing_ids = set()
    skipped_ids = set()
    n_added = 0
    progress = tqdm(total=df.shape[0])
    with db_engine.session() as session:
        for openalex_id_, payload in df.groupby('doc_id'):
            openalex_id = str(openalex_id_)
            payload = clear_empty(payload.drop(columns=['doc_id', 'batch_id', 'revision_id'], errors='ignore').to_dict(orient='records'))
            if payload is None:
                continue
            item_id, enhancement_id = _match(openalex_id, key, settings.PROJECT_ID, session)
            if enhancement_id is not None and item_id is not None:
                skipped_ids.add(openalex_id)
            if item_id is None:
                missing_ids.add(openalex_id)
            else:
                n_added += 1
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
        session.commit()

    logger.info(f'Finished with final score of missed: {len(missing_ids):,} | skipped: {len(skipped_ids):,} | added: {n_added:,}')


def _label_ingest(keys: dict[str, str], id_column: str, source: Path, config: Path, loglevel: str = 'INFO'):
    logger, settings, db_engine = essentials(config=config, loglevel=loglevel, logger_name='ingest', run_log_init=True)

    logger.info(f'Reading data from {source}')
    if source.suffix == '.parquet':
        df = pd.read_parquet(source)
    elif source.suffix == '.arrow':
        df = pd.read_feather(source)
    else:
        raise ValueError(f'Unsupported file type: {source.suffix}')
    ids = df[id_column].unique()
    columns = list(keys.keys())
    n_values = df[columns].count().sum()
    logger.info(f'Found {len(df):,} entries for {len(ids):,} ids ({id_column}) with {n_values:,} labels')

    n_missing_ids = 0
    n_skipped_labels = 0
    n_added_labels = 0
    progress = tqdm(total=len(df))
    with db_engine.session() as session:
        for _, row in df.iterrows():
            item_id = session.scalar(
                select(AcademicItem.item_id).where(AcademicItem.openalex_id == row[id_column], AcademicItem.project_id == settings.PROJECT_ID).limit(1),
            )
            if not item_id:
                n_missing_ids += 1
                continue
            for col_old, col_new in keys.items():
                if pd.isna(row[col_old]):
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
        session.commit()

    logger.info(f'Finished with missed IDs: {n_missing_ids:,} | skipped labels: {n_skipped_labels:,} | added labels: {n_added_labels:,}')


@app.command('dataset')
def dataset(
    source: Annotated[Path, typer.Option(help='source parquet')],
    config: Annotated[Path, typer.Option(help='config file')],
    loglevel: Annotated[str, typer.Option(help='log level')] = 'INFO',
):
    """
    uv run codepile/ingest.py dataset --source data/2025/dataset.parquet --config config/secret.env
      Found 58,307 entries for 58,307 ids (openalex_id) with 1,728,164 labels
      Finished with missed IDs: 44 | skipped labels: 0 | added labels: 1,726,848
    """
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
            't0-0-0|0': 'topic-0-0|26',
            't0-0-0|1': 'topic-0-0|29',
            't0-0-0|2': 'topic-0-0|43',
            't0-0-1|0': 'topic-0-1|11',
            't0-0-1|1': 'topic-0-1|37',
            't0-0-1|2': 'topic-0-1|45',
            't0-0-1|3': 'topic-0-1|64',
            't0-0-1|4': 'topic-0-1|68',
            't0-0-2|0': 'topic-0-2|35',
            't0-0-2|1': 'topic-0-2|47',
            't0-0-2|2': 'topic-0-2|58',
            't0-0-3|0': 'topic-0-3|65',
            't0-0-4|0': 'topic-0-4|1',
            't0-0-4|1': 'topic-0-4|44',
            't0-0-5|0': 'topic-0-5|21',
            't0-0-5|1': 'topic-0-5|31',
            't0-0-5|2': 'topic-0-5|36',
            't0-0-5|3': 'topic-0-5|51',
            't0-0-5|4': 'topic-0-5|52',
            't0-0-6|0': 'topic-0-6|17',
            't0-1-0|0': 'topic-1-0|25',
            't0-1-0|1': 'topic-1-0|61',
            't0-1-1|0': 'topic-1-1|3',
            't0-1-1|1': 'topic-1-1|24',
            't0-1-1|2': 'topic-1-1|67',
            't0-1-2|0': 'topic-1-2|5',
            't0-1-2|1': 'topic-1-2|46',
            't0-1-3|0': 'topic-1-3|7',
            't0-1-3|1': 'topic-1-3|9',
            't0-1-3|2': 'topic-1-3|10',
            't0-1-3|3': 'topic-1-3|40',
            't0-1-4|0': 'topic-1-4|0',
            't0-1-4|1': 'topic-1-4|4',
            't0-1-4|2': 'topic-1-4|6',
            't0-1-4|3': 'topic-1-4|14',
            't0-1-4|4': 'topic-1-4|22',
            't0-1-4|5': 'topic-1-4|55',
            't0-1-4|6': 'topic-1-4|56',
            't0-1-4|7': 'topic-1-4|60',
            't0-1-4|8': 'topic-1-4|66',
            't0-1-5|0': 'topic-1-5|39',
            't0-1-5|1': 'topic-1-5|48',
            't0-1-6|0': 'topic-1-6|27',
            't0-1-6|1': 'topic-1-6|30',
            't0-1-7|0': 'topic-1-7|19',
            't0-1-8|0': 'topic-1-8|20',
            't0-1-8|1': 'topic-1-8|28',
            't0-1-8|2': 'topic-1-8|41',
            't0-1-8|3': 'topic-1-8|59',
            't0-1-8|4': 'topic-1-8|62',
            't0-1-9|0': 'topic-1-9|50',
            't0-2-0|0': 'topic-2-0|53',
            't0-2-1|0': 'topic-2-1|63',
            't0-2-2|0': 'topic-2-2|13',
            't0-2-2|1': 'topic-2-2|23',
            't0-2-3|0': 'topic-2-3|33',
            't0-3-0|0': 'topic-3-0|15',
            't0-3-1|0': 'topic-3-1|2',
            't0-3-1|1': 'topic-3-1|12',
            't0-3-1|2': 'topic-3-1|16',
            't0-3-1|3': 'topic-3-1|69',
            't0-3-2|0': 'topic-3-2|42',
            't0-4-0|0': 'topic-4-0|8',
            't0-4-0|1': 'topic-4-0|18',
            't0-4-0|2': 'topic-4-0|32',
            't0-4-0|3': 'topic-4-0|34',
            't0-4-0|4': 'topic-4-0|38',
            't0-4-0|5': 'topic-4-0|49',
            't0-4-0|6': 'topic-4-0|54',
            't0-4-0|7': 'topic-4-0|57',
        },
        id_column='openalex_id',
        source=source,
        config=config,
        loglevel=loglevel,
    )


@app.command('impacts')
def impacts(
    source: Annotated[Path, typer.Option(help='source parquet')],
    config: Annotated[Path, typer.Option(help='config file')],
    loglevel: Annotated[str, typer.Option(help='log level')] = 'INFO',
):
    """
    uv run codepile/ingest.py impacts --source data/2025/df_impacts.arrow --config config/secret.env
      Found 46,810 entries for 46,810 ids (openalex_id) with 1,778,780 labels
      Finished with missed IDs: 35 | skipped labels: 0 | added labels: 1,777,450
    """
    # ['index', 'publication_year', 'abstract', 'authors', 'doi', 'openalex_id', 'title', 'rel|1', 'revision_id_x', 'cat|1', 'cat|2', 'cat|0', 'revision_id_y',
    # 'batch_id_y', 'place_item_id', 'place_*',
    # 'cont|0', 'cont|1', 'cont|2', 'cont|3', 'cont|4', 'cont|5', 'cont|6', 'affil_item_id', 'affil_*', 'x', 'y', 'Topic|item_id',
    # 't0-2-3|0', 't0-3-0|0', 't0-1-8|4', 't0-0-5|4', 't0-1-8|0', 't0-1-5|1', 't0-3-1|0', 't0-4-0|7', 't0-1-5|0', 't0-3-1|1', 't0-1-4|4', 't0-0-5|0', 't0-2-0|0',
    # 't0-1-0|0', 't0-1-4|8', 't0-2-1|0', 't0-4-0|0', 't0-0-5|1', 't0-1-9|0', 't0-0-0|1', 't0-0-0|0', 't0-0-1|3', 't0-0-1|2', 't0-2-2|0', 't0-4-0|1', 't0-0-2|1',
    # 't0-1-2|1', 't0-0-2|2', 't0-1-2|0', 't0-1-8|1', 't0-0-3|0', 't0-2-2|1', 't0-1-4|1', 't0-1-1|2', 't0-0-5|3', 't0-1-3|0', 't0-0-2|0', 't0-0-5|2', 't0-1-4|6',
    # 't0-1-4|2', 't0-1-4|3', 't0-1-4|7', 't0-1-6|1', 't0-0-4|1', 't0-4-0|6', 't0-0-6|0', 't0-1-0|1', 't0-1-4|5', 't0-1-7|0', 't0-0-1|1', 't0-0-1|0', 't0-0-1|4',
    # 't0-1-3|3', 't0-1-8|3', 't0-1-3|2', 't0-1-8|2', 't0-4-0|2', 't0-3-1|2', 't0-0-4|0', 't0-4-0|3', 't0-1-1|0', 't0-1-6|0', 't0-4-0|4', 't0-4-0|5', 't0-1-1|1',
    # 't0-1-4|0', 't0-3-1|3', 't0-1-3|1', 't0-3-2|0', 't0-0-0|2', 'revision_id', 'batch_id', 'idx', 'Exposure', 'Health impact', 'Intervention option',
    # 'Mediating pathways', 'Other', 'Category', 'pred_climateDrivers|0', 'pred_climateDrivers|1', 'pred_climateDrivers|2', 'pred_climateDrivers|3',
    # 'pred_climateDrivers|4', 'pred_climateDrivers|5', 'pred_climateDrivers|7', 'pred_climateDrivers|6', 'pred_extremeEvent|0', 'pred_extremeEvent|1',
    # 'pred_extremeEvent|2', 'pred_extremeEvent|3', 'pred_extremeEvent|6', 'pred_extremeEvent|4', 'pred_extremeEvent|5', 'pred_impactsHealth|0',
    # 'pred_impactsHealth|1', 'pred_impactsHealth|2', 'pred_impactsHealth|3', 'pred_impactsHealth|4', 'pred_impactsHealth|5', 'pred_impactsHealth|6',
    # 'pred_impactsHealth|7', 'pred_impactsHealth|8', 'pred_impactsHealth|9', 'pred_impactsHealth|10', 'pred_impactsHealth|11', 'pred_impactsHealth|12',
    # 'pred_Exposuretype|0', 'pred_Exposuretype|1', 'pred_Exposuretype|2', 'pred_Exposuretype|3', 'pred_Exposuretype|4', 'pred_Atrributiontype|0',
    # 'pred_Atrributiontype|1', 'pred_Atrributiontype|2', 'pred_Atrributiontype|3', 'pred_Atrributiontype|4']
    _label_ingest(
        keys={
            'pred_climateDrivers|0': 'driver|0',
            'pred_climateDrivers|1': 'driver|1',
            'pred_climateDrivers|2': 'driver|2',
            'pred_climateDrivers|3': 'driver|3',
            'pred_climateDrivers|4': 'driver|4',
            'pred_climateDrivers|5': 'driver|5',
            'pred_climateDrivers|6': 'driver|6',
            'pred_climateDrivers|7': 'driver|7',
            'pred_extremeEvent|0': 'event|0',
            'pred_extremeEvent|1': 'event|1',
            'pred_extremeEvent|2': 'event|2',
            'pred_extremeEvent|3': 'event|3',
            'pred_extremeEvent|4': 'event|4',
            'pred_extremeEvent|5': 'event|5',
            'pred_extremeEvent|6': 'event|6',
            'pred_impactsHealth|0': 'health|0',
            'pred_impactsHealth|1': 'health|1',
            'pred_impactsHealth|2': 'health|2',
            'pred_impactsHealth|3': 'health|3',
            'pred_impactsHealth|4': 'health|4',
            'pred_impactsHealth|5': 'health|5',
            'pred_impactsHealth|6': 'health|6',
            'pred_impactsHealth|7': 'health|7',
            'pred_impactsHealth|8': 'health|8',
            'pred_impactsHealth|9': 'health|9',
            'pred_impactsHealth|10': 'health|10',
            'pred_impactsHealth|11': 'health|11',
            'pred_impactsHealth|12': 'health|12',
            'pred_Exposuretype|0': 'expose|0',
            'pred_Exposuretype|1': 'expose|1',
            'pred_Exposuretype|2': 'expose|2',
            'pred_Exposuretype|3': 'expose|3',
            'pred_Exposuretype|4': 'expose|4',
            'pred_Atrributiontype|0': 'attr|0',
            'pred_Atrributiontype|1': 'attr|1',
            'pred_Atrributiontype|2': 'attr|2',
            'pred_Atrributiontype|3': 'attr|3',
            'pred_Atrributiontype|4': 'attr|4',
        },
        id_column='openalex_id',
        source=source,
        config=config,
        loglevel=loglevel,
    )


if __name__ == '__main__':
    app()

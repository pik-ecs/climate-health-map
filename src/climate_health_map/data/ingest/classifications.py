import logging
import uuid
from pathlib import Path
from typing import Annotated
from itertools import batched

import typer
from sqlalchemy.orm import Session
from tqdm import tqdm
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from nacsos_data.db.schemas import Enhancement, Item

from climate_health_map.data.labels import LABELS_LOOKUP
from climate_health_map.shared import essentials, read_any_pd
from climate_health_map.shared.types import OnConflict


def _ingest_file(
    fn: Path,
    session: Session,
    batch_size: int,
    logger: logging.Logger,
    on_conflict: OnConflict,
) -> None:
    if on_conflict == OnConflict.BREAK:
        raise NotImplementedError('Break on exist is not implemented.')
    logger.info(f'Reading data from {fn.resolve()}...')
    df = read_any_pd(fn)
    keys = list(set(df.columns) & set(LABELS_LOOKUP.keys()))

    for key in keys:
        mask = df[key].notna()
        progress = tqdm(total=df[mask].shape[0], desc=f'Ingesting "{key}"')
        for batch in batched(df[mask].iterrows(), batch_size, strict=False):
            if on_conflict == OnConflict.IGNORE:
                progress.set_postfix_str(f'Dropping existing item/key pairs for key="{key}"')
                session.execute(
                    sa.delete(Enhancement).where(
                        Enhancement.item_id.in_([i['item_id'] for _, i in batch]),
                        Enhancement.key == key,
                    ),
                )
                session.flush()

            data = (
                sa.values(
                    sa.column('enhancement_id', sa.UUID),
                    sa.column('item_id', sa.UUID),
                    sa.column('key', sa.TEXT),
                    sa.column('payload', JSONB),
                )
                .data(
                    [(uuid.uuid4(), enhancement['item_id'], key, float(enhancement[key])) for _, enhancement in batch],
                )
                .alias('data')
            )

            if on_conflict == OnConflict.SKIP:
                stmt_filter = (
                    sa.select(data)
                    .join(Item, Item.item_id == data.c.item_id)
                    .join(Enhancement, sa.and_(Enhancement.item_id == data.c.item_id, Enhancement.key == key), isouter=True)
                    .where(Enhancement.key == None)  # noqa: E711
                )

            progress.set_postfix_str(f'Inserting codes for key="{key}" in mode ({on_conflict.value})')
            session.execute(
                sa.insert(Enhancement.__table__).from_select(['enhancement_id', 'item_id', 'key', 'payload'], stmt_filter),  # type:ignore[arg-type]
            )
            session.flush()
            progress.update(len(batch))
        progress.close()


def ingest_file(
    config: Annotated[Path, typer.Option(help='Path to config.env')],
    source: Annotated[Path, typer.Option(help='Path to the predictions file')],
    on_conflict: Annotated[OnConflict, typer.Option(help='How to handle existing key/value pairs')],
    batch_size: Annotated[int, typer.Option(help='Batch size for import')] = 200,
    loglevel: Annotated[str, typer.Option(help='Verbosity of logger')] = 'INFO',
) -> None:
    logger, settings, db_engine = essentials(config=config, logger_name='ingest', loglevel=loglevel, run_log_init=True)

    with db_engine.session() as session:
        _ingest_file(
            fn=source,
            logger=logger,
            session=session,
            batch_size=batch_size,
            on_conflict=on_conflict,
        )
        session.commit()


def ingest_dir(
    config: Annotated[Path, typer.Option(help='Path to config.env')],
    source_dir: Annotated[Path, typer.Option(help='Path to the predictions directory')],
    on_conflict: Annotated[OnConflict, typer.Option(help='How to handle existing key/value pairs')],
    batch_size: Annotated[int, typer.Option(help='Batch size for import')] = 200,
    loglevel: Annotated[str, typer.Option(help='Verbosity of logger')] = 'INFO',
    filetype: Annotated[str, typer.Option(help='File type')] = 'arrow',
) -> None:
    logger, settings, db_engine = essentials(config=config, logger_name='ingest', loglevel=loglevel, run_log_init=True)

    with db_engine.session() as session:
        for fn in source_dir.glob(f'*.{filetype}'):
            _ingest_file(
                fn=fn,
                logger=logger,
                session=session,
                batch_size=batch_size,
                on_conflict=on_conflict,
            )

        session.commit()

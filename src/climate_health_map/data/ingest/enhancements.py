import re
import uuid
import logging
from pathlib import Path
from typing import Annotated
from itertools import batched

import typer
import pandas as pd
from tqdm import tqdm
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from nacsos_data.util.conf import Settings
from nacsos_data.db import get_engine
from nacsos_data.db.schemas import Enhancement, Item

from climate_health_map.shared.types import OnConflict

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('root').setLevel(logging.INFO)
logging.getLogger('elasticsearch').setLevel(logging.WARNING)

logger = logging.getLogger('ingest')
logger.setLevel(logging.INFO)

raise NotImplementedError('just a copy')


def func(  # type:ignore[unreachable]
    config: Annotated[Path, typer.Option(help='Path to config.env')],
    on_conflict: Annotated[OnConflict, typer.Option(help='How to handle existing key/value pairs')],
    source_dir: Annotated[Path, typer.Option(help='Path to the predictions directory')],
    batch_size: Annotated[int, typer.Option(help='Batch size for import')] = 200,
):
    logger.info('Connecting to database...')
    settings = Settings(_env_file=config, _env_file_encoding='utf-8')
    db_engine = get_engine(settings=settings.DB, debug=False)

    logger.info(f'Reading data from {source_dir.resolve()}...')
    for fn in source_dir.glob('predictions/*_pred.csv'):
        logger.info(f'Reading data from {fn.resolve()}...')
        model, parent, label, repeat = re.fullmatch(r'(.+?)_(.+?)_(.+?)_(\d)_pred\.csv', fn.name).groups()
        df = pd.read_csv(fn, usecols=['item_id', 'score'])

        progress = tqdm(total=df.shape[0])
        with db_engine.session() as session:
            for batch in batched(df.iterrows(), batch_size, strict=False):
                if on_conflict == OnConflict.IGNORE:
                    progress.set_postfix_str(f'Dropping existing item/key pairs for key="{label}"')
                    session.execute(
                        sa.delete(Enhancement).where(
                            Enhancement.item_id.in_([i['item_id'] for _, i in batch]),
                            Enhancement.key == label,
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
                        [(uuid.uuid4(), enhancement['item_id'], label, float(enhancement['score'])) for _, enhancement in batch],
                    )
                    .alias('data')
                )

                if on_conflict == OnConflict.KEEP:
                    stmt_filter = (
                        sa.select(data)
                        .join(Item, Item.item_id == data.c.item_id)
                        .join(Enhancement, sa.and_(Enhancement.item_id == data.c.item_id, Enhancement.key == label), isouter=True)
                        .where(Enhancement.key == None)  # noqa: E711
                    )
                else:  # on_conflict == OnConflict.EXTEND:
                    stmt_filter = sa.select(data).join(Item, Item.item_id == data.c.item_id)

                progress.set_postfix_str(f'Inserting codes for key="{label}" in mode ({on_conflict.value})')
                session.execute(
                    sa.insert(Enhancement.__table__).from_select(['enhancement_id', 'item_id', 'key', 'payload'], stmt_filter),
                )
                session.flush()
                progress.update(len(batch))
            session.commit()
        progress.close()


if __name__ == '__main__':
    typer.run(func)

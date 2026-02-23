import uuid
from pathlib import Path
from typing import Annotated

import typer
from tqdm import tqdm
import sqlalchemy as sa

from mordecai3 import Geoparser

from climate_health_map.shared.env import essentials
from nacsos_data.util import clear_empty
from nacsos_data.db.schemas import Enhancement


def mordecai(
    config: Annotated[Path, typer.Option(help='Path to config.env')],
    batch_size: Annotated[int, typer.Option(help='')] = 100,
    verbose: Annotated[bool, typer.Option(help='Turn on debug log for mordecai')] = False,
    hosts: Annotated[list[str] | None, typer.Option(help='')] = None,
    port: Annotated[int, typer.Option(help='')] = 9200,
    device: Annotated[str, typer.Option(help='')] = 'gpu',
    created_after: Annotated[str, typer.Option(help='Filter to only apply mordecai to items created after that date; format: YYYY-MM-DD')] = None,
    loglevel: Annotated[str, typer.Option(help='Verbosity of logger')] = 'INFO',
):
    logger, settings, db_engine = essentials(config=config, loglevel=loglevel, logger_name='export', run_log_init=True)

    logger.info('Setting up geoparser...')
    geo = Geoparser(debug=verbose, hosts=hosts, port=port, device=device)

    if created_after:
        stmt = """
            SELECT DISTINCT m2mii.item_id,
                   (coalesce(ai.title, '') || '. ' || coalesce(i.text, '')) as txt
            FROM m2m_import_item m2mii
                 JOIN import_revision ir ON m2mii.import_id = ir.import_id AND m2mii.first_revision = ir.import_revision_counter
                 JOIN item i ON m2mii.item_id = i.item_id
                 JOIN academic_item ai ON m2mii.item_id = ai.item_id
                 LEFT OUTER JOIN enhancement e ON i.item_id = e.item_id AND e.key = 'mordecai3'
            WHERE ai.project_id = :project_id
              AND m2mii.import_id::text = ANY(:import_ids)
              AND ir.time_created > :created_after
              AND length((coalesce(ai.title, '') || '. ' || coalesce(i.text, ''))) > 60
              AND e.key IS NULL;"""
    else:
        stmt = """
            SELECT DISTINCT ai.item_id,
                   (coalesce(ai.title, '') || '. ' || coalesce(i.text, '')) as txt
            FROM academic_item ai
                 JOIN item i ON i.item_id = ai.item_id
                 JOIN m2m_import_item ii ON ai.item_id = ii.item_id
                 LEFT OUTER JOIN enhancement e ON i.item_id = e.item_id AND e.key = 'mordecai3'
            WHERE ai.project_id = :project_id
              AND ii.import_id::text = ANY(:import_ids)
              AND length((coalesce(ai.title, '') || '. ' || coalesce(i.text, ''))) > 60
              AND e.key IS NULL;
        """
    with db_engine.session() as session:
        logger.info('Running query...')
        rslt = session.execute(
            sa.text(stmt).execution_options(yield_per=batch_size),
            {'project_id': settings.PROJECT_ID, 'import_ids': settings.IMPORTS, 'created_after': created_after},
        )

        tq = tqdm()
        cnt = 0

        logger.info('Start batched processing...')
        for batch in rslt.mappings().partitions():
            for item in batch:
                tq.update()
                try:
                    places = geo.geoparse_doc(item['txt'])
                    if len(places['geolocated_ents']) > 0:
                        session.add(
                            Enhancement(
                                enhancement_id=uuid.uuid4(),
                                item_id=item['item_id'],
                                key='mordecai3',
                                payload=clear_empty(places['geolocated_ents']),
                            ),
                        )
                        session.flush()
                        cnt += 1
                except Exception as e:
                    logger.error(e)
                session.commit()
            tq.set_description(f'Updated {cnt:,} items')

        tq.close()
        logger.info('Finished processing!')


if __name__ == '__main__':
    typer.run(mordecai)

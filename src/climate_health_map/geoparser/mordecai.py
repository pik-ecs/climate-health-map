import uuid
from pathlib import Path
from typing import Annotated, Any, Generator, TYPE_CHECKING

import typer
from tqdm import tqdm
import sqlalchemy as sa

from nacsos_data.util import clear_empty
from nacsos_data.db.schemas import Enhancement

from climate_health_map.shared.env import essentials
from climate_health_map.shared.text import chunked_text, clean_text
from .filters import TEXT_FILTER

if TYPE_CHECKING:
    from mordecai3 import Geoparser


def apply_mordecai(text: str, geo: 'Geoparser') -> Generator[dict[str, Any], None, None]:
    text_clean = clean_text(text, extra=[TEXT_FILTER])
    for chunk in chunked_text(text_clean, chunk_size=500, overlap=15):
        places = geo.geoparse_doc(chunk)
        yield from places['geolocated_ents']


def mordecai(
    config: Annotated[Path, typer.Option(help='Path to config.env')],
    batch_size: Annotated[int, typer.Option(help='')] = 100,
    verbose: Annotated[bool, typer.Option(help='Turn on debug log for mordecai')] = False,
    hosts: Annotated[list[str] | None, typer.Option(help='')] = None,
    port: Annotated[int, typer.Option(help='')] = 9200,
    device: Annotated[str, typer.Option(help='')] = 'gpu',
    only_incl: Annotated[bool, typer.Option(help='Only apply mordecai to included records')] = True,
    incl_threshold: Annotated[float, typer.Option(help='Only apply mordecai to records with "rel_major|1" > THRESHOLD')] = 0.5,
    min_text_len: Annotated[int, typer.Option(help='Minimum length of title+abstract (in characters)')] = 100,
    show_count: Annotated[bool, typer.Option(help='')] = False,
    created_after: Annotated[str | None, typer.Option(help='Filter to only apply mordecai to items created after that date; format: YYYY-MM-DD')] = None,
    published_after: Annotated[int, typer.Option(help='Filter to only apply mordecai to items publisher after this year (>=)')] = None,
    published_before: Annotated[int, typer.Option(help='Filter to only apply mordecai to items publisher before this year (<>>=)')] = None,
    loglevel: Annotated[str, typer.Option(help='Verbosity of logger')] = 'INFO',
) -> None:
    logger, settings, db_engine = essentials(config=config, loglevel=loglevel, logger_name='export', run_log_init=True)

    logger.info('Setting up geoparser...')
    from mordecai3 import Geoparser

    geo = Geoparser(debug=verbose, hosts=hosts, port=port, device=device)

    extra_joins = []
    extra_wheres = []

    if only_incl:
        extra_joins.append("JOIN enhancement incl ON i.item_id = incl.item_id AND incl.key = 'rel_major|1' AND incl.payload::float > :threshold")
    if created_after:
        extra_joins.append('JOIN import_revision ir ON m2mii.import_id = ir.import_id AND m2mii.first_revision = ir.import_revision_counter')
        extra_wheres.append('ir.time_created > :created_after')

    if published_after:
        extra_wheres.append('ai.publication_year >= :published_after')
    if published_before:
        extra_wheres.append('ai.publication_year <= :published_before')

    extra_wheres_ = ''
    if len(extra_wheres) > 0:
        extra_wheres_ = ' AND '.join(extra_wheres)
        extra_wheres_ = f'AND {extra_wheres_}'

    stmt = f"""
        SELECT DISTINCT m2mii.item_id,
               (coalesce(ai.title, '') || '. ' || coalesce(i.text, '')) as txt
        FROM m2m_import_item m2mii
             JOIN item i ON m2mii.item_id = i.item_id
             JOIN academic_item ai ON m2mii.item_id = ai.item_id
             {'\n '.join(extra_joins)}
             LEFT OUTER JOIN enhancement e ON i.item_id = e.item_id AND e.key = 'mordecai3'
        WHERE ai.project_id = :project_id
          AND m2mii.import_id::text = ANY(:import_ids)
          AND length((coalesce(ai.title, '') || '. ' || coalesce(i.text, ''))) > :min_len
          AND e.key IS NULL
          {extra_wheres_}
    """

    with db_engine.session() as session:
        params = {
            'project_id': settings.PROJECT_ID,
            'import_ids': settings.IMPORTS,
            'created_after': created_after,
            'threshold': incl_threshold,
            'min_len': min_text_len,
            'published_before': published_before,
            'published_after': published_after,
        }
        count = None
        if show_count:
            logger.info('Running count query...')
            count = session.scalar(sa.text(f'SELECT count(1) as n_records FROM ({stmt});').execution_options(yield_per=batch_size), params=params)
            logger.info(f'Will hydrate openalex info for {count:,} records.')

        logger.info('Running query...')
        rslt = session.execute(sa.text(f'{stmt};').execution_options(yield_per=batch_size), params=params)
        logger.info('Start batched processing...')
        tq = tqdm(total=count)
        counters = {'n_processed': 0, 'n_with_place': 0}
        for batch in rslt.mappings().partitions():
            for item in batch:
                tq.update()
                try:
                    places: list[dict[str, Any]] | None = list(apply_mordecai(item['txt'], geo=geo))
                    places = clear_empty(places)
                    session.add(
                        Enhancement(
                            enhancement_id=uuid.uuid4(),
                            item_id=item['item_id'],
                            key='mordecai3',
                            payload=places,
                        ),
                    )
                    counters['n_processed'] += 1
                    counters['n_with_place'] += int(places is not None)
                except Exception as e:
                    logger.error(e)
            session.commit()
            tq.set_postfix(counters)

        tq.close()
        logger.info('Finished processing!')


if __name__ == '__main__':
    typer.run(mordecai)

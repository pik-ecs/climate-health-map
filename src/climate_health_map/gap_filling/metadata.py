import logging
from itertools import batched
from pathlib import Path
from typing import Annotated, Sequence, Iterator, Generator

import typer
from nacsos_data.db.schemas import AcademicItem
from nacsos_data.util.academic.apis import OpenAlexAPI
from sqlalchemy.dialects.postgresql import JSONB
from tqdm import tqdm
import sqlalchemy as sa

from climate_health_map.shared.env import essentials


def unroll_partitions(rslt: Iterator[Sequence[sa.RowMapping]]) -> Generator[sa.RowMapping, None, None]:
    for batch in rslt:
        yield from batch


def hydrate_openalex_metadata(
    config: Annotated[Path, typer.Option(help='Path to config.env')],
    only_incl: Annotated[bool, typer.Option('--only-incl/--all', help='Only apply mordecai to included records')] = True,
    incl_threshold: Annotated[float, typer.Option(help='Only apply mordecai to records with "rel_major|1" > THRESHOLD')] = 0.5,
    created_after: Annotated[str | None, typer.Option(help='Filter to only apply mordecai to items created after that date; format: YYYY-MM-DD')] = None,
    batch_size: Annotated[int, typer.Option(help='', max=100)] = 100,
    show_count: Annotated[bool, typer.Option(help='')] = False,
    overwrite_meta: Annotated[bool, typer.Option(help='Set to update existing `openalex` meta entry via || (concatenate)')] = True,
    loglevel: Annotated[str, typer.Option(help='Verbosity of logger')] = 'INFO',
) -> None:
    logger, settings, db_engine = essentials(config=config, loglevel=loglevel, logger_name='export', run_log_init=True)
    if batch_size > 100:
        raise ValueError('Batch size cannot be greater than 100')
    join_incl = ''
    if only_incl:
        join_incl = "JOIN enhancement incl ON i.item_id = incl.item_id AND incl.key = 'rel_major|1' AND payload::float > :threshold"
    join_created = ''
    if created_after:
        join_created = 'JOIN import_revision ir ON m2mii.import_id = ir.import_id AND m2mii.first_revision = ir.import_revision_counter AND ir.time_created > :created_after'
    # SELECT ai.item_id, ai.openalex_id
    # FROM academic_item ai
    # WHERE ai.project_id = %(project_id)s
    #   AND ai.openalex_id IS NOT NULL
    #   AND ai.meta -> 'openalex' IS NULL
    #   -- Use EXISTS to avoid creating duplicates that require DISTINCT
    #   AND EXISTS (
    #       SELECT 1
    #       FROM m2m_import_item m2mii
    #       WHERE m2mii.item_id = ai.item_id
    #         -- Avoid casting the column; cast the input list instead if possible
    #         AND m2mii.import_id::text = ANY(%(import_ids)s)
    #   )
    #   AND EXISTS (
    #       SELECT 1
    #       FROM enhancement incl
    #       WHERE incl.item_id = ai.item_id
    #         AND incl.key = 'rel_major|1'
    #         -- This cast is still expensive; see index suggestions below
    #         AND (incl.payload)::float > %(threshold)s
    #   );
    stmt = f"""
        SELECT DISTINCT m2mii.item_id, ai.openalex_id
        FROM m2m_import_item m2mii
             JOIN item i ON m2mii.item_id = i.item_id
             JOIN academic_item ai ON m2mii.item_id = ai.item_id
             {join_incl}
             {join_created}
        WHERE ai.project_id = :project_id
          AND m2mii.import_id::text = ANY(:import_ids)
          AND ai.openalex_id IS NOT NULL
          AND ai.meta -> 'openalex' -> 'authorships' IS NULL
    """

    with db_engine.session() as session:
        params = {
            'project_id': settings.PROJECT_ID,
            'import_ids': settings.IMPORTS,
            'created_after': created_after,
            'threshold': incl_threshold,
        }
        count = None
        if show_count:
            logger.info('Running count query...')
            count = session.scalar(sa.text(f'SELECT count(1) as n_records FROM ({stmt});').execution_options(yield_per=batch_size), params=params)
            logger.info(f'Will hydrate openalex info for {count:,} records.')

        logger.info('Running query...')
        rslt = session.execute(sa.text(f'{stmt};').execution_options(yield_per=batch_size), params=params)

        api_logger = logger.getChild('api')
        api_logger.setLevel(logging.WARN)
        api = OpenAlexAPI(api_key=settings.OPENALEX.API_KEY, page_size=batch_size, logger=api_logger)
        logger.info('Start batched processing...')
        tq = tqdm(total=count)
        counters = {'n_processed': 0, 'n_with_oa': 0, 'n_with_authorships': 0}
        for batch in batched(
            # partitions are not necessarily the batch size, so do some acrobatics to ensure exact batch size
            unroll_partitions(rslt.mappings().partitions()),
            n=batch_size,
            strict=False,
        ):
            id_map = {row['openalex_id']: str(row['item_id']) for row in batch}
            counters['n_processed'] += len(batch)
            for item in api.fetch_translated(project_id=settings.PROJECT_ID, query=f'ids.openalex:({"|".join(id_map.keys())})'):
                counters['n_with_oa'] += 1
                if len(item.meta['openalex'].get('authorships', [])) > 0:
                    counters['n_with_authorships'] += 1
                tq.update()
                stmt = sa.update(AcademicItem).where(AcademicItem.openalex_id == item.openalex_id)
                if not overwrite_meta:
                    stmt = stmt.where(AcademicItem.meta['openalex'].astext == None)  # noqa: E711
                session.execute(
                    stmt.values(
                        meta=sa.func.coalesce(AcademicItem.meta, sa.cast({}, JSONB)).concat(item.meta),
                    ),
                )
                logger.debug(f'Wrote update to {item.openalex_id}')
            session.commit()
            tq.set_postfix(counters)
        tq.close()
        logger.info('Finished processing!')


if __name__ == '__main__':
    typer.run(hydrate_openalex_metadata)

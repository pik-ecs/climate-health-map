import asyncio
from pathlib import Path
from typing import Annotated, Generator

import typer
from nacsos_data.db import get_engine_async
from nacsos_data.models.items import AcademicItemModel
from nacsos_data.util.academic.apis import OpenAlexSolrAPI
from nacsos_data.util.academic.importer import import_academic_items

from climate_health_map.shared.env import base_essentials


def get_query(query_file: Path | None) -> str:
    here = Path(__file__).parent.resolve()
    if query_file is None:
        query_file = here / 'query_20241029.txt'
    with open(query_file, 'r') as f:
        return f.read()


def solr_ingest(
    config: Annotated[Path, typer.Option(help='Path to config file')],
    query_file: Annotated[Path | None, typer.Option(help='Query file')] = None,
    batch_size: Annotated[int, typer.Option(help='Batch size')] = 10000,
    min_update_size: Annotated[int, typer.Option(help='Batch size')] = 500,
    loglevel: Annotated[str, typer.Option(help='Loglevel')] = 'INFO',
) -> None:
    """
    This is a clone of `nacsos_data.util.academic.importer.import_openalex`
    with the addition of some filters
    """
    logger, settings = base_essentials(config=config, loglevel=loglevel, logger_name='export', run_log_init=True)
    logger.info('Connecting to database...')
    db_engine = get_engine_async(settings=settings.DB)

    solr_client = OpenAlexSolrAPI(
        openalex_conf=settings.OPENALEX,
        def_type='lucene',
        field='title_abstract',
        op='AND',
        batch_size=batch_size,
        logger=logger,
    )
    logger.info('Reading query...')
    query = get_query(query_file)

    def from_source() -> Generator[AcademicItemModel, None, None]:
        yield from (
            solr_client.fetch_translated(
                query=query,
                project_id=settings.PROJECT_ID,
                params={'fq': '-(source_id:  "S7407052681")'},  # Drop "Data Planet" stuff (~30k+ as of February 2026)
            )
        )

    logger.info('Fetching item count')
    num_new_items = solr_client.get_count(query).num_found

    async def inner() -> tuple[str, int | None]:

        return await import_academic_items(
            db_engine=db_engine,
            project_id=settings.PROJECT_ID,
            new_items=from_source,
            min_update_size=min_update_size,
            num_new_items=num_new_items,
            import_name=None,
            description=None,
            user_id=None,
            import_id=settings.IMPORTS[0],
            vectoriser=None,
            max_slop=0.05,
            batch_size=5000,
            dry_run=False,
            logger=logger,
        )

    asyncio.run(inner())

import csv
import logging
from itertools import batched
from pathlib import Path
from typing import Annotated

import typer
from nacsos_data.util.academic.apis import OpenAlexSolrAPI
from tqdm import tqdm

from climate_health_map.shared.env import base_essentials


def check_openalex_ids(
    config: Annotated[Path, typer.Option(help='Path to config file')],
    source: Annotated[Path, typer.Option(help='Path to read IDs from')],
    target: Annotated[Path, typer.Option(help='Path to write missing DOI/ID pairs to')],
    batch_size: Annotated[int, typer.Option(help='Batch size')] = 1000,
    loglevel: Annotated[str, typer.Option(help='Verbosity of logger')] = 'DEBUG',
) -> None:
    if target.exists():
        raise FileExistsError(f'File {target.name} already exists.')

    logger, settings = base_essentials(config, logger_name='openalex', loglevel=loglevel, run_log_init=True)
    api_logger = logger.getChild('solr')
    api_logger.setLevel(logging.WARNING)
    solr = OpenAlexSolrAPI(openalex_conf=settings.OPENALEX, logger=api_logger, export_fields=['id'], op='AND')

    n_found = 0
    n_checked = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f'Reading IDs from {source}')
    logger.info(f'Writing DOI/IDs to {target}')
    with (
        open(source, 'r') as ids_reader,
        open(target, 'w', newline='') as ids_writer,
    ):
        progress = tqdm(ids_reader)
        csv_file = csv.writer(ids_writer, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_file.writerow(['openalex_id', 'doi'])

        for id_batch in batched(progress, n=batch_size, strict=False):
            cleaned_ids = [oa_id[len('https://openalex.org/') :].strip() for oa_id in id_batch]
            ids = ' OR '.join(cleaned_ids)
            n_checked += len(cleaned_ids)

            # Fetch records with these IDs that have no abstract but a DOI
            for res in solr.fetch_raw(query='', params={'fl': 'id,doi', 'df': 'id', 'q': f'-abstract:* AND doi:* AND id:({ids})'}):
                csv_file.writerow((res['id'], res['doi']))
                n_found += 1
            progress.set_description(f'Checked {n_checked:,} IDs and found {n_found:,} with missing abstract')
        progress.close()

    logger.info(f'Finished, found {n_found:,}/{n_checked:,} records with missing abstract in solr.')

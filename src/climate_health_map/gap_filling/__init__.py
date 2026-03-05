import typer

from openalex_ingest.daily.fix_abstracts import queue_from_file, transfer_abstracts
from openalex_ingest.worker.main import main as api_worker

from .superset import get_openalex_ids
from .check_solr import check_openalex_ids


abstracts_app = typer.Typer(help='Fix missing abstracts')
abstracts_app.command('fetch-ids', help='Fetch IDs from API using superset query where we are likely missing an abstract')(get_openalex_ids)
abstracts_app.command('check-ids', help='Check if list of IDs has an abstract in solr snapshot of OpenAlex')(check_openalex_ids)
abstracts_app.command('queue-ids', help='Put list of ID/DOIs in the meta-cache queue')(queue_from_file)
abstracts_app.command('fetch-abstracts', help='Work on the queue to fetch abstracts')(api_worker)
abstracts_app.command('transfer-abstracts', help='Transfer abstracts to solr')(transfer_abstracts)


if __name__ == '__main__':
    abstracts_app()

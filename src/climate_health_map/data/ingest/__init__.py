import typer
from .import_query import get_query, solr_ingest
from .classifications import ingest_dir, ingest_file

app = typer.Typer()
app.command('query', help='Run solr import with the latest query')(solr_ingest)
app.command('classifications-dir', help='Ingest all classification data in a directory')(ingest_dir)
app.command('classifications-file', help='Ingest classification data from one file')(ingest_file)

__all__ = [
    'get_query',
    'app',
    'ingest_dir',
    'ingest_file',
]

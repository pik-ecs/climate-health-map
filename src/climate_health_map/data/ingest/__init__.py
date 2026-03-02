import typer
from import_query import get_query, solr_ingest

app = typer.Typer()
app.command('query', help='Run solr import with the latest query')(solr_ingest)

__all__ = [
    'get_query',
    'app',
]

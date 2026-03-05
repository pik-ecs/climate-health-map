import typer
from .superset import get_openalex_ids

abstracts_app = typer.Typer(help='Fix missing abstracts')
abstracts_app.command('fetch-ids', help='Fetch IDs from API using superset query where we are likely missing an abstract')(get_openalex_ids)
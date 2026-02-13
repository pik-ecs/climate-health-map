import typer

from climate_health_map.data.annotations.export import dump
from climate_health_map.classification import app as classifier_app


def main():
    app = typer.Typer(no_args_is_help=True)

    app.command('export-annotations', help='Fetch all eligible annotations from NACSOS and prepare a clean csv for training and evaluation')(dump)
    app.add_typer(classifier_app)

    app()

import typer

from climate_health_map.data.annotations.export import dump
from climate_health_map.classification import app as classifier_app


def main():
    app = typer.Typer(no_args_is_help=True)

    app.command('export_annotations')(dump)
    app.add_typer(classifier_app)

    app()

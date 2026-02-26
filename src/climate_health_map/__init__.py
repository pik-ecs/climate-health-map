import typer

from climate_health_map.data.export import app as export_app
from climate_health_map.classification import app as classifier_app
from climate_health_map.shared import get_logger
from climate_health_map.data.geographies.geoparser import mordecai


def main():
    _logger = get_logger('slurm-prep', run_log_init=True, loglevel='DEBUG')
    app = typer.Typer(no_args_is_help=True)

    app.add_typer(export_app, name='export')
    app.add_typer(classifier_app, name='classification')
    app.command('geoparser', help='Extract geolocations using mordecai where the information is missing in the database')(mordecai)

    app()

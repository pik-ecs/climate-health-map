import typer

from climate_health_map.data.annotations.export import dump
from climate_health_map.classification import app as classifier_app
from climate_health_map.shared import get_logger


def main():
    _logger = get_logger('slurm-prep', run_log_init=True, loglevel='DEBUG')
    app = typer.Typer(no_args_is_help=True)

    app.command('export-annotations', help='Fetch all eligible annotations from NACSOS and prepare a clean csv for training and evaluation')(dump)
    app.add_typer(classifier_app)

    app()

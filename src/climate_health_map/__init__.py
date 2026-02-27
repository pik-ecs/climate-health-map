import typer
from rich.console import Console
from rich.tree import Tree

from climate_health_map.data.export import app as export_app
from climate_health_map.classification import app as classifier_app
from climate_health_map.shared import get_logger
from climate_health_map.geoparser import mordecai
from climate_health_map.topics import topic_model


def command_tree(app: typer.Typer) -> Tree:
    """Display all available commands in a tree structure."""

    def walk_commands(tree: Tree, current_app: typer.Typer) -> None:
        """Recursively adds commands and sub-apps to the Rich tree."""
        # Add commands in the current app
        for command in current_app.registered_commands:
            tree.add(f'[bold magenta]{command.name}[/bold magenta] - {command.help or ""}')

        # Recurse into sub-apps
        for group in current_app.registered_groups:
            sub_tree = tree.add(f'[bold blue]{group.name}[/bold blue] (sub-app)')
            walk_commands(sub_tree, group.typer_instance)  # type: ignore [arg-type]

    console = Console()
    root_tree = Tree(':root: [bold white]CLI Root[/bold white]')
    walk_commands(root_tree, app)
    console.print(root_tree)
    return root_tree


def main() -> None:
    _logger = get_logger('slurm-prep', run_log_init=True, loglevel='DEBUG')
    app = typer.Typer(no_args_is_help=True)

    app.add_typer(export_app, name='export')
    app.add_typer(classifier_app, name='classification')
    app.command('geoparser', help='Extract geolocations using mordecai where the information is missing in the database')(mordecai)
    app.command('topicmodel', help='Apply topic model to unseen records')(topic_model)

    # command_tree(app)

    app()

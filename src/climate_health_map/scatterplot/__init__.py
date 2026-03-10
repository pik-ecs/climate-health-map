import typer

from .keywords import project_topic_names
from .util import rescale_projection
from .topic_based import reduce_topic_distribution

app = typer.Typer()

app.command('reduce-topic-scores', help='UMAP projection of topic model scores')(reduce_topic_distribution)


__all__ = [
    'rescale_projection',
    'project_topic_names',
    'reduce_topic_distribution',
    'app',
]

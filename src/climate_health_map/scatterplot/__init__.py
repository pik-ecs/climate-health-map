import typer

from .keywords import project_topic_names, text_based
from .util import rescale_projection
from .reduce_topic_scores import reduce_topic_distribution

app = typer.Typer()

app.command('reduce-topic-scores', help='UMAP projection of topic model scores')(reduce_topic_distribution)
app.command('topic-keywords', help='Find where items for a topic are located and place the topic name there')(project_topic_names)
app.command('text-keywords', help='Recursively segment the space and place dominant keywords')(text_based)

__all__ = [
    'rescale_projection',
    'project_topic_names',
    'reduce_topic_distribution',
    'app',
]

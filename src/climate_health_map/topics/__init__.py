from typing import TYPE_CHECKING


from .topicmodel import TopicModel, topic_model
from .util import rescale_topic_scores

if TYPE_CHECKING:
    from climate_health_map.data.labels import Topic


def get_topic_labels() -> dict[int, 'Topic']:
    from climate_health_map.data.labels import LABELS, Topic

    return {label.topic_id: label for group in LABELS.values() for label in group.labels if type(label) is Topic}


__all__ = [
    'TopicModel',
    'topic_model',
    'rescale_topic_scores',
    'get_topic_labels',
]

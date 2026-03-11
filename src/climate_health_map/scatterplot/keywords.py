from pathlib import Path
from typing import TypedDict, Annotated

import pandas as pd
import typer
import numpy as np

from climate_health_map.shared import read_any_pd, get_logger, write_any_df
from climate_health_map.topics import TopicModel, get_topic_labels


class Keyword(TypedDict):
    x: float
    y: float
    keyword: str
    level: int


def project_topic_names(
    source_topics: Annotated[Path, typer.Option(help='')],
    source_scatter: Annotated[Path, typer.Option(help='')],
    target: Annotated[Path, typer.Option(help='')],
    eps: Annotated[float, typer.Option(help=' Maximum distance between two samples for one to be considered as in the neighborhood of the other.')] = 0.5,
    min_cluster_size: Annotated[int, typer.Option(help='Number of samples (or total weight) in a neighborhood for a point to be considered as centroid')] = 50,
    threshold_quantile: Annotated[float, typer.Option(help=' pick a topic score threshold that keeps X% of records per topic (0.0–1.0)')] = 0.95,
    loglevel: Annotated[str, typer.Option(help='Verbosity of logger')] = 'INFO',
) -> None:
    from scipy.spatial import ConvexHull
    from sklearn.cluster import DBSCAN

    logger = get_logger(loglevel=loglevel, logger_name='keywords', run_log_init=True)
    logger.info('Loading topicmodel info...')
    df_topic_infos = TopicModel.load_topic_infos()
    topic_labels = get_topic_labels()
    topic_columns = [topic.column for topic in topic_labels.values()]

    # dataframe with `item_id` as index and topic scores as columns (using column names defined in `LABELS`)
    df_topic_scores = read_any_pd(source_topics, index_column='item_id')
    logger.info(f'Loaded topic dataframe with shape {df_topic_scores.shape}')
    df_topic_scores = df_topic_scores[df_topic_scores[topic_columns].notna().any(axis=1)][topic_columns]
    logger.info(f'Filtered topic dataframe to shape {df_topic_scores.shape}')

    df_scatter = read_any_pd(source_scatter, index_column='item_id')
    logger.info(f'Loaded scatter dataframe with shape {df_scatter.shape}')

    df = df_topic_scores.join(df_scatter)
    logger.info(f'Joined dataframes into shape {df.shape}')

    keyword_positions: list[Keyword] = []

    for topic_id, topic_info in df_topic_infos.iterrows():
        logger.info(f'Processing topic {topic_id} ({topic_info["Topic (name)"]})')
        topic_label = topic_labels.get(int(str(topic_id)))
        if not topic_label:
            logger.warning(f'Topic ID "{topic_id}" not a defined topic label')
            continue
        if topic_label.column not in df.columns:
            logger.warning('No topic scores available in the provided DataFrame')
            continue

        topic_scores = df[topic_label.column]
        topic_threshold = np.quantile(topic_scores.dropna(), threshold_quantile)
        logger.info(f'Setting threshold to {topic_threshold:.2f}')

        mask = topic_scores > topic_threshold
        points = df.loc[mask, ['x', 'y']].values
        logger.info(f'Using {mask.sum():,}/{topic_scores.notna().sum():,}/{len(topic_scores):,} records for topic "{topic_label.name}"')

        logger.info('Fitting DBSCAN to find clusters...')
        db = DBSCAN(eps=eps, min_samples=min_cluster_size)
        clustering: np.ndarray = db.fit_predict(points)
        logger.info(f' > Found {len(set(clustering)):,} clusters for {len(points):,} points')

        logger.info('Place topic name for each cluster...')
        cluster: int
        for cluster in set(clustering):
            cluster_mask = clustering == cluster
            logger.debug(f' > Cluster {cluster} has {cluster_mask.sum():,} points')

            if cluster < 0:
                logger.debug('   -> ignore the -1 cluster which is the remainder which cannot be clustered')
                continue
            if cluster_mask.sum() < min_cluster_size:
                logger.debug(f'   -> ignore small cluster (< {min_cluster_size})')
                continue

            # Get the smallest shape that can be drawn around the point
            hull = ConvexHull(points[cluster_mask])

            # Get the center of that shape
            cx = np.mean(hull.points[hull.vertices, 0])
            cy = np.mean(hull.points[hull.vertices, 1])

            # Get a short form of the title (just the first term)
            title = topic_info['Topic (short)']
            keyword_positions.append(Keyword(x=float(cx), y=float(cy), keyword=title, level=cluster + 1))

    write_any_df(df=pd.DataFrame(keyword_positions), target=target)


# TODO: add function that's purely based on keywords from the data

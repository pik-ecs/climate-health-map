from pathlib import Path
from typing import TypedDict, Annotated

import pandas as pd
import typer
import numpy as np

from climate_health_map.shared import read_any_pd, get_logger, write_any_df
from climate_health_map.shared.text import text_from_table
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
    threshold_quantile: Annotated[float, typer.Option(help='Pick a topic score threshold that keeps X% of records per topic (0.0–1.0)')] = 0.95,
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


def text_based(
    source_scatter: Annotated[Path, typer.Option(help='')],
    source_items: Annotated[Path, typer.Option(help='')],
    target: Annotated[Path, typer.Option(help='')],
    n_clusters: Annotated[list[int], typer.Option(help='')],
    level_offset: Annotated[int, typer.Option(help='')] = 0,
    limit: Annotated[int | None, typer.Option(help='')] = None,
    loglevel: Annotated[str, typer.Option(help='Verbosity of logger')] = 'INFO',
) -> None:
    from sklearn.cluster import KMeans
    from climate_health_map.shared.text import text_utils
    from sklearn.feature_extraction.text import TfidfVectorizer

    logger = get_logger(loglevel=loglevel, logger_name='keywords', run_log_init=True)

    logger.info('Loading aggressive text util')
    _, process_text_aggressive, _ = text_utils()

    logger.info('Loading data')
    df_scatter = read_any_pd(source_scatter, index_column='item_id').join(
        text_from_table(read_any_pd(source_items, index_column='item_id')).to_frame(name='text'),
    )
    df_scatter = df_scatter[df_scatter['text'].notna()]
    if limit is not None:
        df_scatter = df_scatter.sample(frac=1).iloc[:limit]
    logger.info(f'Processing text for {df_scatter.shape}')
    df_scatter['text'] = df_scatter['text'].map(lambda txt: process_text_aggressive(txt, pos_filter={'ADV', 'DET', 'VERB', 'VBZ', 'RB'}))
    df_scatter['level_0'] = np.zeros(len(df_scatter))

    keyword_positions: list[Keyword] = []
    groups = [['level_0']]

    for level, n_cluster in enumerate(n_clusters, start=1):
        logger.info(f'Working on level {level} (n={n_cluster}) with {len(df_scatter.groupby(groups[level - 1]))} groups')
        df_scatter[f'level_{level}'] = 0
        for _, cluster in df_scatter.groupby(groups[level - 1]):
            if len(cluster) <= n_cluster:
                continue
            df_scatter.loc[cluster.index, f'level_{level}'] = KMeans(n_clusters=n_cluster).fit_predict(cluster[['x', 'y']])
        groups.append([f'level_{li}' for li in range(level + 1)])

    for group in groups[1:]:
        logger.info(f'Working on placements for group {group}')
        pseudo_docs = df_scatter.groupby(group)['text'].apply(lambda grp: ','.join(grp))

        logger.info('Vectorising...')
        vzr = TfidfVectorizer(ngram_range=(1, 3), max_df=0.5, min_df=1, stop_words=None)
        vecs = vzr.fit_transform(pseudo_docs)
        vocab = {v: k for k, v in vzr.vocabulary_.items()}
        logger.info('Placing keywords')
        for (_, grp), vector in zip(df_scatter.groupby(group), vecs, strict=True):
            centroid = grp[['x', 'y']].mean()
            token_idxs = np.asarray(vector.todense())[0].argsort()
            keyword_positions.append(Keyword(x=float(centroid['x']), y=float(centroid['y']), keyword=vocab[token_idxs[-1]], level=len(group) + level_offset))
        logger.info(f'Placed {len(keyword_positions):,} keywords so far')
    write_any_df(df=pd.DataFrame(keyword_positions), target=target)


if __name__ == '__main__':
    typer.run(text_based)

import logging

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN


# This function takes a set of points associated with a given topic label, and adds that label in the center of each cluster of points that are found
def cluster_label_points(
    title: str,
    points: np.ndarray,
    ax: plt.Axes,
    logger: logging.Logger,
    eps: float = 0.001,
    min_cluster: int = 50,
    clabel_size: int = 10,
) -> list[dict[str, int | float | str]]:
    # cluster the points and get the cluster numbers of the points
    db = DBSCAN(eps=eps, min_samples=min_cluster).fit(points)
    labels = db.labels_
    logger.debug(f'Found {len(labels):,} clusters for {len(points):,} points')
    texts: list[dict[str, int | float | str]] = []

    # For each cluster number
    for l in set(labels):
        # ignore the -1 cluster which is the remainder which cannot be clustered
        if l == -1:
            continue

        # get the indices of the points which have this cluster label
        ind = np.argwhere(labels == l).ravel()
        # The label points are those points with those indices
        lpoints = points[ind]
        # As long as the cluster is bigger than the min_cluster parameter, add a label
        if len(ind) > min_cluster:
            # Get the smallest shape that can be drawn around the point
            hull = ConvexHull(lpoints)
            # Get the center of that shape
            cx = float(np.mean(hull.points[hull.vertices, 0]))
            cy = float(np.mean(hull.points[hull.vertices, 1]))
            # Get a short form of the title (just the first term)
            title = title.split(',')[0].replace('{', '')
            # Add the label to the plot
            ax.annotate(
                title,
                (cx, cy),
                fontsize=clabel_size,
                ha='center',
                va='center',
                bbox={'facecolor': 'white', 'alpha': 0.4, 'pad': 0.2, 'boxstyle': 'round'},
            )
            texts.append({'x': cx, 'y': cy, 'keyword': title, 'level': 1})
        else:
            logger.debug(f'Skipping {l} because it is too small {len(ind)} < {min_cluster}')

    return texts


# def cluster_keywords(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame | None:
#     logger.info('Loading topicmodel info')
#     df_topicmodel = pd.read_csv(settings.TM_INFO).rename(columns={'id': 'topic_id'})
#
#     logger.info('Loading and placing topic info into embedding space')
#     logger.debug(list(df.columns))
#     min_cluster = 50
#     t_thresh = 0.95
#     eps = 0.5
#     kws = []
#
#     for i, row in df_topicmodel.iterrows():
#         scores = df[f'Topic|{row["Topic"]}']
#         thresh = np.quantile(scores.dropna(), t_thresh)
#         mask = scores > thresh
#         points = df.loc[mask, ['x', 'y']].values
#         logger.info(f'Topic {i} ({row["Topic"]}) score > {thresh}: {mask.sum():,} / {(~scores.isna()).sum():,} with {points.shape} points')
#
#         db = DBSCAN(eps=eps, min_samples=min_cluster).fit(points)
#         labels = db.labels_
#         logger.info(f' > Found {len(set(labels)):,} clusters for {len(points):,} points')
#
#         # For each cluster number
#         for cluster in set(labels):
#             cluster_mask = labels == cluster
#             logger.debug(f' > Cluster {cluster} has {cluster_mask.sum():,} points')
#
#             if cluster < 0:
#                 logger.debug('   -> ignore the -1 cluster which is the remainder which cannot be clustered')
#                 continue
#             if cluster_mask.sum() < min_cluster:
#                 logger.debug(f'   -> ignore small cluster (< {min_cluster})')
#                 continue
#
#             # Get the smallest shape that can be drawn around the point
#             hull = ConvexHull(points[cluster_mask])
#
#             # Get the center of that shape
#             cx = np.mean(hull.points[hull.vertices, 0])
#             cy = np.mean(hull.points[hull.vertices, 1])
#
#             # Get a short form of the title (just the first term)
#             title = row['short_title'].split(',')[0].replace('{', '')
#             kws.append({'x': cx, 'y': cy, 'keyword': title, 'level': cluster + 1})
#
#     if len(kws) > 0:
#         logger.warning('No keywords found')
#         return None
#
#     return pd.DataFrame(kws)

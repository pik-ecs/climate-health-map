import logging
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm

from climate_health_map.data import LABELS


def rescale_topic_scores(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    logger.info('Rescale topic scores')
    # normalise with scientifically set gamma of .25, which ensures that 99% of documents have at least
    # one topic > 0.5, and the average document has 3.75 topics > 0.5
    topic_cols = [label.column for label in LABELS['topics'].labels]
    scores = df[topic_cols].values
    norm = PowerNorm(0.28, vmin=0, vmax=np.max(scores, where=~np.isnan(scores), initial=-1))
    for col in topic_cols:
        df[col] = df[col].apply(lambda x: norm(x))
        logger.debug(f' > Topic "{col}": {(df[col] > 0.5).sum():,} records')
    return df

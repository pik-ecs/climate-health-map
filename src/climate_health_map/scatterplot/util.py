import logging
import numpy as np
import pandas as pd


def rescale_projection(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    logger.info('Rescale embeddings')
    maxs = df[['x', 'y']].max(axis=0)
    mins = df[['x', 'y']].min(axis=0)
    spans = np.sqrt(np.power(mins - maxs, 2))

    df['x'] = (((df['x'] - mins['x']) / spans['x']) * 2) - 1
    df['y'] = (((df['y'] - mins['y']) / spans['y']) * 2) - 1

    return df

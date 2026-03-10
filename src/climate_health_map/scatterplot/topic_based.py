from pathlib import Path
from typing import Annotated

import typer
import pickle
import pandas as pd

from climate_health_map.shared import read_any_pd, write_any_df, get_logger
from climate_health_map.topics import get_topic_labels


def reduce_topic_distribution(
    source: Annotated[Path, typer.Option(help='Path to file containing topic scores')],
    target: Annotated[Path, typer.Option(help='Path to output file')],
    model_path: Annotated[Path | None, typer.Option(help='Path pickled fitted UMAP')] = None,
    loglevel: Annotated[str, typer.Option(help='Verbosity of logger')] = 'INFO',
) -> None:
    logger = get_logger(loglevel=loglevel, logger_name='dimension-reduction', run_log_init=True)

    logger.info(f'Reading topic distribution from {source}')
    df_source = read_any_pd(source, index_column='item_id')
    logger.info(f'Found data with shape {df_source.shape}')
    topic_columns = [topic.column for topic in get_topic_labels().values()]
    logger.debug(f'Going to use the following columns as topics: {topic_columns}')

    if model_path is None or not model_path.exists():
        logger.info('No reducer found, fitting reducer...')
        import umap

        reducer = umap.UMAP(
            min_dist=0.8,
            n_neighbors=50,
            repulsion_strength=5,
            verbose=True,
        )
        embedding = reducer.fit_transform(df_source[topic_columns].fillna(0))
        logger.info('Storing reducer...')
        with open(model_path, 'wb') as fp_model:
            pickle.dump(reducer, fp_model)
    else:
        logger.info('Loading existing reducer...')
        with open(model_path, 'rb') as fp_model:
            reducer = pickle.load(fp_model)

        logger.info('Transforming data...')
        embedding = reducer.transform(df_source[topic_columns].fillna(0))

    logger.info('Constructing table...')
    df_scatter = pd.DataFrame(
        {
            'item_id': df_source.index,
            'x': embedding[:, 0],
            'y': embedding[:, 1],
        }
    )
    write_any_df(df_scatter, target)

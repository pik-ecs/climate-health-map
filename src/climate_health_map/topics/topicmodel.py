import logging
import re
from typing import Annotated

from joblib import load
from pathlib import Path

import typer
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from climate_health_map.shared import read_any_pd, write_any_df
from climate_health_map.shared.env import base_essentials
from climate_health_map.shared.text import snowball_stemmer, clean_text, text_from_table, ensure_offline_nltk


class TopicModel:
    VOCAB_FILE = 'vocab.csv'
    SCORES_FILE = 'term_topic_scores.csv'
    VECTORIZER_FILE = 'topic_vectorizer.pkl'
    DTM_FILE = 'topic_dtm.parquet'

    def __init__(self, model_path: Path, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger('topic-model')

        self.logger.info(f'Loading vocabulary from {model_path / self.VOCAB_FILE}')
        # The terms in our vocabulary list correspond with those found in the term-topic scores dataframe
        self.vocab = pd.read_csv(model_path / self.VOCAB_FILE)

        self.logger.info(f'Loading topic scores from {model_path / self.SCORES_FILE}')
        # Get the topic-term scores associated with the run_id from our database
        self.tts: pd.DataFrame = pd.read_csv(model_path / self.SCORES_FILE)

        self.logger.info(f'Loading tokenizer from {model_path / self.VECTORIZER_FILE}')
        self.vectorizer: TfidfVectorizer = load(model_path / self.VECTORIZER_FILE)
        self.vectorizer.tokenizer = snowball_stemmer()

        self.logger.info('Preparing DTM...')
        # Transform the dataframe into a matrix in which each row corresponds with a topic and each column corresponds with a term
        # Empty cells are filled with zeroes
        self.priors: np.ndarray = self.tts.pivot(index='topic_id', columns='term_id').fillna(0).values

        self.logger.info('Preparing NMF instance...')
        self.nmf = NMF(n_components=self.priors.shape[0], init='custom', max_iter=10)
        self.nmf.components_ = self.priors  # Use term-topic relationship learned from previous run
        self.nmf.n_components_ = self.priors.shape[0]  # Use number of topics learned from previous run

    def vectorize(self, texts: list[str], min_len: int = 10) -> tuple[np.ndarray, pd.Series]:
        self.logger.info('Cleaning and filtering texts...')
        mask = [len(re.findall(r'(\w+)', text)) > min_len for text in texts]
        texts = [clean_text(text) for incl, text in zip(mask, texts, strict=True)]

        self.logger.info('Vectorizing cleaned texts...')
        tfidf = self.vectorizer.transform(texts).todense()

        # Initialise an empty matrix with size determined by the number of docs and the number of terms with term-topic scores
        doc_topics = np.matrix(np.zeros((len(texts), self.priors.shape[1])))

        self.logger.info('Applying topic model...')
        # Fill this matrix with our data from the new documents
        # Terms not found in our pre-existing vocabulary list are discarded
        vocab_translate = {row['title']: i for i, row in self.vocab.iterrows()}
        for i, word in enumerate(self.vectorizer.get_feature_names_out()):
            try:
                term_idx = vocab_translate[word]
                doc_topics[:, term_idx] = tfidf[:, i]
            except KeyError:
                pass

        return np.asarray(doc_topics), pd.Series(mask)

    def apply(self, df: pd.DataFrame, batch_size: int = 5000) -> pd.DataFrame:
        chunks = []
        for pos in tqdm(range(0, len(df), batch_size), desc=f'Applying topic model to batches ({(batch_size,)} each, total {len(df):,})'):
            batch = df.iloc[pos : pos + batch_size]
            texts = text_from_table(batch)
            doc_topics, mask = self.vectorize(list(texts))

            dtm = pd.DataFrame(doc_topics)
            dtm.columns = self.tts.topic_id.unique()
            dtm.index = batch[mask].index
            chunks.append(dtm.reset_index().melt(id_vars='item_id', var_name='topic_id', value_name='score').query('score>0'))
        return pd.concat(chunks)


def topic_model(
    source: Annotated[Path, typer.Option(help='Path to file to apply topic model to')],
    target: Annotated[Path, typer.Option(help='Path to output file')],
    models_path: Annotated[Path, typer.Option(help='Path to offline model store')],
    config: Annotated[Path, typer.Option(help='Path to config.env')],
    batch_size: Annotated[int, typer.Option(help='')] = 5000,
    loglevel: Annotated[str, typer.Option(help='Verbosity of logger')] = 'INFO',
) -> None:
    logger, settings = base_essentials(config=config, loglevel=loglevel, logger_name='export', run_log_init=True)

    ensure_offline_nltk(target_dir=(models_path / 'nltk_data').resolve(), logger=logger)

    logger.info(f'Initialising topic model from {models_path.resolve()}')
    model = TopicModel((models_path / 'topic_model').resolve(), logger=logger)

    logger.info(f'Reading data from {source.resolve()}')
    df = read_any_pd(source)

    logger.info(f'Applying topic model to data of shape {df.shape}...')
    result = model.apply(df, batch_size=batch_size)

    logger.info(f'Writing data to {target.resolve()}')
    write_any_df(df=result, target=target, kwargs={'compression': 'gzip', 'existing_data_behavior': 'delete_matching'})

    logger.info('All done.')


if __name__ == '__main__':
    typer.run(topic_model)

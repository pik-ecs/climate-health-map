import logging
import re
from typing import Annotated, Optional

from pathlib import Path

import typer
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from climate_health_map.shared import read_any_pd, write_any_df
from climate_health_map.shared.env import base_essentials
from climate_health_map.shared.text import SnowballStemmerClass, clean_text, text_from_table, ensure_offline_nltk


class TopicModel:
    VOCAB_FILE = 'vocab.csv'
    SCORES_FILE = 'term_topic_scores.csv'
    TOPIC_INFOS_FILE = 'topic_infos.csv'
    FALLBACK_PATH = Path(__file__).parent.resolve() / 'models'

    TOPIC_INFOS: Optional[pd.DataFrame] = None

    def __init__(self, model_path: Path | None = None, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger('topic-model')
        model_path = model_path or self.FALLBACK_PATH

        self.logger.info(f'Loading vocabulary from {model_path / self.VOCAB_FILE}')
        # The terms in our vocabulary list correspond with those found in the term-topic scores dataframe
        # Note: topic model and vectorizer vocab have different indexing!!
        self.vocabulary = pd.read_csv(model_path / self.VOCAB_FILE)
        self.vocabulary_tm = {row['token']: row['token_id_tm'] for _, row in self.vocabulary[self.vocabulary['token_id_tm'].notna()].iterrows()}
        self.vectorizer = TfidfVectorizer(
            max_df=0.95,
            min_df=10,
            ngram_range=(1, 1),
            binary=False,
            norm='l2',
            sublinear_tf=False,
            smooth_idf=True,
            use_idf=True,
            tokenizer=SnowballStemmerClass(),
            vocabulary={row['token']: row['token_id_vec'] for _, row in self.vocabulary[self.vocabulary['token_id_vec'].notna()].iterrows()},
        )
        self.vectorizer.idf_ = self.vocabulary[self.vocabulary['idf'].notna()].sort_values('token_id_vec')['idf'].tolist()

        self.logger.info(f'Loading topic scores from {model_path / self.SCORES_FILE}')
        # Get the topic-term scores associated with the run_id from our database
        self.tts: pd.DataFrame = pd.read_csv(model_path / self.SCORES_FILE)

        self.logger.info('Preparing DTM...')
        # Transform the dataframe into a matrix in which each row corresponds with a topic and each column corresponds with a term
        # Empty cells are filled with zeroes
        self.priors: np.ndarray = self.tts.pivot(index='topic_id', columns='term_id').fillna(0).values

        self.logger.info('Preparing NMF instance...')
        self.nmf = NMF(n_components=self.priors.shape[0], init='custom', max_iter=10)
        self.nmf.components_ = self.priors  # Use term-topic relationship learned from previous run
        self.nmf.n_components_ = self.priors.shape[0]  # Use number of topics learned from previous run

    @classmethod
    def load_topic_infos(cls, model_path: Path | None = None) -> pd.DataFrame:
        if cls.TOPIC_INFOS is None:
            cls.TOPIC_INFOS = (
                pd.read_csv((model_path or cls.FALLBACK_PATH) / 'topic_infos.csv', dtype=str, keep_default_na=False)
                .map(str.strip, na_action='ignore')
                .set_index('Topic ID')
            )
        return cls.TOPIC_INFOS

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
        for token_id_vec, token in enumerate(self.vectorizer.get_feature_names_out()):
            if token in self.vocabulary_tm:
                token_id_tm = self.vocabulary_tm[token]
                doc_topics[:, token_id_tm] = tfidf[:, token_id_vec]
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
    write_any_df(df=result, target=target, compression='gzip', existing_data_behavior='delete_matching')

    logger.info('All done.')


if __name__ == '__main__':
    typer.run(topic_model)

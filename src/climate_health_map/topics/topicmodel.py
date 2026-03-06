import logging
import re
from typing import Annotated, Optional

from pathlib import Path

import typer
import numpy as np
import pandas as pd
from tqdm import tqdm

from climate_health_map import get_logger
from climate_health_map.data import LABELS, Topic

from climate_health_map.shared import read_any_pd, write_any_df
from climate_health_map.shared.text import SnowballStemmerClass, clean_text, text_from_table, ensure_offline_nltk


class TopicModel:
    VOCAB_FILE = 'vocabulary.csv'
    SCORES_FILE = 'term_topic_scores.csv'
    TOPIC_INFOS_FILE = 'topic_infos.csv'
    ID2COL = {label.topic_id: label.column for group in LABELS.values() for label in group.labels if type(label) == Topic}
    FALLBACK_PATH = Path(__file__).parent.resolve() / '_model'

    TOPIC_INFOS: Optional[pd.DataFrame] = None

    def __init__(self, model_path: Path | None = None, logger: logging.Logger | None = None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import NMF

        self.logger = logger or logging.getLogger('topic-model')
        model_path = model_path or self.FALLBACK_PATH

        self.logger.info(f'Loading vocabulary from {model_path / self.VOCAB_FILE}')
        # The terms in our vocabulary list correspond with those found in the term-topic scores dataframe
        # Note: topic model and vectorizer vocab have different indexing!!
        self.vocabulary = (
            pd
            .read_csv(model_path / self.VOCAB_FILE, keep_default_na=False)
            .replace({'': np.nan})
            .astype({'token_id_vec': 'Int32', 'token_id_tm': 'Int32'})
        )

        vocab = self.vocabulary[self.vocabulary['token_id_vec'].notna()].fillna(0).groupby('token_id_vec').first().reset_index()
        self.vocabulary_tm = {row['token']: row['token_id_vec'] for _, row in vocab.iterrows()}
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
            vocabulary=self.vocabulary_tm,
        )
        self.vectorizer.idf_ = vocab.sort_values('token_id_vec')['idf'].to_numpy().astype('float64')

        # Mapping to address the vectorizer token IDs are different to the topic model token IDs
        self.vec2tm_ids = {row['token_id_vec']: row['token_id_tm'] for _, row in self.vocabulary[self.vocabulary['token_id_vec'].notna() & self.vocabulary['token_id_tm'].notna()].iterrows()}
        # Topic model token IDs are not continuous, hence we need a mapping to offset gaps
        self.tm_id2idx = {tok_id: tok_idx for tok_idx, tok_id in enumerate(self.vocabulary[self.vocabulary['token_id_tm'].notna()].sort_values('token_id_tm')['token_id_tm'])}

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
        self.logger.debug('Cleaning and filtering texts...')
        mask = [len(re.findall(r'(\w+)', text)) > min_len for text in texts]
        texts = [clean_text(text) for incl, text in zip(mask, texts, strict=True)]

        self.logger.debug('Vectorizing cleaned texts...')
        tfidf = self.vectorizer.transform(texts).todense()

        # Initialise an empty matrix with size determined by the number of docs and the number of terms with term-topic scores
        doc_topics = np.matrix(np.zeros((len(texts), self.priors.shape[1])))

        self.logger.debug('Translating into topic model indexes...')
        # Fill this matrix with our data from the new documents
        # Terms not found in our pre-existing vocabulary list are discarded
        for token_id_vec, token_id_tm in self.vec2tm_ids.items():
            doc_topics[:, self.tm_id2idx[token_id_tm]] = tfidf[:, token_id_vec]

        return np.asarray(doc_topics), pd.Series(mask)

    def apply(self, df: pd.DataFrame, batch_size: int = 5000, clear_below:float=0.001) -> pd.DataFrame:
        chunks = []
        for pos in tqdm(range(0, len(df), batch_size), desc=f'Applying topic model to batches ({batch_size:,} each, total {len(df):,})'):
            # bite off a chunk from the big dataframe
            batch = df.iloc[pos: pos + batch_size]
            index = batch.index.set_names('item_id')

            # Clean and vectorize text
            texts = text_from_table(batch)
            vectors, mask = self.vectorize(texts.tolist())

            # Apply topic model
            topic_scores = self.nmf.transform(vectors[mask])

            # Prepare pretty return format
            dtm = pd.DataFrame(topic_scores, index=index[mask], columns=self.tts.topic_id.unique()).rename(columns=self.ID2COL)

            # Append to our result set; clear all low sores first
            chunks.append(dtm.mask(dtm < clear_below, pd.NA))

        return pd.concat(chunks)


def topic_model(
    source: Annotated[Path, typer.Option(help='Path to file to apply topic model to')],
    target: Annotated[Path, typer.Option(help='Path to output file')],
    offline_models_path: Annotated[Path, typer.Option(help='Path to offline model store')],
    topic_models_path: Annotated[Path, typer.Option(help='Path to offline model store')] | None = None,
    batch_size: Annotated[int, typer.Option(help='')] = 5000,
    loglevel: Annotated[str, typer.Option(help='Verbosity of logger')] = 'INFO',
) -> None:
    logger = get_logger(loglevel=loglevel, logger_name='export', run_log_init=True)

    ensure_offline_nltk(target_dir=(offline_models_path / 'nltk_data').resolve(), logger=logger)

    topic_models_path = topic_models_path or TopicModel.FALLBACK_PATH
    logger.info(f'Initialising topic model from {topic_models_path.resolve()}')
    model = TopicModel(topic_models_path, logger=logger)

    logger.info(f'Reading data from {source.resolve()}')
    df = read_any_pd(source, index_column='item_id')

    logger.info(f'Applying topic model to data of shape {df.shape}...')
    result = model.apply(df, batch_size=batch_size)

    logger.info(f'Writing data to {target.resolve()}')
    write_any_df(df=result.reset_index(), target=target)#, compression='gzip', existing_data_behavior='delete_matching')

    logger.info('All done.')


if __name__ == '__main__':
    typer.run(topic_model)

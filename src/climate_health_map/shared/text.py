import logging
import re
import string
from pathlib import Path
from typing import Generator, Callable

import numpy as np
import pandas as pd


def chunked_text(text: str, chunk_size: int = 500, overlap: int = 15) -> Generator[str, None, None]:
    """
    Split the text into tokens and then into overlapping chunks.

    :param text: input text to be chunked
    :param chunk_size: number of tokens per chunk should contain (512 is mordecai maximum, give it some headroom for slightly differing tokenisation though!)
    :param overlap: number of tokens that overlap between consecutive chunks

    Returns:
    list: A list of chunks, where each chunk is a list of tokens.
    """
    tokens = text.split()
    for pos_begin in range(0, len(tokens), chunk_size - overlap):
        yield ' '.join(tokens[pos_begin : pos_begin + chunk_size + overlap])


def text_utils() -> tuple[Callable[[str, str], str], Callable[[str, set[str]], str], Callable[[str], str]]:
    from nltk import WordNetLemmatizer, pos_tag, wordpunct_tokenize, sent_tokenize, word_tokenize
    from nltk.corpus import stopwords as sw
    from nltk.corpus import wordnet as wn

    lemmatizer = WordNetLemmatizer()
    stopwords = sw.words('english')
    NOALPH = re.compile(r'[^A-Za-z]+')

    def lemmatize(token: str, tag: str) -> str:
        tag = {'N': wn.NOUN, 'V': wn.VERB, 'R': wn.ADV, 'J': wn.ADJ}.get(tag[0], wn.NOUN)
        return lemmatizer.lemmatize(token, tag)  # type:ignore[no-any-return]

    def process_text_aggressive(text: str, pos_filter: set[str] | None = None, min_len: int = 3) -> str:
        return ' '.join(
            [
                lemmatize(tok, tag)
                for sentence in sent_tokenize(text)
                for tok, tag in pos_tag(wordpunct_tokenize(sentence))
                if tok not in stopwords and len(NOALPH.sub('', tok)) >= min_len and (pos_filter is None or tag not in pos_filter)
            ],
        )

    def process_text_light(text: str) -> str:
        return ' '.join([tok for tok in word_tokenize(text) if tok not in stopwords])

    return lemmatize, process_text_aggressive, process_text_light


def tokenize(text: str) -> list[str]:
    from nltk import word_tokenize

    translation_table = {ord(c): None for c in string.punctuation + string.digits}
    # Remove punctuation and digits
    tokens = word_tokenize(text.translate(translation_table))
    # Remove tokens with more than 2 and less than 100 characters
    return [tok for tok in tokens if 2 < len(tok) < 100]


class SnowballStemmerClass(object):
    def __init__(self) -> None:
        from nltk.stem import SnowballStemmer

        self.stemmer = SnowballStemmer('english')

    def __call__(self, doc: str) -> list[str]:
        return [self.stemmer.stem(t) for t in tokenize(doc)]


SUB = [
    # drop URLs
    re.compile(r'https?://[^ ]+'),
    # drop all after "(c) Copyright" or "Copyright (c)"
    re.compile(r'(\(C\)|\(c\)|©)? ?Copyright\.? ?(\(C\)|\(c\)|©)?.+'),
    # drop all after "Published by Elsevier" etc
    re.compile(r'Published by [A-Z]\w+.+'),
    # drop all after "(c) 2024" etc
    re.compile(r'(\(C\)|\(c\)|©) ?[1-2][0-9]{3}.+'),
]


def clean_text(text: str, extra: list[re.Pattern[str]] | None = None) -> str:
    for sub in SUB + (extra or []):
        text = sub.sub('', text)
    return text


def ensure_offline_nltk(logger: logging.Logger, target_dir: Path | None = None) -> None:
    logger.debug('Loading NLTK data...')
    from nltk import download
    import ssl

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context  # type:ignore[assignment]
    download('stopwords', download_dir=str(target_dir))
    download('punkt', download_dir=str(target_dir))
    download('punkt_tab', download_dir=str(target_dir))
    download('wordnet', download_dir=str(target_dir))
    download('averaged_perceptron_tagger_eng', download_dir=str(target_dir))


def text_from_table(df: pd.DataFrame) -> pd.Series:
    if 'title' in df.columns and 'text' in df.columns and df.iloc[0]['title'].str.startswith(f'{df.iloc[0]["title"]}.'):
        return df['text']
    # We have title and abstract, but no dedicated text column
    if 'title' in df.columns and 'abstract' in df.columns and 'text' not in df.columns:
        return df.replace({np.nan: '', None: ''}).apply(lambda row: f'{row["title"]}. {row["abstract"]}', axis=1)
    # We have title and abstract (called `text`), but no dedicated text column
    if 'title' in df.columns and 'abstract' not in df.columns and 'text' in df.columns:
        return df.replace({np.nan: '', None: ''}).apply(lambda row: f'{row["title"]}. {row["text"]}', axis=1)
    # We already seem to have a prepared text column, use that one
    if 'title' in df.columns and 'abstract' in df.columns and 'text' in df.columns:
        return df['text']
    raise ValueError('No title or abstract in the dataframe')

"""
Previously, the topic model was spread across three files.
Pickling the vectorizer throws a bunch of warnings, might as well merge the topicmodel vocab and vectorizer vocab and instantiate fresh.
This is code to produce the merged file

stopwords are implicit
max_df=0.95, min_df=10, ngram_range=(1, 1), binary=False, norm='l2',  sublinear_tf=False, smooth_idf=True, use_idf=True

"""

import pandas as pd
from joblib import load

vec = load('data/models/topicmodel/topic_vectorizer.pkl')
df = pd.read_csv('src/climate_health_map/topics/model/vocab.csv').rename(columns={'title': 'token', 'id': 'token_id_tm'})
df_vec = pd.DataFrame([{'token': k, 'token_id_vec': v, 'idf': vec.idf_[v]} for k, v in vec.vocabulary_.items()])
df_merged = df.merge(df_vec, how='outer').astype({'token_id_tm': 'Int32', 'token_id_vec': 'Int32', 'token': 'str', 'idf': 'Float32'}).set_index('token')

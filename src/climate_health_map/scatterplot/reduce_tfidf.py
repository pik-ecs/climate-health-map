# import numpy as np
# from openTSNE import TSNE
# from sklearn.cluster import HDBSCAN
# from sklearn.feature_extraction.text import TfidfVectorizer
# import re
#
# df_ = df[mask].groupby('item_id').first()
#
# NON_ALPHA = re.compile(r'\\W+')
#
# texts = [NON_ALPHA.sub(re.escape(f'{row['title']} {row['abstract']}'), ' ') for _, row in df_.iterrows()]
# vectoriser = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), min_df=0.02, max_df=0.8)
# vectors = vectoriser.fit_transform(texts)
# print(vectors.shape)
#
# embedding_standard = TSNE(
#     perplexity=30,
#     dof=1,
#     initialization='pca',
#     metric='euclidean',  # cosine,
#     # neighbors='pynndescent',
#     n_jobs=12,
#     random_state=43,
#     verbose=True,
# ).fit(np.asarray(vectors.todense()))
#
# cmodel = HDBSCAN(n_jobs=12, min_cluster_size=500, store_centers='centroid', max_cluster_size=5000)
# clusters = cmodel.fit_predict(embedding_standard)
# print(np.unique_counts(clusters))
# print(len(cmodel.centroids_))
# print(clusters)
#
# grouping = 'technology'
# cols = [li.column for li in LABELS[grouping] if li.column in df.columns]
# df_tsne = pd.DataFrame(
#     {
#         'text': texts,
#         'cluster': clusters,
#         'x': embedding_standard[:, 0],
#         'y': embedding_standard[:, 1],
#         'Method': df_[cols].rename(columns=LABEL_TRANSLATIONS['technology']).idxmax(axis=1),
#     } | {
#         label.name: df_[label.column]
#         for label in LABELS[grouping]
#         if label.column in df.columns
#     },
# )
# print(df_tsne.shape)
# df_tsne.head()
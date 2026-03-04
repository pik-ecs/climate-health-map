# import toml
# import datetime
# import pandas as pd
#
# from utils import get_settings
# from lithub.lithub_types import DatasetInfoFull, SchemeLabel, SchemeGroup
#
# settings = get_settings()
#
# BASE_COLOURS = {  # HSL
#     'Intervention option': [266.25, 38.55, 67.45],  # A88CCC
#     'Exposure': [340.93, 98.17, 78.63],  # FE93B5
#     'Mediating pathways': [45.56, 76.06, 72.16],  # EED482
#     'Other': [127.88, 90.83, 78.63],  # 97FAA4
#     'Health impact': [194.86, 70.32, 69.61],  # 7BCDE8
# }
#
# # id,title,top_words,Topic ID,Aggregated meta-topic,Aggregated topic,Topic,color,short_title
# df = pd.read_csv(settings.TM_INFO)
# c = [
#     [None, r['Aggregated meta-topic'], r['Aggregated topic'], r['Topic']]
#     for _, r in df.iterrows()
# ]
#
# labels = {
#     'rel|1': SchemeLabel(key='rel|1', name='Relevant', value=1, colour=(16.91, 87.13, 66.47)),
#
#     'cat|0': SchemeLabel(key='cat|0', name='Mitigation', value=0, colour=(350.93, 63.24, 26.67)),
#     'cat|1': SchemeLabel(key='cat|1', name='Adaptation', value=1, colour=(290.53, 30.65, 48.63)),
#     'cat|2': SchemeLabel(key='cat|2', name='Impacts', value=2, colour=(143.33, 17.31, 59.22)),
#
#     'cont|0': SchemeLabel(key='cont|0', name='Africa', value=0, colour=(25.18, 97.97, 61.37)),
#     'cont|1': SchemeLabel(key='cont|1', name='Asia', value=1, colour=(338.33, 100.0, 92.94)),
#     'cont|2': SchemeLabel(key='cont|2', name='Europe', value=2, colour=(184.32, 69.06, 35.49)),
#     'cont|3': SchemeLabel(key='cont|3', name='Latin America', value=3, colour=(0.0, 0.0, 38.82)),
#     'cont|4': SchemeLabel(key='cont|4', name='Northern America', value=4, colour=(202.46, 94.48, 35.49)),
#     'cont|5': SchemeLabel(key='cont|5', name='Oceania', value=5, colour=(16.91, 87.13, 66.47)),
#     'cont|6': SchemeLabel(key='cont|6', name='SIDS', value=6, colour=(16.91, 0.0, 66.47))
# }
#
# groups = {
#     'rel': SchemeGroup(name='Relevance', key='rel', type='bool', labels=['rel|1']),
#     'cat': SchemeGroup(name='Category', key='cat', type='single', labels=['cat|0', 'cat|1', 'cat|2']),
#     'cont': SchemeGroup(name='Continent', key='cont', type='single',
#                         labels=['cont|0', 'cont|1', 'cont|2', 'cont|3', 'cont|4', 'cont|5', 'cont|6']),
#     't3': SchemeGroup(name='Meta-topic', key='t3', type='multi', subgroups=[]),
#     't2': SchemeGroup(name='Aggregated topic', key='t2', type='multi', subgroups=[])
# }
#
# topic_renaming = {}
#
# for i, (m_topic, m_group) in enumerate(df.groupby('Aggregated meta-topic')):
#     meta = SchemeGroup(name=m_topic, key=f't2-{i}', type='multi', subgroups=[], colour=BASE_COLOURS[m_topic])
#     groups[meta.key] = meta
#     groups['t3'].subgroups.append(meta.key)
#
#     for j, (ag_topic, ag_group) in enumerate(m_group.groupby('Aggregated topic')):
#         agg = SchemeGroup(name=ag_topic, key=f't1-{i}-{j}', type='multi', labels=[],
#                     colour=(meta.colour[0], meta.colour[1], meta.colour[2] + 5))
#         groups[agg.key] = agg
#         groups[meta.key].subgroups.append(agg.key)
#         groups['t2'].subgroups.append(agg.key)
#         for k, row in ag_group.reset_index().iterrows():
#             lab = SchemeLabel(name=row['Topic'], key=f't0-{i}-{j}|{k}', value=k,
#                               colour=(agg.colour[0], agg.colour[1], agg.colour[2] + 5))
#
#             labels[lab.key] = lab
#             groups[agg.key].labels.append(lab.key)
#             topic_renaming[row['Topic']] = lab.key
#
# info = DatasetInfoFull(
#     name='Climate and Health Map',
#     teaser='Explore scientific papers on climate and health by subject and place of study.',
#     authors=[
#         'Max Callaghan'
#     ],
#     contact=['max.callaghan@pik-potsdam.de'],
#     start_year=1990,
#     end_year=datetime.date.today().year + 1,
#     default_colour='t3',
#     created_date=datetime.date(year=2020, month=7, day=13),
#     last_update=datetime.date.today(),
#     db_filename='documents.sqlite',
#     arrow_filename='slim.arrow',
#     keywords_filename='keywords.arrow',
#     slim_geo_filename='geocodes.minimal.arrow',
#     full_geo_filename='geocodes.full.arrow',
#     # https://pixabay.com/photos/forest-trees-autumn-nature-season-6765636/
#     # @fietzfotos, pixabay
#     figure='teaser.jpg',
#     groups=groups,
#     labels=labels,
#     hidden=False,
# )
#
# if __name__ == '__main__':
#     print('writing info')
#     with open(settings.LITHUB_BASE / 'info.toml', 'w') as f:
#         # f.write(info.model_dump_json(indent=2))
#         toml.dump(info.dict(), f)
# #

from datetime import date

import pandas as pd

from climate_health_map.data.labels import LABELS
from .types import DatasetInfoFull, SchemeLabel, SchemeGroup

exclude_columns = {
    'topic-4-0|8',
    'topic-agg-4|0',
    'topic-agg-agg|4',
}
export_groups = [
    'rel_major',
    'cat',
    'type',
    'cont',
    'rel_impacts',
    'driver',
    'event',
    'health',
    'expose',
    'attr',
    # 'gender_outcome',
    # 'notes_major',
    # 'notes_impacts',
    'sector',
    'topic',
    # 'topic-agg',
    # 'topic-agg-agg',
    'Location_Group (Lancet 2026)',
    'Location_Group (WHO 2026)',
    'Location_Group (HDI 2026)',
    'Location_Region (IPCC AR6, 6)',
    'Location_Region (IPCC AR6, 10)',
    'Location_Region (WorldBank 2026)',
    'Location_Income group (WorldBank 2026)',
    'Location_Lending category (WorldBank 2026)',
    'Location_Continent (Name)',
    'Affiliation_Group (Lancet 2026)',
    'Affiliation_Group (WHO 2026)',
    'Affiliation_Group (HDI 2026)',
    'Affiliation_Region (IPCC AR6, 6)',
    'Affiliation_Region (IPCC AR6, 10)',
    'Affiliation_Region (WorldBank 2026)',
    'Affiliation_Income group (WorldBank 2026)',
    'Affiliation_Lending category (WorldBank 2026)',
    'Affiliation_Continent (Name)',
]
label_groups = {}
for col in export_groups:
    group = LABELS[col]
    if group.type not in {'single', 'multi', 'bool'}:
        continue
    label_groups[col] = SchemeGroup(
        name=group.name,
        key=group.key,
        type=group.type,
        labels=[label.column for label in group.labels if label.column not in exclude_columns],
    )
for col in {
    'topic-agg',
    'topic-agg-agg',
}:
    for group in LABELS[col].labels:
        label_groups[group.column] = SchemeGroup(
            name=group.name,
            key=group.column,
            type='multi',
            subgroups=[topic_key for topic_key in group.topics if topic_key not in exclude_columns],
        )

info = DatasetInfoFull(
    name='Climate and Health Map',
    teaser='Explore scientific papers by subject and place of study',
    authors=['Max Callaghan'],
    contributors=['Tim Repke'],
    contact=['maxcal@pik-potsdam.de'],
    created_date=date(year=2020, month=7, day=13),
    last_update=date.today(),
    figure='teaser.jpg',  # @fietzfotos, pixabay: https://pixabay.com/photos/forest-trees-autumn-nature-season-6765636/
    hidden=False,
    db_filename='documents.sqlite',
    arrow_filename='slim.arrow',
    keywords_filename='keywords.arrow',
    slim_geo_filename='geocodes.minimal.arrow',
    full_geo_filename='geocodes.full.arrow',
    start_year=1990,
    end_year=2025,
    default_colour='cat',
    labels={
        label.column: SchemeLabel(
            key=label.column,
            name=label.name,
            value=label.value,
            colour=(label.colour[0], label.colour[1] * 100, label.colour[2] * 100),
            desc=label.desc,
        )
        for group in LABELS.values()
        for label in group.labels
    },
    groups=label_groups,
    # DatasetInfoWeb:
    # key='healthmap',
    # total=0,
    # columns=set(''),
    # label_columns=set(''),
    # document_columns=set(''),
)


def filter_labels(df: pd.DataFrame, info_: DatasetInfoFull) -> DatasetInfoFull:
    # TODO: subgroups
    info_.labels = {k: v for k, v in info_.labels.items() if k in df.columns}
    keys = list(info_.groups.keys())
    for key in keys:
        if key not in info_.groups:
            continue
        if info_.groups[key].labels is not None:
            info_.groups[key].labels = [col for col in info_.groups[key].labels if col in df.columns]
            if len(info_.groups[key].labels) == 0:
                del info_.groups[key]
        elif info_.groups[key].subgroups is not None:
            info_.groups[key].subgroups = [col for col in info_.groups[key].subgroups if col in df.columns]
            if len(info_.groups[key].subgroups) == 0:
                del info_.groups[key]
    return info_


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

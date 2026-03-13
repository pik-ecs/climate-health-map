from datetime import date

import pandas as pd

from climate_health_map.data.labels import LABELS
from .types import DatasetInfoFull, SchemeLabel, SchemeGroup

exclude_columns = {
    'topic-4-0|8',
    'topic-4-0|18',
    'topic-4-0|38',
    'topic-agg-4|0',
    'topic-agg-agg|4',
    'driver|0',  # CO2rise
    'driver|2',  # Seasonal Change
    'driver|4',  # Sea-level rise
    'health|10',  # Metabolic Disorders
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
    # 'keywords', # -> moved into methods
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


def get_info():
    label_groups = {}
    for col in export_groups:
        group = LABELS[col]
        if group.type not in {'single', 'multi', 'bool'}:
            continue
        label_groups[col] = SchemeGroup(
            name=group.name,
            key=group.key,
            type=group.type,
            colour=(group.colour[0] * 360, group.colour[1] * 100, group.colour[2] * 100),
            labels=[label.column for label in group.labels if label.column not in exclude_columns],
        )

    for _gi, group in enumerate(LABELS['topic-agg'].labels):
        label_groups[group.column] = SchemeGroup(
            name=group.name,
            key=group.column,
            type='multi',
            colour=(group.colour[0] * 360, group.colour[1] * 100, group.colour[2] * 100),
            labels=[topic_key for topic_key in group.topics if topic_key not in exclude_columns],
        )
    # Add keyword filters to the Methods topic group
    label_groups['topic-agg-4|0'].labels += [
        'keywords|0',  # Evidence synthesis
        'keywords|1',  # Evaluation method
    ]

    label_groups['topic-agg-agg'] = SchemeGroup(name='Meta-topic', key='topic-agg-agg', type='multi', colour=(180, 90, 90), subgroups=[])
    for group in LABELS['topic-agg-agg'].labels:
        label_groups[group.column] = SchemeGroup(
            name=group.name,
            key=group.column,
            type='multi',
            colour=(group.colour[0] * 360, group.colour[1] * 100, group.colour[2] * 100),
            subgroups=[topic_key for topic_key in group.topics_agg if topic_key not in exclude_columns],
        )
        label_groups['topic-agg-agg'].subgroups.append(group.column)

    return DatasetInfoFull(
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
                colour=(label.colour[0] * 360, label.colour[1] * 100, label.colour[2] * 100),
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


def filter_labels(df: pd.DataFrame, info: DatasetInfoFull) -> DatasetInfoFull:
    # 1. Filter the top-level labels first
    info.labels = {k: v for k, v in info.labels.items() if k in df.columns}

    # Cache to store whether a group ID is valid (True/False)
    # This prevents redundant work and handles nested dependencies.
    memory = {}

    def check_group_validity(key_: str):
        # If we've already decided if this group is valid, return the result
        if key_ in memory:
            return memory[key_]

        group_ = info.groups.get(key_)
        if not group_:
            memory[key_] = False
            return False

        # Case 1: Group contains direct column labels
        if group_.labels is not None:
            group_.labels = [col for col in group_.labels if col in df.columns]
            is_valid = len(group_.labels) > 0

        # Case 2: Group contains references to other groups (subgroups)
        elif group_.subgroups is not None:
            # We recursively check each subgroup.
            # A subgroup is kept only if check_group_validity returns True.
            group_.subgroups = [s for s in group_.subgroups if check_group_validity(s)]
            is_valid = len(group_.subgroups) > 0

        else:
            is_valid = False

        memory[key_] = is_valid
        return is_valid

    # 2. Iterate through all group keys and trigger the recursive check
    all_keys = list(info.groups.keys())
    for key in all_keys:
        if not check_group_validity(key):
            # If the group (or its nested children) ended up empty, delete it
            if key in info.groups:
                del info.groups[key]

    return info

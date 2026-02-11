import json
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from sqlalchemy import text

from climate_health_map.shared import essentials
from climate_health_map.data import LABELS, LABELS_IMPACTS, LABELS_MAJOR, Group
from .scopes import MAJOR_SCOPES, IMPACT_SCOPES


def label_select(group: Group) -> list[str]:
    if group.type == 'bool':
        return [
            f'''LEAST(1, COUNT(1) FILTER (WHERE ann.key = '{group.nacsos_key}' AND ann.value_bool = true)) as "{group.key}|1"''',
            f'''LEAST(1, COUNT(1) FILTER (WHERE ann.key = '{group.nacsos_key}' AND ann.value_bool = false)) as "{group.key}|0"''',
        ]

    if group.type == 'multi':
        return [
            f'''
            LEAST(1, COUNT(1)
              FILTER (
                WHERE ann.key = '{group.nacsos_key}' AND ann.multi_int @> array [{label.value}]
              )
            ) as "{group.key}|{label.value}"
            '''
            for label in group.labels
        ]

    raise AssertionError(f'Unknown label kind: {group.type}')


def dump(
    config: Annotated[Path, typer.Option(help='Path to config file')],
    target: Annotated[Path, typer.Option(help='Target file')] = 'data/exports/annotations.csv',
    overwrite: Annotated[bool, typer.Option(help='Overwrite target file if it already exists')] = False,
    loglevel: str = 'INFO',
):
    logger, settings, db_engine = essentials(config=config, loglevel=loglevel, logger_name='abstract-transfer', run_log_init=True)

    if target.exists() and not overwrite:
        logger.warning(f'Set --overwrite to overwrite target file: {target}')
        raise FileExistsError(f'Target file {target.resolve()} already exists')
    if target.exists() and overwrite:
        logger.warning(f'Target file already exists, going to overwrite {target.resolve()}')

    selects_major = [sel for key in LABELS_MAJOR for sel in label_select(LABELS[key])]
    selects_impacts = [sel for key in LABELS_IMPACTS for sel in label_select(LABELS[key])]

    columns_major = [f'bm."{label.column}"' for key in LABELS_MAJOR for label in LABELS[key].labels]
    columns_impacts = [f'bi."{label.column}"' for key in LABELS_IMPACTS for label in LABELS[key].labels]

    logger.info('Constructing query...')
    query = text(
        f"""
       WITH
           data_impacts_human as (
              SELECT ass.item_id                                                                                        as label_item_id,
                     --COUNT(1) as                                                             cnt,
                     LEAST(1, COUNT(1) FILTER (WHERE ann.item_id IS NOT NULL))                                          as anno,
                     LEAST(1, COUNT(1) FILTER (WHERE ann.item_id IS NULL))                                              as no_anno,
                     ARRAY_AGG(DISTINCT ann.value_str) FILTER ( WHERE  ann.value_str IS NOT NULL AND ann.key = 'notes') as notes_impacts,
                     ARRAY_AGG(DISTINCT ass.assignment_scope_id::text)                                                  as scope_ids,
                     {',\n'.join(selects_impacts)}
                     FROM assignment ass
              JOIN annotation ann ON ass.assignment_id = ann.assignment_id
              WHERE ass.assignment_scope_id::text = ANY(:impact_scopes)
              GROUP BY ass.item_id),
           data_impacts_robot as (
              SELECT ann.item_id                                                                                        as label_item_id,
                     --COUNT(1) as                                                             cnt,
                     LEAST(1, COUNT(1) FILTER (WHERE ann.item_id IS NOT NULL))                                          as anno,
                     LEAST(1, COUNT(1) FILTER (WHERE ann.item_id IS NULL))                                              as no_anno,
                     ARRAY_AGG(DISTINCT ann.value_str) FILTER ( WHERE  ann.value_str IS NOT NULL AND ann.key = 'notes') as notes_impacts,
                     ARRAY_AGG(DISTINCT ann.bot_annotation_metadata_id::text)                                           as scope_ids,
                     {',\n'.join(selects_impacts)}
              FROM bot_annotation ann
              WHERE ann.bot_annotation_metadata_id::text = ANY(:impact_scopes)
              GROUP BY ann.item_id),
           data_impacts as (
              SELECT r.*
              FROM data_impacts_robot r
                LEFT JOIN data_impacts_human h ON r.label_item_id = h.label_item_id
              UNION ALL
              SELECT h.*
              FROM data_impacts_human h
                LEFT JOIN data_impacts_robot r ON h.label_item_id = r.label_item_id
                WHERE r.label_item_id IS NULL
           ),
           data_major_human as (
               SELECT ass.item_id,
                      --COUNT(1) as                                                                       cnt,
                      LEAST(1, COUNT(1) FILTER (WHERE ann.item_id IS NOT NULL))                          major_anno,
                      LEAST(1, COUNT(1) FILTER (WHERE ann.item_id IS NULL))                              major_no_anno,
                      ARRAY_AGG(DISTINCT ann.value_str) FILTER ( WHERE  ann.value_str IS NOT NULL AND ann.key = 'notes') as notes_major,
                      ARRAY_AGG(DISTINCT ass.assignment_scope_id::text)                                                  as scope_ids,
                     {',\n'.join(selects_major)}
               FROM assignment ass
                    JOIN annotation ann ON ass.assignment_id = ann.assignment_id
               WHERE ass.assignment_scope_id::text = ANY(:major_scopes)
               GROUP BY ass.item_id),
           data_major_robot as (
               SELECT ann.item_id,
                      --COUNT(1) as                                                                       cnt,
                      LEAST(1, COUNT(1) FILTER (WHERE ann.item_id IS NOT NULL))                          major_anno,
                      LEAST(1, COUNT(1) FILTER (WHERE ann.item_id IS NULL))                              major_no_anno,
                      ARRAY_AGG(DISTINCT ann.value_str) FILTER ( WHERE  ann.value_str IS NOT NULL AND ann.key = 'notes') as notes_major,
                      ARRAY_AGG(DISTINCT ann.bot_annotation_metadata_id::text)                                           as scope_ids,
                     {',\n'.join(selects_major)}
               FROM bot_annotation ann
               WHERE ann.bot_annotation_metadata_id::text = ANY(:major_scopes)
               GROUP BY ann.item_id),
           data_major as (
              SELECT r.*
              FROM data_major_robot r
                LEFT JOIN data_major_human h ON r.item_id = h.item_id
              UNION ALL
              SELECT h.*
              FROM data_major_human h
                LEFT JOIN data_major_robot r ON h.item_id = r.item_id
                WHERE r.item_id IS NULL
           ),
           labels as (
               SELECT coalesce(bm.item_id, bi.label_item_id) as item_id,
                      {', '.join(columns_major)},
                      {', '.join(columns_impacts)},
                      (coalesce(bm.scope_ids, array[]::text[]) || coalesce(bi.scope_ids, array[]::text[])) as scope_ids
               FROM data_impacts bi
                    FULL OUTER JOIN data_major bm ON bi.label_item_id = bm.item_id
           )
           SELECT lab.*,
                  ai.publication_year as py,
                  ai.title,
                  i.text,
                  ai.doi,
                  ai.scopus_id,
                  ai.openalex_id,
                  ai.dimensions_id,
                  ai.pubmed_id,
                  ai.wos_id,
                  ai.meta,
                  ai.authors
           FROM labels lab
                JOIN item i ON lab.item_id = i.item_id
                JOIN academic_item ai ON lab.item_id = ai.item_id;
       """,
    )

    logger.debug(query)

    impact_scopes = [scope.scope_id for scope in IMPACT_SCOPES]
    major_scopes = [scope.scope_id for scope in MAJOR_SCOPES]

    logger.debug(f'MAJOR SCOPES: {major_scopes}')
    logger.debug(f'IMPACT SCOPES: {impact_scopes}')

    with db_engine.session() as session:
        logger.info('Running query...')
        res = session.execute(query, {'impact_scopes': impact_scopes, 'major_scopes': major_scopes}).mappings().all()

        logger.info('Constructing DataFrame...')
        df = pd.DataFrame(res).replace({np.nan: None})

    logger.info('Cleaning DataFrame...')
    df['meta_str'] = df['meta'].apply(lambda v: json.dumps(v))
    df['authors_str'] = df['authors'].apply(lambda v: json.dumps(v))

    for grp in LABELS_IMPACTS + LABELS_MAJOR:
        for label in LABELS[grp].labels:
            df[label.column] = df[label.column].astype('Int8')

    logger.info(f'Writing DataFrame to {target}')
    (
        df.drop(columns=['meta', 'authors']).to_csv(
            target,
            index=False,
        )
    )


def main():
    typer.run(dump)


if __name__ == '__main__':
    main()

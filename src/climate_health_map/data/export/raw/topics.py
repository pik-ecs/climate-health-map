from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
import sqlalchemy as sa

from climate_health_map.shared.types import OnConflict
from ..utils import ExportContext


class TopicSource(str, Enum):
    OPENALEX = 'OPENALEX'
    SCOPUS = 'SCOPUS'
    WOS = 'WOS'
    DIMENSIONS = 'DIMENSIONS'


def export(
    config: Annotated[Path, typer.Option(help='Path to config file')],
    source: Annotated[TopicSource, typer.Option(help='Which topic source to use')],
    target: Annotated[Path, typer.Option(help='Target file')],
    batch_size: Annotated[int, typer.Option(help='Batch size')] = 1000,
    project_id: Annotated[str | None, typer.Option(help='Project ID override')] = None,
    import_ids: Annotated[list[str] | None, typer.Option(help='Import ID override')] = None,
    on_exists: Annotated[OnConflict, typer.Option(help='How to react if the target file already exists')] = OnConflict.IGNORE,
    loglevel: Annotated[str, typer.Option(help='Loglevel')] = 'INFO',
) -> None:
    """Export topics that were provided by the original data source"""

    with ExportContext(
        target=target,
        config=config,
        project_id=project_id,
        import_ids=import_ids,
        on_exists=on_exists,
        loglevel=loglevel,
        batch_size=batch_size,
    ) as ctx:
        # ---------------------------------------------------------
        # OPENALEX info
        # ---------------------------------------------------------
        if source == TopicSource.OPENALEX:
            # -- SELECT topic, COUNT(DISTINCT item_id) as cnt
            # -- FROM openalex
            # -- GROUP BY topic
            # -- --
            # -- SELECT subfield, COUNT(DISTINCT item_id) as cnt
            # -- FROM openalex
            # -- GROUP BY subfield
            # -- --
            # -- SELECT domain, COUNT(DISTINCT item_id) as cnt
            # -- FROM openalex
            # -- GROUP BY domain
            # -- --
            # -- ORDER BY cnt;
            #
            # -- {
            # --   "id" : "T13137",
            # --   "display_name" : "Strategic Planning and Analysis",
            # --   "score" : 0.9983,
            # --   "subfield" : {
            # --     "id" : "subfields/1408",
            # --     "display_name" : "Strategy and Management"
            # --   },
            # --   "field" : {
            # --     "id" : "fields/14",
            # --     "display_name" : "Business, Management and Accounting"
            # --   },
            # --   "domain" : {
            # --     "id" : "domains/2",
            # --     "display_name" : "Social Sciences"
            # --   }
            # -- }
            ctx.query = sa.text(
                """
    WITH
        openalex as (
            SELECT ai.item_id,
                   topic ->> 'id'                         as topic_id,
                   topic ->> 'display_name'               as topic,
                   topic ->> 'score'                      as score,
                   topic -> 'subfield' ->> 'id'           as subfield_id,
                   topic -> 'subfield' ->> 'display_name' as subfield,
                   topic -> 'field' ->> 'id'              as field_id,
                   topic -> 'field' ->> 'display_name'    as field,
                   topic -> 'domain' ->> 'id'             as domain_id,
                   topic -> 'domain' ->> 'display_name'   as domain
            FROM academic_item ai
            JOIN m2m_import_item ii ON ai.item_id = ii.item_id
                 CROSS JOIN LATERAL jsonb_array_elements(ai.meta -> 'openalex' -> 'topics') AS topic
            WHERE ai.project_id = :project_id AND ii.import_id::text = ANY(:import_ids)
              AND ai.meta -> 'openalex' is not null)
    SELECT * FROM openalex;
        """
            )

        # ---------------------------------------------------------
        # DIMENSIONS info
        # ---------------------------------------------------------
        elif source == TopicSource.DIMENSIONS:
            # -- SELECT concept, count(DISTINCT item_id) as cnt
            # -- FROM dimensions
            # -- GROUP BY concept
            # -- HAVING count(DISTINCT item_id) > 2
            # -- ORDER BY cnt DESC;
            ctx.query = sa.text("""
    WITH
        dimensions as (
            SELECT ai.item_id,
                   concept ->> 'concept'   as concept,
                   concept ->> 'relevance' as relevance
            FROM academic_item ai
            JOIN m2m_import_item ii ON ai.item_id = ii.item_id
                 CROSS JOIN LATERAL jsonb_array_elements(ai.meta -> 'dimensions' -> 'concepts_scores') AS concept
            WHERE ai.project_id = :project_id AND ii.import_id::text = ANY(:import_ids)
              AND ai.meta -> 'dimensions' is not null)
    SELECT * FROM dimensions;
    """)

        # ---------------------------------------------------------
        # Web of Science info
        # ---------------------------------------------------------
        elif source == TopicSource.WOS:
            #     -- SELECT subheading, count(DISTINCT item_id) as cnt
            #     -- FROM wos_subjects
            #     -- GROUP BY subheading
            #     --
            #     -- SELECT subject, count(DISTINCT item_id) as cnt
            #     -- FROM wos_subjects
            #     -- GROUP BY subject
            #     --
            #     -- SELECT heading, count(DISTINCT item_id) as cnt
            #     -- FROM wos_subjects
            #     -- GROUP BY heading
            #     --
            #     -- ORDER BY cnt;
            #     --

            ctx.query = sa.text("""
    WITH
        wositems AS (
            SELECT ai.item_id, ai.meta
            FROM academic_item ai
            JOIN m2m_import_item ii ON ai.item_id = ii.item_id
            WHERE ai.project_id = :project_id AND ii.import_id::text = ANY(:import_ids)
              AND ai.meta -> 'wos-api' is not null),
        subheadings AS (
            SELECT ai.item_id,
                   subheading
            FROM wositems ai
                 CROSS JOIN LATERAL jsonb_array_elements(
                    ai.meta -> 'wos-api' -> 'static_data' -> 'fullrecord_metadata' -> 'category_info' -> 'subheadings' ->
                    'subheading') AS subheading),
        subjects AS (
            SELECT ai.item_id,
                   subject
            FROM wositems ai
                 CROSS JOIN LATERAL jsonb_array_elements(
                    ai.meta -> 'wos-api' -> 'static_data' -> 'fullrecord_metadata' -> 'category_info' -> 'subjects' ->
                    'subject') AS subject),
        headings AS (
            SELECT ai.item_id,
                   heading
            FROM wositems ai
                 CROSS JOIN LATERAL jsonb_array_elements(
                    ai.meta -> 'wos-api' -> 'static_data' -> 'fullrecord_metadata' -> 'category_info' -> 'headings' ->
                    'heading') AS heading),
        wos_subjects AS (
            SELECT sh.item_id,
                   sh.subheading,
                   h.heading,
                   s.subject ->> 'content'  as subject,
                   s.subject ->> 'ascatype' as ascatype,
                   s.subject ->> 'code'     as code
            FROM subheadings sh
                 LEFT JOIN subjects s ON s.item_id = sh.item_id
                 LEFT JOIN headings h ON h.item_id = sh.item_id)
    SELECT * FROM wos_subjects;""")

        # ---------------------------------------------------------
        # Scopus info
        # ---------------------------------------------------------
        elif source == TopicSource.SCOPUS:
            raise NotImplementedError(f'Unsupported source: {source}')

        else:
            raise NotImplementedError(f'Unsupported source: {source}')


if __name__ == '__main__':
    typer.run(export)

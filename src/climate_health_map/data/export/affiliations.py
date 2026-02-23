from pathlib import Path
from typing import Annotated

import typer
import sqlalchemy as sa

from climate_health_map.shared.types import OnConflict
from ._utils import ExportContext


def export(
    config: Annotated[Path, typer.Option(help='Path to config file')],
    target: Annotated[Path, typer.Option(help='Target file')],
    batch_size: Annotated[int, typer.Option(help='Batch size')] = 1000,
    project_id: Annotated[str | None, typer.Option(help='Project ID override')] = None,
    import_ids: Annotated[list[str] | None, typer.Option(help='Import ID override')] = None,
    only_first_author: Annotated[bool, typer.Option(help='Only include first authors, not all authors')] = False,
    on_exists: Annotated[OnConflict, typer.Option(help='How to react if the target file already exists')] = OnConflict.IGNORE,
    loglevel: Annotated[str, typer.Option(help='Loglevel')] = 'INFO',
):
    """Basic information about author affiliations from all the different sources"""
    with ExportContext(
        target=target,
        config=config,
        project_id=project_id,
        import_ids=import_ids,
        on_exists=on_exists,
        loglevel=loglevel,
        batch_size=batch_size,
    ) as ctx:
        fa_where = 'WHERE author_index = 1 -- limit to first author only' if only_first_author else ''
        ctx.query = sa.text(
            f"""
            WITH
            items as (
                SELECT ai.*
                FROM academic_item ai
                JOIN m2m_import_item ii ON ai.item_id = ii.item_id
                WHERE ai.project_id = :project_id AND ii.import_id::text = ANY(:import_ids)
            ),
            dimensions as (
                SELECT ai.item_id,
                       author_idx                                                                               AS author_index,
                       a.author ->> 'corresponding'                                                             as corresponding,
                       affiliation ->> 'country_code'                                                           as iso2,
                       affiliation ->> 'country'                                                                as country,
                       affiliation ->> 'city'                                                                   as city,
                       null                                                                                     as institution_id,
                       affiliation ->> 'name'                                                                   as institution_name,
                       a.author ->> 'researcher_id'                                                             as author_id,
                       coalesce(a.author ->> 'first_name', '') || ' ' || coalesce(a.author ->> 'last_name', '') as author_name,
                       'dimensions'                                                                             as source
                FROM items ai
                     CROSS JOIN LATERAL jsonb_array_elements(ai.meta -> 'dimensions' -> 'authors') WITH ORDINALITY AS a(author, author_idx)
                     CROSS JOIN LATERAL jsonb_array_elements(a.author -> 'affiliations') AS f(affiliation)
                WHERE ai.meta -> 'dimensions' is not null),
            scopus as (
                SELECT ai.item_id,
                       author_idx                                AS author_index,
                       null                                      as corresponding,
                       null                                      as iso2,
                       aff.affiliation ->> 'affiliation-country' as country,
                       aff.affiliation ->> 'affiliation-city'    as city,
                       aff.affiliation ->> 'afid'                as institution_id,
                       aff.affiliation ->> 'affilname'           as institution_name,
                       author ->> 'authid'                       as author_id,
                       author ->> 'authname'                     as author_name,
                       'scopus'                                  as source
                FROM items ai
                     CROSS JOIN LATERAL jsonb_array_elements(ai.meta -> 'scopus-api' -> 'author') WITH ORDINALITY AS a(author, author_idx)
                     CROSS JOIN LATERAL jsonb_array_elements(a.author -> 'afid') AS institutions(institution)
                     CROSS JOIN LATERAL jsonb_array_elements(ai.meta -> 'scopus-api' -> 'affiliation') AS aff(affiliation)
                WHERE ai.meta -> 'scopus-api' is not null
                  AND aff.affiliation ->> 'afid' = institutions.institution ->> '$'),
            wos as (
                SELECT ai.item_id,
                       (author ->> 'seq_no')::int               AS author_index,
                       null                                     as corresponding,
                       null                                     as iso2,
                       aff -> 'address_spec' ->> 'country'      AS country,
                       aff -> 'address_spec' ->> 'city'         AS city,
                       null                                     as institution_id,
                       aff -> 'address_spec' ->> 'full_address' as institution_name,
                       null                                     as author_id,
                       author ->> 'full_name'                   as author_name,
                       'wos'                                    as source
                FROM items ai
                     CROSS JOIN LATERAL jsonb_array_elements(ai.meta -> 'wos-api' -> 'static_data' -> 'fullrecord_metadata' -> 'addresses' -> 'address_name') AS aff
                     CROSS JOIN LATERAL jsonb_array_elements(aff -> 'names' -> 'name') AS author
                WHERE ai.meta -> 'wos-api' is not null
                  AND (author -> 'addr_no' ->> 0)::int = (aff -> 'address_spec' ->> 'addr_no')::int),
            openalex as (
                SELECT ai.item_id,
                       author_idx                                  AS author_index,
                       a.author ->> 'is_corresponding'             as corresponding,
                       institutions.institution ->> 'country_code' as iso2,
                       null                                        as country,
                       null                                        as city,
                       institutions.institution ->> 'id'           as institution_id,
                       institutions.institution ->> 'display_name' as institution_name,
                       a.author -> 'author' ->> 'id'               as author_id,
                       a.author -> 'author' ->> 'display_name'     as author_name,
                       'openalex'                                  as source
                FROM items ai
                     CROSS JOIN LATERAL jsonb_array_elements(ai.meta -> 'openalex' -> 'authorships') WITH ORDINALITY AS a(author, author_idx)
                     CROSS JOIN LATERAL jsonb_array_elements(a.author -> 'institutions') AS institutions(institution)
                WHERE ai.meta -> 'openalex' is not null),
            collected as (
                SELECT *
                FROM scopus
                UNION ALL
                SELECT *
                FROM wos
                UNION ALL
                SELECT *
                FROM dimensions
                UNION ALL
                SELECT *
                FROM openalex
                )
        SELECT *
        FROM collected
            {fa_where};
            """,
        )


if __name__ == '__main__':
    typer.run(export)

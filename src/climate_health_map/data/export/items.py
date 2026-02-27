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
    label_filter: Annotated[str | None, typer.Option(help='Add filter for this enhancement key')] = None,
    label_threshold: Annotated[float, typer.Option(help='Add filter for this enhancement key above payload threshold')] = 0.5,
    on_exists: Annotated[OnConflict, typer.Option(help='How to react if the target file already exists')] = OnConflict.IGNORE,
    loglevel: Annotated[str, typer.Option(help='Loglevel')] = 'INFO',
):
    """Export items (text + meta-data)"""

    with ExportContext(
        target=target,
        config=config,
        params={'key': label_filter, 'threshold': label_threshold} if label_filter else None,
        project_id=project_id,
        import_ids=import_ids,
        on_exists=on_exists,
        loglevel=loglevel,
        batch_size=batch_size,
    ) as ctx:
        enhancement_join = 'JOIN enhancement en ON ai.item_id = en.item_id'
        enhancement_where = ' AND en.key = :key AND en.payload::float > :threshold'
        ctx.query = sa.text(
            f"""
            SELECT i.item_id::text,
                   ai.openalex_id,
                   ai.doi,
                   ai.wos_id,
                   ai.scopus_id,
                   ai.title as title,
                   i.text   as abstract,
                   ai.publication_year
            FROM academic_item ai
                 JOIN item i on i.item_id = ai.item_id
                 JOIN m2m_import_item ii ON ai.item_id = ii.item_id JOIN enhancement en ON ai.item_id = en.item_id
                 {enhancement_join}
            WHERE ai.project_id = :project_id 
              AND ii.import_id::text = ANY(:import_ids)
              AND ai.meta -> 'openalex' ->> 'source_id' <> 'S7407052681'  -- "source": "Data Planet"
              {enhancement_where};
            """,
        )


if __name__ == '__main__':
    typer.run(export)

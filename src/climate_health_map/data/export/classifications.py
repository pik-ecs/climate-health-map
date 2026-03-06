from pathlib import Path
from typing import Annotated

import typer
import sqlalchemy as sa

from climate_health_map.shared.types import OnConflict
from ._utils import ExportContext
from ..labels import LABELS


def export(
    config: Annotated[Path, typer.Option(help='Path to config file')],
    target: Annotated[Path, typer.Option(help='Target file')],
    batch_size: Annotated[int, typer.Option(help='Batch size')] = 1000,
    project_id: Annotated[str | None, typer.Option(help='Project ID override')] = None,
    import_ids: Annotated[list[str] | None, typer.Option(help='Import ID override')] = None,
    on_exists: Annotated[OnConflict, typer.Option(help='How to react if the target file already exists')] = OnConflict.IGNORE,
    loglevel: Annotated[str, typer.Option(help='Loglevel')] = 'INFO',
) -> None:
    """Export classifications"""
    with ExportContext(
        target=target,
        config=config,
        project_id=project_id,
        import_ids=import_ids,
        on_exists=on_exists,
        loglevel=loglevel,
        batch_size=batch_size,
    ) as ctx:
        filter_selects = [
            # f"MAX(payload::float) FILTER (WHERE key = '{label.column}')  AS \"{label.column}\",  -- {label.name}"
            f'MAX(payload::float) FILTER (WHERE key = \'{label.column}\')  AS "{label.column}"'
            for group in LABELS.values()
            for label in group.labels
        ]
        ctx.query = sa.text(
            f"""
SELECT i.item_id::text,
       {',\n'.join(filter_selects)}
FROM enhancement e
     JOIN item i ON i.item_id = e.item_id
     JOIN m2m_import_item ii ON e.item_id = ii.item_id
WHERE i.project_id::text = :project_id AND ii.import_id::text = ANY(:import_ids)
GROUP BY i.item_id;
            """,
        )


if __name__ == '__main__':
    typer.run(export)

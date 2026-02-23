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
    on_exists: Annotated[OnConflict, typer.Option(help='How to react if the target file already exists')] = OnConflict.IGNORE,
    loglevel: Annotated[str, typer.Option(help='Loglevel')] = 'INFO',
):
    """Export mordecai results

    payload = [
        {
            'lat': 25.7884,
            'lon': -80.1303,
            'name': 'Tropics Hotel',
            'score': 0.8531627655029297,
            'city_id': '4164143',
            'end_char': 686,
            'city_name': 'Miami Beach',
            'geonameid': '6502125',
            'start_char': 679,
            'admin1_code': 'FL',
            'admin1_name': 'Florida',
            'search_name': 'tropics',
            'feature_code': 'HTL',
            'country_code3': 'USA',
            'feature_class': 'S',
        },
        ...
    ]
    """
    with ExportContext(
        target=target,
        config=config,
        project_id=project_id,
        import_ids=import_ids,
        on_exists=on_exists,
        loglevel=loglevel,
        batch_size=batch_size,
    ) as ctx:
        ctx.query = sa.text(
            """
WITH
    unrolled AS (
        SELECT e.item_id,
               jsonb_array_elements(e.payload) as place
        FROM enhancement e
            JOIN item i ON i.item_id = e.item_id
            JOIN m2m_import_item ii ON i.item_id = ii.item_id
        WHERE i.project_id::text  = :project_id
          AND ii.import_id::text = ANY(:import_ids)
          AND e.key = 'mordecai3')
SELECT item_id::text,
       place ->> 'lat'           AS lat,
       place ->> 'lat'           AS lat,
       place ->> 'lon'           AS lon,
       place ->> 'name'          AS name,
       place ->> 'score'         AS score,
       place ->> 'city_id'       AS city_id,
       place ->> 'end_char'      AS end_char,
       place ->> 'city_name'     AS city_name,
       place ->> 'geonameid'     AS geonameid,
       place ->> 'start_char'    AS start_char,
       place ->> 'admin1_code'   AS admin1_code,
       place ->> 'admin1_name'   AS admin1_name,
       place ->> 'search_name'   AS search_name,
       place ->> 'feature_code'  AS feature_code,
       place ->> 'country_code3' AS country_code3,
       place ->> 'feature_class' AS feature_class
FROM unrolled;
            """,
        )


if __name__ == '__main__':
    typer.run(export)

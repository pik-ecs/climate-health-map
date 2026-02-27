from pathlib import Path
from typing import Annotated

import typer

from climate_health_map.shared.types import OnConflict
from .items import export as export_items
from .annotations import export as export_annotations
from .affiliations import export as export_affiliations
from .places import export as export_places
from .labels import export as export_labels
from .topics import export as export_topics

app = typer.Typer(help='Commands to download data from NACSOS into csv files')

app.command('items', help='Export all base information')(export_items)
app.command('affiliations', help='Export author affiliations')(export_affiliations)
app.command('places', help='Export all geo-name information')(export_places)
app.command('places', help='Export all topic information from sources')(export_topics)
app.command('labels', help='Export all classifications')(export_labels)

app.command('annotations', help='Fetch all eligible human annotations from NACSOS and prepare a clean csv for training and evaluation')(export_annotations)


@app.command('all', help='Run all exports')
def export(
    config: Annotated[Path, typer.Option(help='Path to config file')],
    target: Annotated[Path, typer.Option(help='Target directory')],
    batch_size: Annotated[int, typer.Option(help='Batch size')] = 1000,
    project_id: Annotated[str | None, typer.Option(help='Project ID override')] = None,
    import_ids: Annotated[list[str] | None, typer.Option(help='Import ID override')] = None,
    on_exists: Annotated[OnConflict, typer.Option(help='How to react if the target file already exists')] = OnConflict.IGNORE,
    loglevel: Annotated[str, typer.Option(help='Loglevel')] = 'INFO',
) -> None:
    export_items(
        config=config,
        target=target / 'items.csv',
        batch_size=batch_size,
        project_id=project_id,
        import_ids=import_ids,
        on_exists=on_exists,
        loglevel=loglevel,
    )
    export_affiliations(
        config=config,
        target=target / 'affiliations.csv',
        only_first_author=False,
        batch_size=batch_size,
        project_id=project_id,
        import_ids=import_ids,
        on_exists=on_exists,
        loglevel=loglevel,
    )
    export_places(
        config=config,
        target=target / 'places.csv',
        batch_size=batch_size,
        project_id=project_id,
        import_ids=import_ids,
        on_exists=on_exists,
        loglevel=loglevel,
    )
    for source in ['WOS', 'DIMENSIONS', 'OPENALEX']:
        export_topics(
            config=config,
            source=source,  # type:ignore [arg-type]
            target=target / f'topics_{source.lower()}.csv',
            batch_size=batch_size,
            project_id=project_id,
            import_ids=import_ids,
            on_exists=on_exists,
            loglevel=loglevel,
        )
    export_labels(
        config=config,
        target=target / 'labels.csv',
        batch_size=batch_size,
        project_id=project_id,
        import_ids=import_ids,
        on_exists=on_exists,
        loglevel=loglevel,
    )


if __name__ == '__main__':
    app()

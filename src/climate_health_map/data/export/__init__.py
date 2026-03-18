from pathlib import Path
from typing import Annotated

import typer
from climate_health_map.shared.types import OnConflict

from .raw.items import export as export_items
from .raw.affiliations import export as export_affiliations
from .raw.places import export as export_places
from .raw.classifications import export as export_classifications
from .raw.topics import export as export_topics
from .annotations import export as export_annotations
from .lithub import prepare_lithub_export
from .lancet import app as lancet_app

app = typer.Typer(help='Commands to download data from NACSOS into csv files')

app.command('items', help='Export all base information')(export_items)
app.command('affiliations', help='Export author affiliations')(export_affiliations)
app.command('places', help='Export all geo-name information')(export_places)
app.command('topics', help='Export all topic information from sources')(export_topics)
app.command('classifications', help='Export all classifications')(export_classifications)
app.command('annotations', help='Fetch all eligible human annotations from NACSOS and prepare a clean csv for training and evaluation')(export_annotations)
app.command('lithub', help='Consolidate all exports into lithub files')(prepare_lithub_export)

app.add_typer(lancet_app, name='lancet', help='Lancet Countdown data and figure exports')


@app.command('all', help='Run all exports')
def export(
    config: Annotated[Path, typer.Option(help='Path to config file')],
    target: Annotated[Path, typer.Option(help='Target directory')],
    batch_size: Annotated[int, typer.Option(help='Batch size')] = 1000,
    project_id: Annotated[str | None, typer.Option(help='Project ID override')] = None,
    import_ids: Annotated[list[str] | None, typer.Option(help='Import ID override')] = None,
    only_first_author: Annotated[bool, typer.Option('--only-first-author/--all-authors', help='Only include first authors, not all authors')] = False,
    run_items: Annotated[bool, typer.Option('--export-items/--skip-items', help='Run items export')] = True,
    run_affiliations: Annotated[bool, typer.Option('--export-affiliations/--skip-affiliations', help='Run affiliations export')] = True,
    run_places: Annotated[bool, typer.Option('--export-places/--skip-places', help='Run places export')] = True,
    run_topics: Annotated[bool, typer.Option('--export-topics/--skip-topics', help='Run topics export')] = True,
    run_classifications: Annotated[bool, typer.Option('--export-classifications/--skip-classifications', help='Run classifications export')] = True,
    filter_classifications: Annotated[bool, typer.Option(help='Only export items that are classified as impact-relevant')] = False,
    filter_num_classifications: Annotated[int | None, typer.Option(help='Only export items with at least this many classification/topic enhancements')] = None,
    on_exists: Annotated[OnConflict, typer.Option(help='How to react if the target file already exists')] = OnConflict.IGNORE,
    filetype: Annotated[str, typer.Option(help='File type override')] = 'csv',
    loglevel: Annotated[str, typer.Option(help='Loglevel')] = 'INFO',
) -> None:
    if run_items:
        export_items(
            config=config,
            target=target / f'items.{filetype}',
            label_filter='rel_major|1',
            batch_size=batch_size,
            project_id=project_id,
            import_ids=import_ids,
            on_exists=on_exists,
            loglevel=loglevel,
        )
    if run_affiliations:
        export_affiliations(
            config=config,
            target=target / f'affiliations.{filetype}',
            only_first_author=only_first_author,
            batch_size=batch_size,
            project_id=project_id,
            import_ids=import_ids,
            on_exists=on_exists,
            loglevel=loglevel,
        )
    if run_places:
        export_places(
            config=config,
            target=target / f'places.{filetype}',
            batch_size=batch_size,
            project_id=project_id,
            import_ids=import_ids,
            on_exists=on_exists,
            loglevel=loglevel,
        )
    if run_topics:
        for source in ['WOS', 'DIMENSIONS', 'OPENALEX']:
            export_topics(
                config=config,
                source=source,  # type:ignore [arg-type]
                target=target / f'topics_{source.lower()}.{filetype}',
                batch_size=batch_size,
                project_id=project_id,
                import_ids=import_ids,
                on_exists=on_exists,
                loglevel=loglevel,
            )
    if run_classifications:
        export_classifications(
            config=config,
            target=target / f'classifications.{filetype}',
            batch_size=batch_size,
            project_id=project_id,
            import_ids=import_ids,
            on_exists=on_exists,
            filter_classifications=filter_classifications,
            filter_num_classifications=filter_num_classifications,
            loglevel=loglevel,
        )


if __name__ == '__main__':
    app()

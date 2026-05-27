import typer

from .excel import prepare_lancet_excel_export
from .excel_regional import prepare_lancet_excel_export as prepare_lancet_excel_export_regional
from .loaders import read_base_data

app = typer.Typer()

app.command('excel', help='Write Excel datasheet exports')(prepare_lancet_excel_export)
app.command('excel-regional', help='Write regional indicator Excel datasheet exports')(prepare_lancet_excel_export_regional)

__all__ = [
    'prepare_lancet_excel_export',
    'read_base_data',
]

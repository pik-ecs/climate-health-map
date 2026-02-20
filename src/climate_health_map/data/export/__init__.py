import typer
from .annotations import dump

app = typer.Typer()

app.command('export-annotations', help='Fetch all eligible annotations from NACSOS and prepare a clean csv for training and evaluation')(dump)

if __name__ == '__main__':
    app()

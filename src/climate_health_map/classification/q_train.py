from pathlib import Path
from typing import Annotated

import typer
import pandas as pd

from climate_health_map.classification.util import read_tuning_info

pd.options.display.max_columns = 650
pd.options.display.max_rows = 200
pd.options.display.width = 100000


def main(
    source: Annotated[Path, typer.Option(help='Path to folder containing all the validation and tuning parameters')],
    target: Annotated[Path, typer.Option(help='Path to folder to write quality summary to')],
):
    target.mkdir(parents=True, exist_ok=True)
    # TODO


if __name__ == '__main__':
    typer.run(main)

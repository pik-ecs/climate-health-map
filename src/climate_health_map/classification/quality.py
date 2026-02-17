from pathlib import Path
from typing import Annotated

from tqdm import tqdm
import pandas as pd
import json
import typer

pd.options.display.max_columns = 650
pd.options.display.max_rows = 200
pd.options.display.width = 100000


def main(
    source: Annotated[Path, typer.Option(help='Path to folder containing all the validation and tuning parameters')],
    target: Annotated[Path, typer.Option(help='Path to file to write quality summary to')],
):
    infos = []
    for file in tqdm(source.glob('*.json')):
        with open(file) as f:
            infos.append(json.load(f))
    df = pd.DataFrame.from_records(infos)  # , exclude=['params', 'repeat', 'train_proportion', 'random_state', 'n_train', 'n_test'])

    df_agg = df.groupby(['column', 'model']).describe()
    df_aggagg = df_agg.sort_values(('f1_test', 'mean'), ascending=False).groupby('column').first().sort_values('column')
    print(df_aggagg[[('f1_test', 'mean'), ('f1_test', 'std')]])

    with pd.ExcelWriter(target) as writer:
        df.to_excel(writer, index=False, sheet_name='Full dataset')
        df_agg.T.to_excel(writer, sheet_name='Quality (model, column)')
        df_aggagg.T.to_excel(writer, index=True, sheet_name='Quality (column)')


if __name__ == '__main__':
    typer.run(main)

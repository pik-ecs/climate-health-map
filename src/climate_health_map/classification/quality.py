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
        name = file.stem.split('_')[0]
        label = file.stem.split('_')[-2]
        with open(file) as f:
            infos.append(json.load(f) | {'name': name, 'label': label})
    df = pd.DataFrame.from_records(infos)  # , exclude=['params', 'repeat', 'train_proportion', 'random_state', 'n_train', 'n_test'])

    with pd.ExcelWriter(target) as writer:
        df.to_excel(writer, index=False, sheet_name='Full dataset')
        df_agg = df.groupby(['label', 'name']).describe()
        df_agg.T.to_excel(writer, sheet_name='Quality (model, label)')
        (
            df_agg.sort_values(('f1_test', 'mean'), ascending=False)
            .groupby('label')
            .first()
            .sort_values('label')
            .T.to_excel(writer, index=False, sheet_name='Quality (label)')
        )

    print(df)


if __name__ == '__main__':
    typer.run(main)

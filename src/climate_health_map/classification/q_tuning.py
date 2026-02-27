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
) -> None:
    target.mkdir(parents=True, exist_ok=True)

    df = read_tuning_info(source)

    df_agg = df.groupby(['column', 'model']).describe().drop(columns=['repeat', 'random_state', 'train_proportion'])
    df_aggagg = df_agg.reset_index().sort_values(('f1_test', 'mean'), ascending=False).groupby('column').first().sort_values('column')
    print(df_aggagg[[('model', ''), ('f1_test', 'mean'), ('f1_test', 'std')]])

    with pd.ExcelWriter(target / 'summary.xlsx') as writer:
        df.to_excel(writer, index=False, sheet_name='Full dataset')
        df.to_csv(target / 'complete.csv', index=False)

        df_agg.T.to_excel(writer, sheet_name='Quality (model, column)')
        df_agg.to_csv(target / 'column_model_best.csv')

        df_aggagg.T.to_excel(writer, sheet_name='Quality (column)')
        df_aggagg.to_csv(target / 'column_best.csv')


if __name__ == '__main__':
    typer.run(main)

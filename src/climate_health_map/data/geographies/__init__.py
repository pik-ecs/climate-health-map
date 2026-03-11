"""Definitions of countries with ISO2/3 codes, names, and groupings.

https://en.wikipedia.org/wiki/ISO_3166-1_numeric
https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3

Dummy snippet to help extending the countries.csv
Afterwards, check that Continent column is still intact (North America = NA might be empty)

Income groups and lending categories based on World Bank data from 2026
https://datahelpdesk.worldbank.org/knowledgebase/articles/906519-world-bank-country-and-lending-groups

```python
import pandas as pd
df1 = pd.read_csv('src/climate_health_map/data/geographies/_countries.csv', keep_default_na=False)  # dtype=str,
df2 = pd.read_csv('src/climate_health_map/data/geographies/tmp.csv')  # , sep='\t')
(
    df1
    .merge(df2[['iso3', 'population']], left_on='iso3', right_on='iso3', how='outer')
    .astype({'population': 'Int32', 'iso_num': 'Int32'})
    .rename(columns={'population': 'Population (Lancet, 2025)'})
    .to_csv('src/climate_health_map/data/geographies/updated.csv', index=False)
)
df2 = pd.read_csv('src/climate_health_map/data/geographies/tmp.csv')  # , sep='\t')
(
    df1
    .merge(df2.iloc[:218].rename(columns={'Economy': 'Name (WorldBank 2026)', 'Code': 'iso3', 'Region': 'Region (WorldBank 2026)', 'Income group': 'Income group (WorldBank 2026)', 'Lending category': 'Lending category (WorldBank 2026)'}), left_on='iso3', right_on='iso3', how='outer')
    .replace({'': None})
    .astype({'Population (Lancet, 2025)': 'Int32', 'iso_num': 'Int32'})
    .to_csv('src/climate_health_map/data/geographies/updated.csv', index=False)
)
```
"""

from pathlib import Path
import pandas as pd

from .continents import CONTINENT_MAP
from .variations import country_names as COUNTRY_NORMALISATION
from .places import fix_geographies, get_naming_mask, get_publisher_mask, load_df_places
from .features import FEATURES, FEATURE_LOOKUP

here = Path(__file__).parent.resolve()


def load_country_infos() -> pd.DataFrame:
    return pd.read_csv(here / '_countries.csv', dtype=str, keep_default_na=False).map(str.strip, na_action='ignore').replace({'': pd.NA})


def load_grid_data() -> pd.DataFrame:
    return pd.read_csv(here / '_grid_data.csv', dtype=str, keep_default_na=False).map(str.strip, na_action='ignore').replace({'': pd.NA})


def flatten_country_groups(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    # return pd.DataFrame(
    #     [
    #         {'item_id': item_id}
    #         | {
    #             f'{prefix}_{column}|{grouping}': count
    #             for column in [
    #                 'Group (Lancet 2026)',
    #                 'Group (WHO 2026)',
    #                 'Group (HDI 2026)',
    #                 'Region (IPCC AR6, 6)',
    #                 'Region (IPCC AR6, 10)',
    #                 'Region (WorldBank 2026)',
    #                 'Income group (WorldBank 2026)',
    #                 'Lending category (WorldBank 2026)',
    #                 'Continent (Name)',
    #             ]
    #             for grouping, count in group[group[column].notna()][column].value_counts().items()
    #         }
    #         for item_id, group in tqdm(df.groupby('item_id'))
    #     ],
    # ).set_index('item_id')
    #
    # Gemini translation of the above:
    cols = [
        'Group (Lancet 2026)',
        'Group (WHO 2026)',
        'Group (HDI 2026)',
        'Region (IPCC AR6, 6)',
        'Region (IPCC AR6, 10)',
        'Region (WorldBank 2026)',
        'Income group (WorldBank 2026)',
        'Lending category (WorldBank 2026)',
        'Continent (Name)',
    ]
    # 2. "Melt" the dataframe so columns become a single categorical variable
    # This is much faster than looping over columns manually
    df_melted = df.melt(id_vars=['item_id'], value_vars=cols, var_name='column', value_name='grouping')
    # 3. Drop NaNs once globally
    df_melted = df_melted.dropna(subset=['grouping'])
    # 4. Perform a single GroupBy + Size (Vectorized value_counts)
    counts = df_melted.groupby(['item_id', 'column', 'grouping']).size()
    # 5. Reshape to get your specific naming convention
    # Unstack 'column' and 'grouping' into the header
    result = counts.unstack(level=[1, 2]).fillna(0).astype(int)
    # 6. Fix column names to match your '{prefix}_{column}|{grouping}' format
    result.columns = [f'{prefix}_{col}|{grp}' for col, grp in result.columns]
    return result


__all__ = [
    'load_country_infos',
    'load_grid_data',
    'fix_geographies',
    'get_publisher_mask',
    'get_naming_mask',
    'load_df_places',
    'CONTINENT_MAP',
    'COUNTRY_NORMALISATION',
    'FEATURES',
    'FEATURE_LOOKUP',
    'flatten_country_groups',
]

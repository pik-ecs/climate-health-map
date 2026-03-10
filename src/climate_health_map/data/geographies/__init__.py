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
    return pd.read_csv(here / '_countries.csv', dtype=str, keep_default_na=False).map(str.strip, na_action='ignore')


def load_grid_data() -> pd.DataFrame:
    return pd.read_csv(here / '_grid_data.csv', dtype=str, keep_default_na=False).map(str.strip, na_action='ignore')


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
]

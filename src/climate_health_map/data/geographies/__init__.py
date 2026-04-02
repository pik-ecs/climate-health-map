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
-----------------------
df1 = pd.read_csv('src/climate_health_map/data/geographies/_countries.csv', keep_default_na=False)  # dtype=str,
df2 = pd.read_csv('src/climate_health_map/data/geographies/tmp.csv')  # , sep='\t')
(
    df1
    .merge(df2[['iso3', 'population']], left_on='iso3', right_on='iso3', how='outer')
    .astype({'population': 'Int32', 'iso_num': 'Int32'})
    .rename(columns={'population': 'Population (Lancet, 2025)'})
    .to_csv('src/climate_health_map/data/geographies/updated.csv', index=False)
)
-----------------------
df2 = pd.read_csv('src/climate_health_map/data/geographies/tmp.csv')  # , sep='\t')
(
    df1
    .merge(df2.iloc[:218].rename(columns={'Economy': 'Name (WorldBank 2026)', 'Code': 'iso3', 'Region': 'Region (WorldBank 2026)', 'Income group': 'Income group (WorldBank 2026)', 'Lending category': 'Lending category (WorldBank 2026)'}), left_on='iso3', right_on='iso3', how='outer')
    .replace({'': None})
    .astype({'Population (Lancet, 2025)': 'Int32', 'iso_num': 'Int32'})
    .to_csv('src/climate_health_map/data/geographies/updated.csv', index=False)
)
-----------------------
df1 = pd.read_csv('src/climate_health_map/data/geographies/_countries.csv', keep_default_na=False, dtype=str)
df2 = pd.read_csv('src/climate_health_map/data/geographies/europe.csv' , keep_default_na=False, sep='\t')
df1.merge(df2[['iso3', 'EEA sub-region division', 'European sub-region (UN geoscheme)', 'EU', 'EEA ']].rename(columns={
    'EEA sub-region division': 'Region (EEA sub-division)',
    'European sub-region (UN geoscheme)': 'Region (UN Europe sub-region)',
    'EU': 'EU member (2025)',
    'EEA ':'EEA member (2025)'
}), left_on='iso3', right_on='iso3', how='left').to_csv('src/climate_health_map/data/geographies/updated.csv', index=False)
```
"""

from .continents import CONTINENT_MAP
from .variations import country_names as COUNTRY_NORMALISATION
from .utils import fix_geographies, get_naming_mask, get_publisher_mask, flatten_country_groups, join_shapes, join_grid_shapes, LOCATION_FEATURE_CODES
from .readers import read_places_export, load_df_places, load_grid_data, load_country_infos, load_annual_population
from .features import FEATURES, FEATURE_LOOKUP

__all__ = [
    'load_country_infos',
    'load_grid_data',
    'load_annual_population',
    'fix_geographies',
    'get_publisher_mask',
    'get_naming_mask',
    'load_df_places',
    'CONTINENT_MAP',
    'COUNTRY_NORMALISATION',
    'FEATURES',
    'FEATURE_LOOKUP',
    'flatten_country_groups',
    'read_places_export',
    'join_shapes',
    'join_grid_shapes',
    'LOCATION_FEATURE_CODES',
]

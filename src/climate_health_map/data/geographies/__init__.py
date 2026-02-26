"""Definitions of countries with ISO2/3 codes, names, and groupings.

https://en.wikipedia.org/wiki/ISO_3166-1_numeric
https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3

Dummy snippet to help extending the countries.csv
Afterwards, check that Continent column is still intact (North America = NA might be empty)
```python
import pandas as pd
df1 = pd.read_csv('src/climate_health_map/data/countries/countries.csv')
df2 = pd.read_csv('src/climate_health_map/data/countries/tmp.csv', sep='\t')
df1.merge(df2, left_on='iso3', right_on='ISO3', how='outer').to_csv('src/climate_health_map/data/countries/updated.csv', index=False)
```
"""

from pathlib import Path
import pandas as pd

from .geoparser import mordecai
from .continents import CONTINENT_MAP
from .variations import country_names as COUNTRY_NORMALISATION

here = Path(__file__).parent.resolve()

def load_country_infos()->pd.DataFrame:
    return pd.read_csv(here / '_countries.csv', dtype=str, keep_default_na=False).map(str.strip, na_action='ignore')

def load_grid_data()->pd.DataFrame:
    return pd.read_csv(here / '_grid_data.csv', dtype=str, keep_default_na=False).map(str.strip, na_action='ignore')

__all__ = [
    'load_country_infos',
    'load_grid_data',
    'mordecai',
    'CONTINENT_MAP',
    'COUNTRY_NORMALISATION',
]

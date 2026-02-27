"""Definitions of countries with ISO2/3 codes, names, and groupings.

https://en.wikipedia.org/wiki/ISO_3166-1_numeric
https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3

Dummy snippet to help extending the countries.csv
Afterwards, check that Continent column is still intact (North America = NA might be empty)
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
```
"""

from pathlib import Path
import pandas as pd

from .geoparser import mordecai
from .continents import CONTINENT_MAP
from .variations import country_names as COUNTRY_NORMALISATION
from .filters import fix_geographies, get_naming_mask, get_publisher_mask

here = Path(__file__).parent.resolve()


def load_country_infos() -> pd.DataFrame:
    return pd.read_csv(here / '_countries.csv', dtype=str, keep_default_na=False).map(str.strip, na_action='ignore')


def load_grid_data() -> pd.DataFrame:
    return pd.read_csv(here / '_grid_data.csv', dtype=str, keep_default_na=False).map(str.strip, na_action='ignore')


def load_df_places(source: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a clean version of extracted places and a filter mask.

    Don't forget to get the additional `get_publisher_mask` after merging df_places with df_base!
    """
    if source.suffix == '.csv':
        df = pd.read_csv(source, dtype=str, keep_default_na=False)  # keep_default_na handles cells that contain "NA" (which is valid)
    elif source.suffix == '.feather':
        df = pd.read_feather(source)
    elif source.suffix == '.parquet':
        df = pd.read_parquet(source)
    else:
        raise ValueError(f'Unsupported file type: {source.suffix}')
    df = fix_geographies(df)
    mask = get_naming_mask(df)
    return df, mask


__all__ = [
    'load_country_infos',
    'load_grid_data',
    'mordecai',
    'fix_geographies',
    'get_publisher_mask',
    'get_naming_mask',
    'load_df_places',
    'CONTINENT_MAP',
    'COUNTRY_NORMALISATION',
]

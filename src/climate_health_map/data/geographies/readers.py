from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .utils import fix_geographies, get_naming_mask
from climate_health_map.shared import read_any_pd

here = Path(__file__).parent.resolve()

HDI_MAP = {
    'Very High': 'Very High or High',
    'High': 'Very High or High',
    'Medium': 'Low or Medium',
    'Low': 'Low or Medium',
}


def load_country_infos() -> pd.DataFrame:
    return (
        pd.read_csv(here / '_countries.csv', dtype=str, keep_default_na=False)
        .map(str.strip, na_action='ignore')
        .replace({'': pd.NA, np.nan: pd.NA})
        .astype(
            {'Population (Lancet, 2025)': 'Int32', 'iso_num': 'Int32'},
        )
        .assign(
            **{
                'Group (HDI-2 2026)': lambda data: data['Group (HDI 2026)'].map(lambda v: HDI_MAP[v] if v in HDI_MAP else pd.NA),
                'Group (HDI-2 2025)': lambda data: data['Group (HDI 2025)'].map(lambda v: HDI_MAP[v] if v in HDI_MAP else pd.NA),
            }
        )
    )


def load_grid_data() -> pd.DataFrame:
    df_grid = (
        pd.read_csv(here / '_grid_data.csv', dtype=str, keep_default_na=False)
        .map(str.strip, na_action='ignore')
        .replace(
            {'': pd.NA, np.nan: pd.NA},
        )
        .astype(
            {
                'index': 'Int32',
                'LAT': 'Float64',
                'LON': 'Float64',
                'area': 'Float64',
                #'is_land': 'bool',
                'precip_da': 'Float32',
                'temp_da': 'Float32',
                'population': 'Float32',
            },
        )
        # we need two steps before getting to int
        .astype({'precip_da': 'Int8', 'temp_da': 'Int8', 'population': 'Int32'})
        .set_index('index')
    )
    df_grid['grid_cooler'] = df_grid['temp_da'].isin([-2, -3])
    df_grid['grid_warmer'] = df_grid['temp_da'].isin([2, 3])
    df_grid['grid_wetter'] = df_grid['precip_da'].isin([2, 3])
    df_grid['grid_drier'] = df_grid['precip_da'].isin([-2, -3])
    df_grid['grid_attributable'] = df_grid[['grid_cooler', 'grid_warmer', 'grid_wetter', 'grid_drier']].any(axis=1)

    df_grid['attribution'] = 0  # no attribution
    df_grid.loc[df_grid['grid_attributable'], 'attribution'] = 1  # temp or humidity
    df_grid.loc[df_grid[['grid_cooler', 'grid_warmer']].any(axis=1) & df_grid[['grid_wetter', 'grid_drier']].any(axis=1), 'attribution'] = 2

    df_grid['is_land'] = df_grid['is_land'].map({'True': True, 'False': False})
    return df_grid.reset_index(names='grid_id')


def load_annual_population():
    """
    ```bash
    wget -O "data/shapes_2026/population.json" "https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?format=json&date=1990:2025&per_page=20000"
    ```

    ```python
    import json
    import pandas as pd
    pop = json.load(open('data/shapes_2026/population.json'))
    (
        pd.DataFrame(pop[1])
        .drop(columns=['indicator', 'country', 'unit', 'obs_status', 'decimal'])
        .astype({'value': 'Int64'})
        .rename(columns={'countryiso3code': 'iso3', 'value': 'Population'})
        .to_csv('src/climate_health_map/data/geographies/_annual_population.csv', index=False)
    )
    ```
    """
    df_population = pd.read_csv(here / '_annual_population.csv').astype({'year': 'Int32', 'Population': 'Int64'})
    return df_population[df_population['iso3'].notna() & df_population['Population'].notna()]


def _read_places_df(source: Path, index_column: str | None = 'item_id', resolution: float = 2.5, merge_taiwan_china: bool = True) -> pd.DataFrame:
    df_places = (
        read_any_pd(source, dtype=str, keep_default_na=False, index_column=index_column)
        .replace(
            {'': pd.NA, np.nan: pd.NA},
        )
        .astype(
            {
                'lat': 'Float64',
                'lon': 'Float64',
                'score': 'Float64',
                # FIXME: Failed to parse string: 'Berenguela' as a scalar of type int32: Error while type casting for column 'city_id'
                # 'city_id': 'Int32',
                'end_char': 'Int32',
                'start_char': 'Int32',
                'geonameid': 'Int32',
            },
        )
        .replace(
            {'': pd.NA, np.nan: pd.NA},
        )
    )
    df_places['LAT'] = df_places['lat'] // resolution * resolution + (resolution / 2)
    df_places['LON'] = df_places['lon'] // resolution * resolution + (resolution / 2)
    df_places['location_id'] = np.arange(df_places.shape[0])
    if merge_taiwan_china:
        df_places.loc[df_places['country_code3'] == 'TWN', 'country_code3'] = 'CHN'

    return df_places


def load_df_places(
    source: Path,
    index_column: str | None = None,
    merge_taiwan_china: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load a clean version of extracted places and a filter mask.

    Don't forget to get the additional `get_publisher_mask` after merging df_places with df_base!
    Example usage:

    ```
    df_base = pd.read(main-data)
    df_places, mask_places = load_df_places(source)
    df = df_base.merge(df_places, on='search_name')
    ```

    """
    df = _read_places_df(source, index_column=index_column, merge_taiwan_china=merge_taiwan_china)
    df = fix_geographies(df)
    mask = get_naming_mask(df) & df['geonameid'].notna()
    return df, mask


def read_places_export(
    source: Path,
    df_countries: Optional[pd.DataFrame] = None,
    df_grid: Optional[pd.DataFrame] = None,
    resolution: float = 2.5,
    include_grid: bool = False,
    merge_taiwan_china: bool = True,
    index_column: str | None = None,
) -> pd.DataFrame:
    df_countries = load_country_infos() if df_countries is None else df_countries
    df_places = _read_places_df(source, index_column=index_column, resolution=resolution, merge_taiwan_china=merge_taiwan_china)
    df = df_places.merge(df_countries, left_on='country_code3', right_on='iso3', how='left')

    if include_grid:
        df_grid = load_grid_data() if df_grid is None else df_grid
        df = df.merge(df_grid, left_on=['LAT', 'LON'], right_on=['LAT', 'LON'], how='outer')
    return df.replace({np.nan: pd.NA})

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .utils import fix_geographies, get_naming_mask
from climate_health_map.shared import read_any_pd

here = Path(__file__).parent.resolve()


def load_country_infos() -> pd.DataFrame:
    return (
        pd.read_csv(here / '_countries.csv', dtype=str, keep_default_na=False)
        .map(str.strip, na_action='ignore')
        .replace({'': pd.NA, np.nan: pd.NA})
        .astype(
            {'Population (Lancet, 2025)': 'Int32', 'iso_num': 'Int32'},
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
                'lat': 'Float64',
                'lon': 'Float64',
                'area': 'Float64',
                'is_land': 'bool',
                'precip_da': 'Int8',
                'temp_da': 'Int8',
                'population': 'Int32',
            },
        )
        .set_index('index')
    )
    df_grid['grid_cooler'] = df_grid['temp_da'].isin([-2, -3])
    df_grid['grid_warmer'] = df_grid['temp_da'].isin([2, 3])
    df_grid['grid_wetter'] = df_grid['precip_da'].isin([2, 3])
    df_grid['grid_drier'] = df_grid['precip_da'].isin([-2, -3])
    df_grid['grid_attributable'] = df_grid[['cooler', 'warmer', 'wetter', 'drier']].any(axis=1)
    return df_grid


def _read_places_df(source: Path, index_column: str | None = 'item_id', resolution: float = 2.5) -> pd.DataFrame:
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
    return df_places


def load_df_places(source: Path, index_column: str = 'item_id') -> tuple[pd.DataFrame, pd.Series]:
    """Load a clean version of extracted places and a filter mask.

    Don't forget to get the additional `get_publisher_mask` after merging df_places with df_base!
    Example usage:

    ```
    df_base = pd.read(main-data)
    df_places, mask_places = load_df_places(source)
    df = df_base.merge(df_places, on='search_name')
    ```

    """
    df = _read_places_df(source)
    df = fix_geographies(df)
    mask = get_naming_mask(df)
    return df, mask


def read_places_export(
    source: Path,
    df_countries: Optional[pd.DataFrame] = None,
    df_grid: Optional[pd.DataFrame] = None,
    resolution: float = 2.5,
    include_grid: bool = False,
) -> pd.DataFrame:
    df_countries = load_country_infos() if df_countries is None else df_countries
    df_places = _read_places_df(source, index_column=None, resolution=resolution)
    df = df_places.merge(df_countries, left_on='country_code3', right_on='iso3', how='left')

    if include_grid:
        df_grid = load_grid_data() if df_grid is None else df_grid
        df = df.merge(df_grid, left_on=['LAT', 'LON'], right_on=['LAT', 'LON'], how='outer')
    return df

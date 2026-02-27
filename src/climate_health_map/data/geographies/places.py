from pathlib import Path

import pandas as pd

from climate_health_map.shared import read_any_pd

EXCLUDED_SEARCH_NAMES = {
    'B.V.',
    'MMT',
    'Yellow',
    'Hadley',
    'Western North',
    'colonies',
    'TN',
    'NH',
    'Mn',
    'Tx',
    'TX',
    'Tn',
    'FL',
    'Spartina',
    'Tamarix',
    'Eurasia',
    'Phillyrea',
    'N-15',
    'LT50',
    'POSEIDON',
    'LC50',
    'El Nio',
    'La Nia',
    'Red',
    'Gulf Stream',
    'NH 1',
    'Quercus',
    'ZJP',
    'MSW',
    'CCS',
    'Tier-3',
    'N2O',
    'VKT',
    'OECD',
    'States',
    'North to South',
    'Stabilising',
    'Mass Railway',
    'City',
}
EXCLUDED_NAMES = {
    'Pacific County',
}


def get_naming_mask(place_df: pd.DataFrame) -> pd.Series:
    """Create mask to remove places that are, based on anecdotal evidence, not actually clean place names."""
    return ~place_df['search_name'].isin(EXCLUDED_SEARCH_NAMES) & ~place_df['name'].isin(EXCLUDED_NAMES) & (place_df['search_name'].str.len() > 2)


def get_publisher_mask(merged_df: pd.DataFrame) -> pd.Series:
    """Remove texts associated with certain publishers; input is df_base.merge(df_places) or similar"""
    pubs = [('S. Karger AG, Basel', ['Basel', 'Switzerland']), ('Licensee MDPI, Basel', ['Basel', 'Switzerland'])]
    mask = merged_df['item_id'].notna()
    for snippet, place in pubs:
        mask &= ~(merged_df['text'].str.contains(snippet) & merged_df['name'].isin(place))
    return mask


def fix_geographies(place_df: pd.DataFrame) -> pd.DataFrame:
    geocolumns = ['feature_code', 'lat', 'lon', 'name', 'feature_class', 'geonameid', 'country_code3']

    place_df.loc[place_df['search_name'] == 'Pakistan', geocolumns] = ['PCLI', 30, 70, 'Islamic Republic of Pakistan', 'A', 1168579, 'PAK']
    place_df.loc[place_df['search_name'] == 'Colombia', geocolumns] = ['PCLI', 4, -73.25, 'Colombia', 'A', 3686110, 'COL']
    place_df.loc[place_df['search_name'] == 'Argentina', geocolumns] = ['PCLI', -34, -64, 'Argentine Republic', 'A', 3865483, 'ARG']
    place_df.loc[place_df['search_name'] == 'East China', geocolumns] = ['PCLI', 35, 105, 'China', 'A', 1814991, 'CHN']
    place_df.loc[place_df['search_name'] == 'South China', geocolumns] = ['PCLI', 35, 105, 'China', 'A', 1814991, 'CHN']
    place_df.loc[place_df['search_name'] == 'Ireland', geocolumns] = ['PCLI', 53, -8, 'Ireland', 'A', 2963597, 'IRL']
    place_df.loc[place_df['search_name'] == 'United States', geocolumns] = ['PCLI', 39.76, -98.5, 'United States', 'A', 6252001, 'USA']
    place_df.loc[place_df['search_name'] == 'Czech Republic', geocolumns] = ['PCLI', 49.75, 15, 'Czechia', 'A', 3077311, 'CZE']
    place_df.loc[place_df['search_name'] == 'Czechia', geocolumns] = ['PCLI', 49.75, 15, 'Czechia', 'A', 3077311, 'CZE']
    place_df.loc[place_df['search_name'] == 'China', geocolumns] = ['PCLI', 35, 105, 'China', 'A', 1814991, 'CHN']
    place_df.loc[place_df['search_name'] == 'United Arab Emirates', geocolumns] = ['PCLI', 23.75, 54.5, 'United Arab Emirates', 'A', 290557, 'ARE']

    place_df.loc[place_df['search_name'] == 'Sahara', geocolumns] = ['DSRT', 26, 13, 'Sahara', 'T', 2212709, None]

    place_df.loc[place_df['search_name'] == 'Alps', geocolumns] = ['MTS', 46.41667, 10, 'Alps', 'T', 2661786, None]
    place_df.loc[place_df['search_name'] == 'Himalayan', geocolumns] = ['MTS', 28, 84, 'Himalayas', 'T', 1252558, None]
    place_df.loc[place_df['search_name'] == 'Himalayas', geocolumns] = ['MTS', 28, 84, 'Himalayas', 'T', 1252558, None]

    place_df.loc[place_df['search_name'] == 'Mediterranean Sea', geocolumns] = ['SEA', 35, 20, 'Mediterranean Sea', 'T', 2661786, None]
    place_df.loc[place_df['search_name'] == 'MEDITERRANEAN', geocolumns] = ['SEA', 35, 20, 'Mediterranean Sea', 'T', 2661786, None]
    place_df.loc[place_df['search_name'] == 'Red Sea', geocolumns] = ['SEA', 20.26735, 38.53455, 'Red Sea', 'H', 350155, None]
    place_df.loc[place_df['search_name'] == 'North Sea', geocolumns] = ['SEA', 55, 3, 'North Sea', 'P', 2960848, None]
    place_df.loc[place_df['search_name'] == 'Philippine Sea', geocolumns] = ['SEA', 20, 135, 'Philippine Sea', 'P', 1818190, None]
    place_df.loc[place_df['search_name'] == 'Black Sea', geocolumns] = ['SEA', 43, 34, 'Black Sea', 'H', 630673, None]
    place_df.loc[place_df['search_name'] == 'Coral Sea', geocolumns] = ['SEA', -20, 155, 'Coral Sea', 'H', 2194166, None]
    place_df.loc[place_df['search_name'] == 'Timor Sea', geocolumns] = ['SEA', -11, 127, 'Timor Sea', 'H', 2078065, None]
    place_df.loc[place_df['search_name'] == 'Bering Sea', geocolumns] = ['SEA', 60, -175, 'Bering Sea', 'H', 4031788, None]
    place_df.loc[place_df['search_name'] == 'Okhotsk Sea', geocolumns] = ['SEA', 55, 150, 'Sea of Okhotsk', 'H', 2127380, None]
    place_df.loc[place_df['search_name'] == 'Ionian Sea', geocolumns] = ['SEA', 39, 19, 'Ionian Sea', 'H', 2463713, None]

    place_df.loc[place_df['search_name'] == 'South Pacific', geocolumns] = ['OCN', -45, -130, 'South Pacific Ocean', 'H', 4030483, None]
    place_df.loc[place_df['search_name'] == 'Atlantic Ocean', geocolumns] = ['OCN', 10, -25, 'Atlantic Ocean', 'H', 3373405, None]
    place_df.loc[place_df['search_name'] == 'North Pacific', geocolumns] = ['OCN', 30, -170, 'North Pacific Ocean', 'H', 4030875, None]
    place_df.loc[place_df['search_name'] == 'Indian Ocean', geocolumns] = ['OCN', -10, 70, 'Indian Ocean', 'P', 1545739, None]

    place_df.loc[place_df['search_name'] == 'Great Lakes', geocolumns] = ['LK', 45.68751, -84.43753, 'Great Lakes', 'H', 4994594, 'USA']

    place_df.loc[place_df['search_name'] == 'Catalonia', geocolumns] = ['ADM1', 41.82046, 1.86768, 'Catalunya', 'A', 3336901, 'ESP']
    place_df.loc[place_df['search_name'] == 'California (USA', geocolumns] = ['ADM1', 37.25022, -119.75126, 'California', 'A', 5332921, 'USA']
    place_df.loc[place_df['search_name'] == 'California, USA', geocolumns] = ['ADM1', 37.25022, -119.75126, 'California', 'A', 5332921, 'USA']
    place_df.loc[place_df['name'] == 'Central Upper Nile', geocolumns] = ['ADM1', 10, 32.7, 'Upper Nile', 'A', 381229, 'SSD']

    place_df.loc[place_df['search_name'] == 'Gulf Coast', geocolumns] = ['AREA', 29.36901, -95.00565, 'Gulf Coast', 'L', 7287689, 'USA']
    place_df.loc[place_df['search_name'] == 'Gulf coast', geocolumns] = ['AREA', 29.36901, -95.00565, 'Gulf Coast', 'L', 7287689, 'USA']

    place_df.loc[place_df['search_name'] == 'Hainan Island', geocolumns] = ['ISL', 19.2, 109.7, 'Hainan Dao', 'T', 1809055, 'CHN']

    place_df.loc[place_df['search_name'] == "North America's", geocolumns] = ['CONT', 46.07323, -100.54688, 'North America', 'L', 6255149, None]

    place_df.loc[place_df['search_name'] == 'Scandinavia', geocolumns] = ['RGN', 63, 12, 'Scandinavia', 'L', 2614165, None]

    place_df.loc[place_df['search_name'] == 'Huai', geocolumns] = ['STM', 33.133333, 118.5, 'Huai He', 'H', 1807690, 'CHN']
    place_df.loc[place_df['search_name'] == 'Washington, DC', geocolumns] = ['PPLC', 38.89511, -77.03637, 'Washington', 'P', 4140963, 'USA']
    place_df.loc[place_df['search_name'] == 'Messinian', geocolumns] = ['ADM2', 37.25, -21.83333, 'Nomos Messinias', 'A', 257149, 'GRC']

    place_df.loc[place_df['search_name'] == 'NYC', geocolumns] = ['PPL', 40.71427, -74.00597, 'New York City', 'P', 5128581, 'USA']

    place_df.loc[place_df['search_name'] == 'Hudson Bay', geocolumns] = ['BAY', 60, -85, 'Hudson Bay', 'H', 5978134, 'CAN']

    return place_df


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
    df = read_any_pd(
        source,
        kwargs={'dtype': str, 'keep_default_na': False},  # keep_default_na handles cells that contain "NA" (which is valid)
        index_column=index_column,
    )
    df = fix_geographies(df)
    mask = get_naming_mask(df)
    return df, mask

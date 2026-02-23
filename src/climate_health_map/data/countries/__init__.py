"""Definitions of countries with ISO2/3 codes, names, and groupings.

https://en.wikipedia.org/wiki/ISO_3166-1_numeric
https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3
"""

from pathlib import Path
import pandas as pd

from .geoparser import mordecai
from .continents import CONTINENT_MAP
from .variations import country_names as COUNTRY_NORMALISATION

here = Path(__file__).parent.resolve()
countries = pd.read_csv(here / 'countries.csv', dtype=str).map(str.strip, na_action='ignore')

__all__ = [
    'countries',
    'mordecai',
    'CONTINENT_MAP',
    'COUNTRY_NORMALISATION',
]

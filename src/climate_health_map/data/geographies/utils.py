import logging
from typing import TYPE_CHECKING

import pandas as pd
import numpy as np

from climate_health_map.data import LABELS

if TYPE_CHECKING:
    from geopandas import GeoDataFrame

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
    'the Urban Heat Island',
    'Munich Re',
    'HKH',
    'WC-RC',
}
EXCLUDED_NAMES = {
    'Pacific County',
}
LOCATION_FEATURE_CODES = [
    {
        'PCLI',  # (56478) Independent political entity
        'ADM1',  # (20834) Primary administrative division of a country, such as a state in the United States
        'ADM1H',  # (49) Historical first-order administrative division (a former primary administrative division)
    },
    {
        'SEA',  # (606) Sea (Large body of salt water more or less confined by continuous land or chains of islands forming a subdivision of an ocean)
        'MTS',  # (477) Mountains (a mountain range or a group of mountains or high ridges)
        'PLN',  # (318) Extensive area of comparatively level to gently undulating land, lacking surface irregularities, and usually adjacent to a higher area
        'OCN',  # (305) Ocean (one of the major divisions of the vast expanse of salt water covering the earth)
        'DSRT',  # (84) Desert (a large area with little or no vegetation due to lack of precipitation)
        'GULF',  # Large recess in the coastline, larger than a bay
        'PLAT',  # Elevated plain with steep slopes on one or more sides, and often with incised streams
        'CHN',  # Deepest part of a stream, bay, lagoon, or strait, through which the main current flows
        'BSNU',  # Depression more or less equidimensional in plan and of variable extent
    },
    {
        'ADM2',  # (5156) Second-order administrative division (a subdivision of a first-order administrative division)
        'ADM3',  # (1730) Third-order administrative division (a subdivision of a second-order administrative division)
        'RGN',  # (3101) Region (an area with some type of unity, often administrative or economic)
        'STM',  # (1347) Stream (a body of running water moving to a lower level in a channel on land)
        'CONT',  # (576) Continent (one of the major land masses of the earth)
        'ISL',  # (501) Island (a tract of land, smaller than a continent, surrounded by water at high water)
        'PCLD',  # (345) Dependent political entity
        'MT',  # (271) Mountain (an elevation standing high above the surrounding area with small summit area)
        'LK',  # (262) Lake (a large inland body of standing water)
        'AREA',  # (200) Area (a tract of land without homogeneous character or boundaries)
        'BAY',  # (173) Bay (a coastal indentation between two capes or headlands)
        'PEN',  # (154) Peninsula (an elongate area of land projecting into a body of water)
        'TERR',  # (93) Territory (an area under the jurisdiction of a government)
        'ADMD',  # (86) Administrative division (undifferentiated as to administrative level)
        'PCLS',  # (43) Semi-independent political entity
        'ZN',  # (40) Zone (an area with a particular characteristic or function)
        'PCLIX',  # (36) Section of independent political entity
        'BSNU',  # (34) Undersea basin (a depression in the seafloor, more or less equidimensional)
        'ADM4H',  # (34) Historical fourth-order administrative division
        'PCLF',  # (30) Freely associated state
        'WAD',  # (13) Wadi (a valley or ravine that becomes a watercourse during the rainy season)
        'PCL',  # (12) Political entity (a country or similar entity)
        'CNL',  # (11) Canal (an artificial watercourse)
    },
    {
        'PPLA',  # (10727) Seat of a first-order administrative division (e.g., a state capital)
        'PPL',  # (8311) Populated place (a city, town, village, or other agglomeration of buildings)
        'PPLC',  # (8237) Capital of a political entity
        'PPLA2',  # (6188) Seat of a second-order administrative division
        'PPLX',  # (632) Section of a populated place
        'HTL',  # (232) Hotel (a building providing lodging and/or meals for the public)
        'PPLL',  # (177) Populated locality (an area or place of unspecified or mixed character)
        'VAL',  # (173) Valley (an elongated depression usually traversed by a stream)
        'AIRP',  # (159) Airport (a place where aircraft regularly land and take off, with major facilities)
        'RSTN',  # (158) Railroad station (a facility comprising ticket office, platforms, etc., for train stops)
        'PPLA3',  # (157) Seat of a third-order administrative division
        'LCTY',  # (153) Locality (a minor area or place of unspecified or mixed character)
        'ADM4',  # (141) Fourth-order administrative division
        'BLDG',  # (135) Building(s) (a structure erected by man as a whole or in part)
        'ISLS',  # (112) Islands (tracts of land surrounded by water)
        'DLTA',  # (110) Delta (the triangular area of alluvium at the mouth of a river)
        'PPLQ',  # (89) Abandoned populated place
        'RFU',  # (80) Refugee camp (a facility for people who have fled their country)
        'PLAT',  # (73) Plateau (an elevated plain with steep slopes on one or more sides)
        'PRK',  # (71) Park (an area, often of aromatic or scenic value, set aside for public recreation)
        'PPLA4',  # (63) Seat of a fourth-order administrative division
        'RGNE',  # (62) Economic region
        'CH',  # (60) Church (a building for public Christian worship)
        'FCL',  # (60) Facility (a building or place that provides a particular service or is used for a particular industry)
        'PK',  # (48) Peak (the pointed top of a mountain or other elevation)
        'HSP',  # (44) Hospital (a building in which the sick or injured are given medical or surgical care)
        'RSV',  # (33) Reservoir(s) (an artificial pond or lake)
        'SCH',  # (30) School (an institution for teaching and learning)
        'STRT',  # (25) Strait (a relatively narrow waterway connecting two larger bodies of water)
        'BDG',  # (23) Bridge (a structure erected across an obstacle such as a river or road)
        'UNIV',  # (22) University (an institution for higher learning)
        'MUS',  # (21) Museum (a building where objects of interest are stored and exhibited)
        'MALL',  # (20) Shopping mall (a large enclosed shopping complex)
        'HLL',  # (19) Hill (a rounded elevation of limited extent rising above the surrounding land)
        'CAPE',  # (19) Cape (a land area, more or less prominent, extending into a body of water)
        'CST',  # (19) Coast (a zone of variable width straddling the shoreline)
        'MFGB',  # (17) Manufacturing building (a building where goods are produced)
        'HMSD',  # (16) Homestead (a residence, including outbuildings and land)
        'HBR',  # (16) Harbor(s) (a haven of deep water sheltered by land to afford safe anchorage)
        'STMI',  # (16) Intermittent stream (a stream that flows only part of the time)
        'ADMF',  # (15) Administrative facility (a government building)
        'RD',  # (15) Road (an open way for the passage of vehicles, persons, or animals)
        'BCH',  # (15) Beach (the gently sloping shore of an ocean, lake, or river covered by sand or gravel)
        'MN',  # (14) Mine (a place where minerals are extracted from the ground)
        'FRM',  # (14) Farm (a tract of land used for agricultural purposes)
        'STMD',  # (12) Distributary (a branch which flows away from the main stream)
        'PND',  # (12) Pond (a small standing waterbody)
        'DPR',  # (12) Depression (a low area surrounded by higher ground)
        'TRANT',  # (12) Transit terminal (a facility for the transfer of passengers or goods)
        'RDGU',  # (11) Road junction (a place where two or more roads join or cross)
        'TOWR',  # (11) Tower (a tall, narrow structure)
        'MSQE',  # (11) Mosque (a building for public Islamic worship)
        'TRIG',  # (10) Triangulation station (a point on the earth whose position is determined by triangulation)
        'AIRF',  # (10) Airfield (a place on land where aircraft land and take off without commercial facilities)
        'PPLG',  # (10) Seat of government
        'SQR',  # (10) Square (a public open space in a town or city)
        'EST',  # (9) Estate (a large piece of privately owned land, often with a large house)
        'GULF',  # (9) Gulf (a large recess in the coastline, larger than a bay)
        'DAM',  # (8) Dam (a barrier constructed across a stream to impound water)
        'WLL',  # (8) Well (a cylindrical hole drilled or dug to access water, oil, or gas)
        'ST',  # (8) Street (a paved public road in a village, town, or city)
        'TRB',  # (8) Tribe (a political and social subdivision of a people)
        'CMP',  # (8) Camp (a site with temporary structures or tents)
        'INDS',  # (7) Industrial area (a tract of land used for industrial purposes)
    },
]


def get_naming_mask(place_df: pd.DataFrame) -> pd.Series:
    """Create mask to remove places that are, based on anecdotal evidence, not actually clean place names."""
    return (
        # ~place_df['name'].str.lower().str.strip().isin({name.lower().strip() for name in EXCLUDED_NAMES})
        # & ~place_df['search_name'].str.lower().str.strip().isin({name.lower().strip() for name in EXCLUDED_SEARCH_NAMES})
        (place_df['search_name'].str.len() > 2)
        & ~place_df['name'].str.strip().isin({name.strip() for name in EXCLUDED_NAMES})
        & ~place_df['search_name'].str.strip().isin({name.strip() for name in EXCLUDED_SEARCH_NAMES})
    )


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

    # Additional examples for potential fixes
    # > item_id  	name 	search_name 	feature_code 	country_code3 	lat 	lon 	start_char 	end_char 	geonameid
    # > | Republic of Zambia | Northern Egypt.(Table | PCLI | ZMB | \-14.33333 | 28.5 | 493 | 523 | 895949 |
    # > | Syrian Arab Republic | the Northern Delta Region | PCLI | SYR | 35.0 | 38.0 | 220 | 264 | 163843 |
    # > | 06521812-a904-403d-8ecd-501e2b9fdfe2 | Sunway | Sunway City | HTL | CHN | 22.64952 | 113.82104 | 1599 | 1617 | 9952455 |
    # > | b687e55e-8eb0-4e96-b20b-d5b3d489c791 | West Coast | the U.S. East Coast | ADM1 | NZL | \-42.45658 | 171.22081 | 52 | 85 | 6612113 |
    # > | b687e55e-8eb0-4e96-b20b-d5b3d489c791 | United States | the United States | PCLI | USA | 39.76 | \-98.5 | 146 | 174 | 6252001 |
    # > | b687e55e-8eb0-4e96-b20b-d5b3d489c791 | West Coast | the U.S. East Coast | ADM1 | NZL | \-42.45658 | 171.22081 | 1035 | 1068 | 6612113 |
    # > de5874e7-0389-4336-a9fb-a34b1a64bb39 	West Asia Hotel Shanghai 	West Asia 	HTL 	CHN 	31.1166 	121.3666 	375 	389 	6522100
    # > 5b6c8ffb-896a-4293-bbd5-271c552c83e4 	Bosnia and Herzegovina 	the Black Sea Region of 	PCLI 	BIH 	44.25 	17.83333 	606 	650 	3277605
    # > ee1d28bf-7e32-427a-9a41-5b9b29aea400 	Colorado 	East 23rd to 34th Streets 	ADM1 	USA 	39.00027 	-105.50083 	822 	865 	5417618
    # > ee1d28bf-7e32-427a-9a41-5b9b29aea400 	New York City 	NYC 	PPL 	USA 	40.71427 	-74.00597 	605 	608 	5128581
    # > 13ecac4d-d0c0-48e7-8398-7a30d74fcea8 	Republic of Namibia 	South and East Africa 	PCLI 	NAM 	-22.0 	17.0 	2124 	2145 	3355338
    # > f7ef8ef4-256d-4af2-8580-b2c86957cce4 	Sabah 	Sub-Sahara Africa 	ADM1 	MYS 	5.5 	117.0 	312 	340 	1733039
    # > 2d797363-894f-4267-bc75-c5e0b56502ab 	Tropics Hotel 	tropics 	HTL 	BGR 	42.71021 	27.75489 	997 	1004 	10120934
    # > 297d77a4-b264-4962-8a9c-fbb3bee2eb36 	Shatskiy Rise 	Northwest Pacific 	RISU 	NA 	35.0 	160.0 	150 	167 	2113215
    # > 5c1d7d12-a988-490e-8b5c-bfd954445bef 	Guangzhou 	Kiang Wu Hospital 	PPLA 	CHN 	23.11667 	113.25 	595 	621 	1809858
    # > bd3ad050-6e03-47b7-991f-286d6f91efa5 	Greater New Orleans Church 	Greater New Orleans 	CH 	USA 	29.97321 	-90.10179 	2062 	2093 	7533772
    # > bd3ad050-6e03-47b7-991f-286d6f91efa5 	Rapides Parish 	the Rapid Triage Area 	ADM2 	USA 	31.19862 	-92.5332 	1525 	1563 	4338356
    # > bd3ad050-6e03-47b7-991f-286d6f91efa5 	New Orleans 	East Jeff 	PPLA2 	USA 	29.95465 	-90.07507 	1768 	1782 	4335045
    # > b8dea545-191a-4521-9103-5f8429b99571 	Porto 	Porto 	ADM2 	PRT 	41.15556 	-8.62672 	1310 	1315 	6458924
    # > b8dea545-191a-4521-9103-5f8429b99571 	North America 	North 	CONT 	NA 	46.07323 	-100.54688 	1317 	1322 	6255149
    # > 1da02a15-c3e2-4503-8251-92e576149b87 	Republic of Cabo Verde 	Cape Hedo 	PCLI 	CPV 	16.0 	-24.0 	900 	914 	3374766
    # > aa9ae2e1-7499-4ab9-bbb8-d1c190508167 	Port of Spain 	Port of Spain 	PPLC 	TTO 	10.66668 	-61.51889 	1105 	1126 	3573890
    # > c6331fbf-641c-4b56-b77f-e0b32089853e 	Ehime 	MEA 	ADM1 	JPN 	33.61667 	132.85 	607 	610 	1864226
    # > 2c7f3ccb-8fc0-4899-a4be-de0b1e1a897e 	United States Virgin Islands 	THE UNITED STATES 	PCLD 	VIR 	18.34829 	-64.98348 	57 	85 	4796775
    # > 04fd36b2-b45e-4b64-bd25-41ef44f16bb0 	Southern District 	Southern Hemisphere 	ADM1 	ISR 	30.66667 	34.83333 	994 	1022 	294952
    # > 04fd36b2-b45e-4b64-bd25-41ef44f16bb0 	Pohjois-Pohjanmaa 	Northern Hemisphere 	ADM1 	FIN 	65.0 	26.41667 	1110 	1138 	830667
    # > b1b534a8-0de6-4656-b66b-bddd4f00fb98 	Bairin Youqi 	Guanzhong Plain 	ADM3 	CHN 	43.55045 	118.95073 	101 	116 	2038538
    # > b1b534a8-0de6-4656-b66b-bddd4f00fb98 	Great Dividing Range 	The Guanzhong Plain 	MTS 	AUS 	-25.0 	147.0 	146 	165 	2164625
    # > b1b534a8-0de6-4656-b66b-bddd4f00fb98 	Bairin Zuoqi 	the Guanzhong Plain 	ADM3 	CHN 	44.18835 	119.24984 	564 	583 	2038537
    # > b1b534a8-0de6-4656-b66b-bddd4f00fb98 	Chang’an Qu 	the Chang'an District 	ADM3 	CHN 	34.03702 	108.93366 	594 	615 	1815800
    # > b1b534a8-0de6-4656-b66b-bddd4f00fb98 	Dihok 	Hu County 	ADM1 	IRQ 	37.06667 	43.13333 	620 	629 	97270
    # >
    return place_df


def flatten_country_groups(df: pd.DataFrame, prefix: str) -> tuple[pd.DataFrame, dict[str, list[tuple[str, str, str]]]]:
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

    if 'item_id' not in df.columns:
        df.reset_index(inplace=True)

    # "Melt" the dataframe so columns become a single categorical variable
    df_melted = df.melt(id_vars=['item_id'], value_vars=cols, var_name='column', value_name='grouping')
    # Drop NaNs once globally
    df_melted = df_melted.dropna(subset=['grouping'])
    # Perform a single GroupBy + Size (Vectorized value_counts)
    counts = df_melted.groupby(['item_id', 'column', 'grouping']).size()
    # Reshape to get specific naming convention /  Unstack 'column' and 'grouping' into the header
    result = counts.unstack(level=[1, 2]).fillna(0).astype(int)

    # Fix column names to match '{prefix}_{column}|{grouping}' format and build lookup
    groups: dict[str, list[tuple[str, str, str]]] = {}
    renamed_columns = {f'{prefix}_{col}|{label.name}': label.column for col in cols for label in LABELS[f'{prefix}_{col}'].labels}
    for col, grp in result.columns:
        column = f'{prefix}_{col}|{grp}'
        groups.setdefault(col, []).append((grp, column, renamed_columns[column]))
    result.columns = [f'{prefix}_{col}|{grp}' for col, grp in result.columns]
    result.index.name = 'item_id'
    return result.rename(columns=renamed_columns), groups


def join_shapes(df_locations: pd.DataFrame, shapes: 'GeoDataFrame', keep_columns: list[str]) -> 'GeoDataFrame':
    from shapely.geometry import Point
    import geopandas as gpd

    mask = df_locations['lat'].notna() & df_locations['lon'].notna()
    return (
        gpd.GeoDataFrame(
            df_locations[mask][keep_columns + ['lat', 'lon', 'item_id']].assign(
                geometry=lambda table: table.apply(lambda row: Point(row['lon'], row['lat']), axis=1),
            ),
            # df[mask].apply(lambda row: Point(row['lon'], row['lat']), axis=1).reset_index(name='geometry'),
            crs='EPSG:4326',
        )
        .reset_index(drop=True)  # drop if switching back to single column
        .sjoin(shapes, how='left', predicate='within')
        .astype({'index_right': 'Int32'})
        .rename(columns={'index_right': 'shape_id'})
    )


def join_grid_shapes(df_grid: pd.DataFrame, shapes: 'GeoDataFrame', resolution: float = 2.5) -> 'GeoDataFrame':
    from shapely.geometry import Polygon
    import geopandas as gpd

    HR = resolution / 2  # half resolution
    return (
        gpd.GeoDataFrame(
            df_grid.apply(
                lambda row: Polygon(
                    [
                        (row['LON'] - HR, row['LAT'] - HR),
                        (row['LON'] - HR, row['LAT'] + HR),
                        (row['LON'] + HR, row['LAT'] + HR),
                        (row['LON'] + HR, row['LAT'] - HR),
                    ],
                ),
                axis=1,
            ).reset_index(name='geometry'),
            crs='EPSG:4326',
        )
        .sjoin(shapes, how='left', predicate='intersects')
        .astype({'index_right': 'Int32'})
        .rename(columns={'index_right': 'shape_id', 'index': 'grid_id'})
    )


def merge_grid_info(
    df: pd.DataFrame,
    df_locations: pd.DataFrame,
    df_grid: pd.DataFrame,
    df_population: pd.DataFrame,
    shapes: 'GeoDataFrame',
    location_codes_abstracted: set[str],
    location_codes_direct: set[str],
    logger: logging.Logger,
    reference_year: int = 2024,
    resolution: float = 2.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    location_mask = df_locations['incl_major'] & df_locations['incl_impacts'] & df_locations['incl_location']
    mask_abstracted = df_locations['feature_code'].isin(location_codes_abstracted)
    mask_direct = df_locations['feature_code'].isin(location_codes_direct)

    # Match dataframes to shapes
    points_locations = join_shapes(df_locations[location_mask], shapes=shapes, keep_columns=['location_id']).set_index('item_id')

    # Match df_grid to shapes
    points_grid = join_grid_shapes(df_grid, shapes=shapes, resolution=resolution)
    # Merge in [reference_year] population based on ISO code (note, some shape files might not have `ISO_A3`, then you need to edit this
    points_grid = points_grid.merge(
        df_population[df_population['year'] == reference_year][['iso3', 'Population']],
        left_on='ISO_A3',
        right_on='iso3',
        how='left',
    )
    # Some cells might span multiple iso codes/shapes, collapse things back down and take average population
    points_grid = points_grid.reset_index(drop=True).groupby('grid_id').agg({**{col: 'first' for col in points_grid.columns}, 'Population': 'mean'})

    # Match low-level locations to grid
    locations_direct = df_locations[mask_direct].merge(df_grid.reset_index(drop=True), left_on=['LAT', 'LON'], right_on=['LAT', 'LON'])[
        ['grid_id', 'item_id', 'location_id']
    ]

    # Get locations that have a high-level feature code and are matched to a shape
    locations_abstracted = points_locations[
        points_locations.index.isin(df_locations[mask_abstracted].index) & points_locations['shape_id'].notna()
    ].reset_index()[['item_id', 'shape_id', 'location_id']]
    # Match high-level locations to shape-matched grid
    locations_abstracted = points_grid[points_grid['shape_id'].notna()][['grid_id', 'shape_id']].merge(
        locations_abstracted,
        left_on='shape_id',
        right_on='shape_id',
    )

    logger.info(
        f'Found {locations_abstracted["item_id"].nunique():,} records in {locations_abstracted["shape_id"].nunique():,} shapes / '
        f'{locations_abstracted["grid_id"].nunique():,} grid cells (matched shape entries: {locations_abstracted.shape[0]:,} / '
        f'{(location_mask & mask_abstracted).sum():,})',
    )
    logger.info(
        f'Found {locations_direct["item_id"].nunique():,} records in {locations_direct["grid_id"].nunique():,} grid cells '
        f'(filtered locations: {mask_direct.sum():,})',
    )

    # Count number of studies with specific location per grid cell and normalise by grid cell population
    counts_direct = df_grid[['grid_id', 'population']].merge(
        locations_direct.groupby('grid_id')['item_id'].nunique().rename('study_count'),
        left_on='grid_id',
        right_on='grid_id',
        how='left',
    )
    counts_direct['studies_per_capita'] = (
        counts_direct.replace({0: np.nan})
        .apply(lambda row: row['study_count'] / row['population'], axis=1)
        .replace(
            {pd.NA: np.nan},
        )
        .astype('float')
    )
    counts_direct.drop(columns=['population', 'grid_id'], inplace=True)
    counts_direct.rename_axis('grid_id', inplace=True)

    # Count number of studies with region/abstracted location per grid cell and normalise by df_population
    counts_abstracted = locations_abstracted.groupby('grid_id')['item_id'].nunique().rename('study_count')
    counts_abstracted = points_grid[['grid_id', 'Population']].set_index('grid_id').join(counts_abstracted)
    counts_abstracted['studies_per_capita'] = counts_abstracted['study_count'] / counts_abstracted['Population']
    counts_abstracted.drop(columns='Population', inplace=True)

    # Merge both counts and per capita values
    counts = df_grid.join(counts_direct, how='left').join(counts_abstracted, how='left', lsuffix='_direct', rsuffix='_abstracted')
    counts['study_count'] = (counts['study_count_direct'].fillna(0) + counts['study_count_abstracted'].fillna(0)).replace({0: np.nan})
    counts['studies_per_capita'] = (counts['studies_per_capita_direct'].fillna(0) + counts['studies_per_capita_abstracted'].fillna(0)).replace({0: np.nan})

    df_locations = df_locations.merge(
        pd.concat([locations_abstracted, locations_direct])
        .merge(df_grid, left_on='grid_id', right_on='grid_id', how='left')
        .groupby('location_id')[['grid_cooler', 'grid_warmer', 'grid_wetter', 'grid_drier', 'grid_attributable', 'is_land']]
        .any()
        .reset_index(),
        left_on='location_id',
        right_on='location_id',
        how='left',
    )

    df = df.join(
        pd.concat([locations_abstracted, locations_direct])
        .merge(df_grid, left_on='grid_id', right_on='grid_id', how='left')
        .groupby('item_id')[['grid_cooler', 'grid_warmer', 'grid_wetter', 'grid_drier', 'grid_attributable', 'is_land']]
        .any(),
        how='left',
    )

    return df, df_locations, counts

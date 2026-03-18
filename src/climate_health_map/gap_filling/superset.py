import logging
import re
from pathlib import Path
from typing import Annotated

import typer
from nacsos_data.util.academic.apis import OpenAlexAPI
from tqdm import tqdm

from climate_health_map.shared.env import base_essentials

# Term lookup:
# http://10.10.12.41:8983/solr/#/openalex/query?q=*:*&q.op=AND&defType=lucene&indent=true&fl=id&rows=100&facet=true&terms.fl=title_abstract&terms.limit=100&terms.stats=true&terms.ttf=true&terms.prefix=hydroclim&useParams=&qt=%2Fterms

# TODO: verify precipitat*
# TODO: verify drought*
# TODO: verify flood*
# TODO: verify storm*
CLIMATE_QUERY = """
       climate
    OR "global warming"
    OR "greenhouse effect"
    OR "greenhouse gas"
    OR "greenhouse emission"
    OR GHG
    OR temperature
    OR precipitation
    OR rainfall
    OR "heat index"
    OR "heat indices"
    OR "extreme heat event"
    OR "heat wave"
    OR heatwave
    OR "extreme cold"
    OR "cold index"
    OR "cold indices"
    OR humidity
    OR drought
    OR hydroclimate
    OR hydroclimatic
    OR hydroclimatology
    OR hydroclimatological
    OR monsoon
    OR "el nino"
    OR ENSO
    OR "sea surface temperature"
    OR SST
    OR snowmelt
    OR flood
    OR storm
    OR cyclone
    OR hurricane
    OR typhoon
    OR "sea level"
    OR wildfire
    OR "wild fire"
    OR "forest fire"
    OR (
        disaster
        AND (
            risk
            OR management
            OR manage
            OR managing
            OR natural
        )
    )
    OR (
        (
            extreme
            AND event
        )
        NOT paleo
    )
    OR (
        (
               hydrochloroflourocarbon
            OR pm2.5
            OR ammonia
            OR nox
            OR HFC
            OR SO4
            OR carbon
            OR n20
            OR halogen
            OR chlorocarbon
            OR nh3
            OR SOX
            OR O3
            OR ccl4
            OR NMVOC
            OR SO2
            OR HFC
            OR CO
            OR nitrous
            OR methane
            OR ch4
            OR co2
            OR sulphur
            OR VOC
            OR ozone
            OR chlorocarbons
        )
        AND (
            emit
            OR mitigate
            OR emission
            OR mitigation
        )
    )
"""

# TODO: check infect*
# TODO: check hospital*
# TODO: check parasit*
# TODO: check bronchi*
# TODO: check psycho*
HEALTH_QUERY = """
       health
    OR wellbeing
    OR "well being"
    OR ill
    OR illness
    OR disease
    OR syndrome
    OR infect
    OR infection
    OR infectious
    OR medical
    OR mortality
    OR DALY
    OR morbidity
    OR injury
    OR death
    OR hospital
    OR hospitalization
    OR hospitalisation
    OR accident
    OR accidental
    OR emergency
    OR emergent
    OR doctor
    OR GP
    OR obesity
    OR obese
    OR overweight
    OR "over weight"
    OR underweight
    OR "under weight"
    OR hunger
    OR stunting
    OR wasting
    OR undernourishment
    OR undernourish
    OR undernutrition
    OR anthropometric
    OR anthropometry
    OR malnutrition
    OR malnourishment
    OR malnourish
    OR anemia
    OR anaemia
    OR "micro nutrient"
    OR hypertension
    OR "blood pressure"
    OR stroke
    OR renovascular
    OR cerebrovascular
    OR "heart disease"
    OR cardiovascular
    OR "cardio vascular"
    OR "heart attack"
    OR ischemic
    OR ischaemic
    OR coronary
    OR CHD
    OR diabetes
    OR diabetic
    OR CKD
    OR renal
    OR cancer
    OR kidney
    OR lithogenesis
    OR lithogenes
    OR skin
    OR fever
    OR feverish
    OR renal
    OR rash
    OR eczema
    OR eczematous
    OR "thermal stress"
    OR hyperthermia
    OR hyperthermic
    OR hypothermia
    OR hypothermic
    OR preterm
    OR stillbirth
    OR "birth weight"
    OR LBW
    OR maternal
    OR pregnant
    OR pregnancy
    OR gestation
    OR "pre eclampsia"
    OR preeclampsia
    OR sepsis
    OR oligohydramnios
    OR encephalitis
    OR encephalitic
    OR placenta
    OR haemorrhage
    OR hemorrhage
    OR malaria
    OR dengue
    OR mosquito
    OR chikungunya
    OR leishmaniasis
    OR "vector borne"
    OR pathogen
    OR zoonosis
    OR zoonose
    OR zika
    OR "west nile"
    OR onchocerciasis
    OR filiariasis
    OR waterborne
    OR diarrhoeal
    OR diarrheal
    OR gastrointestinal
    OR gastroenterology
    OR gastroesophageal
    OR "vibrio bacteria"
    OR cyanobacteria
    OR parasitic
    OR borrelia
    OR paralysis
    OR paralyzed
    OR paralytic
    OR paralysed
    OR neurotoxic
    OR neurotoxicity
    OR neurotoxin
    OR viral
    OR rotavirus
    OR noravirus
    OR hantavirus
    OR cholera
    OR protozoa
    OR protozoan
    OR protozoal
    OR lyme
    OR "tick borne"
    OR salmonella
    OR giardia
    OR shigella
    OR campylobacter
    OR "food borne"
    OR aflatoxin
    OR poison
    OR poisonous
    OR ciguatera
    OR respiratory
    OR allergic
    OR allergen
    OR allergy
    OR lung
    OR asthma
    OR asthmatic
    OR bronchial
    OR bronchitis
    OR pulmonary
    OR COPD
    OR rhinitis
    OR wheezing
    OR wheeze
    OR mental
    OR depression
    OR depressive
    OR depressed
    OR anxiety
    OR PTSD
    OR psychological
    OR psychosocial
    OR psychometric
    OR psychotic
    OR suicide
    OR "pre trauma"
    OR pretrauma
    OR "post trauma"
    OR posttrauma
     OR (
            enteric
        NOT (
               fermentation
            OR "enteric CH4"
            OR "enteric methane"
        )
    )
    OR (
            heat
        AND (
               stress
            OR fatigue
            OR burn
            OR stroke
            OR exhaustion
            OR cramp
        )
        NOT cattle
    )
    OR (
        CVD
        NOT (
               vapor
            OR vapour
        )
    )
"""

QUERY = re.sub(r'\s+', ' ', f'({CLIMATE_QUERY}) AND ({HEALTH_QUERY})').replace('( ', '(').replace(' )', ')')


def get_openalex_ids(
    config: Annotated[Path, typer.Option(help='Path to config file')],
    target: Annotated[Path, typer.Option(help='Path to write IDs to')],
    loglevel: Annotated[str, typer.Option(help='Verbosity of logger')] = 'DEBUG',
) -> None:
    if target.exists():
        raise FileExistsError(f'File {target.name} already exists.')

    logger, settings = base_essentials(config, logger_name='openalex', loglevel=loglevel, run_log_init=True)
    logger.debug(QUERY)
    logger.info(f'Running query with {len(QUERY):,} characters')
    api_logger = logger.getChild('api')
    api_logger.setLevel(logging.WARNING)
    api = OpenAlexAPI(api_key=settings.OPENALEX.API_KEY, logger=api_logger)

    results = api.fetch_raw(
        QUERY,
        params={
            'select': 'id',
            'filter': (
                # Do not include xpac records
                f'is_xpac:false'
                # Only include non-open-access records (the others are highly likely to have an abstract)
                f',is_oa:false'
                # Filter for publication year
                f',publication_year:>2024'
                # FIXME: use `from_created_date:2024-10-01` for more accurate filtering (needs premium or institutional key)
                f',title_and_abstract.search:{QUERY}'
                # f'title_and_abstract.search.no_stem: {query}'  # raw/no stemming
                # FIXME: supposedly `title_and_abstract.search` is deprecated, but `search` param throws error 500
                # f',search:{QUERY}'
                # Published by springer or elsevier
                # f',primary_location.source.publisher_lineage:p4310320990|p4310319965'
            ),
        },
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, 'w') as ids_file:
        row = next(results)
        ids_file.write(f'{row["id"]}\n')
        for row in tqdm(results, total=int((api.meta or {}).get('count', 0)) - 1):
            ids_file.write(f'{row["id"]}\n')


# On NACSOS: 1,290,164 (in Oct 2024)
#
# Trivial query adaptation
# 1,065,881 query only
#   559,515 not oa
#    90,043 springer|elsevier
#    11,371 not oa & springer|elsevier
#
# after replacing wildcards and simplifying
# 1,334,707 query only
#   709,750 not oa
#   112,715 springer|elsevier
#    16,506 not oa & springer|elsevier
#
# Estimated query time:
# 1,334,707 records, 200 per page -> ~6674 pages
# max 10 requests per second -> ~667 seconds -> ~11min
#
#
# NACSOS database:
# SELECT academic_item.openalex_id
# FROM academic_item
#      JOIN m2m_import_item m2mii on academic_item.item_id = m2mii.item_id
# WHERE m2mii.import_id = '5b18344c-6c30-40ba-92ad-48a2136efe6b';
#
# SELECT DISTINCT ai.openalex_id
# FROM academic_item ai
#      JOIN m2m_import_item m2mii ON ai.item_id = m2mii.item_id
#      JOIN annotation a ON a.item_id = ai.item_id
# WHERE m2mii.import_id = '5b18344c-6c30-40ba-92ad-48a2136efe6b'
#   AND a.key = 'relevant' AND a.value_bool = TRUE;
#
# Evaluate overlaps:
# ids = {}
# with open('data/climate_health_ids.txt') as f:
#     ids['OpenAlex'] = set(f.readlines())
# with open('data/climate_health_ids_db.txt') as f:
#     ids['NACSOS'] = set(f.readlines())
# with open('data/climate_health_ids_db_rel.txt') as f:
#     ids['NACSOS (rel)'] = set(f.readlines())
# with open('data/climate_health_pred_incl.txt') as f:
#     ids['Predicted'] = set(f.readlines())
# for k1, v1 in ids.items():
#     print(f'{k1} has {len(v1):,} records')
#     for k2, v2 in ids.items():
#         if k1 != k2:
#             print(f'  {k2} has {len(v2):,} records')
#             print(f'   > Union: {len(v1 | v2):,} '
#                   f'| intersect: {len(v1 & v2):,} '
#                   f'| {k1} exclusive: {len(v1 - v2):,} '
#                   f'| {k2} exclusive {len(v2 - v1):,}')

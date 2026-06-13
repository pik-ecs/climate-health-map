from collections import OrderedDict
from pathlib import Path
from typing import Annotated

import typer
from nacsos_data.util.academic.apis import OpenAlexSolrAPI

from climate_health_map.shared.env import base_essentials

# Used by Finn et al
#    * in https://iopscience.iop.org/article/10.1088/2515-7620/ae0099/meta
#    * works on dimensions
# Derived from https://www.nature.com/articles/s41558-019-0684-5
#    Web of Science query (see methods):
#    (SO=(Climate Alert OR Climate Dynamics OR Climate Policy OR Climatic Change OR Global and Planetary Change OR Global Change Biology OR International Journal of Greenhouse Gas Control OR Mitigation and Adaptation Strategies for Global Change) OR TS=(((CO2 OR “carbon dioxide” OR methane OR CH4 OR “carbon cycle” OR “carbon cycles” OR “carbon cycling” OR “carbon budget*” OR “carbon flux*” OR “carbon mitigation”) AND (climat*)) OR ((“carbon cycle” OR “carbon cycles” OR “carbon cycling” OR “carbon budget*” OR “carbon flux*” OR “carbon mitigation”) AND (atmospher*))) OR TS=(“carbon emission*” OR “sequestration of carbon” OR “sequester* carbon” OR “sequestration of CO2” OR “sequester* CO2” OR “carbon tax*” OR “CO2 abatement” OR “CO2 capture” OR “CO2 storage” OR “CO2 sequester*” OR “CO2 sequestration” OR “CO2 sink*” OR “anthropogenic carbon” OR “captur* of carbon dioxide” OR “captur* of CO2” OR “climat* variability” OR “climat* dynamic*” OR “chang* in climat*” OR “climat* proxies” OR “climat* proxy” OR “climat* sensitivity” OR “climat* shift*” OR “coupled ocean-climat*” OR “early climat*” OR “future climat*” OR “past climat*” OR “shift* climat*” OR “shift in climat*”) OR TS=(“atmospheric carbon dioxide” OR “atmospheric CH4” OR “atmospheric CO2” OR “atmospheric methane” OR “atmospheric N2O” OR “atmospheric nitrous oxide” OR “carbon dioxide emission*” OR “carbon sink*” OR “CH4 emission*” OR “climat* policies” OR “climat* policy” OR “CO2 emission*” OR dendroclimatolog* OR (“emission* of carbon dioxide” NOT nanotube*) OR “emission* of CH4” OR “emission* of CO2” OR “emission* of methane” OR “emission* of N2O” OR “emission* of nitrous oxide” OR “historical climat*” OR IPCC OR “methane emission*” OR “N2O emission*” OR “nitrous oxide emission*”) OR TS=(“climat* change*” OR “global warming” OR “greenhouse effect” OR “greenhouse gas*” OR “Kyoto Protocol” OR “warming climat*” OR “cap and trade” OR “carbon capture” OR “carbon footprint*” OR “carbon neutral” OR “carbon offset” OR “carbon sequestration” OR “carbon storage” OR “carbon trad*” OR “changing climat*” OR “climat* warming”)) NOT PY=2019
# Alternative for "all literature" growth rates:
#    * https://www.nature.com/articles/s41599-021-00903-w
query = """
(((CO2 OR "carbon dioxide" OR methane OR CH4 OR "carbon cycle" OR "carbon cycles" OR "carbon cycling" OR
    "carbon budget" OR "carbon budgets" OR "carbon flux" OR "carbon fluxes" OR "carbon mitigation") AND
    (climate OR climatology OR climatic OR climates OR climatological)) OR

(("carbon cycle" OR "carbon cycles" OR "carbon cycling" OR "carbon budget" OR "carbon budgets" OR
    "carbon flux" OR "carbon fluxes" OR "carbon mitigation") AND
    (atmosphere OR atmospheres OR atmospheric OR atmospherical OR atmospherically)) OR

("carbon emission" OR "carbon emissions" OR "sequestration of carbon" OR "sequestered carbon" OR "sequestering carbon" OR
    "sequestration of CO2" OR "sequestered CO2" OR "sequestering CO2" OR "carbon tax" OR "carbon taxes" OR "CO2 abatement" OR
    "CO2 capture" OR "CO2 storage" OR "CO2 sequestration" OR "CO2 sink" OR "CO2 sinks" OR "anthropogenic carbon" OR
    "capture of carbon dioxide" OR "capture of CO2" OR "climate variability" OR "climatic variability" OR
    "climate dynamics" OR "change in climate" OR "change in climatic" OR "climate proxies" OR "climate proxy"
    OR "climate sensitivity" OR "climate shift" OR "climatic shift" OR "coupled ocean-climate" OR "early climate" OR
    "future climate" OR "past climate" OR "shifting climate" OR "shifting climatic" OR "shift in climate" OR "shift in climatic") OR

("atmospheric carbon dioxide" OR "atmospheric CH4" OR "atmospheric CO2" OR "atmospheric methane" OR "atmospheric N2O" OR
    "atmospheric nitrous oxide" OR "carbon dioxide emission" "carbon dioxide emissions" OR "carbon sink" OR "carbon sinks"
    OR "CH4 emission" OR "CH4 emissions" OR "climate policies" OR "climate policy" OR "CO2 emissions" OR "CO2 emission" OR
    dendroclimatology OR dendroclimatological OR ("emission of carbon dioxide" NOT (nanotube OR nanotubes)) OR
    ("emissions of carbon dioxide" NOT (nanotube OR nanotubes)) OR "emission of CH4" OR "emissions of CH4" OR
    "emission of CO2" OR "emissions of CO2" OR "emission of methane" OR "emissions of methane" OR "emission of N2O"
    OR "emissions of N20" OR "emission of nitrous oxide" OR "emissions of nitrous oxide" OR "historical climate" OR
    "historical climatic" OR IPCC OR "Intergovernmental Panel on Climate Change" OR "methane emission" OR
    "methane emissions" OR "N2O emission" OR "N20 emissions" OR "nitrous oxide emission" OR "nitrous oxide emissions") OR

("climate change" OR "climatic change" OR "climate changes" OR "climatic changes" OR "global warming" OR
    "greenhouse effect" OR "greenhouse gas" OR "greenhouse gases" OR "Kyoto Protocol" OR "warming climate" OR
    "warming climatic" OR "cap and trade" OR "carbon capture" OR "carbon footprint" OR "carbon footprints" OR
    "carbon neutral" OR "carbon neutrality" OR "carbon offset" OR "carbon sequestration" OR "carbon storage" OR
    "carbon trading" OR "carbon trade" OR "changing climate" OR "changing climatic" OR "climate warming" OR "climatic warming"))
"""


def main(
    config: Annotated[Path, typer.Option(help='Path to config file')],
    loglevel: Annotated[str, typer.Option(help='Loglevel')] = 'INFO',
) -> None:

    logger, settings = base_essentials(config=config, loglevel=loglevel, logger_name='export', run_log_init=True)

    api = OpenAlexSolrAPI(
        openalex_conf=settings.OPENALEX,
        include_histogram=True,
        histogram_from=1990,
        histogram_to=2027,
    )

    logger.info('Requesting CC counts')
    result_cc = api.query(query=query, limit=1, params={'is_xpac': False})
    logger.info(f'  -> Found {result_cc.num_found:,} records')

    logger.info('Requesting OA counts')
    result_oa = api.query(query='*', limit=1, params={'is_xpac': False})
    logger.info(f'  -> Found {result_oa.num_found:,} records')

    result = {}
    for yr, cnt in result_cc.histogram.items():
        result[int(yr)] = {'CC': result_cc.histogram[yr], 'OA': result_oa.histogram[yr]}

    print(result)


if __name__ == '__main__':
    typer.run(main)

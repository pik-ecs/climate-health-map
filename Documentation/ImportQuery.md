# Data import
Currently based on self-hosted openalex in solr
Planning to transition to [API](https://developers.openalex.org/guides/searching) with subsequent abstract fixing

Import via web interface seems to fail for some reason. Alternatively, try to run via nacsos package command:
```bash
uv run  --no-sources-package nacsos_data nacsos import SOLR --import-id 5b18344c-6c30-40ba-92ad-48a2136efe6b --source src/climate_health_map/data/query/query_20241029.txt --project-id 52f62f17-cefb-4152-bcca-2fcb77541e83 --config-file /data/nacsos2/nacsos-core/config/server.env
```
Note, this needs milvus, connection to database on se164 and to solr on srv-mcc-apsis (10.10.12.41)

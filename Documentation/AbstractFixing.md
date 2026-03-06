```bash
export UV_NO_SOURCES_PACKAGE="nacsos_data openalex_ingest"

# Ask OpenAlex API which abstracts we might be missing abstracts for
uv run healthmap abstracts fetch-ids --config config/secret.env --target data/gap_filling/ids.csv
# Check that list of OpenAlex IDs against our local snapshot and prepare a list to queue for abstract fixing
uv run healthmap abstracts check-ids --config config/secret.env --source data/gap_filling/ids.csv --target data/gap_filling/dois.csv
# Queue OpenAlex-ID/DOI pairs to meta-cache
uv run healthmap abstracts queue-ids --config config/secret.env --source data/gap_filling/dois.csv --sources DIMENSIONS,TRY --sources SCOPUS,TRY

# if you are not in the PIK network, forward a port, e.g. `ssh -N -D 1080 foote`
# the following assumes, that you are the sole user of the queue at the moment and can filter by date
# Let the abstract worker loose on the queue
uv run healthmap abstracts fetch-abstracts --config config/secret.env --sources DIMENSIONS --sources SCOPUS --created-after 2026-03-04 --created-before 2026-03-06

# Write abstracts to solr
uv run healthmap abstracts transfer-abstracts --batch-size 1000 --config config/secret.env --created-after 2026-03-04 --created-before 2026-03-07
```

Assuming you can filter by time, use this to see how many abstracts to retrieved
```sql
SELECT count(1) as                                                                     total,
       count(DISTINCT queue_id)                                                        num_queue_ids,
       count(1) filter ( where abstract IS NOT NULL )                                  has_abstract,
       count(1) filter ( where abstract IS NULL )                                      has_no_abstract,
       count(1) filter ( where abstract IS NOT NULL AND lower(wrapper) = 'scopus')     has_abstract_scopus,
       count(1) filter ( where abstract IS NOT NULL AND lower(wrapper) = 'dimensions') has_abstract_dimensions
FRom request
where time_created > '2026-03-01';
```

In case the queries are blocking:
```sql
-- Check what's going on
SELECT pid AS process_id,
query AS active_query,
query_start
FROM pg_stat_activity
WHERE state = 'active';

-- kill a query
SELECT pg_cancel_backend(1661578);
```
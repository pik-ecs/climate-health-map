-- PROJECT_ID: str = '52f62f17-cefb-4152-bcca-2fcb77541e83'  # Climate and Health Literature Landscape
-- MAJOR_SCHEME_ORIG: str = '7dd7ac85-f472-4d95-ae63-a41de368ad95'  # Original annotations
-- MAJOR_SCHEME_NEW: str = 'fbb0d34b-3232-48ab-9d9d-78164692e4e9'  # Climate and health major categories
-- IMPACTS_SCHEME: list[str] = [
--     '9cadee95-14c2-4790-b39f-7125f1f63346',  # Health impacts of climate change version 2
--     'a8aba098-44e3-471a-87de-3d1a07c2b7ba',  # Health impacts of climate change (soon to be archived and translated/relabelled into 2)
-- ]
-- IMPORTS: list[str] = [
--     '5b18344c-6c30-40ba-92ad-48a2136efe6b',  # oa climate health v4
-- ]

-- Number of records with mordecai info
-- Count as of Mar 05, 2026: 101,443
SELECT count(distinct e.item_id)
FROM enhancement e
     JOIN item i ON i.item_id = e.item_id
     JOIN m2m_import_item ii ON i.item_id = ii.item_id
WHERE i.project_id::text = '52f62f17-cefb-4152-bcca-2fcb77541e83'
  AND ii.import_id::text = '5b18344c-6c30-40ba-92ad-48a2136efe6b'
  AND e.key = 'mordecai3'
  AND e.payload IS NOT NULL;

-- Unrolled mordecai places
WITH
    unrolled AS (
        SELECT e.item_id,
               jsonb_array_elements(e.payload) as place
        FROM enhancement e
             JOIN item i ON i.item_id = e.item_id
             JOIN m2m_import_item ii ON i.item_id = ii.item_id
        WHERE i.project_id::text = '52f62f17-cefb-4152-bcca-2fcb77541e83'
          AND ii.import_id::text = '5b18344c-6c30-40ba-92ad-48a2136efe6b'
          AND e.key = 'mordecai3'
          AND e.payload IS NOT NULL)
SELECT item_id::text,
       place ->> 'lat'           AS lat,
       place ->> 'lat'           AS lat,
       place ->> 'lon'           AS lon,
       place ->> 'name'          AS name,
       place ->> 'score'         AS score,
       place ->> 'city_id'       AS city_id,
       place ->> 'end_char'      AS end_char,
       place ->> 'city_name'     AS city_name,
       place ->> 'geonameid'     AS geonameid,
       place ->> 'start_char'    AS start_char,
       place ->> 'admin1_code'   AS admin1_code,
       place ->> 'admin1_name'   AS admin1_name,
       place ->> 'search_name'   AS search_name,
       place ->> 'feature_code'  AS feature_code,
       place ->> 'country_code3' AS country_code3,
       place ->> 'feature_class' AS feature_class
FROM unrolled;

-- Number of xpac records
-- num_xpac,num_total (2026-03-06)
-- 201,322    1,723,893
SELECT count(distinct ai.item_id) FILTER (
    WHERE ai.meta -> 'openalex' ->> 'is_xpac' is not null
        AND (ai.meta -> 'openalex' ->> 'is_xpac')::bool = TRUE) as num_xpac,
       count(distinct ai.item_id)                               as num_total
FROM academic_item ai
     JOIN m2m_import_item ii ON ai.item_id = ii.item_id
WHERE ai.project_id::text = '52f62f17-cefb-4152-bcca-2fcb77541e83'
  AND ii.import_id::text = '5b18344c-6c30-40ba-92ad-48a2136efe6b';

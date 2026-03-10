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

-- Clear all labels in project (except mordecai)
DELETE
FROM enhancement e
WHERE e.item_id IN (
    SELECT i.item_id from item i WHERE i.project_id::text = '52f62f17-cefb-4152-bcca-2fcb77541e83')
  AND key = ANY
      (array ['rel_major|1', 'rel_major|0', 'cat|0', 'cat|1', 'cat|2', 'type|0', 'type|1', 'type|2', 'type|3', 'type|4', 'type|5', 'cont|0', 'cont|1', 'cont|2', 'cont|3', 'cont|4', 'cont|5', 'cont|6', 'rel_impacts|1', 'rel_impacts|0', 'driver|0', 'driver|1', 'driver|2', 'driver|3', 'driver|4', 'driver|5', 'driver|7', 'driver|6', 'event|0', 'event|1', 'event|2', 'event|3', 'event|6', 'event|4', 'event|5', 'health|0', 'health|1', 'health|2', 'health|3', 'health|4', 'health|5', 'health|6', 'health|7', 'health|8', 'health|9', 'health|10', 'health|11', 'health|12', 'expose|0', 'expose|1', 'expose|2', 'expose|3', 'expose|4', 'attr|0', 'attr|1', 'attr|2', 'attr|3', 'attr|4', 'gender_outcome|1', 'gender_outcome|0', 'sector|0', 'sector|1', 'sector|2', 'sector|3', 'sector|4', 'sector|5', 'sector|6', 'topic-0-0|26', 'topic-0-0|29', 'topic-0-0|43', 'topic-0-1|11', 'topic-0-1|37', 'topic-0-1|45', 'topic-0-1|64', 'topic-0-1|68', 'topic-0-2|35', 'topic-0-2|47', 'topic-0-2|58', 'topic-0-3|65', 'topic-0-4|1', 'topic-0-4|44', 'topic-0-5|21', 'topic-0-5|31', 'topic-0-5|36', 'topic-0-5|51', 'topic-0-5|52', 'topic-0-6|17', 'topic-1-0|25', 'topic-1-0|61', 'topic-1-1|3', 'topic-1-1|24', 'topic-1-1|67', 'topic-1-2|5', 'topic-1-2|46', 'topic-1-3|7', 'topic-1-3|9', 'topic-1-3|10', 'topic-1-3|40', 'topic-1-4|0', 'topic-1-4|4', 'topic-1-4|6', 'topic-1-4|14', 'topic-1-4|22', 'topic-1-4|55', 'topic-1-4|56', 'topic-1-4|60', 'topic-1-4|66', 'topic-1-5|39', 'topic-1-5|48', 'topic-1-6|27', 'topic-1-6|30', 'topic-1-7|19', 'topic-1-8|20', 'topic-1-8|28', 'topic-1-8|41', 'topic-1-8|59', 'topic-1-8|62', 'topic-1-9|50', 'topic-2-0|53', 'topic-2-1|63', 'topic-2-2|13', 'topic-2-2|23', 'topic-2-3|33', 'topic-3-0|15', 'topic-3-1|2', 'topic-3-1|12', 'topic-3-1|16', 'topic-3-1|69', 'topic-3-2|42', 'topic-4-0|8', 'topic-4-0|18', 'topic-4-0|32', 'topic-4-0|34', 'topic-4-0|38', 'topic-4-0|49', 'topic-4-0|54', 'topic-4-0|57', 'topic-agg-0|0', 'topic-agg-0|1', 'topic-agg-0|2', 'topic-agg-0|3', 'topic-agg-0|4', 'topic-agg-0|5', 'topic-agg-0|6', 'topic-agg-1|0', 'topic-agg-1|1', 'topic-agg-1|2', 'topic-agg-1|3', 'topic-agg-1|4', 'topic-agg-1|5', 'topic-agg-1|6', 'topic-agg-1|7', 'topic-agg-1|8', 'topic-agg-1|9', 'topic-agg-2|0', 'topic-agg-2|1', 'topic-agg-2|2', 'topic-agg-2|3', 'topic-agg-3|0', 'topic-agg-3|1', 'topic-agg-3|2', 'topic-agg-4|0', 'topic-agg-agg|0', 'topic-agg-agg|1', 'topic-agg-agg|2', 'topic-agg-agg|3', 'topic-agg-agg|4']);


SELECT *
FROM academic_item ai
LEFT JOIN LATERAL (
    SELECT string_agg(elem->>'name', ', ') AS names
    FROM jsonb_array_elements(ai.authors) AS elem
) AS author_stuff ON true
LIMIT 10;
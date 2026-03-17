SELECT ai.publication_year, count(distinct ai.item_id)
FROM enhancement e
     JOIN academic_item ai ON ai.item_id = e.item_id
     JOIN m2m_import_item ii ON ai.item_id = ii.item_id
WHERE e.key = 'mordecai3'
  AND ai.project_id = '52f62f17-cefb-4152-bcca-2fcb77541e83'
  AND ii.import_id = '5b18344c-6c30-40ba-92ad-48a2136efe6b'
GROUP BY ai.publication_year;

SELECT ai.publication_year, count(distinct ai.item_id)
FROM enhancement e
     JOIN academic_item ai ON ai.item_id = e.item_id
     JOIN m2m_import_item ii ON ai.item_id = ii.item_id
WHERE e.key = 'mordecai3'
  AND ai.project_id = '52f62f17-cefb-4152-bcca-2fcb77541e83'
  AND ii.import_id = '5b18344c-6c30-40ba-92ad-48a2136efe6b'
GROUP BY ai.publication_year;

-- +----------------+------+
-- |publication_year|count |
-- +----------------+------+
-- |2002            |718   |
-- |2003            |833   |
-- |2004            |1031  |
-- |2005            |1363  |
-- |2006            |1803  |
-- |2007            |1947  |
-- |2008            |2319  |
-- |2009            |2853  |
-- |2010            |3051  |
-- |2011            |3935  |
-- |2012            |3960  |
-- |2013            |5122  |
-- |2014            |5316  |
-- |2015            |5909  |
-- |2016            |6546  |
-- |2017            |6407  |
-- |2018            |7571  |
-- |2019            |8546  |
-- |2020            |10807 |
-- |2021            |12280 |
-- |2022            |13083 |
-- |2023            |15354 |
-- |2024            |115526|
-- |2025            |103283|
-- |2026            |2151  |
-- +----------------+------+

SELECT jsonb_array_length(payload), payload
FROM enhancement
WHERE key = 'mordecai3';
SELECT ai.publication_year,
       count(distinct ai.item_id)                                                                     as total,
       count(distinct e.item_id) filter ( where e.key = 'mordecai3' )                                 as has_mordecai,
       count(distinct e.item_id) filter ( where e.key = 'mordecai3' and payload IS NOT NULL)          as mordecai_notna,
       sum(jsonb_array_length(e.payload)) filter ( where e.key = 'mordecai3' and payload IS NOT NULL) as sum_len
FROM academic_item ai
     JOIN m2m_import_item ii ON ai.item_id = ii.item_id
     LEFT OUTER JOIN enhancement e ON ai.item_id = e.item_id
WHERE ai.project_id = '52f62f17-cefb-4152-bcca-2fcb77541e83'
  AND ii.import_id = '5b18344c-6c30-40ba-92ad-48a2136efe6b'
  AND ai.publication_year > 1990
GROUP BY ai.publication_year;
-- +----------------+------+------------+--------------+-------+
-- |publication_year|total |has_mordecai|mordecai_notna|sum_len|
-- +----------------+------+------------+--------------+-------+
-- |2014            |56334 |5316        |4197          |18659  |
-- |2015            |63815 |5909        |4580          |19824  |
-- |2016            |67840 |6546        |5082          |20801  |
-- |2017            |62070 |6407        |5066          |22057  |
-- |2018            |68231 |7571        |5947          |25649  |
-- |2019            |76307 |8546        |6690          |29976  |
-- |2020            |92309 |10807       |8327          |37864  |
-- |2021            |101228|12280       |9248          |40446  |
-- |2022            |105471|13083       |9727          |40889  |
-- |2023            |108599|15354       |11199         |46724  |
-- |2024            |117051|115526      |55769         |284939 |
-- |2025            |104510|103283      |47191         |196305 |
-- |2026            |2190  |2151        |1102          |4668   |
-- +----------------+------+------------+--------------+-------+
-- 2014.0     8237
-- 2015.0     9086
-- 2016.0     9577
-- 2017.0     9627
-- 2018.0    11468
-- 2019.0    14290
-- 2020.0    17045
-- 2021.0    17455
-- 2022.0    17658
-- 2023.0    18645
-- 2024.0    16611
-- 2025.0     2753
-- 2014.0    1852
-- 2015.0    1996
-- 2016.0    2221
-- 2017.0    2235
-- 2018.0    2635
-- 2019.0    3086
-- 2020.0    3677
-- 2021.0    3941
-- 2022.0    4086
-- 2023.0    4353
-- 2024.0    4301
-- 2025.0    5588


SELECT COUNT(1),
       COUNT(distinct item_id)
FROM (
         SELECT e.item_id,
                jsonb_array_elements(e.payload) as place
         FROM enhancement e
              JOIN item i ON i.item_id = e.item_id
              JOIN m2m_import_item ii ON i.item_id = ii.item_id
         WHERE i.project_id = '52f62f17-cefb-4152-bcca-2fcb77541e83'
           AND ii.import_id = '5b18344c-6c30-40ba-92ad-48a2136efe6b'
           AND e.key = 'mordecai3'
           AND e.payload IS NOT NULL
           AND e.payload <> '[]'::jsonb) a;
-- +-------+-------+
-- |count  |count  |
-- +-------+-------+
-- |912,321|202,277|
-- +-------+-------+


WITH
    unrolled AS (
        SELECT e.item_id,
               jsonb_array_elements(e.payload) as place
        FROM enhancement e
             JOIN item i ON i.item_id = e.item_id
             JOIN m2m_import_item ii ON i.item_id = ii.item_id
        WHERE i.project_id = '52f62f17-cefb-4152-bcca-2fcb77541e83'
          AND ii.import_id = '5b18344c-6c30-40ba-92ad-48a2136efe6b'
          AND e.key = 'mordecai3'
          AND e.payload IS NOT NULL
          AND e.payload <> '[]'::jsonb)
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

WITH
    unrolled as (
        SELECT e.item_id,
               ai.publication_year,
               jsonb_array_elements(e.payload) as place
        FROM enhancement e
             JOIN enhancement incl
                  ON e.item_id = incl.item_id AND incl.key = 'rel_major|1' AND incl.payload::float > 0.5
             JOIN academic_item ai ON ai.item_id = e.item_id
             JOIN m2m_import_item ii ON ai.item_id = ii.item_id
        WHERE ai.project_id = '52f62f17-cefb-4152-bcca-2fcb77541e83'
          AND ii.import_id = '5b18344c-6c30-40ba-92ad-48a2136efe6b'
          AND ai.publication_year > 2020
          AND e.key = 'mordecai3')
-- SELECT publication_year,
--        count(distinct item_id)                                                        as cnt,
--        count(distinct item_id) filter ( where place ->> 'country_code3' IS NOT NULL ) as cnt3
-- FROM unrolled
-- GROUP BY publication_year;
SELECT *, place ->> 'country_code3'
FROM unrolled
WHERE place ->> 'country_code3' IS NULL;

SELECT *
FROM enhancement
WHERE item_id = '009fef2e-6ea4-4d12-85f4-2dd50643017c'
  and key = 'mordecai3';


SELECT e.item_id,
       (
           SELECT jsonb_agg(elems)
           FROM jsonb_array_elements(e.payload) AS sublists,
                jsonb_array_elements(sublists) AS elems) AS new_payload
FROM enhancement e
     JOIN academic_item ai ON ai.item_id = e.item_id
     JOIN m2m_import_item ii ON ai.item_id = ii.item_id
WHERE ai.project_id = '52f62f17-cefb-4152-bcca-2fcb77541e83'
  AND ii.import_id = '5b18344c-6c30-40ba-92ad-48a2136efe6b'
  AND e.key = 'mordecai3'
  AND jsonb_typeof(e.payload -> 0) = 'array';

-- For some reason, some mordecai results are nested list[list[obj]]
-- This flattens those into a list of objects
-- WITH flattened_data AS (
--     SELECT
--         e.item_id,
--         (SELECT jsonb_agg(elems)
--          FROM jsonb_array_elements(e.payload) AS sublists,
--               jsonb_array_elements(sublists) AS elems) AS new_payload
--     FROM enhancement e
--     JOIN academic_item ai ON ai.item_id = e.item_id
--     JOIN m2m_import_item ii ON ai.item_id = ii.item_id
--     WHERE ai.project_id = '52f62f17-cefb-4152-bcca-2fcb77541e83'
--       AND ii.import_id = '5b18344c-6c30-40ba-92ad-48a2136efe6b'
--       AND e.key = 'mordecai3'
--       AND jsonb_typeof(e.payload->0) = 'array'
-- )
-- UPDATE enhancement
-- SET payload = fd.new_payload
-- FROM flattened_data fd
-- WHERE enhancement.item_id = fd.item_id
--   AND enhancement.key = 'mordecai3';
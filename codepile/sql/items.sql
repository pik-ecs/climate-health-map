SELECT ai.*
FROM academic_item ai
     JOIN m2m_import_item m2m ON ai.item_id = m2m.item_id
WHERE import_id = '5b18344c-6c30-40ba-92ad-48a2136efe6b'
  AND ai.meta -> 'openalex' ->> 'type' = 'libguides'
LIMIT 10;

-- +-----------------------+-------+
-- |kind                   |count  |
-- +-----------------------+-------+
-- |article                |1322060|
-- |book                   |12985  |
-- |book-chapter           |55420  |
-- |book-section           |5      |
-- |database               |9      |
-- |dataset                |25031  |
-- |dissertation           |56344  |
-- |editorial              |1228   |
-- |erratum                |203    |
-- |grant                  |165    |
-- |journal-article        |24     |
-- |letter                 |1313   |
-- |libguides              |130    |
-- |other                  |80948  |
-- |paratext               |2512   |
-- |peer-review            |5234   |
-- |posted-content         |1      |
-- |preprint               |12849  |
-- |reference-entry        |994    |
-- |report                 |8965   |
-- |report-component       |19     |
-- |retraction             |4      |
-- |review                 |4954   |
-- |software               |14     |
-- |standard               |337    |
-- |supplementary-materials|2      |
-- |null                   |155845 |
-- +-----------------------+-------+
SELECT ai.meta -> 'openalex' ->> 'type' as kind, count(distinct ai.item_id)
FROM academic_item ai
     JOIN m2m_import_item m2m ON ai.item_id = m2m.item_id
WHERE import_id = '5b18344c-6c30-40ba-92ad-48a2136efe6b'
GROUP BY kind;


-- +---------------+-----+
-- |kind           |count|
-- +---------------+-----+
-- |article        |62712|
-- |book           |606  |
-- |book-chapter   |3253 |
-- |book-section   |1    |
-- |database       |3    |
-- |dataset        |1833 |
-- |dissertation   |2205 |
-- |editorial      |93   |
-- |erratum        |15   |
-- |grant          |42   |
-- |letter         |69   |
-- |libguides      |25   |
-- |other          |4375 |
-- |paratext       |96   |
-- |peer-review    |270  |
-- |preprint       |1515 |
-- |reference-entry|69   |
-- |report         |559  |
-- |review         |624  |
-- |standard       |5    |
-- |null           |5877 |
-- +---------------+-----+
WITH
    target_items AS (
        SELECT e.item_id, meta
        FROM m2m_import_item m2m
             JOIN enhancement e ON e.item_id = m2m.item_id AND e.key = 'rel_major|1' AND e.payload::float > 0.5
             JOIN academic_item ai ON ai.item_id = e.item_id
        WHERE import_id = '5b18344c-6c30-40ba-92ad-48a2136efe6b'),
    classifications AS (
        SELECT e.item_id,
               MAX(payload::float) FILTER (WHERE key = 'rel_major|1') AS "rel_major|1",
               MAX(payload::float) FILTER (WHERE key = 'cat|0')       AS "cat|0",
               MAX(payload::float) FILTER (WHERE key = 'cat|1')       AS "cat|1",
               MAX(payload::float) FILTER (WHERE key = 'cat|2')       AS "cat|2"
        FROM enhancement e
             JOIN target_items ti ON e.item_id = ti.item_id
        GROUP BY e.item_id)
SELECT ti.meta -> 'openalex' ->> 'type' as kind, count(distinct ti.item_id)
FROM classifications c
     JOIN target_items ti ON ti.item_id = c.item_id
GROUP BY kind;



-- +-----+--------+-----+-----+----------+-----------+
-- |rel  |not_xpac|is_en|mai  |valid_type|all_filters|
-- +-----+--------+-----+-----+----------+-----------+
-- |84247|71825   |83245|73793|81567     |60805      |
-- +-----+--------+-----+-----+----------+-----------+
WITH
    target_items AS (
        SELECT e.item_id
        FROM m2m_import_item m2m
             JOIN enhancement e ON e.item_id = m2m.item_id AND e.key = 'rel_major|1' AND e.payload::float > 0.5
        WHERE import_id = '5b18344c-6c30-40ba-92ad-48a2136efe6b'),
    classifications AS (
        SELECT e.item_id,
               MAX(payload::float) FILTER (WHERE key = 'rel_major|1') AS "rel_major|1",
               MAX(payload::float) FILTER (WHERE key = 'cat|0')       AS "cat|0",
               MAX(payload::float) FILTER (WHERE key = 'cat|1')       AS "cat|1",
               MAX(payload::float) FILTER (WHERE key = 'cat|2')       AS "cat|2"
        FROM enhancement e
             JOIN target_items ti ON e.item_id = ti.item_id
        GROUP BY e.item_id
        HAVING COUNT(e.item_id) > 3),
    query AS (
        SELECT (ai.meta -> 'openalex' ->> 'is_xpac' IS NULL OR
                (ai.meta -> 'openalex' ->> 'is_xpac')::bool = FALSE) AS not_xpac,
               (ai.meta -> 'openalex' ->> 'language' IS NULL OR
                ai.meta -> 'openalex' ->> 'language' = 'en')         AS is_en,
               (ai.meta -> 'openalex' ->> 'type' IS NULL OR
                ai.meta -> 'openalex' ->> 'type' = ANY (array ['article',
                    'book-chapter',
                    'dataset',
--                     'dissertation',
                    'preprint',
                    'other',
                    'book',
                    'review',
                    --                     'paratext',
--                     'libguides',
--                     'letter',
                    'report',
                    'reference-entry',
--                     'peer-review',
                    'editorial',
--                     'erratum',
                    'standard',
                    'supplementary-materials',
                    --                     'retraction',
--                     'software',
                    'database',
                    'book-section',
                    'report-component',
                    'grant']))                                       AS valid_type,
               ("cat|0" > 0.5
                   OR "cat|1" > 0.5
                   OR "cat|2" > 0.5)                                 as mai
        FROM academic_item ai
             JOIN target_items ti ON ai.item_id = ti.item_id
             JOIN classifications c ON ai.item_id = c.item_id
        WHERE ai.project_id = '52f62f17-cefb-4152-bcca-2fcb77541e83'
          AND (ai.meta -> 'openalex' ->> 'source_id' IS NULL OR
               ai.meta -> 'openalex' ->> 'source_id' <> 'S7407052681'))
SELECT count(1)                                                              rel,
       count(1) filter ( where not_xpac )                                    not_xpac,
       count(1) filter ( where is_en )                                       is_en,
       count(1) filter ( where mai )                                         mai,
       count(1) filter ( where valid_type )                                  valid_type,
       count(1) filter ( where not_xpac AND is_en AND mai AND valid_type) as all_filters
from query;
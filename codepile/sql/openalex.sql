select count(1)
from (
         SELECT ai.item_id,
                i.time_edited,
                ai.meta -> 'openalex' -> 'authorships',
                ai.openalex_id,
                meta -> 'openalex'
         FROM academic_item ai
              JOIN item i ON i.item_id = ai.item_id
              JOIN m2m_import_item ii ON ai.item_id = ii.item_id
              JOIN enhancement incl ON i.item_id = incl.item_id AND incl.key = 'rel_major|1' AND payload::float > 0.5
         WHERE ai.project_id = '52f62f17-cefb-4152-bcca-2fcb77541e83'
           AND ii.import_id = '5b18344c-6c30-40ba-92ad-48a2136efe6b'
--   AND ai.publication_year = 2024
           AND ai.meta -> 'openalex' -> 'authorships' IS NULL) a;

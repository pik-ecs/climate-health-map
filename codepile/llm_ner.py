from ollama import chat
import json

TEXT = """
The study area includes Berlin, Alabama.

This study is applicable to areas with similar political climate for example Tokyo has similar conditions.
"""

PROMPT = f"""
You are an expert geographic entity extraction system for scientific literature.

Your task is to extract ALL geographic entities mentioned in the text and classify their contextual role.

Return ONLY valid JSON.

OUTPUT SCHEMA:

{{
  "locations": [
    {{
      "name": "string",
      "feature_type": "string",
      "admin_type": "string or null",
      "country": "string or null",
      "admin_region": "string or null",
      "latitude": "number or null",
      "longitude": "number or null",
      "confidence": "number between 0 and 1",
      "reason_for_mention": "string",
      "source_text": "exact source snippet"
    }}
  ]
}}

FIELD DEFINITIONS:

- name:
  Canonical place name inferred from context.

- feature_type:
  Geographic feature category.

  Allowed values:
  - city
  - country
  - region
  - river
  - lake
  - mountain
  - forest
  - island
  - administrative_area
  - protected_area
  - climate_region
  - continent
  - ocean
  - sea
  - unknown

- admin_type:
  Administrative level/type if applicable.

  Allowed values:
  - capital_city
  - state
  - province
  - county
  - municipality
  - district
  - region
  - federal_state
  - territory
  - prefecture
  - null

- country:
  Infer country from context when confidence is high.
  Example:
  "Berlin" near Germany → Germany
  If ambiguous, use null.

- admin_region:
  Administrative region/state/province if known.

- latitude / longitude:
  Use coordinates only if explicitly stated or confidently inferred.
  Otherwise use null.

- confidence:
  Confidence score between 0 and 1.

  Guidelines:
  - 0.95+ explicit mention with coordinates
  - 0.8+ strong contextual inference
  - <0.6 ambiguous references

- reason_for_mention:
  Contextual role of the geographic entity.

  Allowed values:
  - Study_Subject
  - Study_Location
  - Example_Mention
  - Comparison_Location
  - Background_Context
  - Climate_Reference
  - Methodology_Context
  - Data_Source
  - Related_Region
  - Historical_Context
  - Unknown

- source_text:
  Exact minimal text span that triggered extraction.

DISAMBIGUATION RULES:

- Resolve ambiguous place names using nearby context.
- Prefer the geographically relevant interpretation.
- Example:
  "Berlin" mentioned with Germany or European climate context
  → Berlin, Germany
- Do NOT hallucinate locations.
- If uncertain, lower confidence and use null fields.

STRICT RULES:

- Return ONLY valid JSON
- Do NOT explain
- Do NOT use markdown
- Every object MUST contain ALL fields
- Missing values MUST be null
- Use only allowed enum values where specified
- Extract all geographic entities even if incidental


1. Explicit geographic qualifiers ALWAYS override world knowledge.

Examples:
- "Paris, Texas" → Paris in Texas, USA
- "Athens, Georgia" → Athens in Georgia, USA

2. If a city is followed by a state/province/region name,
infer the country from the administrative region.

3. NEVER resolve a city to a famous global location if a more specific local qualifier exists.

4. Administrative regions take precedence over popularity.

5. If the administrative region uniquely identifies a country:
- Alabama → USA
- Bavaria → Germany
- Ontario → Canada

6. Lower confidence when ambiguity remains.

7. Use source text literally before applying semantic inference.

TEXT:
{TEXT}
"""

schema = {
    'type': 'object',
    'properties': {
        'locations': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                    'feature_type': {'type': 'string'},
                    'admin_type': {'type': ['string', 'null']},
                    'country': {'type': ['string', 'null']},
                    'admin_region': {'type': ['string', 'null']},
                    'latitude': {'type': ['number', 'null']},
                    'longitude': {'type': ['number', 'null']},
                    'confidence': {'type': 'number'},
                    'reason_for_mention': {'type': 'string'},
                    'source_text': {'type': 'string'},
                },
                'required': [
                    'name',
                    'feature_type',
                    'admin_type',
                    'country',
                    'admin_region',
                    'latitude',
                    'longitude',
                    'confidence',
                    'reason_for_mention',
                    'source_text',
                ],
            },
        }
    },
    'required': ['locations'],
}
response = chat(model='qwen2.5-coder:7b', messages=[{'role': 'user', 'content': PROMPT}], format=schema, options={'temperature': 0})

content = response['message']['content']

# print(content)

data = json.loads(content)

print(json.dumps(data, indent=2))

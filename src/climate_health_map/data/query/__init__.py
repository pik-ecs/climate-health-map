from pathlib import Path

here = Path(__file__).parent.resolve()
with open(here / 'query_20241029.txt', 'r') as f:
    query = f.read()

__all__ = [
    'query',
]

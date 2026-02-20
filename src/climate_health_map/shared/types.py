from enum import Enum


class OnConflict(str, Enum):
    # Pretend we never checked for conflicts and just run with it (and possibly overwrite)
    IGNORE = 'IGNORE'
    # When we check for potential conflicts, skip task if output exists (larger pipeline will keep running)
    SKIP = 'SKIP'
    # When we check for potential conflicts, fail task if output exists (larger pipeline will break running)
    BREAK = 'BREAK'

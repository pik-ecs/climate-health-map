from enum import Enum


class OnConflict(str, Enum):
    IGNORE = 'IGNORE'
    BREAK = 'BREAK'
    SKIP = 'SKIP'

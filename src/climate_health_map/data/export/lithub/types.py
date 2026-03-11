from datetime import date
from typing import Literal, Annotated

from pydantic import BaseModel, ConfigDict, AfterValidator


class SchemeLabel(BaseModel):
    key: str  # Column in the database
    name: str  # Humanly readable name
    value: int | str  # label value (e.g. technology = 1)
    colour: tuple[float, float, float]  # label colour in HSL
    desc: str | None = None  # Description


class SchemeGroup(BaseModel):
    name: str
    key: str
    type: Literal['single', 'bool', 'multi']
    colour: tuple[float, float, float] | None = None  # if this is a subgroup, this needs to be set
    desc: str | None = None  # Description

    labels: list[str] | None = None  # list of SchemeLabel.key (if empty, will be inferred from subgroups)
    subgroups: list[str] | None = None  # list of SchemeGroup.key


class DatasetInfo(BaseModel):
    model_config = ConfigDict(extra='ignore')

    name: str
    teaser: str

    authors: list[str] | None = None
    contributors: list[str] | None = None

    created_date: date
    last_update: date

    figure: str | None = None

    hidden: bool = False  # if True, dataset will not be displayed in the frontend


class _DatasetInfoFull(DatasetInfo):
    model_config = ConfigDict(extra='ignore')
    db_filename: str
    arrow_filename: str
    keywords_filename: str | None = None

    slim_geo_filename: str | None = None
    full_geo_filename: str | None = None

    start_year: int = 1990
    end_year: int = 2025

    labels: dict[str, SchemeLabel]
    groups: dict[str, SchemeGroup]

    default_colour: str


class DatasetInfoFull(_DatasetInfoFull):
    model_config = ConfigDict(extra='ignore')
    contact: list[str] | None = None


class DatasetInfoWeb(DatasetInfoFull):
    model_config = ConfigDict(extra='ignore')
    key: str
    total: int
    columns: set[str]
    label_columns: set[str]
    document_columns: set[str]


class Document(BaseModel):
    idx: int
    title: str | None = None
    abstract: str | None = None
    publication_year: int | None = None
    openalex_id: str | None = None
    nacsos_id: str | None = None
    doi: str | None = None
    authors: str | None = None
    institutions: str | None = None


# This way, floats are rounded to two decimals to save space in transit
RoundedFloat = Annotated[float, AfterValidator(lambda f: round(f, 2))]


class AnnotatedDocument(Document):
    manual: bool = False  # True, iff labels are from manual annotation
    labels: dict[str, RoundedFloat]

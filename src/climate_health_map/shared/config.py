import os
from typing import Any
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from nacsos_data.util.conf import OpenAlexConfig, DatabaseConfig


class Settings(BaseSettings):
    DB: DatabaseConfig = DatabaseConfig()  # NACSOS-core database
    OPENALEX: OpenAlexConfig = OpenAlexConfig()  # OpenAlex (solr/api) config

    LOG_CONF_FILE: str = 'config/logging.toml'
    LOGGING_CONF: dict[str, Any] | None = None

    PROJECT_ID: str = '52f62f17-cefb-4152-bcca-2fcb77541e83'  # Climate and Health Literature Landscape
    MAJOR_SCHEME_ORIG: str = '7dd7ac85-f472-4d95-ae63-a41de368ad95'  # Original annotations
    MAJOR_SCHEME_NEW: str = 'fbb0d34b-3232-48ab-9d9d-78164692e4e9'  # Climate and health major categories
    IMPACTS_SCHEME: list[str] = [
        '9cadee95-14c2-4790-b39f-7125f1f63346',  # Health impacts of climate change version 2
        'a8aba098-44e3-471a-87de-3d1a07c2b7ba',  # Health impacts of climate change (soon to be archived and translated/relabelled into 2)
    ]
    IMPORTS: list[str] = [
        '5b18344c-6c30-40ba-92ad-48a2136efe6b',  # oa climate health v4
    ]

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_prefix='NACSOS_',
        env_nested_delimiter='__',
        extra='allow',
    )


def load_settings(conf_file: Path | str | None = None) -> Settings:
    if conf_file is None:
        conf_file = os.environ.get('CH_CONFIG', 'config/default.env')
    if not Path(conf_file).is_file():
        raise FileNotFoundError(f'Configuration file not found: {conf_file}')
    return Settings(_env_file=conf_file, _env_file_encoding='utf-8')

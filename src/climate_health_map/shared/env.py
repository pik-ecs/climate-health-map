import logging
from pathlib import Path
from nacsos_data.db import DatabaseEngine, get_engine
from climate_health_map.shared.config import Settings, load_settings


def get_logger(logger_name: str, run_log_init=True, loglevel: str = 'INFO') -> logging.Logger:
    if run_log_init:
        logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s (%(process)d): %(message)s', level=loglevel)
        logging.getLogger('elasticsearch').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('filelock').setLevel(logging.WARNING)

        logging.getLogger('root').setLevel(loglevel)

    logger = logging.getLogger(logger_name)
    logger.setLevel(loglevel)

    return logger


def essentials(config: Path, logger_name: str, run_log_init=True, loglevel: str = 'INFO') -> tuple[logging.Logger, Settings, DatabaseEngine]:
    logger = get_logger(logger_name=logger_name, run_log_init=run_log_init, loglevel=loglevel)

    logger.info(f'Loading config from {config.resolve()}...')
    if not config.exists():
        raise AssertionError(f'Config file does not exist at {config.resolve()}!')
    settings = load_settings(config)

    logger.info('Connecting to database...')
    db_engine = get_engine(settings=settings.DB)
    return logger, settings, db_engine


__all__ = ['essentials', 'get_logger']

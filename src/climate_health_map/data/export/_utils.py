from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import sqlalchemy as sa

from climate_health_map.shared import essentials
from climate_health_map.shared.types import OnConflict


class ExportContext:
    def __init__(
        self,
        config: Path,
        target: Path,
        params: dict[str, Any] | None = None,
        batch_size: int = 1000,
        project_id: str | None = None,
        import_ids: str | None = None,
        on_exists: OnConflict = OnConflict.IGNORE,
        loglevel: str = 'INFO',
    ):
        self.logger, self.settings, self.db_engine = essentials(config=config, loglevel=loglevel, logger_name='export', run_log_init=True)
        if target.exists() and on_exists == OnConflict.SKIP:
            self.logger.warning(f'Target file already exists (ending silently): {target.resolve()}')
            return
        if target.exists() and on_exists == OnConflict.BREAK:
            raise FileExistsError(f'Target file {target.resolve()} already exists')
        self.logger.info(f'Going to write result to {target.resolve()}')
        self.target = target
        self.params = params or {}
        self.batch_size = batch_size
        if import_ids is None and self.params.get('import_ids') is None:
            self.params['import_ids'] = self.settings.IMPORTS
        if project_id is None and self.params.get('project_id') is None:
            self.params['project_id'] = self.settings.PROJECT_ID
        self.query: sa.TextClause | None = None

    def __enter__(self):
        # return the instance so it can be used inside the with-block
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.query is None:
            raise RuntimeError('Query must be defined')

        with self.db_engine.session() as session:
            self.logger.info('Running query...')
            rslt = session.execute(self.query.execution_options(yield_per=self.batch_size), self.params)

            columns: list[str] | None = None
            for batch in tqdm(rslt.mappings().partitions(self.batch_size)):
                sub_df = pd.DataFrame(batch).replace({np.nan: None})
                if columns is None:
                    columns = sub_df.columns
                    sub_df.to_csv(self.target, index=False)
                else:
                    sub_df.to_csv(self.target, index=False, header=False, columns=columns, mode='a')

        # return False to propagate exceptions
        return False

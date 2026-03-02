import json
import math
from pathlib import Path
from typing import Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if math.isnan(obj) or obj == np.nan or (type(obj) is float and np.isnan(obj)):
            return None
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def json_dump(file_path: Path, obj: dict[str, Any], **kwargs: Any) -> None:
    with open(file_path, 'w') as f:
        f.write(json.dumps(obj, cls=NumpyEncoder, **kwargs))

import math
from pathlib import Path
import pyarrow as pa
import pandas as pd


def write_keywords(df_kws: pd.DataFrame, fname: Path, chunk_size: int = 10000) -> None:
    schema = pa.schema(
        [('x', pa.float16()), ('y', pa.float16()), ('level', pa.uint8()), ('keyword', pa.string())],
    )

    print(f'Writing {fname}')
    with pa.OSFile(str(fname), 'wb') as sink:
        with pa.ipc.new_stream(sink, schema) as writer:
            n_chunks = int(math.ceil(df_kws.shape[0] / chunk_size))
            for chunk in range(n_chunks):
                tmp = df_kws[chunk * chunk_size : (chunk + 1) * chunk_size]
                batch = pa.record_batch(tmp, schema)
                writer.write(batch)
    print(f'Wrote {fname}')

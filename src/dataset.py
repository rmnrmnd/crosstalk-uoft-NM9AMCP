"""Dataset loader to load features and labels from Parquet files.

Example:
    import src.dataset
    X, y = src.dataset.load_data("data/train.parquet", ["AVALON", "MACCS"])
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse
import tqdm.auto

FINGERPRINTS: list[str] = [
    "ATOMPAIR",
    "MACCS",
    "ECFP6",
    "ECFP4",
    "FCFP4",
    "FCFP6",
    "TOPTOR",
    "RDK",
    "AVALON",
]


def load_data(
    path: str,
    x_cols: list[str] | str,
    y_col: str | None = "DELLabel",
    max_rows: int | None = None,
    chunk_size: int = 20000,
) -> scipy.sparse.csr_matrix | tuple[scipy.sparse.csr_matrix, np.ndarray]:
    """Loads Parquet datasets into scipy sparse matrices."""
    if isinstance(x_cols, str):
        x_cols = [x_cols]

    for col in x_cols:
        if col not in FINGERPRINTS:
            raise ValueError(f"Unknown fingerprint: {col}")

    pf = pq.ParquetFile(path)
    cols = x_cols + ([y_col] if y_col is not None else [])
    
    total = pf.metadata.num_rows
    limit = total if max_rows is None or max_rows > total else max_rows

    mats = []
    y_list = []
    loaded = 0

    pbar = tqdm.auto.tqdm(
        total=int(np.ceil(limit / chunk_size)),
        desc="Loading chunks",
    )

    for batch in pf.iter_batches(columns=cols, batch_size=min(chunk_size, limit)):
        df = pa.Table.from_batches([batch]).to_pandas()
        remaining = limit - loaded
        if len(df) > remaining:
            df = df.iloc[:remaining]
            
        # Parse and horizontally stack all requested feature columns
        exploded_list = [
            scipy.sparse.csr_matrix(
                df[col].str.split(",", expand=True).astype(float, copy=False)
            )
            for col in x_cols
        ]
        mats.append(scipy.sparse.hstack(exploded_list))
        
        if y_col is not None:
            y_list.append(df[y_col].values)
            
        loaded += len(df)
        pbar.update(1)
        if loaded >= limit:
            break

    pbar.n = pbar.total
    pbar.refresh()
    pbar.close()

    X = scipy.sparse.vstack(mats)
    if y_col is not None and y_list:
        return X, np.concatenate(y_list)
    return X

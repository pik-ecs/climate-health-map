import logging
from pathlib import Path

import numpy as np
import pandas as pd
from climate_health_map.data.labels import LABELS, Label, COLUMNS_MAJOR, COLUMN_GROUP, COLUMNS_IMPACTS


def downsampling_mask(y: np.ndarray, sampling: float | None, min_n_majority: int = 3, threshold: float = 0.5) -> np.ndarray:
    """Produce downsampling mask.
    This figures out which one the majority class is and reduces it's size to `sampling`% of the original number

    Example:
       - sampling = 0.6
       - y has 20x 0 and 1200x 1
        --> then 1 is the majority class
        --> mask will be True for all indexes of `y` where it is `0`
        --> mask will be True for 60% of indexes of `y` where it is `1`

    `sampling == 1.0` -> keep all
    `sampling == 0.0` -> keep `min_n_majority` of majority class
    """
    mask = np.ones(len(y), dtype=bool)

    # Ensure we are doing this on binary labels
    y_ = y > threshold

    # Find majority class, we only downsample on that one
    counts = np.unique_counts(y_)
    n_majority = counts.counts.max()
    majority_class = counts.values[counts.counts.argmax()]

    # Ensure that we always keep at least `min_n_majority` of the majority class
    sample_size = int(n_majority * sampling)
    if sample_size < min_n_majority:
        sample_size = min_n_majority

    # Find indexes of items of the majority class
    sample = np.argwhere(y_ == majority_class)

    # Shuffle to keep up the random spirit
    np.random.shuffle(sample)
    mask[sample[:sample_size]] = False
    return mask


def is_label_eligible(df: pd.DataFrame, label: Label, min_minor_class: int = 10, logger: logging.Logger | None = None) -> bool:
    logger = logger or logging.getLogger('dataset')
    if label.column not in df.columns:
        logger.debug(f'{label.column} not in columns')
        return False
    column_stats = df[df[label.column].notna()][label.column].value_counts()
    logger.debug(f'Class balance: {column_stats.to_dict()}')
    if len(column_stats) != 2:
        return False
    return min(column_stats) >= min_minor_class


def get_filtered_labels(
    dataset_path: Path | None = None,
    dataset_df: pd.DataFrame | None = None,
    min_minor_class: int = 10,
):
    """Filter `LABELS` based on the available training data.

    Pass data through ONE of the following:
    :param dataset_path: Path to annotations (typically `data/exports/annotations.csv`)
    :param dataset_df: Pandas DataFrame containing training data

    :param min_minor_class: How many of the under-represented class for a label have to exist to include the column
    :return:
    """
    if dataset_path is not None:
        df = pd.read_csv(dataset_path)
    elif dataset_df is not None:
        df = dataset_df
    else:
        raise AssertionError('Must provide either `dataset_path` or `dataset_df`')

    labels = {}
    for key, group in LABELS.items():
        clone = group.copy(deep=True)
        clone.labels = [label for label in clone.labels if is_label_eligible(df, label, min_minor_class)]
        if len(clone.labels) > 0:
            labels[key] = clone

    return labels


class Dataset:
    def __init__(
        self,
        dataset_path: Path | None = None,
        logger: logging.Logger | None = None,
    ):
        self.logger = logger or logging.getLogger('dataset')
        self.logger.info(f'Loading dataset from {dataset_path}')
        self.df = pd.read_csv(dataset_path).replace({np.nan: None}).set_index('item_id', drop=False)
        self.logger.info(f'Loaded {self.df.shape} records')

    def is_label_eligible(self, label: Label, min_minor_class: int = 10) -> bool:
        return is_label_eligible(df=self.df, label=label, min_minor_class=min_minor_class)

    def get_simplified_df(self, column: str, mask: pd.Series[bool] | None = None) -> pd.DataFrame:
        masked = self.df[mask] if mask is not None else self.df
        return pd.DataFrame(
            [
                {
                    'id': row['item_id'],
                    'text': f'{row["title"] or ""} {row["abstract"] or ""}',
                    'label': int(row[column] > 0.5) if row[column] is not None else 0,
                }
                for _, row in masked.iterrows()
            ],
        ).set_index('id')

    def get_mask(self, column: str, ensure_text: bool = False) -> pd.Series[bool]:
        if column in {'rel_major|1', 'rel_major|0', 'rel_impacts|1', 'rel_impacts|0'}:
            mask = self.df[column].notna()
        elif column in COLUMNS_MAJOR:
            mask = (self.df['rel_major|1'] > 0.5) & self.df[[col for col in COLUMN_GROUP[column] if col in self.df.columns]].any(axis=1)
        elif column in COLUMNS_IMPACTS:
            mask = (self.df['rel_impacts|1'] > 0.5) & self.df[[col for col in COLUMN_GROUP[column] if col in self.df.columns]].any(axis=1)
        else:
            raise KeyError(f'No standard filter available for column "{column}"')

        if ensure_text:
            mask &= self.df['abstract'].notna()

        self.logger.info(f'Filtered data down to {mask.sum():,} records for column {column}')
        return mask


__all__ = [
    'downsampling_mask',
    'is_label_eligible',
    'get_filtered_labels',
    'Dataset',
]

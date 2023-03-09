import json
import pickle
import random
from enum import Enum
from pathlib import Path
from typing import Counter, Dict, Iterable, List, TypeVar, Union

import datasets
import joblib
import numpy as np
import yaml
from torch import nn

DatasetT = TypeVar("DatasetT", datasets.Dataset, datasets.DatasetDict)


def get_params(params_path: Union[None, str, Path] = None):
    if params_path is None:
        params_file_path = Path.cwd() / "params.yaml"
    else:
        params_file_path = Path(params_path)

    with open(params_file_path) as f:
        params = yaml.safe_load(f)

    return params


def datasets_exclusive_map(dataset: DatasetT, function, *args, **kwargs) -> DatasetT:
    if isinstance(dataset, datasets.Dataset):
        columns: List[str] = dataset.column_names
        kwargs["remove_columns"] = columns
        mapped_dataset: DatasetT = dataset.map(function, *args, **kwargs)
    elif isinstance(dataset, datasets.DatasetDict):
        mapped_dataset = datasets.DatasetDict()
        for split_name in dataset:
            dataset_split = dataset[split_name]
            columns = dataset_split.column_names

            kwargs["remove_columns"] = columns
            mapped_dataset[split_name] = dataset_split.map(function, *args, **kwargs)
    else:
        raise ValueError("Wrong dataset type %s" % type(dataset))
    return mapped_dataset

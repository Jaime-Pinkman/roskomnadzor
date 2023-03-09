from pathlib import Path

import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import get_params


def xlsx_to_dataframe(data_file):
    df = pd.read_excel(data_file, index_col=[0]).reset_index(drop=True)
    df["label"] = df["class"].apply(lambda x: 0 if x == "не-самоубийство" else 1)
    df.drop("class", inplace=True, axis=1)
    return df


def make_dataset(df):
    return datasets.Dataset.from_dict(
        {
            # Text Classification required
            "text": df.text.to_list(),
            "label": df.label.to_list(),
        }
    )


def make_splits(df):
    traindf, testdf = train_test_split(df, test_size=0.2, random_state=42)
    traindf, devdf = train_test_split(traindf, test_size=0.2, shuffle=True, random_state=42)
    return traindf, devdf, testdf


def read(raw_dataset_dir):
    raw_dataset_dir = Path(raw_dataset_dir)
    dataset_path = raw_dataset_dir / "dataset.xlsx"

    df = xlsx_to_dataframe(dataset_path)

    train_df, dev_df, test_df = make_splits(df)

    dataset = datasets.DatasetDict(
        {
            "train": make_dataset(train_df),
            "dev": make_dataset(dev_df),
            "test": make_dataset(test_df),
        }
    )

    return dataset


if __name__ == "__main__":
    params = get_params()

    raw_dataset_dir = Path(params["files"]["raw_datasets"])
    hf_dataset_dir = Path(params["files"]["converted_datasets"])

    read_dataset = read(raw_dataset_dir=raw_dataset_dir)
    read_dataset.save_to_disk(hf_dataset_dir)

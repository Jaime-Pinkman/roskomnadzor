from pathlib import Path

import datasets
from src.utils import get_params
from transformers import AutoTokenizer


def prepare_text_classifier(dataset):
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    return encoded_dataset


def preprocess_function(examples):
    return tokenizer(examples["text"], max_length=512, padding=True, truncation=True)


if __name__ == "__main__":
    params = get_params()

    hf_dataset_dir = Path(params["files"]["converted_datasets"])
    prepared_dataset_dir = Path(params["files"]["prepared_datasets"])

    hf_dataset = datasets.load_from_disk(str(hf_dataset_dir))

    tokenizer = AutoTokenizer.from_pretrained(params["model"])

    dataset = prepare_text_classifier(hf_dataset)

    dataset.save_to_disk(prepared_dataset_dir)

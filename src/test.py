from pathlib import Path

import datasets
import evaluate
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer

from src.utils import get_params

if __name__ == "__main__":
    params = get_params()
    prepared_dataset_dir = Path(params["files"]["prepared_datasets"])
    artifacts_dir = Path(params["output"]) / "checkpoint-713"

    tokenized_dataset = datasets.load_from_disk(str(prepared_dataset_dir))
    model = AutoModelForSequenceClassification.from_pretrained(artifacts_dir).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(params["model"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    model.eval()
    text = "Мир так прекрасен я люблю в нем жить"
    inputs = tokenizer(text, max_length=512, truncation=True, padding=True)
    predictions = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
    ).predict(test_dataset=tokenized_dataset["test"])
    preds = np.argmax(predictions.predictions, axis=-1)
    print(preds)
    print(tokenized_dataset["test"])
    print(inputs)

    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    results = clf_metrics.compute(predictions=preds, references=tokenized_dataset["test"]["label"])
    print(results)

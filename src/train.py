from pathlib import Path

import datasets
import evaluate
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.utils import get_params


def compute_metrics(eval_preds):
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return clf_metrics.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    params = get_params()
    prepared_dataset_dir = Path(params["files"]["prepared_datasets"])
    artifacts_dir = Path(params["artifacts"])

    tokenized_dataset = datasets.load_from_disk(str(prepared_dataset_dir))

    model = AutoModelForSequenceClassification.from_pretrained(params["model"], num_labels=2).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(params["model"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    training_args = TrainingArguments(
        output_dir="output",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["dev"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(artifacts_dir)

    Trainer(
        model=model,
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    ).evaluate()

"""utils.py"""
import csv
import json
import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import Dataset, DatasetDict
import matplotlib.pyplot as plt


def plot_metrics(metrics: dict[str, list[float]], has_validation: bool, output_dir: Path) -> None:
    """plot_metrics"""
    epochs = range(1, len(metrics['train_losses']) + 1)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics['train_losses'], label="Training Loss")
    if has_validation:
        plt.plot(epochs, metrics['valid_losses'], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss" if has_validation else "Training Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics['train_metric'], label="Training Metric")
    if has_validation:
        plt.plot(epochs, metrics['valid_metric'], label="Validation Metric")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training and Validation Metric" if has_validation else "Training Metric")
    plt.legend()
    plt.grid(True)

    plot_path = output_dir / "metrics_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def setup_logging(output_dir: Path) -> None:
    """setup_logging"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    log_file = output_dir / "training.log"
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S"
    ))
    logging.getLogger().addHandler(file_handler)

def load_dataset(
    data_files: dict[str, str], context_file: str, end_to_end: bool = False
    ) -> DatasetDict:
    """load_dataset"""

    with open(context_file, "r", encoding="utf8") as file:
        context_data = json.load(file)

    def _read_and_fix(file_path: str) -> Dataset:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            entry.setdefault("relevant", 0)
            entry.setdefault("answer", {"text": "", "start": 0})
            entry.setdefault("paragraphs", [0, 0, 0, 0])
            entry["answers"] = {
                    "text": [entry["answer"]["text"]],
                    "answer_start": [entry["answer"]["start"]]
                }

            if end_to_end:
                entry["context"] = "".join([context_data[i] for i in entry["paragraphs"]])

            else:
                entry["sent1"] = entry["question"]
                entry["sent2"] = entry["question"]
                for i in range(4):
                    entry[f"ending{i}"] = context_data[entry["paragraphs"][i]]

                if entry["relevant"] in entry["paragraphs"]:
                    entry["label"] = entry["paragraphs"].index(entry["relevant"])
                else:
                    entry["label"] = 0
                entry.setdefault("context", context_data[entry["relevant"]])

        return Dataset.from_dict({key: [entry[key] for entry in data] for key in data[0]})

    processed_datasets = DatasetDict(
        {split: _read_and_fix(file_path) for split, file_path in data_files.items()}
    )

    print(processed_datasets)
    return processed_datasets

def paragraph_selection(
    model: nn.Module,
    test_dataloader: DataLoader,
    test_dataset: Dataset,
    prediction_path: Path
    ) -> None:
    """paragraph_selection"""
    model.eval()
    predicted_labels = []
    predictions = []
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    for batch in tqdm(test_dataloader, desc="Paragraph Selection", colour="red"):
        with torch.no_grad():
            outputs = model(**batch)
            predicted_labels += outputs.logits.argmax(dim=-1).cpu().tolist()

    for i, predicted_label in enumerate(predicted_labels):
        example = test_dataset[i]

        predictions.append({
            "id": example["id"],
            "question": example["question"],
            "context": example[f"ending{predicted_label}"]
        })

    with open(prediction_path, "w", encoding="utf-8") as json_file:
        json.dump(predictions, json_file, ensure_ascii=False, indent=4)

    print(f"The prediction results have been saved to {prediction_path}")

def span_selection(predictions: list[dict[str, str]], prediction_path: Path) -> None:
    """span_selection"""

    with prediction_path.open("w", newline="") as csv_file:
        fieldnames = ['id', 'answer']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        for prediction in predictions:
            csv_writer.writerow({'id': prediction['id'], 'answer': prediction['prediction_text']})

    print(f"The prediction results have been saved to {prediction_path}")

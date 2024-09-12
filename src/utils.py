"""utils.py"""
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
        plt.plot(epochs, metrics['val_losses'], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss" if has_validation else "Training Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics['train_accuracies'], label="Training Accuracy")
    if has_validation:
        plt.plot(epochs, metrics['val_accuracies'], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy" if has_validation else "Training Accuracy")
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

    def _read_and_fix(file_path: str, is_test: bool = False) -> Dataset:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if is_test:
            for entry in data:
                entry.setdefault('relevant', "")
                entry.setdefault('answer', {"text": "", "start": 0})

        column_based_data = {key: [entry[key] for entry in data] for key in data[0]}

        return Dataset.from_dict(column_based_data)

    def _preprocess_entry(
        entry: dict, context_data: dict[str, str], end_to_end: bool, is_test: bool
        ) -> dict:
        processed = {
            "id": entry["id"],
            "question": entry["question"]
        }

        if end_to_end:
            label = 0 if is_test else entry["paragraphs"].index(entry["relevant"])
            answer_start = 0 if is_test else entry["answer"]["start"]

            context = "".join([context_data[para] for para in entry["paragraphs"][:4]])
            for paragraph in entry["paragraphs"][:label]:
                answer_start += len(context_data[paragraph])

            processed["context"] = context
            processed["answers"] = {
                "answer_start": [0] if is_test else [answer_start],
                "text": [""] if is_test else [entry["answer"]["text"]]
            }
        else:
            processed.update({
                "sent1": entry["question"],
                "sent2": entry["question"],
                "ending0": context_data[entry["paragraphs"][0]],
                "ending1": context_data[entry["paragraphs"][1]],
                "ending2": context_data[entry["paragraphs"][2]],
                "ending3": context_data[entry["paragraphs"][3]],
                "label": 0 if is_test else entry["paragraphs"].index(entry["relevant"]),
                "context": "" if is_test else context_data[entry["relevant"]],
                "answers": {
                    "text": [""] if is_test else [entry["answer"]["text"]],
                    "answer_start": [0] if is_test else [entry["answer"]["start"]]
                }
            })

        return processed

    raw_datasets = {}
    for split, file_path in data_files.items():
        is_test = split == "test"
        raw_datasets[split] = _read_and_fix(file_path, is_test=is_test)

    processed_datasets = {}
    for split, dataset in raw_datasets.items():
        is_test = split == 'test'
        processed_split_data = [
            _preprocess_entry(entry, context_data, end_to_end, is_test) for entry in dataset
        ]
        column_based_data = {
            key: [entry[key] for entry in processed_split_data] for key in processed_split_data[0]
        }
        processed_datasets[split] = Dataset.from_dict(column_based_data)

    processed_datasets = DatasetDict(processed_datasets)
    print(processed_datasets)
    return processed_datasets

def predict_paragraph_selection(
    model: nn.Module,
    test_dataloader: DataLoader,
    test_dataset: Dataset,
    output_file_path: str
    ) -> None:
    """predict_paragraph_selection"""
    model.eval()
    predicted_labels = []
    predictions = []

    for batch in tqdm(test_dataloader, desc="Paragraph Selection", colour="red"):
        with torch.no_grad():
            outputs = model(**batch)
            predicted_labels += outputs.logits.argmax(dim=-1).cpu().tolist()

    for i, predicted_label in enumerate(predicted_labels):
        example = test_dataset[i]

        predictions.append({
            "id": example["id"],
            "question": example["question"],
            "context": example[f"ending{predicted_label}"],
            "answers": example["answers"]
        })

    with open(output_file_path, "w", encoding="utf-8") as json_file:
        json.dump(predictions, json_file, ensure_ascii=False, indent=4)

    print(f"Predictions saved to {output_file_path}")

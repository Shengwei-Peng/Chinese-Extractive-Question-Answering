#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a Transformers model for Paragraph Selection using Accelerate.
"""
import argparse
import json
import math
import os
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Optional, Union

import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import PaddingStrategy

from src.utils import setup_logging, load_dataset, plot_metrics, paragraph_selection

logger = get_logger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for paragraph selection received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side 
            and padding index) among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if 
              only a single sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or 
              to the maximum acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with 
              sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            # If set, will pad the sequence to a multiple of the provided value. 
            # This is especially useful to enable the use of Tensor Cores on 
            # NVIDIA hardware with compute capability >= 7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)]
            for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

def parse_args():
    """parse_args"""
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a paragraph selection task"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None,
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None,
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None,
        help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--context_file", type=str, default=None,
        help="A csv or a json file containing the context data."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization."
            "Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--num_train_epochs", type=int, default=3,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts",
            "polynomial", "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub."
    )
    parser.add_argument(
        "--hub_model_id", type=str,
        help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub. "
            "This option should only be set to `True` for repositories you trust "
            "and in which you have read the code, "
            "as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help=(
            "Whether the various states should be saved at the end of every n steps, "
            "or 'epoch' for each epoch."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. '
            'Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. '
            'Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--use_pretrained",
        action="store_true",
        help="If passed, use a pretrained model. Otherwise, train from scratch."
    )
    args = parser.parse_args()

    if args.push_to_hub:
        assert args.output_dir is not None, (
            "Need an `output_dir` to create a repo when `--push_to_hub` is passed."
        )

    return args

def main():
    """main"""
    args = parse_args()

    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs
    )

    if args.output_dir:
        args.output_dir = Path(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(args.output_dir)

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.push_to_hub:
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = args.output_dir.absolute().name
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            gitignore_path = args.output_dir / ".gitignore"
            with gitignore_path.open("w+", encoding="utf-8") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")

    accelerator.wait_for_everyone()

    data_files = {k: v for k, v in {
        "train": args.train_file,
        "validation": args.validation_file,
        "test": args.test_file
    }.items() if v is not None}

    raw_datasets = load_dataset(data_files, args.context_file)

    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))

    if raw_datasets["train"] is not None:
        column_names = raw_datasets["train"].column_names
    elif raw_datasets["validation"] is not None:
        column_names = raw_datasets["validation"].column_names
    elif raw_datasets["test"] is not None:
        column_names = raw_datasets["test"].column_names

    ending_names = [f"ending{i}" for i in range(4)]
    context_name = "sent1"
    question_header_name = "sent2"
    label_column_name = "label" if "label" in column_names else "labels"

    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path, trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path, trust_remote_code=args.trust_remote_code
        )
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            use_fast=not args.use_slow_tokenizer,
            trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=not args.use_slow_tokenizer,
            trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch."
            "This is not supported by this script."
            "You can create it in another script,"
            "save it, and load it from here using --tokenizer_name."
        )

    if args.model_name_or_path and args.use_pretrained:
        model = AutoModelForMultipleChoice.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMultipleChoice.from_config(
            config, trust_remote_code=args.trust_remote_code
        )

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        first_sentences = [[context] * 4 for context in examples[context_name]]
        question_headers = examples[question_header_name]
        second_sentences = [
            [f"{header} {examples[end][i]}" for end in ending_names]
            for i, header in enumerate(question_headers)
        ]
        labels = examples[label_column_name]

        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            max_length=args.max_seq_length,
            padding=padding,
            truncation=True,
        )
        tokenized_inputs = {
            k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()
        }
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    with accelerator.main_process_first():
        if args.train_file is not None:
            processed_datasets = raw_datasets.map(
                preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names
            )
        elif args.test_file is not None:
            processed_datasets = raw_datasets.map(
                preprocess_function, batched=True, remove_columns=raw_datasets["test"].column_names
            )

    if args.train_file is not None:
        train_dataset = processed_datasets["train"]

    if args.validation_file is not None:
        eval_dataset = processed_datasets["validation"]

    if args.test_file is not None:
        test_dataset = processed_datasets["test"]

    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForMultipleChoice(
            tokenizer, pad_to_multiple_of=(8 if accelerator.mixed_precision == "fp16" else None)
        )

    if args.train_file is not None:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator,
            batch_size=args.per_device_train_batch_size
        )
    if args.validation_file  is not None:
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
    if args.test_file  is not None:
        test_dataloader = DataLoader(
            test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )

    if args.num_train_epochs > 0 and args.train_file is not None:
        metrics = {
            "train_losses": [],
            "valid_losses": [],
            "train_metric": [],
            "valid_metric": []
        }

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        device = accelerator.device
        model.to(device)

        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps
            if overrode_max_train_steps
            else args.max_train_steps * accelerator.num_processes,
        )

        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )

        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        checkpointing_steps = args.checkpointing_steps
        if checkpointing_steps is not None and checkpointing_steps.isdigit():
            checkpointing_steps = int(checkpointing_steps)

        if args.with_tracking:
            experiment_config = vars(args)
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            accelerator.init_trackers("swag_no_trainer", experiment_config)

        metric = evaluate.load("./src/accuracy.py")

        total_batch_size = (
            args.per_device_train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        progress_bar = tqdm(
            range(args.max_train_steps), disable=not accelerator.is_local_main_process
        )
        completed_steps = 0
        starting_epoch = 0

        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
                checkpoint_path = args.resume_from_checkpoint
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]
                checkpoint_path = path
                path = os.path.basename(checkpoint_path)

            accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
            accelerator.load_state(checkpoint_path)
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
                completed_steps = starting_epoch * num_update_steps_per_epoch
            else:
                step_value = int(training_difference.replace("step_", ""))
                resume_step = step_value * args.gradient_accumulation_steps
                starting_epoch = resume_step // len(train_dataloader)
                completed_steps = resume_step // args.gradient_accumulation_steps
                resume_step -= starting_epoch * len(train_dataloader)

        progress_bar.update(completed_steps)

        for epoch in range(starting_epoch, args.num_train_epochs):
            model.train()
            total_loss = 0
            if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
                active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            else:
                active_dataloader = train_dataloader
            for batch in active_dataloader:
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    total_loss += loss.detach().cpu().item()

                    predictions, references = accelerator.gather_for_metrics(
                        (outputs.logits.argmax(dim=-1), batch["labels"])
                    )

                    metric.add_batch(
                        predictions=predictions,
                        references=references,
                    )

                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                        output_dir = args.output_dir / f"step_{completed_steps}"
                        accelerator.save_state(output_dir)

                if completed_steps >= args.max_train_steps:
                    break

            train_loss = total_loss / len(train_dataloader)
            train_accuracy = metric.compute()["accuracy"]
            metrics["train_losses"].append(train_loss)
            metrics["train_metric"].append(train_accuracy)
            accelerator.print(
                f"Train Loss: {metrics['train_losses'][-1]}, "
                f"Exact Match: {metrics['train_metric'][-1]}"
            )

            if args.validation_file is not None:
                logger.info("***** Running Evaluation *****")
                logger.info(f"  Num examples = {len(eval_dataset)}")
                logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

                model.eval()
                valid_total_loss = 0

                for batch in eval_dataloader:
                    with torch.no_grad():
                        outputs = model(**batch)
                    valid_total_loss += outputs.loss.detach().cpu().item()

                    predictions, references = accelerator.gather_for_metrics(
                        (outputs.logits.argmax(dim=-1), batch["labels"])
                    )

                    metric.add_batch(
                        predictions=predictions,
                        references=references,
                    )

                metrics["valid_losses"].append(valid_total_loss / len(eval_dataloader))
                metrics["valid_metric"].append(metric.compute()["accuracy"])
                accelerator.print(
                    f"Valid Loss: {metrics['valid_losses'][-1]}, "
                    f"Exact Match: {metrics['valid_metric'][-1]}"
                )

                if args.with_tracking:
                    accelerator.log(
                        {
                            "accuracy": metrics["valid_metric"][-1],
                            "train_loss": metrics["train_losses"][-1],
                            "epoch": epoch+1,
                            "step": completed_steps,
                        },
                        step=completed_steps,
                    )

            if args.push_to_hub and epoch < args.num_train_epochs - 1:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    args.output_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
                    api.upload_folder(
                        commit_message=f"Training in progress epoch {epoch}",
                        folder_path=args.output_dir,
                        repo_id=repo_id,
                        repo_type="model",
                        token=args.hub_token,
                    )

            if args.checkpointing_steps == "epoch":
                output_dir = args.output_dir / f"epoch_{epoch}"
                accelerator.save_state(output_dir)

        plot_metrics(metrics, args.validation_file is not None, args.output_dir)

        if args.with_tracking:
            accelerator.end_training()

    if args.test_file is not None:
        model, test_dataloader = accelerator.prepare(model, test_dataloader)
        paragraph_selection(
            model=model,
            test_dataloader=test_dataloader,
            test_dataset=raw_datasets["test"],
            prediction_path=args.output_dir / "predictions.json"
        )

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        for param in unwrapped_model.parameters():
            param.data = param.data.contiguous()

        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )

            metrics_path = args.output_dir / "metrics.json"
            with metrics_path.open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()

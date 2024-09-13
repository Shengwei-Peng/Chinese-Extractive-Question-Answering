#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning a Transformers model for span selection (extractive QA) using Accelerate.
"""
import argparse
import json
import math
import os
from pathlib import Path

import datasets
import evaluate
import numpy as np
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
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

from src.utils import setup_logging, load_dataset, plot_metrics, span_selection
from src.utils_qa import postprocess_qa_predictions

logger = get_logger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def parse_args():
    """parse_args"""
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a span selection task"
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
        "--preprocessing_num_workers", type=int, default=1,
        help="A csv or a json file containing the training data."
    )
    parser.add_argument("--do_predict", action="store_true",
                        help="To do prediction on the question answering model")
    parser.add_argument(
        "--validation_file", type=str, default=None,
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None,
        help="A csv or a json file containing the Prediction data."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=384,
        help=(
            "The maximum total input sequence length after tokenization. "
            "Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.",
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
        "--doc_stride",
        type=int,
        default=128,
        help=(
            "When splitting up a long document into chunks how much stride to take between chunks."
        ),
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="The total number of n-best predictions to generate when looking for an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help=(
            "The threshold used to select the null answer: "
            "if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, "
            "the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        ),
    )
    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, some of the examples do not have an answer.",
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=30,
        help=(
            "The maximum length of an answer that can be generated. "
            "This is needed because the start and "
            "end predictions are not conditioned on one another."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, "
            "truncate the number of training examples to this value if set."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, "
            "truncate the number of evaluation examples to this value if set."
        ),
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true",
        help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_predict_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, "
            "truncate the number of prediction examples to this"
        ),
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
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
        "--context_file", type=str, default=None,
        help="A csv or a json file containing the context data."
    )
    parser.add_argument(
        "--use_pretrained",
        action="store_true",
        help="If passed, use a pretrained model. Otherwise, train from scratch."
    )
    parser.add_argument(
        "--prediction_path",
        type=str,
        default="prediction.csv",
        help="Path to the output prediction file. Default is 'prediction.csv'.",
    )
    args = parser.parse_args()

    if (
        args.dataset_name is None
        and args.train_file is None
        and args.validation_file is None
        and args.test_file is None
    ):
        raise ValueError("Need either a dataset name or a training/validation/test file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if args.test_file is not None:
            extension = args.test_file.split(".")[-1]
            assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."

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

    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name, trust_remote_code=args.trust_remote_code
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
            args.tokenizer_name, use_fast=True, trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=True, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. "
            "This is not supported by this script. "
            "You can do it from another script, save it, "
            "and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path and args.use_pretrained:
        model = AutoModelForQuestionAnswering.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForQuestionAnswering.from_config(
            config, trust_remote_code=args.trust_remote_code
        )

    if args.train_file is not None:
        column_names = raw_datasets["train"].column_names
    if args.test_file is not None:
        column_names = raw_datasets["test"].column_names

    question_column_name = "question" if "question" in column_names else column_names[1]
    context_column_name = "context" if "context" in column_names else column_names[2]
    answer_column_name = "answers" if "answers" in column_names else column_names[3]

    pad_on_right = tokenizer.padding_side == "right"

    if args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({args.max_seq_length})"
            " is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length})."
            " Using max_seq_length={tokenizer.model_max_length}."
        )

    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    def prepare_features(examples, is_training=True):
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []
        if is_training:
            tokenized_examples["start_positions"] = []
            tokenized_examples["end_positions"] = []
            for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
                input_ids = tokenized_examples["input_ids"][i]
                if tokenizer.cls_token_id in input_ids:
                    cls_index = input_ids.index(tokenizer.cls_token_id)
                elif tokenizer.bos_token_id in input_ids:
                    cls_index = input_ids.index(tokenizer.bos_token_id)
                else:
                    cls_index = 0
                sequence_ids = tokenized_examples.sequence_ids(i)
                sample_index = sample_mapping[i]
                answers = examples[answer_column_name][sample_index]
                if len(answers["answer_start"]) == 0:
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    start_char = answers["answer_start"][0]
                    end_char = start_char + len(answers["text"][0])
                    token_start_index = 0
                    while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                        token_start_index += 1
                    token_end_index = len(input_ids) - 1
                    while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                        token_end_index -= 1
                    if not (
                        offsets[token_start_index][0] <= start_char
                        and offsets[token_end_index][1] >= end_char
                        ):
                        tokenized_examples["start_positions"].append(cls_index)
                        tokenized_examples["end_positions"].append(cls_index)
                    else:
                        while (
                            token_start_index < len(offsets)
                            and offsets[token_start_index][0] <= start_char
                            ):
                            token_start_index += 1
                        tokenized_examples["start_positions"].append(token_start_index - 1)
                        while offsets[token_end_index][1] >= end_char:
                            token_end_index -= 1
                        tokenized_examples["end_positions"].append(token_end_index + 1)
        else:
            tokenized_examples["offset_mapping"] = tokenized_examples["offset_mapping"]
        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    if args.num_train_epochs > 0 and args.train_file is not None:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_examples = raw_datasets["train"]
        if args.max_train_samples is not None:
            train_examples = train_examples.select(range(args.max_train_samples))

        with accelerator.main_process_first():
            train_dataset = train_examples.map(
                prepare_features,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            if args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(args.max_train_samples))

        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if args.max_eval_samples is not None:
            eval_examples = eval_examples.select(range(args.max_eval_samples))
        with accelerator.main_process_first():
            eval_dataset = eval_examples.map(
                prepare_features,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
            if args.max_eval_samples is not None:
                eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    if args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if args.max_predict_samples is not None:
            predict_examples = predict_examples.select(range(args.max_predict_samples))
        with accelerator.main_process_first():
            predict_dataset = predict_examples.map(
                lambda examples: prepare_features(examples, is_training=False),
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
            if args.max_predict_samples is not None:
                predict_dataset = predict_dataset.select(range(args.max_predict_samples))

    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=(8 if accelerator.mixed_precision == "fp16" else None)
        )

    if args.num_train_epochs > 0 and args.train_file is not None:
        train_dataset_for_model = train_dataset.remove_columns(["example_id", "offset_mapping"])
        train_dataloader = DataLoader(
            train_dataset_for_model,
            collate_fn=data_collator,
            batch_size=args.per_device_train_batch_size
        )

        eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
        eval_dataloader = DataLoader(
            eval_dataset_for_model,
            collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size
        )
        metric = evaluate.load("squad_v2" if args.version_2_with_negative else "squad")

    if args.do_predict:
        predict_dataset_for_model = predict_dataset.remove_columns(["example_id", "offset_mapping"])
        predict_dataloader = DataLoader(
            predict_dataset_for_model,
            collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size
        )

    def post_processing_function(examples, features, predictions, stage="eval"):
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=args.version_2_with_negative,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            null_score_diff_threshold=args.null_score_diff_threshold,
            output_dir=args.output_dir,
            prefix=stage,
        )
        if args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
                for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [
                {"id": k, "prediction_text": v} for k, v in predictions.items()
            ]

        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        step = 0
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        for output_logit in start_or_end_logits:

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat

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
            accelerator.init_trackers("qa_no_trainer", experiment_config)

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
            train_total_loss = 0
            all_start_logits = []
            all_end_logits = []

            if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
                active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            else:
                active_dataloader = train_dataloader

            for batch in active_dataloader:
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    train_total_loss += loss.detach().cpu().item()

                    start_logits = outputs.start_logits
                    end_logits = outputs.end_logits
                    if not args.pad_to_max_length:
                        start_logits = accelerator.pad_across_processes(
                            start_logits, dim=1, pad_index=-100
                        )
                        end_logits = accelerator.pad_across_processes(
                            end_logits, dim=1, pad_index=-100
                        )

                    all_start_logits.append(
                        accelerator.gather_for_metrics(start_logits).detach().cpu().numpy()
                    )
                    all_end_logits.append(
                        accelerator.gather_for_metrics(end_logits).detach().cpu().numpy()
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

            max_len = max(x.shape[1] for x in all_start_logits)
            start_logits_concat = create_and_fill_np_array(all_start_logits, train_dataset, max_len)
            end_logits_concat = create_and_fill_np_array(all_end_logits, train_dataset, max_len)
            del all_start_logits, all_end_logits

            outputs_numpy = (start_logits_concat, end_logits_concat)
            prediction = post_processing_function(train_examples, train_dataset, outputs_numpy)
            eval_metric = metric.compute(
                predictions=prediction.predictions, references=prediction.label_ids
            )

            metrics["train_losses"].append(train_total_loss / len(train_dataset))
            metrics["train_metric"].append(eval_metric["exact_match"])
            accelerator.print(
                f"Train Loss: {metrics['train_losses'][-1]}, "
                f"Exact Match: {metrics['train_metric'][-1]}"
            )

            if args.checkpointing_steps == "epoch":
                output_dir = args.output_dir / f"epoch_{epoch}"
                accelerator.save_state(output_dir)

            if args.push_to_hub and epoch < args.num_train_epochs - 1:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    args.output_dir,
                    is_main_process=accelerator.is_main_process, save_function=accelerator.save
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

            if args.validation_file is not None:
                logger.info("***** Running Evaluation *****")
                logger.info(f"  Num examples = {len(eval_dataset)}")
                logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

                all_start_logits = []
                all_end_logits = []
                model.eval()
                valid_total_loss = 0

                for batch in eval_dataloader:
                    with torch.no_grad():
                        outputs = model(**batch)
                        loss = outputs.loss
                        valid_total_loss += outputs.loss.detach().cpu().item()
                        start_logits = outputs.start_logits
                        end_logits = outputs.end_logits
                        if not args.pad_to_max_length:
                            start_logits = accelerator.pad_across_processes(
                                start_logits, dim=1, pad_index=-100
                            )
                            end_logits = accelerator.pad_across_processes(
                                end_logits, dim=1, pad_index=-100
                            )
                        all_start_logits.append(
                            accelerator.gather_for_metrics(start_logits).cpu().numpy()
                        )
                        all_end_logits.append(
                            accelerator.gather_for_metrics(end_logits).cpu().numpy()
                        )

                max_len = max(x.shape[1] for x in all_start_logits)
                start_logits_concat = create_and_fill_np_array(
                    all_start_logits, eval_dataset, max_len
                )
                end_logits_concat = create_and_fill_np_array(
                    all_end_logits, eval_dataset, max_len
                )
                del all_start_logits, all_end_logits

                outputs_numpy = (start_logits_concat, end_logits_concat)
                prediction = post_processing_function(eval_examples, eval_dataset, outputs_numpy)
                eval_metric = metric.compute(
                    predictions=prediction.predictions, references=prediction.label_ids
                )

                metrics["valid_losses"].append(valid_total_loss / len(eval_dataloader))
                metrics["valid_metric"].append(eval_metric["exact_match"])
                accelerator.print(
                    f"Valid Loss: {metrics['valid_losses'][-1]}, "
                    f"Exact Match: {metrics['valid_metric'][-1]}"
                )

    plot_metrics(metrics, args.validation_file is not None, args.output_dir)

    if args.do_predict:
        model, predict_dataloader = accelerator.prepare(model, predict_dataloader)
        logger.info("***** Running Prediction *****")
        logger.info(f"  Num examples = {len(predict_dataset)}")
        logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

        all_start_logits = []
        all_end_logits = []
        model.eval()

        for batch in predict_dataloader:
            with torch.no_grad():
                outputs = model(**batch)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                if not args.pad_to_max_length:
                    start_logits = accelerator.pad_across_processes(
                        start_logits, dim=1, pad_index=-100
                    )
                    end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)
                all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
                all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())

        max_len = max(x.shape[1] for x in all_start_logits)
        start_logits_concat = create_and_fill_np_array(all_start_logits, predict_dataset, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, predict_dataset, max_len)
        del all_start_logits, all_end_logits

        outputs_numpy = (start_logits_concat, end_logits_concat)
        prediction = post_processing_function(predict_examples, predict_dataset, outputs_numpy)
        span_selection(prediction.predictions, Path(args.prediction_path))

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        for param in unwrapped_model.parameters():
            param.data = param.data.contiguous()

        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process, save_function=accelerator.save
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

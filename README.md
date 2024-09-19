# Chinese Extractive Question Answering

## Table of Contents 📚

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Inference](#inference)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Overview 🌟

This project focuses on Chinese extractive question answering (QA), a task where a model is trained to select a relevant span of text from a given context to answer a specific question. The model first identifies the most relevant paragraphs, and then extracts the precise answer spans from those paragraphs.

## Installation 💻

1. Clone the repository:
    ```sh
    git clone https://github.com/Shengwei-Peng/chinese-extractive-question-answering.git
    ```
2. Navigate to the project directory:
    ```sh
    cd chinese-extractive-question-answering
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage ⚙️

To perform Chinese extractive question answering, you can either follow two steps: **Paragraph Selection** and **Span Selection**, or use the **End-to-End** approach to complete the entire process in one step.

### Option 1: Two-Step Process ➡️

#### Step 1: Paragraph Selection

The first step is to select relevant paragraphs from the context file. Use the following command to train and predict paragraph selection:

```bash
python paragraph_selection.py \
    --model_name_or_path google-bert/bert-base-chinese \
    --train_file ./ntu-adl-2024-hw-1-chinese-extractive-qa/train.json \
    --validation_file ./ntu-adl-2024-hw-1-chinese-extractive-qa/valid.json \
    --context_file ./ntu-adl-2024-hw-1-chinese-extractive-qa/context.json \
    --test_file ./ntu-adl-2024-hw-1-chinese-extractive-qa/test.json \
    --max_seq_length 512 \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 8 \
    --seed 11207330 \
    --use_pretrained \
    --output_dir ./paragraph_selection/bert \
    --prediction_path ./paragraph_selection/bert/prediction.json
```

#### Step 2: Span Selection

Once you have the paragraph predictions from Step 1, proceed to the span selection step to extract the answer spans from the selected paragraphs. Use the following command:

```bash
python span_selection.py \
    --model_name_or_path google-bert/bert-base-chinese \
    --train_file ./ntu-adl-2024-hw-1-chinese-extractive-qa/train.json \
    --validation_file ./ntu-adl-2024-hw-1-chinese-extractive-qa/valid.json \
    --context_file ./ntu-adl-2024-hw-1-chinese-extractive-qa/context.json \
    --test_file ./paragraph_selection/bert/prediction.json \
    --max_seq_length 512 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 8 \
    --seed 11207330 \
    --use_pretrained \
    --do_predict \
    --output_dir ./span_selection/bert \
    --prediction_path ./prediction.csv
```
#### ⚠️ Special Note:

- In **Span Selection**, the `test_file` parameter should be set to the **prediction file from the Paragraph Selection** step (i.e., `./test/paragraph_selection/prediction.json`).
- **Paragraph Selection** saves its predictions in a **JSON file**, while **Span Selection** saves the final answer predictions in a **CSV file**. Make sure to use the appropriate formats for each step.

### Option 2: End-to-End Process 🚀

You can run the entire process in one step using the `--end_to_end` argument in the `span_selection.py` script. This will directly train and predict the extractive QA task without running `paragraph_selection` separately.

```bash
python span_selection.py \
    --model_name_or_path google-bert/bert-base-chinese \
    --train_file ./ntu-adl-2024-hw-1-chinese-extractive-qa/train.json \
    --validation_file ./ntu-adl-2024-hw-1-chinese-extractive-qa/valid.json \
    --context_file ./ntu-adl-2024-hw-1-chinese-extractive-qa/context.json \
    --test_file ./ntu-adl-2024-hw-1-chinese-extractive-qa/test.json \
    --max_seq_length 512 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 8 \
    --seed 11207330 \
    --use_pretrained \
    --do_predict \
    --output_dir ./end_to_end/bert \
    --prediction_path ./prediction.csv \
    --end_to_end
```

## Inference 🔮

If you already have trained models and wish to only perform inference without retraining, you can use the following commands:

### Step 1: Paragraph Selection (Only Inference)

```bash
python paragraph_selection.py \
    --model_name_or_path ./paragraph_selection/bert \
    --context_file ./ntu-adl-2024-hw-1-chinese-extractive-qa/context.json \
    --test_file ./ntu-adl-2024-hw-1-chinese-extractive-qa/test.json \
    --max_seq_length 512 \
    --per_device_eval_batch_size 8 \
    --seed 11207330 \
    --use_pretrained \
    --prediction_path ./paragraph_selection/bert/prediction.json
```

### Step 2: Span Selection (Only Inference)

```bash
python span_selection.py \
    --model_name_or_path ./span_selection/bert \
    --context_file ./ntu-adl-2024-hw-1-chinese-extractive-qa/context.json \
    --test_file ./paragraph_selection/bert/prediction.json \
    --max_seq_length 512 \
    --per_device_eval_batch_size 8 \    
    --seed 11207330 \
    --use_pretrained \
    --do_predict \
    --prediction_path ./prediction.csv
```

### End-to-End (Only Inference)

```bash
python span_selection.py \
    --model_name_or_path ./end_to_end/bert \
    --context_file ./ntu-adl-2024-hw-1-chinese-extractive-qa/context.json \
    --test_file ./ntu-adl-2024-hw-1-chinese-extractive-qa/test.json \
    --max_seq_length 512 \
    --per_device_eval_batch_size 8 \
    --seed 11207330 \
    --use_pretrained \
    --do_predict \
    --prediction_path ./prediction.csv \
    --end_to_end
```

## Acknowledgements 🙏

This project is based on the example code provided by Hugging Face in their [Transformers repository](https://github.com/huggingface/transformers/tree/main/examples/pytorch). We have made modifications to adapt the code for our specific use case.

Special thanks to the [NTU Miulab](http://adl.miulab.tw) professors and teaching assistants for providing the dataset and offering invaluable support throughout the project.

## Contact ✉️

For any questions or inquiries, please contact m11207330@mail.ntust.edu.tw
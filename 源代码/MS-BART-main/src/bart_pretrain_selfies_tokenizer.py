# Refer: https://github.com/huggingface/transformers/issues/18030#issuecomment-1193251528
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py
# https://github.com/morganmcg1/rotobart/blob/main/data_collator.py

#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import random
import psutil

import datasets
from datasets import load_dataset, DatasetDict
import numpy as np
import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    BartForConditionalGeneration,
    BartTokenizer,
    BartConfig,
)
from transformers.trainer_utils import get_last_checkpoint


sys.path.append(".")
from src.model.data_collator import DataCollatorForDenoisingTasks

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    min_lr: Optional[float] = field(default=1e-5, metadata={"help": "Min learning rate"})
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    pretrain_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the pretraining data."}
    )
    val_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the validation data."}
    )
    mask_ratio: Optional[float] = field(default=0.3, metadata={"help": "Data collator arguments for bart pretrain denoising task."})
    poisson_lambda: Optional[float] = field(default=3.0, metadata={"help": "Data collator arguments for bart pretrain denoising task."})
    pad_to_multiple_of: Optional[int] = field(default=16, metadata={"help": "Data collator arguments for bart pretrain denoising task."})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(default=2, metadata={"help": "Number of workers for preprocessing."})
    max_seq_length: Optional[int] = field(default=512, metadata={"help": "Max input length."})
    debug_flag: bool = field(default=False, metadata={"help": "Debug mode."})

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    set_seed(training_args.seed)

    delimiter = "\t" if "tsv" in data_args.pretrain_path else ","
    with training_args.main_process_first(desc="load data"):
        pretrain_datasets = load_dataset(
            "csv",
            data_files= {
                "train": data_args.pretrain_path,
                "val": data_args.val_path
            },
            delimiter=delimiter
        )
        logger.info(f"RAM:{psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

        if "val" not in pretrain_datasets:
            pretrain_datasets = pretrain_datasets.train_test_split(test_size=0.01)
            pretrain_datasets = DatasetDict({
                "train": pretrain_datasets["train"],
                "val": pretrain_datasets["test"]
            })
        if data_args.debug_flag: 
            pretrain_datasets = pretrain_datasets.select(range(1000))
            training_args.report_to = []
            training_args.overwrite_output_dir = True

    # Initialize tokenizer and model
    tokenizer_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
    
    config = BartConfig.from_pretrained(model_args.model_name_or_path)
    config.vocab_size = len(tokenizer)
    model = BartForConditionalGeneration(config)

    if training_args.should_log:
        print("=="*24, "Model Config", "=="*24)
        logger.info(f"{model.config}")
        logger.info(f"Tokenizer is fast: {tokenizer.is_fast}")
        logger.info(f"Model vocab size matches tokenizer: {model.config.vocab_size == len(tokenizer)}")
        logger.info(f"Model dtype: {next(model.parameters()).dtype}")

    def tokenize_function(examples):
        # Type 1 Data: <fps> + <selfies> → <selfies> (Random order of <fps><selfies>, MLM)
        # Type 2 Data: <selfies> → <selfies> (MLM)
        # Type 3 Data: <fps> → <selfies> (Translation Task)
        sep_token = "<fps_sep>"
        examples_num = len(examples["fps"])
        first_data = {
            "inputs": [],
            "labels": []
        }
        second_data = {
            "inputs": [],
            "labels": []
        }
        third_data = {
            "inputs": [],
            "labels": []
        }
        for i in range(examples_num):
            fps = examples["fps"][i]
            selfies = examples["selfies"][i]
            if random.randint(0, 1) == 0:
                first_data["inputs"].append(f"{fps}{sep_token}{selfies}")
            else:
                first_data["inputs"].append(f"{selfies}{sep_token}{fps}")
            first_data["labels"].append(selfies)

            second_data["inputs"].append(selfies)
            second_data["labels"].append(selfies)

            third_data["inputs"].append(fps)
            third_data["labels"].append(selfies)
        
        all_inputs = first_data["inputs"] + second_data["inputs"] + third_data["inputs"]
        all_labels = first_data["labels"] + second_data["labels"] + third_data["labels"]

        final_output = {}
        inputs_ids =  tokenizer(
            all_inputs,
            truncation=True,
            max_length=data_args.max_seq_length
        )
        final_output["input_ids"] = inputs_ids["input_ids"]

        labels = tokenizer(
            all_labels,
            truncation=True,
            max_length=data_args.max_seq_length
        )
        final_output["labels"] = labels["input_ids"]
        final_output["data_class"] = [1] * examples_num + [2] * examples_num + [3] * examples_num

        # Filter out samples with a length less than max_seq_length.
        filtered_indices = [i for i, ids in enumerate(final_output["input_ids"]) if len(ids) < data_args.max_seq_length]
        if len(filtered_indices) != 3*examples_num:
            logger.warning("some selfies are too long, will be filter")
            final_output["input_ids"] = [final_output["input_ids"][i] for i in filtered_indices]
            final_output["labels"] = [final_output["labels"][i] for i in filtered_indices]
            final_output["data_class"] = [final_output["data_class"][i] for i in filtered_indices]
        
        return final_output


    with training_args.main_process_first(desc="dataset map processing"):
        tokenized_dataset = pretrain_datasets.map(
            function=tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns= pretrain_datasets["train"].column_names
        )

    # https://huggingface.co/docs/transformers/main/en/trainer#customize
    class BartPretrainTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch = None):
            input_ids = inputs["input_ids"]
            decoder_input_ids = inputs["decoder_input_ids"]
            attention_mask = input_ids != self.processing_class.pad_token_id
            labels = inputs["labels"]
            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids, 
                            labels=labels)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss
    
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)
            
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        labels_flat = labels.reshape(-1)
        preds_flat = preds.reshape(-1)
        mask = labels_flat != -100
        labels_flat = labels_flat[mask]
        preds_flat = preds_flat[mask]
        correct_predictions = np.sum(labels_flat == preds_flat)
        total_predictions = len(labels_flat)
        token_level_accuracy = correct_predictions / total_predictions
        
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = [label.strip() for label in decoded_labels]

        cnt = 0
        for pred, label in zip(decoded_preds, decoded_labels):
            if pred == label:
                cnt += 1
        accuracy = cnt / len(decoded_preds)
        
        return {"token_level_accuracy": token_level_accuracy, "sequence_level_accuracy": accuracy}

    data_collator = DataCollatorForDenoisingTasks(
        tokenizer=tokenizer,
        mask_ratio=data_args.mask_ratio,
        poisson_lambda=data_args.poisson_lambda,
        pad_to_multiple_of=data_args.pad_to_multiple_of,
    )
    training_args.remove_unused_columns = False
    training_args.lr_scheduler_kwargs={"min_lr": model_args.min_lr} if training_args.lr_scheduler_type == 'cosine_with_min_lr' else None
    trainer = BartPretrainTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
    )

    # Training
    if training_args.do_train:
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
        elif training_args.resume_from_checkpoint is not None:
            last_checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        
        metrics = train_result.metrics
        metrics["train_samples"] = len(tokenized_dataset["train"])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(tokenized_dataset["val"])

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "bart pretrain"}
    trainer.create_model_card(**kwargs)

if __name__ == "__main__":
    main()
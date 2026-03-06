#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
# Refer to: https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import datasets
import numpy as np
import matchms
import matchms.filtering as ms_filters
from datasets import load_dataset, DatasetDict, load_from_disk, Dataset
from collections import defaultdict
import torch
from copy import deepcopy
import json
import torch.nn.functional as F

import transformers
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    HfArgumentParser,
    EarlyStoppingCallback,
    GenerationConfig,
    DataCollatorWithPadding
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
import warnings

sys.path.append(".")
from src.metric import MoleculeEvaluator, TopkMoleculeEvaluator
from src.utils import save_arr, read_jsonl, print_trainable_parameters
from src.rank_rl.trainer import RankLossSeq2SeqTrainer

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    finetune_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the finetune data."}
    )

    val_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the val data."}
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=10,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    debug_flag: bool = field(default=False, metadata={"help": "Debug mode."})

@dataclass
class CustomTrainingArguments(Seq2SeqTrainingArguments):
    cand_margin: float = field(default=0.001, metadata={"help": "Margin for candidate ranking loss"})
    gold_margin: float = field(default=0, metadata={"help": "Margin for gold ranking loss"})
    gold_weight: float = field(default=0, metadata={"help": "Weight for gold ranking loss"})
    rank_weight: float = field(default=1.0, metadata={"help": "Weight for ranking loss"})
    mle_weight: float = field(default=1.0, metadata={"help": "Weight for MLE loss"})
    normalize: bool = field(default=True, metadata={"help": "Normalize predicited likelihood"})
    score_mode: str = field(default="log", metadata={"help": "Use log-likelihood for ranking loss"})
    smooth: float = field(default=0.1, metadata={"help": "label smoothing"})
    length_penalty: float = field(default=1.0, metadata={"help": "length penalty"})
    num_generation: int = field(default=3, metadata={"help": "use num of generation for ranking loss < 10"})
    freeze_encoder: bool = field(default=False, metadata={"help": "freeze encoder"})
    loss_type: str = field(default="mle", metadata={"help": "the secondary loss type, can be 'mle' or 'kl' "})
    kl_weight: float = field(default=0.04, metadata={"help": "Weight for KL loss"})
    temperature: float = field(default=0.4, metadata={"help": "Temperature for generation"})
    early_stopping_patience: int = field(default=3, metadata={"help": "Step patience for early stopping"})
    

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    set_seed(training_args.seed)
    if training_args.should_log:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
            + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
        )
        logger.info(f"Training/evaluation parameters {training_args}")
        logger.info(f"Model parameters {model_args}")
        logger.info(f"Data parameters {data_args}")

    finetune_dataset = load_dataset("csv", data_files={"train": data_args.finetune_path}, delimiter="\t")["train"]
    val_dataset = load_dataset("csv", data_files={"val": data_args.val_path}, delimiter="\t")["val"]

    if data_args.debug_flag: 
        # finetune_dataset = finetune_dataset.select(range(1000))
        finetune_dataset = finetune_dataset.select(range(100))
        val_dataset = val_dataset.select(range(100))
        training_args.report_to = []
        training_args.overwrite_output_dir = True

    tokenizer_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
    model = BartForConditionalGeneration.from_pretrained(model_args.model_name_or_path)

    if training_args.freeze_encoder:
        for param in model.model.encoder.parameters():
            param.requires_grad = False
        if training_args.should_log:
            print_trainable_parameters(model)

    if training_args.should_log:
        print("=="*24, "Model Config", "=="*24)
        logger.info(f"{model.config}")
        logger.info(f"Tokenizer is fast: {tokenizer.is_fast}")
        logger.info(f"Model vocab size matches tokenizer: {model.config.vocab_size == len(tokenizer)}")
        logger.info(f"Model dtype: {next(model.parameters()).dtype}")

    def tokenize_function(examples):
        # remove pairs where at least one record is None
        inputs, targets = [], []
        for i in range(len(examples["fps"])):
            if examples["fps"][i] and examples["selfies"][i]:
                inputs.append(examples["fps"][i])
                targets.append(examples["selfies"][i])

        model_inputs = tokenizer(
            inputs,
            max_length=data_args.max_source_length,
            padding="longest", 
            truncation=True, 
            add_special_tokens=True
        )

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=targets,
            max_length=data_args.max_target_length,
            padding="longest", 
            truncation=True, 
            add_special_tokens=True
        )

        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
    
        return model_inputs

    # https://github.com/huggingface/transformers/blob/800510c67bfc5cedd0bb7635648a07f39719be43/examples/pytorch/summarization/run_summarization.py#L581
    with training_args.main_process_first(desc="Training dataset map processing"):
        finetune_dataset = finetune_dataset.map(
            function=tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns= finetune_dataset.column_names,
            desc="Running tokenizer on training dataset",
        )

    def tokenize_function_val(examples):
        inputs, targets = [], []
        for i in range(len(examples["fps"])):
            if examples["fps"][i] and examples["selfies"][i]:
                inputs.append(examples["fps"][i])
                targets.append(examples["selfies"][i])
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding="longest", truncation=True, add_special_tokens=True)
        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(targets, max_length=data_args.max_target_length, padding="longest",  truncation=True, add_special_tokens=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    with training_args.main_process_first(desc="Val dataset map processing"):
        val_dataset = val_dataset.map(
            function=tokenize_function_val,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns= val_dataset.column_names,
            desc="Running tokenizer on validation dataset",
        )
    
    if training_args.should_log: 
        logger.info("Training Examples")
        for index in range(3):
            sample = finetune_dataset[index]
            input_ids = [id if id != -100 else tokenizer.pad_token_id for id in sample["input_ids"]]
            input_text = tokenizer.decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            label_ids = [id if id != -100 else tokenizer.pad_token_id for id in sample["labels"]]
            label_text = tokenizer.decode(label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            logger.info(f"Input text: {input_text}")
            logger.info(f"Labels: {label_text}")
            logger.info("="*50)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        batch_size = len(labels)
        if len(preds) > batch_size:  # Top-K generation
            num_return_sequences = len(preds) // batch_size
        else:
            num_return_sequences = 1
        logger.info(f"Batch Size: {batch_size}, Num Return Sequences: {num_return_sequences}")
        
        if num_return_sequences  > 1:
            preds = [
                preds[i * num_return_sequences : (i + 1) * num_return_sequences] 
                for i in range(batch_size)
            ]
            top1_preds = [pred[0] for pred in preds]
        else:
            top1_preds = preds

        logger.info(f"Decoded Predictions: {top1_preds[:5]}")
        logger.info(f"Decoded Labels: {labels[:5]}")

        evaluator = MoleculeEvaluator()
        top1_results = evaluator.evaluate_de_novo_step_selfies(top1_preds, labels)
        total_results = top1_results

        if num_return_sequences > 1:
            topk_evaluator = TopkMoleculeEvaluator()
            top10_results = topk_evaluator.evaluate_de_novo_step_selfies_top_k(preds, labels)
            total_results = {**total_results, **top10_results}
        return total_results

    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
    )

    # # Initialize our Trainer
    # training_args.remove_unused_columns = False # Disable automatic column deletion.
    trainer = RankLossSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=finetune_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=[
            # If there is no improvement in 3 evaluation steps, trigger early stopping. Ensure to set load_best_model_at_end=True, and it is recommended that save_step and eval_step be consistent.
            EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience),
        ],
    )

    # Training
    if training_args.do_train:
        last_checkpoint = None
        # if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if training_args.resume_from_checkpoint is not None:
            last_checkpoint = training_args.resume_from_checkpoint
        else:
            if os.path.isdir(training_args.output_dir):
                import shutil
                try:
                    shutil.rmtree(training_args.output_dir)
                    print(f"Folder {training_args.output_dir} has been deleted successfully.")
                except Exception as e:
                    print(f"An error occurred while deleting the folder: {e}")
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        
        metrics = train_result.metrics
        metrics["train_samples"] = len(finetune_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(val_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        logger.info(f"Generation config: {model.generation_config}")
        predict_dataset = val_dataset
        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
        metrics = predict_results.metrics
        metrics["predict_samples"] = len(predict_dataset)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = predict_results.predictions
                predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                predictions = tokenizer.batch_decode(
                    predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip().replace(" ", "") for pred in predictions]

                 # Get and process labels
                labels = predict_results.label_ids
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                labels = tokenizer.batch_decode(
                    labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                labels = [label.strip().replace(" ", "") for label in labels]

                batch_size = len(labels)
                if len(predictions) > batch_size:  #  Top-K generation
                    num_return_sequences = len(predictions) // batch_size
                else:
                    num_return_sequences = 1

                save_jsonl = []
                for i in range(batch_size):
                    if num_return_sequences > 1:
                        pred = predictions[i * num_return_sequences : (i + 1) * num_return_sequences]
                    else:
                        pred = [predictions[i]]
                    save_jsonl.append({"pred": pred, "label": labels[i]})

                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.jsonl")
                save_arr(save_jsonl, output_prediction_file)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "MS Token Fine-tuning" }
    trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()

import os
os.environ["WANDB_DISABLED"] = "true"
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

import evaluate
import shutil

import pandas as pd
from datasets import Dataset

import logging
logger = logging.getLogger("fine_tune_pipeline")

EPOCHS=5

def compute_metrics(eval_pred):
    metric = evaluate.load("f1")
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return metric.compute(predictions=preds, references=labels)


def to_hf_dataset(claim_list_1, claim_list_2, label_list):
    return Dataset.from_dict({"sentence1": claim_list_1,
                              "sentence2": claim_list_2,
                              "label": label_list})

def train_and_save(model:AutoModelForSequenceClassification,
                   tokenized_train:Dataset,
                   tokenized_val:Dataset,
                   tokenizer:AutoTokenizer,
                   project_path:str,
                   filename:str,
                   learning_rate:float):
    model_name = filename.split("/")[-2]
    checkpoint_dir = f"{project_path}/checkpoints/{model_name}"
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="no",
        #logging_steps=50,
        learning_rate=learning_rate,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(filename)
    tokenizer.save_pretrained(filename)
    logger.info(f"Model saved to: {filename}")
    # remove checkpointing dir
    shutil.rmtree(checkpoint_dir)



def fine_tune_bert_model(model:AutoModelForSequenceClassification,
                         tokenizer:AutoTokenizer,
                         claim_list_1:list[str],
                         claim_list_2:list[str],
                         label_list:list[int],
                         project_path:str,
                         filename:str,
                         learning_rate:float=2e-5):
    """
        hf_model_name: checkpoint on huggingface, like "xlm-roberta-large"
    """
    hf_dataset = to_hf_dataset(claim_list_1, claim_list_2, label_list)
    split = hf_dataset.train_test_split(test_size=0.1)
    hf_df_train = split["train"]
    hf_df_val = split["test"]

    #tokenizer = AutoTokenizer.from_pretrained(hf_model_name,
    #                                          use_fast=True)
    #model = AutoModelForSequenceClassification.from_pretrained(hf_model_name,
    #                                                           num_labels=2)
    def preprocess_function(df):
        return tokenizer(
            df["sentence1"],
            df["sentence2"],
            truncation=True,
            max_length=256,
        )

    tokenize = lambda hf_df: hf_df.map(preprocess_function, batched=True)
    tokenized_train = tokenize(hf_df_train)
    tokenized_val = tokenize(hf_df_val)

    train_and_save(model,
                   tokenized_train,
                   tokenized_val,
                   tokenizer,
                   project_path,
                   filename,
                   learning_rate)


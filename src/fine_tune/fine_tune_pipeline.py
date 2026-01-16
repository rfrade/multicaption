import random

import pandas as pd
from src.config import config_dict
from src.fine_tune.fine_tune_encoder import fine_tune_bert_model
import numpy as np
import torch
import os
from pathlib import Path
import logging
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification)

from src.util.logging_util import init_log


def load_multicaption(project_path: str, ablation: str | None) -> tuple[pd.DataFrame]:
    multicaption_train = pd.read_csv(f"{project_path}/dataset/multicaption_train.csv")

    remove_map = {
        "PARAPHRASE": ["GPT5-paraphrase"],
        "NEGATION": ["Negation-Filter"],
        "PARAPHRASE-NEGATION": ["GPT5-paraphrase", "Negation-Filter"],
    }

    if ablation in remove_map:
        multicaption_train = multicaption_train[~multicaption_train["label_strategy"].isin(remove_map[ablation])]
        print(f"Filtered out for ablation {ablation}. Filtered training dataset size: {multicaption_train.shape[0]}")

    return multicaption_train

def get_nli_config(hf_model_name: str):
    """hf_model_name: name of the model in the huffing face"""
    config = AutoConfig.from_pretrained(hf_model_name)
    config.num_labels = 2
    config.id2label = {0: "non-contradiction", 1: "contradiction"}
    config.label2id = {"non-contradiction": 0, "contradiction": 1}
    return config

def finetune_nli_classifiers(project_path:str,
                             multicaption_train:pd.DataFrame) -> None:
    path_ft_models = f"{project_path}/fine_tuned_models/"
    Path(path_ft_models).mkdir(parents=True, exist_ok=True)

    multilingual_models = [
        #"sentence-transformers/nli-distilbert-base",#for local testing only
        "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "joeddav/xlm-roberta-large-xnli"
    ]

    learning_rate=1e-6

    for i in range(5):
        for model_name in multilingual_models:
            # train on original captions
            logger.info(f"Starting fine-tune for: {model_name}")
            config = get_nli_config(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                      use_fast=True)
            ### use original claims
            model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                       config=config,
                                                                       ignore_mismatched_sizes=True)
            fine_tune_bert_model(model=model,
                                 tokenizer=tokenizer,
                                 claim_list_1=multicaption_train["claim_1"],
                                 claim_list_2=multicaption_train["claim_2"],
                                 label_list=multicaption_train["label"],
                                 project_path=project_path,
                                 filename=f"{project_path}/fine_tuned_models/{model_name.split('/')[-1]}_original/{i}",
                                 learning_rate=learning_rate)

            ### use translated claims
            model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                       config=config,
                                                                       ignore_mismatched_sizes=True)
            fine_tune_bert_model(model=model,
                                 tokenizer=tokenizer,
                                 claim_list_1=multicaption_train["claim_1_en"],
                                 claim_list_2=multicaption_train["claim_2_en"],
                                 label_list=multicaption_train["label"],
                                 project_path=project_path,
                                 filename=f"{project_path}/fine_tuned_models/{model_name.split('/')[-1]}_en/{i}",
                                 learning_rate=learning_rate)


def set_seed(i:int):
    np.random.seed(i)
    random.seed(i)
    torch.manual_seed(i)
    transformers.set_seed(i)

def finetune_bert_classifiers(project_path:str,
                              multicaption_train:pd.DataFrame) -> None:
    path_ft_models = f"{project_path}/fine_tuned_models/"
    Path(path_ft_models).mkdir(parents=True, exist_ok=True)

    model_list = [#"google-bert/bert-base-multilingual-cased",
                  #"xlm-roberta-base",
                  #"microsoft/mdeberta-v3-base",
                  #"xlm-roberta-large"
                 ]

    for model_name in model_list:
        for i in range(5):
            set_seed(42+i)
            ## train on original captions
            logger.info(f"\n ## Starting fine-tune for: {model_name}-{i} ## \n")
#
            #tokenizer = AutoTokenizer.from_pretrained(model_name,
            #                                          use_fast=True)
            #model = AutoModelForSequenceClassification.from_pretrained(model_name,
            #                                                           num_labels=2)
#
            #fine_tune_bert_model(model=model,
            #                     tokenizer=tokenizer,
            #                     claim_list_1=multicaption_train["claim_1_en"],
            #                     claim_list_2=multicaption_train["claim_2_en"],
            #                     label_list=multicaption_train["label"],
            #                     project_path=project_path,
            #                     filename=f"{project_path}/fine_tuned_models/{model_name.split('/')[-1]}_en/{i}")

            tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                      use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                       num_labels=2)

            fine_tune_bert_model(model=model,
                                 tokenizer=tokenizer,
                                 claim_list_1=multicaption_train["claim_1"],
                                 claim_list_2=multicaption_train["claim_2"],
                                 label_list=multicaption_train["label"],
                                 project_path=project_path,
                                 filename=f"{project_path}/fine_tuned_models/{model_name.split('/')[-1]}_original/{i}")
            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # clear cached memory
                torch.cuda.ipc_collect()  # force CUDA to release unused memory


def sample_negations(fact_checks_neg: pd.DataFrame,
                     n:int) -> pd.DataFrame:
    negations_sample = fact_checks_neg.query("similarity > 0.5 & similarity < 0.95").sample(n)

    rename_dict = {"fact_check_id": "cid_1",
                   "claim": "claim_1",
                   "title": "claim_2",
                   "claim_en": "claim_1_en",
                   "title_en": "claim_2_en",
                   "claim_detected_language_iso": "language_1",
                   "title_detected_language_iso": "language_2",
                   "similarity": "cosine_similarity_en"
                   }
    negations_sample = negations_sample.rename(columns=rename_dict)
    negations_sample["cid_2"] = negations_sample["cid_1"]
    negations_sample["cos_bin_id"] = pd.NA
    negations_sample["label"] = 1
    negations_sample["label_name"] = "contradiction"
    negations_sample["label_strategy"] = "Negation-Filter"
    negations_sample["type_1"] = "claim"
    negations_sample["type_2"] = "title"

    cols_multicaption = ['cid_1', 'claim_1', 'claim_1_en', 'language_1', 'type_1', 'cid_2',
                         'claim_2', 'claim_2_en', 'language_2', 'type_2', 'label_name',
                         'label_strategy', 'cosine_similarity_en', 'label', 'cos_bin_id']
    negations_sample = negations_sample[cols_multicaption]
    return negations_sample

def fine_bert_for_ablation(data_config):
    for dataset, description in data_config:
        model_name = "xlm-roberta-large"
        for i in range(5):
            logger.info(f"Starting fine-tune for: {model_name}/{i}")
            set_seed(42+i)
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
            fine_tune_bert_model(model=model,
                                 tokenizer=tokenizer,
                                 claim_list_1=dataset["claim_1_en"].tolist(),
                                 claim_list_2=dataset["claim_2_en"].tolist(),
                                 label_list=dataset["label"].tolist(),
                                 project_path=project_path,
                                 filename=f"{project_path}/ablation_models_en/{model_name.split('/')[-1]}_{description}/{i}")

def fine_nli_for_ablation(data_config):
    model_name = "joeddav/xlm-roberta-large-xnli"
    nli_config = get_nli_config(model_name)

    for dataset, description in data_config:
        for i in range(5):
            logger.info(f"Starting fine-tune for: {model_name}/{i}")
            set_seed(42+i)
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                       config=nli_config,
                                                                       ignore_mismatched_sizes=True)
            fine_tune_bert_model(model=model,
                                 tokenizer=tokenizer,
                                 claim_list_1=dataset["claim_1"].tolist(),
                                 claim_list_2=dataset["claim_2"].tolist(),
                                 label_list=dataset["label"].tolist(),
                                 project_path=project_path,
                                 filename=f"{project_path}/ablation_models/{model_name.split('/')[-1]}_{description}/{i}",
                                 learning_rate=1e-6)


def ablation(project_path: str):
    """ try finetuning with different sample sizes"""
    multicaption_all = load_multicaption(project_path)
    logger.info(f"shape multicaltion_train: {len(multicaption_all)}")

    #no paraphrase, no negation
    multicaption_original = multicaption_all.query("label_strategy != 'GPT5-paraphrase' & label_strategy != 'Negation-Filter'")
    logger.info(f"shape multicaltion original data: {len(multicaption_original)}")
    # with paraphrase
    multicaption_paraphrase = multicaption_all.query("label_strategy != 'Negation-Filter'")
    logger.info(f"shape multicaltion with paraphrase: {len(multicaption_original)}")

    # with negation
    multicaption_negation = multicaption_all.query("label_strategy != 'GPT5-paraphrase'")
    logger.info(f"shape multicaltion with negation: {len(multicaption_negation)}")

    # (dataset, description)
    data_config = [(multicaption_original, "original"),
                    (multicaption_paraphrase, "paraphrase"),
                    (multicaption_negation, "negation"),
                    (multicaption_all, "all")
                    ]
    #fine_nli_for_ablation(data_config)
    fine_bert_for_ablation(data_config)

    # model_name, config, learning_rate
    #train_config = [("sentence-transformers/nli-distilbert-base", None, 1e-6),
    #                ("sentence-transformers/nli-distilbert-base", None, 2e-5)]


if __name__=="__main__":
    os.environ["WANDB_DISABLED"] = "true"
    project_path = config_dict["project_path"]
    init_log(logger_name="fine_tune_pipeline")
    logger = logging.getLogger("fine_tune_pipeline")

    multicaption_train = load_multicaption(project_path)
    #multicaption_train = multicaption_train.sample(50)

    #finetune_nli_classifiers(project_path, multicaption_train)
    #finetune_bert_classifiers(project_path, multicaption_train)
    ablation(project_path)

    #bert_classifiers_experiment(project_path)


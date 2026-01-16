from src.baselines.nli_methods import gen_nli_function
from src.baselines.llms import gen_llm_function
from src.baselines.finetuned_llms import gen_finetuned_llm_function
from src.evaluation.evaluation import eval_methods
import pandas as pd
from src.config import config_dict
from src.evaluation.evaluation_config import EvaluationConfig
from transformers import logging as transformers_log
transformers_log.set_verbosity_error()

import traceback
from src.baselines.finetuned_berts import gen_bert_function
import torch
import glob
import os

import logging
from src.util.logging_util import init_log

def load_datasets(project_path: str) -> tuple[pd.DataFrame]:

    multicaption_test_original = pd.read_csv(f"{project_path}/dataset/multicaption_test.csv")
    multicaption_test_en = pd.read_csv(f"{project_path}/dataset/multicaption_test.csv")
    multicaption_test_en["claim_1"] = multicaption_test_en["claim_1_en"]
    multicaption_test_en["claim_2"] = multicaption_test_en["claim_2_en"]
    cosmos_test = pd.read_csv(f"{project_path}/dataset/cosmos_test.csv")
    cosmos_test["claim_1"] = cosmos_test["claim_1_en"]
    cosmos_test["claim_2"] = cosmos_test["claim_2_en"]
    #cosmos_synthetic = pd.read_csv(f"{project_path}/dataset/cosmos_synthetic.csv")

    return cosmos_test, multicaption_test_original, multicaption_test_en


def run_pretrained_nli_methods(cosmos_test,
                                multicaption_test_original,
                                multicaption_test_en) -> None:

    multilingual = [
        "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "joeddav/xlm-roberta-large-xnli"
    ]

    #for dataset, dataset_name in dataset_list:
    for model_name_hf in multilingual:
        nli_function = gen_nli_function(model_name_hf)
        model_name = model_name_hf.split("/")[1]
        eval_cosmos = EvaluationConfig(function=nli_function, function_name="NLI",
                                        model_name=model_name,
                                        dataset=cosmos_test, dataset_name="cosmos_test",
                                        description="baseline,neutral=1", use_translation=False,
                                        iterations=1)
        eval_methods([eval_cosmos])
        eval_multicaption_en = EvaluationConfig(function=nli_function, function_name="NLI",
                                                model_name=model_name,
                                                dataset=multicaption_test_en, dataset_name="multicaption_test_en",
                                                description="baseline,neutral=1", use_translation=False,
                                                iterations=1)
        eval_methods([eval_multicaption_en])
        nli_function = gen_nli_function(model_name_hf)
        model_name = model_name_hf.split("/")[1]
        eval_multilingual = EvaluationConfig(function=nli_function, function_name="NLI", model_name=model_name,
                                             dataset=multicaption_test_original, dataset_name="multicaption_test_original",
                                             description="baseline,neutral=1", use_translation=False,
                                             iterations=1)
        eval_methods([eval_multilingual])


def run_llms(dataset_list: list[tuple[pd.DataFrame, str]], config: dict) -> None:

    multilingual = [
        "microsoft/Phi-4-mini-instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Llama-3.1-8B-Instruct",
        "google/gemma-7b",
        "Qwen/Qwen2.5-7B-Instruct"
        ]

    monolingual = [
        ]

    for dataset, dataset_name in dataset_list:
        for model_name_hf in monolingual:
            llm_function = gen_llm_function(model_name_hf, config)
            model_name = model_name_hf.split("/")[1]
            eval = EvaluationConfig(function=llm_function, function_name="LLM-ZS", model_name=model_name,
                                     dataset=dataset, dataset_name=dataset_name,
                                     description="prompt1", use_translation=True)
            eval_methods([eval])

        for translation in [True, False]:
            if (dataset_name != "multi_caption_test") and (not translation):
                continue

            for model_name_hf in multilingual:
                print(f"Running {model_name_hf} on dataset {dataset_name} with {'English' if translation else 'original'} claims")

                if translation:
                    llm_function = gen_llm_function(model_name_hf, config)
                else:
                    lang_list1 = dataset['language_1'].tolist()
                    lang_list2 = dataset['language_2'].tolist()
                    llm_function = gen_llm_function(model_name_hf, config, translation, lang_list1, lang_list2)
                model_name = model_name_hf.split("/")[1]
                eval = EvaluationConfig(function=llm_function, function_name="LLM-ZS", model_name=model_name,
                                        dataset=dataset, dataset_name=dataset_name,
                                        description="prompt1" if translation else "prompt2", use_translation=translation)
                try:
                    eval_methods([eval])
                except Exception as e:
                    error_details = traceback.format_exc()
                    print("Full error details:\n", error_details)

def run_finetuned_llms(dataset_list: list[tuple[pd.DataFrame, str]], config: dict) -> None:

    multilingual = [
        "microsoft/Phi-4-mini-instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Llama-3.1-8B-Instruct",
        "google/gemma-7b",
        "Qwen/Qwen2.5-7B-Instruct",
        ]

    monolingual = [
        ]

    for dataset, dataset_name in dataset_list:
        for model_name_hf in monolingual:
            llm_function = gen_finetuned_llm_function(model_name_hf, config)
            model_name = model_name_hf.split("/")[1] + "-FT-MO"

            eval = EvaluationConfig(function=llm_function, function_name="LLM-FT", model_name=model_name,
                                     dataset=dataset, dataset_name=dataset_name,
                                     description="prompt1", use_translation=True)
            eval_methods([eval])

        for translation in [True, False]:
            if (dataset_name != "multi_caption_test") and (not translation):
                continue

            for model_name_hf in multilingual:
                print(f"Running {model_name_hf} on dataset {dataset_name} with {'English' if translation else 'original'} claims")

                if translation:
                    llm_function = gen_finetuned_llm_function(model_name_hf, config)
                else:
                    lang_list1 = dataset['language_1'].tolist()
                    lang_list2 = dataset['language_2'].tolist()
                    llm_function = gen_finetuned_llm_function(model_name_hf, config, translation, lang_list1, lang_list2)

                if translation:
                    model_name = model_name_hf.split("/")[1] + "-FT-MO"
                else:
                    model_name = model_name_hf.split("/")[1] + "-FT-MU"

                eval = EvaluationConfig(function=llm_function, function_name="LLM-FT", model_name=model_name,
                                        dataset=dataset, dataset_name=dataset_name,
                                        description="prompt1" if translation else "prompt2", use_translation=translation,
                                        save_predictions=False)
                try:
                    eval_methods([eval])
                except Exception as e:
                    error_details = traceback.format_exc()
                    print("Full error details:\n", error_details)


def run_finetuned_llms_ablation(dataset_list: list[tuple[pd.DataFrame, str]], config: dict) -> None:

    multilingual = [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "microsoft/Phi-4-mini-instruct"
        ]

    monolingual = [
        ]

    for ablation in ["PARAPHRASE", "NEGATION", "PARAPHRASE-NEGATION"]:
        for dataset, dataset_name in dataset_list:
            for model_name_hf in monolingual:
                llm_function = gen_finetuned_llm_function(model_name_hf, config, True, None, None, ablation)
                model_name = model_name_hf.split("/")[1] + "-FT-MO-" + ablation

                eval = EvaluationConfig(function=llm_function, function_name="LLM-FT", model_name=model_name,
                                         dataset=dataset, dataset_name=dataset_name,
                                         description="prompt1", use_translation=True)
                eval_methods([eval])

            for translation in [True, False]:
                if (dataset_name != "multi_caption_test") and (not translation):
                    continue

                for model_name_hf in multilingual:
                    print(f"Running {model_name_hf} on dataset {dataset_name} with {'English' if translation else 'original'} claims")

                    if translation:
                        llm_function = gen_finetuned_llm_function(model_name_hf, config, True, None, None, ablation)
                    else:
                        lang_list1 = dataset['language_1'].tolist()
                        lang_list2 = dataset['language_2'].tolist()
                        llm_function = gen_finetuned_llm_function(model_name_hf, config, translation, lang_list1, lang_list2, ablation)

                    if translation:
                        model_name = model_name_hf.split("/")[1] + "-FT-MO-" + ablation
                    else:
                        model_name = model_name_hf.split("/")[1] + "-FT-MU-" + ablation

                    eval = EvaluationConfig(function=llm_function, function_name="LLM-FT", model_name=model_name,
                                            dataset=dataset, dataset_name=dataset_name,
                                            description="prompt1" if translation else "prompt2", use_translation=translation)
                    try:
                        eval_methods([eval])
                    except Exception as e:
                        error_details = traceback.format_exc()
                        print("Full error details:\n", error_details)



def run_finetuned_bert_classifiers(project_path:str,
                                   dataset_list:list[pd.DataFrame],
                                   device:str):

    model_paths = [f for f in glob.glob(os.path.join(path_ft_models, '*')) if os.path.isdir(f)]

    datasets = [(cosmos, "cosmos_test"),
                (multicaption_en, "multicaption_en"),
                (multicaption_original, "multicaption_original")]
    # Run every model for each dataset
    for dataset, dataset_name in datasets:
        for model_path in model_paths:
            #model_path = f"{path_ft_models}/{model_path}"
            bert_function = gen_bert_function(model_root=model_path,
                                              device=device)
            model_name = model_path.split("/")[-1]
            eval = EvaluationConfig(function=bert_function, function_name="BERT",
                                    model_name=model_name,
                                    dataset=dataset, dataset_name=dataset_name,
                                    description="fine-tuned", use_translation=False)
            eval_methods([eval])

def save_predictions(project_path:str,
                     cosmos,
                     multicaption_en,
                     multicaption_original,
                     device:str):
    logger.info("\n### Saving predictions for error analysis ###\n")
    #path_ft_deberta = f"{project_path}/fine_tuned_models/mdeberta-v3-base_original"
    path_ft_deberta_nli = f"{project_path}/fine_tuned_models/mDeBERTa-v3-base-mnli-xnli_original"

    error_analysis_path = f"{project_path}/error_analysis"

    datasets = [(cosmos, "cosmos_test"),
                (multicaption_en, "multicaption_en"),
                (multicaption_original, "multicaption_original")]
    # Run every model for each dataset

    for model_path in [path_ft_deberta_nli]:# [path_ft_deberta, path_ft_xlmr_nli]:
        for dataset, dataset_name in datasets:

            bert_function = gen_bert_function(model_root=model_path,
                                              device=device)
            model_name = model_path.split("/")[-1]
            logger.info(f"\n### {model_name}-{dataset_name} ###\n")
            eval = EvaluationConfig(function=bert_function,
                                    function_name="BERT",
                                    model_name=model_name,
                                    dataset=dataset,
                                    dataset_name=dataset_name,
                                    description="fine-tuned",
                                    use_translation=False,
                                    iterations=1,
                                    save_predictions_to=error_analysis_path)
            eval_methods([eval])


if __name__=="__main__":
    init_log(logger_name="eval_pipeline")
    logger = logging.getLogger("eval_pipeline")
    pd.set_option('display.max_columns', 12)
    pd.set_option('display.width', 300)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    project_path = config_dict["project_path"]

    cosmos_test, multicaption_test_original, multicaption_test_en = load_datasets(project_path)
    logger.info("Datasets loaded")

    #dataset_list = [(cosmos_test, "cosmos_test"), (multi_caption_test, "multi_caption_test")]
    # Run baselines
    #run_llms(dataset_list, config_dict["llm_config"])
    #run_finetuned_bert_classifiers(project_path, dataset_list, device)
    run_finetuned_llms(dataset_list, config_dict)
    
    ### OTHER BRANCH
    #run_finetuned_llms_ablation(dataset_list, config_dict)
    #run_pretrained_nli_methods(cosmos_test, multicaption_test_original, multicaption_test_en)

    #path_ft_models = f"{project_path}/fine_tuned_models"
    #run_finetuned_bert_classifiers(path_ft_models=path_ft_models,
    #                               cosmos=cosmos_test,
    #                               multicaption_en=multicaption_test_en,
    #                               multicaption_original=multicaption_test_original,
    #                               device=device)

    #path_ft_models = f"{project_path}/ablation_models"
    #run_finetuned_bert_classifiers(path_ft_models=path_ft_models,
    #                               cosmos=cosmos_test,
    #                               multicaption_en=multicaption_test_en,
    #                               multicaption_original=multicaption_test_original,
    #                               device=device)

    #run_finetuned_llms(dataset_list, config_dict)

    #save_predictions(project_path=project_path,
    #                 cosmos=cosmos_test,
    #                 multicaption_en=multicaption_test_en,
    #                 multicaption_original=multicaption_test_original,
    #                 device=device)


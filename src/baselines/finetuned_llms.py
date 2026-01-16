import torch
from src.baselines.llms import (generate_test_prompt, generate_multilingual_test_prompt, generate_response,
                                getPossibleTokenIds)
from src.fine_tune.fine_tune_pipeline import load_multicaption
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from datasets import Dataset
from datetime import datetime
import gc
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
import os
from pathlib import Path
import pycountry
import re

def gen_finetuned_llm_function(hf_model_name, config, translation=True, lang_list1=None, lang_list2=None, ablation=None):
    """ Wrapper class to get llm
        Parameters:
            hf_model_name: model name on huggingface
        Example:
            xlm_roberta = gen_llm_function("joeddav/xlm-roberta-large-xnli")
            # Then pass xlm_roberta to eval_methods
    """

    def llm_impl(claim_list_1,
                 claim_list_2, iteration):
        predictions = get_finetuned_llm_predictions(claim_list_1,
                                                    claim_list_2,
                                                    hf_model_name,
                                                    config,
                                                    translation,
                                                    lang_list1,
                                                    lang_list2,
                                                    iteration,
                                                    ablation)
        return predictions
    return llm_impl


def get_finetuned_llm_predictions(claim_list_1: list[str],
                                  claim_list_2: list[str],
                                  hf_model_name: str,
                                  config: dict,
                                  translation: bool,
                                  lang_list1: list[str],
                                  lang_list2: list[str],
                                  iteration: int,
                                  ablation: str|None) -> list[int]:
    prediction_list = []
    size = len(claim_list_1)

    model, tokenizer = getFinetunedModelAndTokenizer(hf_model_name, config, translation, iteration, ablation) #Get the finetuned model
    torch.cuda.empty_cache()
    possible_tokens = getPossibleTokenIds(tokenizer)

    with torch.no_grad():
        for i in range(size):
            if translation:
                prompt = generate_test_prompt(claim_list_1[i], claim_list_2[i])
            else:
                prompt = generate_multilingual_test_prompt(claim_list_1[i],claim_list_2[i],lang_list1[i],lang_list2[i])
            predictions = generate_response(prompt, model, tokenizer, possible_tokens)
            prediction_list.append(predictions)

    del model, tokenizer
    torch.cuda.empty_cache()

    return prediction_list


def generate_train_prompt(sample):
    prompt = (
        "You are an expert fact-checking assistant. "
        "Your task is to decide if two claims about the same image or video contradict each other.\n\n"
        "Definition of Contradict:\n"
        "Answer 'Yes' if the two claims cannot both be true for the same image or video that is, they describe opposing or mutually exclusive facts "
        "(different events, people, locations, or circumstances). "
        "Answer 'No' if they could both be true, or if one simply adds compatible information. "
        "Two claims are contradicting only when both cannot be true simultaneously, not when one is just less specific or a partial correction.\n\n"
        f"Claim 1: {sample['claim_1_en'].strip()}\n"
        f"Claim 2: {sample['claim_2_en'].strip()}\n\n"
        "Question: Do Claim 1 and Claim 2 contradict each other according to the definition above?\n\n"
        "Answer with only one word: Yes or No\n\n"
        f"Answer: {'Yes' if int(sample['label']) == 1 else 'No'}"
    )

    sample['prompt'] = prompt
    return sample

def code_to_language(code):
    try:
        lang = pycountry.languages.get(alpha_3=code)
        return lang.name
    except:
        return "Unknown"


def generate_multilingual_train_prompt(sample):
    prompt = (
        "You are an expert multilingual fact-checking assistant. "
        "Your task is to decide if two claims about the same image or video contradict each other.\n\n"
        "The two claims below may be written in different languages, but always produce the final answer **in English only**." 
        "Definition of Contradict:\n"
        "Answer 'Yes' if the two claims cannot both be true for the same image or video that is, they describe opposing or mutually exclusive facts "
        "(different events, people, locations, or circumstances). "
        "Answer 'No' if they could both be true, or if one simply adds compatible information. "
        "Two claims are contradicting only when both cannot be true simultaneously, not when one is just less specific or a partial correction.\n\n"
        f"Claim 1 written in {code_to_language(sample['language_1'])}: {sample['claim_1'].strip()}\n"
        f"Claim 2 written in {code_to_language(sample['language_2'])}: {sample['claim_2'].strip()}\n\n"
        "Question: Do Claim 1 and Claim 2 contradict each other according to the definition above?\n\n"
        "Answer with only one word in English: Yes or No \n\n"
        f"Answer: {'Yes' if int(sample['label']) == 1 else 'No'}"
    )

    sample['prompt'] = prompt
    return sample


def get_largest_checkpoint(checkpoints_dir):
    """
    Returns the full path to the checkpoint folder
    with the largest step number.
    """
    # List all checkpoint directories
    ckpts = [
        d for d in os.listdir(checkpoints_dir)
        if os.path.isdir(os.path.join(checkpoints_dir, d)) and d.startswith("checkpoint-")
    ]

    if not ckpts:
        return None

    # Extract step numbers and find largest
    def extract_step(ckpt_name):
        match = re.search(r"checkpoint-(\d+)", ckpt_name)
        return int(match.group(1)) if match else -1

    largest = max(ckpts, key=extract_step)
    return os.path.join(checkpoints_dir, largest)


def is_lengthy(prompt):
    # Must end exactly with 'Answer: Yes' or 'Answer: No' (ignoring leading/trailing spaces)
    return prompt.strip().endswith("Answer: Yes") or prompt.strip().endswith("Answer: No")


def getFinetunedModelAndTokenizer(hf_model_name, config, translation, iteration, ablation):
    cache_dir = config['llm_config']['cache_dir']
    project_path = config['project_path']
    path_ft_models = f"{config['model_dir']}/fine_tuned_models/"
    Path(path_ft_models).mkdir(parents=True, exist_ok=True)
    token = config['llm_config']['hf_token']

    prefix = hf_model_name.split("/")[1]

    if translation:
        ft_type = "-FT-MO-"
    else:
        ft_type = "-FT-MU-"

    if not ablation:
        model_name = f"{prefix}{ft_type}{iteration}"
    else:
        model_name = f"{prefix}{ft_type}{ablation}-{iteration}"

    model_path = os.path.join(path_ft_models, model_name)

    # ===============================================================
    # 1. CHECK IF THE FINETUNED MODEL ALREADY EXISTS
    # ===============================================================
    if os.path.exists(model_path) and os.path.isdir(model_path):
        best_ckpt = get_largest_checkpoint(model_path)

        if best_ckpt is not None:
            print(f"Loading existing finetuned model from: {model_path}")
            print("Using checkpoint:", best_ckpt)

            tokenizer = AutoTokenizer.from_pretrained(best_ckpt)

            model = AutoModelForCausalLM.from_pretrained(
                hf_model_name,
                device_map="auto",
                cache_dir=cache_dir,
                token=token
            )

            # Load LoRA adapter
            model = PeftModel.from_pretrained(
                model,
                best_ckpt,
            )

            model.eval()
            return model, tokenizer

    # ===============================================================
    # 2. IF NOT AVAILABLE → TRAIN A NEW FINETUNED MODEL
    # ===============================================================
    print("Finetuned model not found. Starting training...")

    # 4-bit Quantization Configuration
    compute_dtype = getattr(torch, "float16")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype
    )
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        device_map='auto',
        quantization_config=quant_config,
        cache_dir=cache_dir,
        token=token,
        offload_folder=cache_dir + "./offload"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, cache_dir=cache_dir,
                                              token=token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    multicaption_train = load_multicaption(project_path, ablation)

    if translation:
        multicaption_train = multicaption_train.apply(generate_train_prompt, axis=1)
    else:
        multicaption_train = multicaption_train.apply(generate_multilingual_train_prompt, axis=1)

    # Iterate through all rows and print prompts that don't match
    multicaption_train = multicaption_train[multicaption_train['prompt'].apply(is_lengthy)]

    hf_dataset = Dataset.from_pandas(multicaption_train)
    tokenized_dataset = hf_dataset.map(
        lambda samples: tokenizer(samples["prompt"], truncation=True),
        batched=True,
        remove_columns=hf_dataset.column_names
    )

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear"
    )

    args = TrainingArguments(
        output_dir=model_path,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=100,
        optim="paged_adamw_8bit",
        learning_rate=2e-5,
        fp16=True,
        weight_decay=0.001,
        max_grad_norm=0.3, warmup_ratio=0.03, group_by_length=True,
        run_name=f"{model_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
        lr_scheduler_type='constant'
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=tokenized_dataset,
        peft_config=peft_config,
    )

    gc.collect()

    torch.cuda.empty_cache()

    print("Training started at : ", datetime.now().strftime('%Y-%m-%d-%H-%M'))
    trainer.train()
    print("Training ended at: ", datetime.now().strftime('%Y-%m-%d-%H-%M'))

    model.eval()

    return model, tokenizer
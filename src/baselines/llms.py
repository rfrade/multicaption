from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LogitsProcessor
import torch
import pycountry


def generate_test_prompt(claim1, claim2):
    prompt = (
        "You are an expert fact-checking assistant. "
        "Your task is to decide if two claims about the same image or video contradict each other.\n\n"
        "Definition of Contradict:\n"
        "Answer 'Yes' if the two claims cannot both be true for the same image or video that is, they describe opposing or mutually exclusive facts "
        "(different events, people, locations, or circumstances). "
        "Answer 'No' if they could both be true, or if one simply adds compatible information. "
        "Two claims are contradicting only when both cannot be true simultaneously, not when one is just less specific or a partial correction.\n\n"
        f"Claim 1: {claim1.strip()}\n"
        f"Claim 2: {claim2.strip()}\n\n"
        "Question: Do Claim 1 and Claim 2 contradict each other according to the definition above?\n\n"
        "Answer with only one word: Yes or No \n\n"
        "Answer: "
    )

    return prompt

def code_to_language(code):
    try:
        lang = pycountry.languages.get(alpha_3=code)
        return lang.name
    except:
        return "Unknown"


def generate_multilingual_test_prompt(claim1, claim2, lang1, lang2):
    prompt = (
        "You are an expert multilingual fact-checking assistant. "
        "Your task is to decide if two claims about the same image or video contradict each other.\n\n"
        "The two claims below may be written in different languages, but always produce the final answer **in English only**." 
        "Definition of Contradict:\n"
        "Answer 'Yes' if the two claims cannot both be true for the same image or video that is, they describe opposing or mutually exclusive facts "
        "(different events, people, locations, or circumstances). "
        "Answer 'No' if they could both be true, or if one simply adds compatible information. "
        "Two claims are contradicting only when both cannot be true simultaneously, not when one is just less specific or a partial correction.\n\n"
        f"Claim 1 written in {code_to_language(lang1)}: {claim1.strip()}\n"
        f"Claim 2 written in {code_to_language(lang2)}: {claim2.strip()}\n\n"
        "Question: Do Claim 1 and Claim 2 contradict each other according to the definition above?\n\n"
        "Answer with only one word in English: Yes or No \n\n"
        "Answer: "
    )

    return prompt


class YesOrNoTokens(LogitsProcessor):
    def __init__(self, allowed):
        self.allowed = allowed

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float('-inf'))
        for t in self.allowed:
            mask[:, t] = scores[:, t]
        return mask


def generate_response(prompt, model, tokenizer, possible_tokens):
    encoded_input = tokenizer(prompt, return_tensors="pt", truncation=True, padding=False)
    device = next(model.parameters()).device
    model_inputs = encoded_input.to(device)

    del encoded_input


    output = model.generate(
        **model_inputs,
        max_new_tokens=1,
        do_sample=True,
        temperature=0.1,
        logits_processor=[possible_tokens]
    )

    decoded_output = tokenizer.decode(output[0].cpu())

    del output
    torch.cuda.empty_cache()

    response = decoded_output.split('Answer: ')[1].lower()
    if "yes" in response:
        return 1
    elif "no" in response:
        return 0
    else:
        print(f"Last decoded output: {decoded_output}")
        raise ValueError("No correct prediction made after max attempts.")


def gen_llm_function(hf_model_name, config, translation=True, lang_list1=None, lang_list2=None):
    """ Wrapper class to get llm
        Parameters:
            hf_model_name: model name on huggingface
        Example:
            xlm_roberta = gen_llm_function("joeddav/xlm-roberta-large-xnli")
            # Then pass xlm_roberta to eval_methods
    """

    def llm_impl(claim_list_1,
                 claim_list_2):
        predictions = get_llm_predictions(claim_list_1,
                                          claim_list_2,
                                          hf_model_name,
                                          config,
                                          translation,
                                          lang_list1,
                                          lang_list2)
        return predictions
    return llm_impl


def get_model_and_tokenizer(hf_model_name, config):
    """
    Loads a Hugging Face model with 4-bit quantization (if supported) and its tokenizer,
    patches rope_scaling if needed, and sets it to eval mode.
    Returns: model, tokenizer
    """

    # Load config separately (patch rope_scaling if present)
    model_config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True, token=config["hf_token"])

    if hasattr(model_config, "rope_scaling"):
        model_config.rope_scaling = None  # avoids ValueError for Phi-4 / LLaMA-3

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        config=model_config,
        device_map="auto",  # automatically moves model to correct device(s)
        cache_dir=config["cache_dir"],
        token=config["hf_token"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )

    # Adjust config
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name,
        cache_dir=config["cache_dir"],
        use_auth_token=config["hf_token"]
    )

    model.eval()
    return model, tokenizer


def getPossibleTokenIds(tokenizer):
    yes_id = tokenizer("Yes", add_special_tokens=False).input_ids[0]
    no_id = tokenizer("No", add_special_tokens=False).input_ids[0]
    allowed = {yes_id, no_id}

    possible_tokens = YesOrNoTokens(allowed)
    return possible_tokens


def get_llm_predictions(claim_list_1: list[str],
                        claim_list_2: list[str],
                        hf_model_name: str,
                        config: dict,
                        translation: bool,
                        lang_list1: list[str],
                        lang_list2: list[str]) -> list[int]:
    prediction_list = []
    size = len(claim_list_1)

    model, tokenizer = get_model_and_tokenizer(hf_model_name, config)
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
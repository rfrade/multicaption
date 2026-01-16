import openai
import json
from pydantic import BaseModel
from typing import List
import json
from src.config import config_dict
import pandas  as pd
from tqdm import tqdm
import logging
from src.util.logging_util import init_log

class ParaphrasesOutput(BaseModel):
    paraphrases: List[str]

def chunks(lst, size):
    """Yield successive size-sized chunks from lst."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def generate_paraphrases(sentence_pairs:list[tuple[str]],
                         api_key:str,
                         prefix_param:str) -> list[str]:
    logger.info(f"starting paraphrasing for the {prefix_param}. Total claims: {len(sentence_pairs)}")

    client = openai.OpenAI(api_key=api_key)

    # ---------- CONFIG ----------
    CHUNK_SIZE = 20
    MODEL_NAME = "gpt-5-mini"
    # ----------------------------

    # ---------- PROMPT PREFIX (CACHED) ----------
    PROMPT_PREFIX = f"""You are given a list of image caption pairs.
    For each pair, generate a paraphrase of {prefix_param}.
    The paraphrase should keep the same meaning, but be as different as possible.\n"""
    PROMPT_PREFIX += """
    Return results as JSON with the following structure:

    {
      "paraphrases": ["...", "..."]
    }

    Sentence pairs:
    """
    # --------------------------------------------

    all_paraphrases = []

    for chunk in tqdm(chunks(sentence_pairs, CHUNK_SIZE)):
        prompt = PROMPT_PREFIX
        for i, (s1, s2) in enumerate(chunk, start=1):
            prompt += f"{i}. A: {s1}\n   B: {s2}\n"
        prompt += "\nReturn ONLY the JSON object."

        response = client.responses.parse(
            model=MODEL_NAME,
            input=[{"role": "user", "content": prompt}],
            text_format=ParaphrasesOutput
        )

        # `response.output_parsed` holds the Pydantic model instance
        paraphrases = response.output_parsed.paraphrases
        all_paraphrases.extend(paraphrases)
    return all_paraphrases

def load_multicaption(project_path: str) -> tuple[pd.DataFrame]:
    multicaption_train = pd.read_csv(f"{project_path}/dataset/multicaption_train.csv")
    return multicaption_train

def create_paraphrase_dataset(multicaption_train,
                              prefix_param,
                              claim_number:int):
    """
    claim_number has to be 1 or 2
    """
    pairs = [(i,j) for i,j in zip(multicaption_train["claim_1_en"], multicaption_train["claim_2_en"])]

    paraphrased_captions = generate_paraphrases(pairs,
                                                api_key,
                                                prefix_param)
    multicaption_copy = multicaption_train.copy()
    multicaption_copy[f"claim_{claim_number}"] = paraphrased_captions
    multicaption_copy[f"claim_{claim_number}_en"] = paraphrased_captions
    multicaption_copy[f"language_{claim_number}"] = "eng"
    multicaption_copy["label_strategy"] = "gpt-5-mini"
    return multicaption_copy


if __name__=="__main__":
    init_log(logger_name="paraphrase_generation")
    logger = logging.getLogger("paraphrase_generation")

    project_path = config_dict["project_path"]
    api_key = config_dict["openai_api"]
    multicaption_train = load_multicaption(project_path)
    multicaption_train = multicaption_train.query("label == 1")
    # first sentence
    prefix_param = "the FIRST caption only"
    multicaption_synthetic_1 = create_paraphrase_dataset(multicaption_train,
                                                         prefix_param,
                                                         claim_number=1)

    # second sentence
    prefix_param = "the SECOND caption only"
    multicaption_synthetic_2 = create_paraphrase_dataset(multicaption_train,
                                                         prefix_param,
                                                         claim_number=2)
    multicaption_synthetic = pd.concat([multicaption_synthetic_1,
                                        multicaption_synthetic_2], axis=0)

    multicaption_synthetic.to_csv(f"{project_path}/dataset/multicaption_train_synthetic.csv",
                                  index=False, mode="w")


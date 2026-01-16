from tqdm.auto import tqdm
from sentence_transformers import CrossEncoder
import numpy as np

"""
List of NLI models:

    multilingual:
        "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "joeddav/xlm-roberta-large-xnli",
        "tasksource/ModernBERT-base-nli",

    monolingual:
        "tasksource/deberta-small-long-nli",
        "cross-encoder/nli-deberta-v3-small",
        "sentence-transformers/nli-distilbert-base",
"""

def get_nli_scores(claim_list_1: list[str],
                        claim_list_2: list[str],
                        model: CrossEncoder) -> tuple[list[str], np.array]:
    """
        returns a prediction in the set ['contradiction', 'entailment', 'neutral']
        for each pair and the scores for each label
        # 0: not out of context
        # 1: out of context
    """
    prediction_list = []
    prediction_scores = []

    label_mapping = ['contradiction', 'entailment', 'neutral']

    for claim1, claim2 in tqdm(zip(claim_list_1, claim_list_2), total=len(claim_list_1)):
        scores = model.predict([(claim1, claim2)])[0]

        label = label_mapping[scores.argmax()]
        prediction_scores.append(scores)
        prediction_list.append(label)

    prediction_scores = np.stack(prediction_scores)
    return prediction_list, prediction_scores

def get_nli_predictions(claim_list_1: list[str],
                        claim_list_2: list[str],
                        hf_model_name: str,
                        neutral: int) -> list[int]:
    """
    Tries the model: xlm-roberta-large-xnli
    Maps Neutral to Contradict
    Parameters:
        neutral: pass 0 or 1 > indicates if neutral is to be considered
                 contradiction or not
    """
    model = CrossEncoder(hf_model_name)
    prediction_list, prediction_scores = get_nli_scores(claim_list_1,
                                                        claim_list_2,
                                                        model)
    # entailment means sentences refer to diferent things
    # which in our context means contradiction
    mapping = {"contradiction": 1,
               "entailment": 0,
               "neutral": neutral}
    # Vectorize the mapping
    map_function = np.vectorize(lambda x: mapping.get(x, x))
    prediction_list = map_function(prediction_list)
    return prediction_list

def gen_nli_function(hf_model_name):
    """ Wrapper class to get nli
        Parameters:
            hf_model_name: model name on huggingface
        Example:
            xlm_roberta = gen_nli_function("joeddav/xlm-roberta-large-xnli")
            # Then pass xlm_roberta to eval_methods
    """
    def nli_impl(claim_list_1:list[str],
                 claim_list_2:list[str],
                 iteration:int):
        """iteration is not used"""
        predictions = get_nli_predictions(claim_list_1,
                                          claim_list_2,
                                          hf_model_name,
                                          neutral=1)
        return predictions
    return nli_impl

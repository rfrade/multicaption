import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)


def predict_pairs(model,
                  tokenizer,
                  pairs,
                  device,
                  batch_size: int = 32):
    """
    pairs: list[tuple[str, str]]  ->  [(sentence1, sentence2), ...]
    returns: (pred_labels, probs) where
             pred_labels: list[int]
             probs: torch.Tensor [N, num_labels]
    """
    all_probs = []

    model.to(device)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            s1_list = [p[0] for p in batch_pairs]
            s2_list = [p[1] for p in batch_pairs]

            enc = tokenizer(
                s1_list,
                s2_list,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )

            enc = {k: v.to(device) for k, v in enc.items()}

            outputs = model(**enc)
            probs = torch.softmax(outputs.logits, dim=-1)
            all_probs.append(probs.cpu())

    probs = torch.cat(all_probs, dim=0)      # [N, 2]
    pred_labels = probs.argmax(dim=-1).tolist()
    return pred_labels

def gen_bert_function(model_root:str,
                      device):

    def bert_predictions(claim_list_1:list[str],
                         claim_list_2:list[str],
                         iteration:int) -> list[int]:
        """ """
        model_path = f"{model_root}/{iteration}"
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                   num_labels=2,
                                                                   # reference_compile=False #for modern_bert
                                                                   )
        model.eval()

        pairs = [(c1,c2) for c1, c2 in zip(claim_list_1, claim_list_2)]
        predicted_pairs = predict_pairs(model,
                                        tokenizer,
                                        pairs,
                                        device)
        return predicted_pairs
    return bert_predictions

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from src.config import config_dict
import os
import pathlib
from src.evaluation.evaluation_config import EvaluationConfig
import logging
import traceback

logger = logging.getLogger("eval_pipeline")

def get_metrics(eval_config: EvaluationConfig,
                predictions: list[int],
                ground_truth_labels: list[int]):
    """
    Given a list of predictions and ground_truth_labels, returns a dictionary
    of metrics.
    """
    y_true = ground_truth_labels
    y_pred = predictions

    metrics_model = {
        "model": eval_config.model_name,
        "dataset": eval_config.dataset_name,
        "translation": eval_config.use_translation,
        "description": eval_config.description,
        "precision": np.round(precision_score(y_true, y_pred), 3),
        "recall": np.round(recall_score(y_true, y_pred), 3),
        "f1": np.round(f1_score(y_true, y_pred), 3),
        "accuracy": np.round(accuracy_score(y_true, y_pred), 3)
    }
    return metrics_model


def print_confusion_matrix(y_true:list, y_pred:list):
    y_true = np.where(np.array(y_true) == 0, "NC", "C")
    y_pred = np.where(np.array(y_pred) == 0, "NC", "C")

    cm = confusion_matrix(y_true, y_pred, labels=None)
    #labels = labels if labels is not None else sorted(set(y_true) | set(y_pred))
    labels = sorted(set(y_true) | set(y_pred))

    # Header
    cm_str = (f"Confusion Matrix:\n")
    header = "Pred ->".ljust(10) + " ".join(f"{label:>7}" for label in labels)
    cm_str += header + "\n"
    #cm_str += "-" * len(header) + "\n"

    # Rows
    for i, label in enumerate(labels):
        row = f"True {label:<5}" + " ".join(f"{cm[i, j]:>7}" for j in range(len(labels)))
        cm_str += row + "\n"
    logger.info(cm_str)


def log_dataset_info(ground_truth_labels):
    n_contradict = np.sum(ground_truth_labels)
    size = len(ground_truth_labels)
    n_non_contradict = size - n_contradict
    info = f"Total pairs: {size}, Contradicting: {n_contradict}, Non-Contradicting: {n_non_contradict}"
    logger.info(info)


def get_claims_and_labels(eval_config: EvaluationConfig):
    if eval_config.use_translation:
        claim_list_1 = eval_config.dataset["claim_1_en"].to_list()
        claim_list_2 = eval_config.dataset["claim_2_en"].to_list()
    else:
        claim_list_1 = eval_config.dataset["claim_1"].to_list()
        claim_list_2 = eval_config.dataset["claim_2"].to_list()

    labels = eval_config.dataset["label"].to_list()
    log_dataset_info(labels)

    return claim_list_1, claim_list_2, labels


def log_results(results_eval, ground_truth_labels, predictions):
    logger.info(f'precision: {results_eval["precision"]}, recall: {results_eval["recall"]}, f1:{results_eval["f1"]}, '
                f'accuracy:{results_eval["accuracy"]}')
    print_confusion_matrix(y_true=ground_truth_labels,
                           y_pred=predictions)

def log_eval_info(eval_config):
    logger.info(f"** {eval_config.model_name.upper()} - {eval_config.dataset_name.upper()} **")

    info = (f"description: {eval_config.description}, use_translation: {eval_config.use_translation}")
    logger.info(info)

def save_results(results):
    logger.info("Final results:")
    project_path = config_dict["project_path"]
    os.makedirs(f"{project_path}/results", exist_ok=True)

    results_df = pd.DataFrame(results)

    for i, df in results_df.groupby("dataset"):
        dataset_name = df["dataset"].iloc[0]
        csvfile = pathlib.Path(f"{project_path}/results/{dataset_name}.csv")
        df.drop("dataset", axis=1).to_csv(csvfile,
                                          mode="a",
                                          header=not csvfile.exists(),
                                          index=False)
    logger.info("\n" + str(results_df))

    save_excel(project_path)
    logger.info(f'\n Saved results to {f"{project_path}/results/"}')


def save_predictions(eval_config, iteration, predictions):
    project_path = config_dict["project_path"]
    os.makedirs(f"{project_path}/results", exist_ok=True)

    csvfile = pathlib.Path(f"{project_path}/results/{eval_config.dataset_name}_predictions.csv")

    col_name = (
        f"{eval_config.model_name}_"
        f"{iteration}"
    )

    if csvfile.exists():
        df = pd.read_csv(csvfile)
    else:
        # Start with a fresh DF the size of predictions
        df = eval_config.dataset.copy()

    df[col_name] = predictions
    df.to_csv(csvfile, index=False)

    logger.info(f'\n Saved predictions to {f"{project_path}/results/{eval_config.dataset_name}_predictions.csv"}')


def save_excel(project_path: str):
    """ save to an excel file where the dataset results are saved to different spreasheets.
        resuls are not appended, every time csvs are read and the file is recreated"""
    csv_files = [f for f in os.listdir(f"{project_path}/results") if f.endswith('.csv')]
    with pd.ExcelWriter(f"{project_path}/results/results.xls", engine='openpyxl') as writer:
        for csv_file in csv_files:
            file_path = os.path.join(f"{project_path}/results", csv_file)
            sheet_name = csv_file[:31]  # Excel sheet names max 31 chars
            df = pd.read_csv(file_path)
            df.to_excel(writer, sheet_name=sheet_name, index=False)

def save_predictions_file(eval_config:EvaluationConfig,
                          predictions:list[int]):
    destination_folder = eval_config.save_predictions_to
    if destination_folder is not None:
        dict_predictions = {"line_number":range(len(predictions)),
                            "predicted_label":predictions}
        pd.DataFrame(dict_predictions).to_csv(f"{destination_folder}/{eval_config.model_name}_{eval_config.dataset_name}.csv",
                                              index=False,
                                              mode="w")

def eval_methods(eval_config_list: list[EvaluationConfig]):
    """
        Calls the functions that return the predictions and compute the metrics.
        Saves the results in the log folder.
    """
    results = []

    for eval_config in eval_config_list:
        function = eval_config.function
        finetuning = True if "-FT" in eval_config.function_name else False

        log_eval_info(eval_config)

        claim_list_1, claim_list_2, ground_truth_labels = get_claims_and_labels(eval_config)

        results_per_iteration = []
        iterations = 1 if eval_config.function_name == "NLI" else 5

        try:
            for i in range(eval_config.iterations):
                logger.info(f"Iteration: {i}")
                predictions = function(claim_list_1, claim_list_2, i)

                results_eval = get_metrics(eval_config, predictions, ground_truth_labels)
                results_per_iteration.append(results_eval)
                log_results(results_eval, ground_truth_labels, predictions)
                save_predictions_file(eval_config, predictions)
        except Exception as e:
            error_details = traceback.format_exc()
            print("Full error details:\n", error_details)

            if eval_config.save_predictions:
                save_predictions(eval_config, i, predictions)

    results.append(summarize_metrics(results_per_iteration))
    save_results(results)


def summarize_metrics(metrics_list):
    # metric keys to aggregate
    metric_keys = ["precision", "recall", "f1", "accuracy"]

    summary = {
        "model": metrics_list[0]["model"],
        "dataset": metrics_list[0]["dataset"],
        "translation": metrics_list[0]["translation"],
        "description": metrics_list[0]["description"]
    }

    # compute mean and std for each metric
    for key in metric_keys:
        values = [m[key] for m in metrics_list]
        summary[f"{key}_mean"] = np.round(np.mean(values), 3)
        summary[f"{key}_std"] = np.round(np.std(values), 3)

    return summary











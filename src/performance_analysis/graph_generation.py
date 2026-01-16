import pandas as pd
import numpy as np
import logging
from src.util.logging_util import init_log
from src.config import config_dict
import seaborn as sns
import matplotlib.pyplot as plt

def crosslingual_plot(test_long:pd.DataFrame,
                      palette:str,
                      save_file_to:str):
    """
        test_long: dataset in the long format
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axhline(0.6, color='black', linestyle='-', linewidth=0.5, zorder=0)
    ax.axhline(0.8, color='black', linestyle='-', linewidth=0.5, zorder=0)

    sns.barplot(
        data=test_long,
        x="crosslingual",
        y="correct",
        hue="model",
        palette=palette,
        ax=ax
    )

    new_width = 0.2
    for bar in []:  # ax.patches:
        x = bar.get_x()
        width = bar.get_width()
        center = x + width / 2
        bar.set_x(center - new_width / 2)
        bar.set_width(new_width)

    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("accuracy")
    # axes[0].get_legend().remove()
    # ax.get_legend().remove()
    sns.despine()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower center",
              ncol=len(labels), frameon=False,
              bbox_to_anchor=(0.5, -0.25), )
    plt.savefig(f"{save_file_to}/graph_cross_lingual.png", bbox_inches="tight")
    #plt.show()

def dataset_wide_to_long(multicaption_test:pd.DataFrame):
    cols = ["label_strategy", "language_1", "language_2", "crosslingual",
            "cosine_similarity_en", "mdeberta_correct", "nli_correct", "mistral_correct", "cid_1", "cid_2"]
    id_vars = ["label_strategy", "language_1", "language_2", "crosslingual", "cosine_similarity_en", "cid_1", "cid_2"]
    test_long = multicaption_test[cols]
    test_long = test_long.rename(columns={"mdeberta_correct": "mDeBERTa",
                                          "nli_correct": "mDeBERTa-NLI",
                                          "mistral_correct": "Mistral"})

    test_long = pd.melt(test_long,
                        id_vars=id_vars,
                        value_vars=['mDeBERTa', 'mDeBERTa-NLI', 'Mistral'])
    test_long = test_long.rename(columns={"variable": "model", "value": "correct"})
    test_long["crosslingual"] = np.where(test_long["crosslingual"] == 0, "Monolingual", "Crosslingual")
    return test_long

if __name__=="__main__":
    init_log(logger_name="graphs")
    TEXT_COLOR = "black"
    plt.rcParams['text.color'] = TEXT_COLOR  # all text
    plt.rcParams['axes.labelcolor'] = TEXT_COLOR  # axis labels
    plt.rcParams['xtick.color'] = TEXT_COLOR  # x-axis tick labels
    plt.rcParams['ytick.color'] = TEXT_COLOR  # y-axis tick labels
    plt.rcParams['axes.titlecolor'] = TEXT_COLOR  # plot title

    #sns.color_palette("Paired")
    PALETTE = "pastel"  # "colorblind"#

    logger = logging.getLogger("eval_pipeline")
    project_path = config_dict["project_path"]
    multicaption_predictions = pd.read_csv(f"{project_path}/error_analysis/multicaption_multilingual_predictions.csv")
    test_long = dataset_wide_to_long(multicaption_predictions)
    crosslingual_plot(test_long=test_long,
                      palette=PALETTE,
                      save_file_to=f"{project_path}/error_analysis/graphs")
    logger.info("graphs generated")


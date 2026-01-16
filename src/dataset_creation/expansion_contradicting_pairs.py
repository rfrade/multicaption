import pandas as pd

def create_pairs_through_post_id():
    """
        Non-contradicting pairs were created through claim_post_id mappings.
        We use these mapped pairs to create contradicting pairs.
        The logic is: If claim A has a title A, and claim A and B are mapped
        in the non-contradicting set, we create a new mapping (A,B)

    """
    # SET ROOT DATA
    root = "."

    # Multiclaim V2
    fc_post_v2 = pd.read_csv(f"{root}/MultiClaimV2/fact_check_post_mapping.csv")
    fact_checks_v2 = pd.read_csv(f"{root}/MultiClaimV2/fact_checks.csv")


    ## Manual annotation
    contradict_pairs = (pd.read_csv(f"{root}/ours/contradicting_pairs_annotated_2025.csv")
                         .query("discard == 0")
                         .drop("discard", axis=1))

    cols = ["fact_check_id", "claim", "claim_en", "claim_detected_language_iso",
            "title", "title_en", "title_detected_language_iso"]
    contradict_pairs = contradict_pairs[["fact_check_id", "instances"]].merge(fact_checks_v2[cols],
                           on="fact_check_id",
                           how="inner",
                           #indicator=True
                           ).drop("instances", axis=1)

    contradict_pairs = contradict_pairs.rename(columns={"fact_check_id":"fact_check_id_claim"})

    contradict_pairs["fact_check_id_title"] = contradict_pairs["fact_check_id_claim"]
    contradict_pairs["label"] = "contradiction"
    contradict_pairs["label_strategy"] = "Manual"

    ## Self-expasion

    path_ours = f"{root}/ours"
    contradict_pairs_ann = (pd.read_csv(f"{root}/ours/contradicting_pairs_annotated_2025.csv")
                         .query("discard == 0")
                         .drop("discard", axis=1))

    noncontradict_df = pd.read_csv(f"{path_ours}/NonContradiction_Pairs_All.csv")

    nonc_concat = pd.DataFrame({"fact_check_id_1":pd.concat([noncontradict_df["CID_1"],
                                                             noncontradict_df["CID_2"]]),
                                "fact_check_id_2":pd.concat([noncontradict_df["CID_2"],
                                                             noncontradict_df["CID_1"]])})

    expanded_ids = (contradict_pairs_ann.merge(nonc_concat,
                           left_on="fact_check_id",
                           right_on="fact_check_id_1")
                      .rename(columns={"fact_check_id":"fact_check_id_title",
                                       "fact_check_id_2":"fact_check_id_claim"})
                      .drop("fact_check_id_1", axis=1))[["fact_check_id_title", "fact_check_id_claim"]]

    expanded_pairs = (expanded_ids.merge(fact_checks_v2[["fact_check_id", "claim", "claim_en", "claim_detected_language_iso"]],
                                         left_on="fact_check_id_claim",
                                         right_on="fact_check_id",
                                         how="inner",
                                         #indicator=True
                                         ).drop("fact_check_id", axis=1)
                                  .merge(fact_checks_v2[["fact_check_id", "title", "title_en", "title_detected_language_iso"]],
                                         left_on="fact_check_id_title",
                                         right_on="fact_check_id",
                                         how="inner",
                                         #indicator=True
                                         ).drop("fact_check_id", axis=1))

    expanded_pairs["label"] = "contradiction"
    expanded_pairs["label_strategy"] = "Self-Expansion"

    (pd.concat([contradict_pairs, expanded_pairs], axis=0)
        .reset_index(drop=True)
        .to_csv(f"{root}/ours/contradicting_pairs_expanded.csv",
                index=False, mode="w"))

if __name__=="__main__":
    create_pairs_through_post_id()

import pandas as pd
import polars as pl
from src.config import config_dict

def extract_title_and_translation(title: str) -> tuple[str, str]:
    """ returns title in original language and translation in english """
    delimiter_original = title[1]
    splitted = title.split(delimiter_original + ",")
    original = splitted[0].replace(f"({delimiter_original}", "")
    translation = splitted[1].split("[(")[0].strip()
    translation = translation.replace("'","").replace('"',"").replace(",","")
    return original, translation

def test_extract_title_and_translation():
    t1 = """("original", "translation", [('eng', 1.0)])"""
    t2 = """('original', 'translation', [('eng', 1.0)])"""
    t3 = """("original", 'translation', [('eng', 1.0)])"""
    t4 = """('original', "translation", [('eng', 1.0)])"""

    for title in [t1, t2, t3, t4]:
        original, translation = extract_title_and_translation(title)
        assert original == "original"
        assert translation == "translation"
test_extract_title_and_translation()

def extract_title_from_col(fc_posts: pl.DataFrame) -> pl.DataFrame:
    original_list = []
    translated_list = []
    for row in fc_posts.iter_rows(named=True):
        title = row["title"]
        if title is not None:
            original, translation = extract_title_and_translation(title)
        else:
            original, translation = None, None
        original_list.append(original)
        translated_list.append(translation)
    return fc_posts.with_columns(original_title=pl.Series(values=original_list),
                                 translated_title=pl.Series(values=translated_list))

def filter_media_expressions(fc_posts):
    negation_terms = ["fake", "?", "mislead", "false", "wrong", "manipulat", "edited", "altered",
                      " not ", " no ",  "photoshop", "isn’t", "purport",
                      "fabricated", "forged", "mistaken", "claim that", "doesn't",
                      "doesn’t", "wasn't", "watsn’t", "morphed", "doctored", "misuse",
                      "unrelated", "rumor", "nothing", "generated", "misrepresent",
                      "misinterpret", "didn’t", "shared as", "fact check:", "fraudulent", "deceive", "misinform",
                      "misguide", "invalid", "inaccurate", "untrue", "erroneous", "tamper",
                      "distort", "modified", "portrayed as", "falsified", "concoct", "bogus", "phony",
                      # new terms
                      "hoax", "associated with", "aren’t", "isn't", "neither", "miscaption",
                      "context", "since", "fictional", "montage", ",not ", "actually", "incorrect", "old",
                      "retouched", "reject", "as if", "doesnt", "didnt", "apocryphal",
                      "didnt", "doesnt", "yes", "predate", "before", "previous", "prior",
                      "another", "none", "far from", "in reports about", "satirical", "attributed",
                      "was taken in", "not ", "no ", "attention", "never",
                      "claiming", "staged", "claim", "joke", "in fact",
                      "is from", "are from", "dates from", "date from"
                      ]

    keywords = ["picture", "image", "photo", "photograph",
                "footage", "document", "video", "clip"]

    # showing
    keywords_plural = [f"{k}s" for k in keywords]
    media_expressions = keywords + keywords_plural
    expressions = [f"{k} show" for k in keywords]
    expressions = expressions + [f"{k} show" for k in keywords_plural]

    # either title or claim contain media references
    # title doesn't contain negation
    cc_candidates = fc_posts.filter(
            pl.col("translated_title_lower").str.contains_any(expressions) | # or
            pl.col("claim_translation_lower").str.contains_any(expressions)
        ).filter(
            pl.col("translated_title_lower").str.contains_any(negation_terms).not_()
        ).to_pandas()
    cc_candidates = cc_candidates.drop_duplicates("fact_check_id")
    print(f"Claims to annotate: {len(cc_candidates)}")
    return cc_candidates

def gen_contradicting_pairs_to_annotate(root):
    datapath = f"{root}/MultiClaim"
    multiclaim_cleaned = f"{root}/MultiClaim/Cleaned"

    # load multiclaim datasets
    factchecks = pl.read_csv(f"{datapath}/fact_checks.csv")
    posts = (pl.read_csv(f"{datapath}/posts.csv")
             .select(pl.col(["post_id", "ocr"])))
    factcheck_posts = pl.read_csv(f"{datapath}/fact_check_post_mapping.csv")

    # load cleaned versions of multiclaim«
    posts_cleaned = pl.read_csv(f"{multiclaim_cleaned}/Posts_Text.csv")
    claims_cleaned = pl.read_csv(f"{multiclaim_cleaned}/Claims.csv")

    posts_cleaned = posts_cleaned.rename({"Post": "post_text",
                                          "Translation": "post_translation",
                                          "Language": "post_language",
                                          "Verdict": "post_verdict",
                                          "PID": "post_id"})

    claims_cleaned = claims_cleaned.rename({"CID": "claim_id",
                                            "Claim": "claim",
                                            "Language": "claim_language",
                                            "Translation": "claim_translation"})

    fc_posts = (factchecks
                .drop("claim")
                .join(factcheck_posts, on="fact_check_id", how="inner")
                .join(posts, on="post_id", how="inner")
                .join(claims_cleaned, left_on="fact_check_id", right_on="claim_id")
                .join(posts_cleaned, left_on="post_id", right_on="post_id")
                .pipe(extract_title_from_col)
                )

    fc_posts = fc_posts.with_columns(
        translated_title_lower=pl.col("translated_title").str.to_lowercase(),
        claim_translation_lower=pl.col("claim_translation").str.to_lowercase()
    )

    contradicting_pairs_to_annotate = filter_media_expressions(fc_posts)
    cols = ["fact_check_id", "claim_translation", "translated_title", "instances"]
    contradicting_pairs_to_annotate = contradicting_pairs_to_annotate[cols]
    contradicting_pairs_to_annotate.to_csv(f"{root}/ours/contradicting_pairs_to_annotate.csv",
                                           index=False, mode="w")
    print(f"contradicting_pairs_to_annotate saved to {root}/ours")

if __name__=="__main__":
    path = config_dict["data_path"]
    gen_contradicting_pairs_to_annotate(path)

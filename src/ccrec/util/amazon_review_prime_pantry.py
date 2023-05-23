import pandas as pd, numpy as np, torch
from sklearn.feature_extraction.text import TfidfVectorizer


def _shorten_brand_name_function(x):
    if not isinstance(x, str):
        return x
    for i, part in enumerate(x.split(" ")):
        try:
            int(part)
            continue
        except Exception:
            return " ".join(x.split(" ")[: i + 1])


def _join_title_description(x):
    text = []
    if isinstance(x["title"], str):
        text.append(x["title"])
    if isinstance(x["description"], str):
        text.append(x["description"])
    return " ".join(text)


def get_item_df(
    data_root="data/amazon_review_prime_pantry",
    meta_file="meta_Prime_Pantry.json.gz",
    landingImageURL_file="landingImageURL.csv.gz",
    landingImageURL_folder=None,  # "landingImage",
    shorten_brand_name=False,
    return_tfidf_csr=True,
    nrows=None,
):
    item_df = pd.read_json(
        f"{data_root}/{meta_file}", lines=True, nrows=nrows
    ).drop_duplicates(subset=["asin"])
    item_df = item_df.set_index("asin").assign(
        TITLE=lambda df: df.apply(_join_title_description, axis=1),
        BRAND=lambda df: df.brand,
    )[["TITLE", "BRAND"]]

    if shorten_brand_name:
        item_df["BRAND"] = item_df["BRAND"].apply(_shorten_brand_name_function)
    print(f'# items {len(item_df)}, # brands {item_df["BRAND"].nunique()}')

    item_df = item_df.join(
        pd.read_csv(
            f"{data_root}/{landingImageURL_file}", names=["asin", "landingImage"]
        ).set_index("asin")
    )
    item_df = item_df[item_df["landingImage"].notnull()]
    item_df = item_df[item_df["landingImage"].apply(lambda x: x.endswith(".jpg"))]
    if landingImageURL_folder is not None:
        item_df["landingImage"] = [
            f"{data_root}/{landingImageURL_folder}/{x}.jpg"
            for x in item_df.index.values
        ]

    tfidf_fit = TfidfVectorizer().fit(item_df["TITLE"].tolist())
    tfidf_csr = tfidf_fit.transform(item_df["TITLE"].tolist())

    item_df["tfidf_indices"] = np.split(tfidf_csr.indices, tfidf_csr.indptr[1:-1])
    item_df["tfidf_words"] = np.split(
        np.array(tfidf_fit.get_feature_names_out())[tfidf_csr.indices],
        tfidf_csr.indptr[1:-1],
    )
    item_df["tfidf_data"] = np.split(tfidf_csr.data, tfidf_csr.indptr[1:-1])

    item_df = item_df.join(
        pd.concat(
            {
                "words": item_df["tfidf_words"].explode(),
                "data": item_df["tfidf_data"].explode(),
            },
            axis=1,
        )
        .sort_values("data", ascending=False)
        .groupby("asin")["words"]
        .apply(lambda x: x[:5].tolist())
        .to_frame("sorted_words")
    ).drop("tfidf_words", axis=1)

    return (item_df, tfidf_csr) if return_tfidf_csr else item_df

import ast
import hashlib
import os

import numpy as np
import pandas as pd


def clean_text(value):
    if pd.isna(value):
        return ""
    text = str(value)
    if len(text) > 1 and text[0] == "[":
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return " ".join(map(str, parsed))
        except Exception:
            pass
    return text


def main():
    dataset_name = "amazon_musical_instruments"
    data_dir = os.path.join("data", dataset_name)
    csv_path = os.path.join(data_dir, "processed_data.csv")
    out_path = os.path.join(data_dir, f"{dataset_name}_embedding_llama3.parquet")

    print(f"reading {csv_path}", flush=True)
    df = pd.read_csv(csv_path, usecols=["ItemId", "title", "description", "features"])

    # Match AmazonUserSequencesDataset's two-step remapping so embedding ids line up with training ids.
    unique_item_ids = df["ItemId"].unique()
    first_map = {old_id: new_id for new_id, old_id in enumerate(unique_item_ids, start=0)}
    item_ids = df["ItemId"].map(first_map)
    unique_mapped = item_ids.unique()
    second_map = {old_id: new_id for new_id, old_id in enumerate(unique_mapped, start=1)}
    df["ItemId"] = item_ids.map(second_map).astype(int)

    dedup = df.drop_duplicates("ItemId", keep="first").copy()
    texts = (
        dedup["title"].fillna("")
        + " "
        + dedup["description"].map(clean_text)
        + " "
        + dedup["features"].map(clean_text)
    ).tolist()

    dim = 768
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for token in text.lower().split()[:512]:
            bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16) % dim
            vectors[i, bucket] += 1.0
        norm = np.linalg.norm(vectors[i])
        if norm > 0:
            vectors[i] /= norm

    out_df = pd.DataFrame(
        {
            "ItemId": dedup["ItemId"].tolist(),
            "prompt": texts,
            "item_text_embedding": [row.tolist() for row in vectors],
        }
    )
    out_df.to_parquet(out_path, compression="snappy")
    print(
        f"saved {out_path}; rows={len(out_df)} dim={dim} max_item_id={out_df['ItemId'].max()} size={os.path.getsize(out_path)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

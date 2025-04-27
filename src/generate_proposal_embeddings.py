import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE


def generate_text_string(contract_df: pd.DataFrame) -> str:
    text_columns = [
        "Item Description",
        "Item Description Subset",
        "Quantity",
        "Unit",
    ]
    proposal_type = contract_df.at[0, "Proposal Type"]
    proposal_description = contract_df.at[0, "Proposal Description"]
    item_types = contract_df["Item Type"].unique().tolist()
    item_types = ", ".join(item_types)
    unique_item_info = contract_df[text_columns].astype(str).drop_duplicates()
    unique_item_info.replace(
        {"Item Description Subset": {"nan": ""}}, inplace=True
    )
    item_description_list = unique_item_info["Item Description"].tolist()
    item_description_subset_list = unique_item_info[
        "Item Description Subset"
    ].tolist()
    quantity_list = unique_item_info["Quantity"].tolist()
    unit_list = unique_item_info["Unit"].tolist()
    vendor_name_list = contract_df["Vendor Name"].unique().tolist()

    total_string = " ".join(
        [
            proposal_type,
            proposal_description,
            item_types,
        ]
    )
    total_string += "\n" + ", ".join(vendor_name_list) + "\n"
    for (
        item_description,
        item_description_subset,
        quantity,
        unit,
    ) in zip(
        item_description_list,
        item_description_subset_list,
        quantity_list,
        unit_list,
    ):
        total_string += " ".join(
            [
                item_description.strip(),
                item_description_subset.strip(),
                quantity.strip(),
                unit.strip(),
            ]
        )
        total_string += "\n"
    total_string += ", ".join(vendor_name_list) + "\n"
    total_string = total_string.replace("\n", "")
    return total_string


def rename_json_index(response: dict, csv_files: list):
    for i, csv_embedding in enumerate(response["data"]):
        csv_embedding["index"] = f"{csv_files[i]}"
    return response


def main():
    data_path = Path("nc_csv")
    texts = []
    csv_files = [csv_file.stem for csv_file in sorted(data_path.glob("*.csv"))]
    tokens = 0

    for csv_file in sorted(data_path.glob("*.csv")):
        contract_df = pd.read_csv(csv_file)
        text = generate_text_string(contract_df)
        texts.append(text)
        tokens += len(text.split())

    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer jina_e1a13a3200f946a9ab26f2639bf018772eiDIwyE7mgdMZi1bZQD2KGzgIBg",
    }
    data = {
        "model": "jina-embeddings-v3",
        "task": "text-matching",
        "late_chunking": False,
        "dimensions": 1024,
        "embedding_type": "float",
        "input": [{"text": text} for text in texts],
    }
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # This will raise an HTTPError for 4xx/5xx status codes
    except requests.exceptions.HTTPError as e:
        # Get the actual response from the exception
        response = e.response
        print(f"Status code: {response.status_code}")
        print(f"Reason: {response.reason}")
        print(f"Response Headers: {response.headers}")
        print(f"Response Content: {response.text}")
    with open("original_jina_embedding_api_response.json", "w") as f:
        json.dump(response.json(), f, indent=2)
    json_response = response.json()
    modified_response = rename_json_index(json_response, csv_files)

    with open("jina_embedding_proposalIndex_response.json", "w") as f:
        json.dump(modified_response, f, indent=2)


if __name__ == "__main__":
    main()

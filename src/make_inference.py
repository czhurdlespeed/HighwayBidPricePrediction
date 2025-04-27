import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import requests
from geopy.distance import geodesic as GD
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from calcs import predict_with_confidence
from contract_locations import query_google_maps
from inferenceExcel import excel_predictions
from model_helpers import (
    combine_csv_files,
    create_embedding_dict,
    estimate_inference_contract_size,
    load_distance,
)
from models import ImprovedKNNRegression


@dataclass
class InferenceProposal:
    proposal: str
    contractors: list[str]
    embedding: np.ndarray
    location: tuple
    estimated_contract_size: float = 0.0


class Inference:
    def __init__(
        self,
        embedding_dict: dict,
        combined_df: pd.DataFrame,
        inference_proposal: InferenceProposal,
    ):
        self.embedding_dict = embedding_dict
        self.combined_df = combined_df
        self.inference_proposal = inference_proposal
        self.inference()

    def inference(self):
        os.makedirs("InferenceResults", exist_ok=True)
        output: dict = {}
        output.setdefault(self.inference_proposal.proposal, {})
        contractor_list = self.inference_proposal.contractors
        payitems_df = pd.read_csv(
            f"InferenceProposals/{self.inference_proposal.proposal}.csv"
        )
        payitems_df["Item"] = payitems_df["Item"].str.strip()
        print(payitems_df)
        est_inference_contract_size = estimate_inference_contract_size(
            self.combined_df,
            (
                payitems_df["Item"].tolist(),
                payitems_df["Quantity"].astype(float).tolist(),
            ),
            self.inference_proposal.proposal,
        )
        self.inference_proposal.estimated_contract_size = (
            est_inference_contract_size
        )
        print(contractor_list)
        for contractor in contractor_list:
            output[self.inference_proposal.proposal].setdefault(contractor, {})
            for payitem in payitems_df["Item"]:
                print(f"Starting inference for {payitem} in {contractor}")
                inference_payitem_data = payitems_df[
                    payitems_df["Item"] == payitem
                ]
                line_item = inference_payitem_data["Line"]
                payitem_desc = inference_payitem_data["Description"]
                inference_quantity = inference_payitem_data["Quantity"]
                inference_unit = inference_payitem_data["Unit"].iloc[0]
                historic_data = self.combined_df[
                    (
                        self.combined_df["Proposal"].astype(str)
                        != self.inference_proposal.proposal
                    )
                    & (self.combined_df["Vendor Name"] == contractor)
                    & (self.combined_df["Item"] == payitem)
                ]
                if historic_data.empty:
                    print(f"No data found for {payitem} in {contractor}")
                    for line, quantity, desc in zip(
                        line_item, inference_quantity, payitem_desc
                    ):
                        output[self.inference_proposal.proposal][contractor][
                            f"{payitem}:{line}:{desc}"
                        ] = [
                            "N/A",
                            quantity,
                            "N/A",
                            0,
                            "N/A",
                            [],
                            inference_unit,
                        ]
                    continue

                historic_data = historic_data.copy()

                historic_data["Similarity"] = historic_data["Proposal"].apply(
                    lambda x: np.dot(
                        self.inference_proposal.embedding,
                        self.embedding_dict[x],
                    )
                    / (
                        np.linalg.norm(self.inference_proposal.embedding)
                        * np.linalg.norm(self.embedding_dict[x])
                    )
                )
                # compute distances
                historic_data["Distance"] = historic_data.apply(
                    lambda row: float(
                        GD(
                            self.inference_proposal.location,
                            (row["Latitude"], row["Longitude"]),
                        ).miles
                    ),
                    axis=1,
                )
                number_payitem_occurances = len(historic_data)
                # Scaling
                quantity_scaler = Pipeline(
                    [
                        ("robust", RobustScaler()),
                        ("minmax", MinMaxScaler()),
                    ]
                )
                distance_scaler = Pipeline(
                    [
                        ("robust", RobustScaler()),
                        ("minmax", MinMaxScaler()),
                    ]
                )
                contract_dollar_size_scaler = Pipeline(
                    [
                        ("robust", RobustScaler()),
                        ("minmax", MinMaxScaler()),
                    ]
                )
                features = [
                    "Quantity Scaled",
                    "Distance Scaled",
                    "Similarity",
                    "Total Contract Amount Scaled",
                ]

                historic_data["Quantity Scaled"] = (
                    quantity_scaler.fit_transform(historic_data[["Quantity"]])
                )
                historic_data["Distance Scaled"] = (
                    distance_scaler.fit_transform(historic_data[["Distance"]])
                )
                historic_data["Total Contract Amount Scaled"] = (
                    contract_dollar_size_scaler.fit_transform(
                        historic_data[["Total Contract Amount"]]
                    )
                )

                inference_quantity = pd.DataFrame(
                    inference_payitem_data["Quantity"], columns=["Quantity"]
                )

                contract_amount = pd.DataFrame(
                    [self.inference_proposal.estimated_contract_size],
                    columns=["Total Contract Amount"],
                )

                quantity_scaled = quantity_scaler.transform(inference_quantity)
                contract_amount_scaled = contract_dollar_size_scaler.transform(
                    contract_amount
                )

                inference_df = pd.DataFrame(
                    {
                        "Quantity Scaled": quantity_scaled.ravel(),
                    }
                )
                inference_df["Distance Scaled"] = 0
                inference_df["Similarity"] = 1
                inference_df["Total Contract Amount Scaled"] = (
                    contract_amount_scaled.item()
                )
                print(f"{'Inference data':-^50}")
                print(inference_df)
                print("\n")

                inference_df = inference_df[features]

                kpca = KernelPCA(
                    n_components=None, kernel="rbf", random_state=42
                )
                historic_kpca = kpca.fit_transform(historic_data[features])
                inference_kpca = kpca.transform(inference_df[features])

                eigenvalues = kpca.eigenvalues_
                total_variance = np.sum(eigenvalues)

                if total_variance > 0:
                    variance_ratio = eigenvalues / total_variance
                    cumulative_variance_ratio = np.cumsum(variance_ratio)
                    if np.any(cumulative_variance_ratio > 0.95):
                        n_components = (
                            np.argmax(cumulative_variance_ratio > 0.95) + 1
                        )
                    else:
                        n_components = len(cumulative_variance_ratio)
                    historic_kpca = historic_kpca[:, :n_components]
                    inference_kpca = inference_kpca[:, :n_components]
                    historic_kpca_df = pd.DataFrame(
                        historic_kpca,
                        columns=[f"PC{i+1}" for i in range(n_components)],
                    )
                    inference_kpca_df = pd.DataFrame(
                        inference_kpca,
                        columns=[f"PC{i+1}" for i in range(n_components)],
                    )

                    model_used = "KNN with Kernel PCA"
                    knn = ImprovedKNNRegression(
                        historic_kpca_df, historic_data["Unit Price"]
                    )
                    predictions, confidences, details = predict_with_confidence(
                        knn,
                        inference_kpca_df,
                        historic_kpca_df,
                        k=knn.best_k,
                    )
                    distances, indices = knn.model.kneighbors(
                        inference_kpca_df,
                        n_neighbors=knn.best_k,
                        return_distance=True,
                    )

                    for prediction, confidence in zip(predictions, confidences):
                        print(f"{model_used:-^50}")
                        print(f"Pay item: {payitem:10s}")
                        print(f"Description: {payitem_desc}")
                        print(f"Prediction: {prediction:10.2f}")
                        print(f"Confidence: {confidence:10.2f}")
                        print(f"Shape of input data: {historic_kpca.shape}")
                else:
                    print(
                        "Warning: All eigenvalues are zero or very close to zero. Using original features."
                    )
                    model_used = "KNN with original features"
                    knn = ImprovedKNNRegression(
                        historic_data[features],
                        historic_data["Unit Price"],
                    )
                    predictions, confidences, details = predict_with_confidence(
                        knn,
                        inference_df,
                        historic_data[features],
                        k=knn.best_k,
                    )
                    distances, indices = knn.model.kneighbors(
                        inference_df,
                        n_neighbors=knn.best_k,
                        return_distance=True,
                    )
                    for prediction, confidence in zip(predictions, confidences):
                        print(f"{model_used:-^50}")
                        print(f"Pay item: {payitem:10s}")
                        print(f"Description: {payitem_desc}")
                        print(f"Prediction: {prediction:10.2f}")
                        print(f"Confidence: {confidence:10.2f}")
                        print(
                            f"Shape of input data: {historic_data[features].shape}"
                        )
                for (
                    prediction,
                    quantity,
                    confidence,
                    distance,
                    index,
                    line,
                    description,
                ) in zip(
                    predictions,
                    inference_quantity["Quantity"].tolist(),
                    confidences,
                    distances,
                    indices,
                    line_item,
                    payitem_desc,
                ):
                    print(f"{'Historic data':-^50}")
                    print(historic_data)
                    print(f"{'Index':-^50}")
                    print(index)

                    nearest_neighbors = (
                        historic_data.iloc[index]
                        .sort_values("Distance")["Proposal"]
                        .tolist()
                    )
                    extension = round(quantity * prediction, 0)
                    output[self.inference_proposal.proposal][contractor][
                        f"{payitem}:{line}:{description}"
                    ] = [
                        round(prediction, 2),
                        quantity,
                        extension,
                        number_payitem_occurances,
                        round(confidence, 1),
                        list(zip(distance, nearest_neighbors)),
                        historic_data["Unit"].iloc[0],
                    ]
                    nearest_neighbors = None
            print(f"Finished inference for {contractor}")
        print(f"Finished inference for {self.inference_proposal.proposal}")
        excel_predictions(
            output, f"InferenceResults/{self.inference_proposal.proposal}.xlsx"
        )


def generate_text_string(
    contract_df: pd.DataFrame, contractors: List[str], proposal_description: str
) -> str:
    text_columns = [
        "Description",
        "Quantity",
        "Unit",
    ]
    unique_item_info = contract_df[text_columns].astype(str).drop_duplicates()
    item_description_list = unique_item_info["Description"].tolist()
    quantity_list = unique_item_info["Quantity"].tolist()
    unit_list = unique_item_info["Unit"].tolist()

    total_string = proposal_description + " "
    total_string += ", ".join(contractors)
    for (
        item_description,
        quantity,
        unit,
    ) in zip(
        item_description_list,
        quantity_list,
        unit_list,
    ):
        total_string += " ".join(
            [
                item_description.strip(),
                quantity.strip(),
                unit.strip(),
            ]
        )
        total_string += ", "
    total_string += ", ".join(contractors)
    total_string = total_string.replace("\n", "")
    return total_string


def generate_inference_proposal_embedding(
    contract: str, vendors: List[str], proposal_description: str
) -> np.ndarray:
    inference_csv = Path("InferenceProposals") / f"{contract}.csv"
    proposal_df = pd.read_csv(inference_csv)
    text = generate_text_string(proposal_df, vendors, proposal_description)
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
        "input": [{"text": text}],
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
    response = response.json()
    return response["data"][0]["embedding"]


if __name__ == "__main__":
    embedding_dict = create_embedding_dict()
    combined_df, _ = combine_csv_files(training=False)
    combined_df["Item"] = combined_df["Item"].str.strip()
    combined_df["Vendor Name"] = combined_df["Vendor Name"].str.strip()
    combined_df = combined_df.dropna(subset=["Unit Price"])
    contractors_list = [
        "BARNHILL CONTRACTING CO",
        "FSC II LLC DBA FRED SMITH COMPANY",
        "HIGHLAND PAVING CO LLC",
    ]
    if not os.path.exists("pickles/C204988_inference_proposal.pkl"):
        print("Generating inference proposal")
        inference_proposal = InferenceProposal(
            proposal="C204988",
            contractors=contractors_list,
            embedding=generate_inference_proposal_embedding(
                "C204988",
                contractors_list,
                "GRADING, DRAINAGE, PAVING,SIGNALS, AND WALLS.",
            ),
            location=query_google_maps("Cumberland"),
        )
        with open("pickles/C204988_inference_proposal.pkl", "wb") as f:
            pickle.dump(inference_proposal, f)
    else:
        print("Loading inference proposal from pickle")
        with open("pickles/C204988_inference_proposal.pkl", "rb") as f:
            inference_proposal = pickle.load(f)
    inference = Inference(embedding_dict, combined_df, inference_proposal)

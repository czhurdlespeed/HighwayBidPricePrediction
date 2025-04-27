import csv
import json
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from geopy.distance import geodesic as GD


@dataclass
class PayItemBid:
    """Represents a single pay item bid instance"""

    contract: str  # Contract ID
    county: str  # County
    payitem: str  # Pay item name
    embedding: List[float]  # Pay item embedding
    distance: float  # Project location distance
    quantity: float  # Bid quantity
    unit_price: float  # Bid unit price
    contractor_id: str  # Who made the bid
    is_winning: bool  # Whether this was part of winning bid
    bid_date: str  # When the bid was made
    competitors: List[str]  # List of competitor IDs for this bid
    bid_hash: str
    bid_size: float


@cache
def create_embedding_dict():
    embedding_dict = {}
    embedding_path = (
        Path().cwd()
        / "embeddings"
        / "jina_embedding_proposalIndex_response.json"
    )
    with open(embedding_path, "r") as f:
        embeddings = json.load(f)
    data: List = embeddings["data"]
    for info_dict in data:
        embedding_dict[info_dict["index"]] = np.array(
            info_dict["embedding"]
        ).tolist()
    return embedding_dict


@cache
def load_embedding(contract_id: str):
    embeddings = create_embedding_dict()
    return embeddings.get(contract_id, None)


contract_locations: Dict[str, Tuple[float, float]] = {}


def get_contract_location(contract: str) -> Tuple[float, float]:
    """Cache and return contract locations"""
    if contract not in contract_locations:
        df = pd.read_csv(f"nc_csv/{contract}.csv")
        contract_locations[contract] = (
            df.at[0, "Latitude"],
            df.at[0, "Longitude"],
        )
    return contract_locations[contract]


@lru_cache(maxsize=10000)
def load_distance(contract_one: str, contract_two: str) -> float:
    """
    Compute distance between contracts with multiple levels of caching:
    1. Contract locations are cached to avoid repeated CSV reads
    2. Distance computations are cached with lru_cache
    3. Symmetric computation (A->B same as B->A)
    """
    # Sort contracts to ensure symmetric caching (A->B same as B->A)
    if contract_one > contract_two:
        contract_one, contract_two = contract_two, contract_one

    location_one = get_contract_location(contract_one)
    location_two = get_contract_location(contract_two)

    return float(GD(location_one, location_two).miles)


@cache
def load_winning_status(contract_id: str, contractor: str):
    rankings_df = pd.read_csv(
        Path("contract_rankings") / f"{contract_id}_rankings.csv"
    )
    contractor = contractor.lower().strip()
    winning_contractor = rankings_df.at[0, "Vendor Name"].lower().strip()
    return contractor == winning_contractor


@cache
def load_bid_date(contract_id: str):
    contract_df = pd.read_csv(f"nc_csv/{contract_id}.csv")
    return str(contract_df.at[0, "Date"])


@cache
def load_competitors(contract_id: str, contractor: str):
    rankings_df = pd.read_csv(
        Path("contract_rankings") / f"{contract_id}_rankings.csv"
    )
    competitors = rankings_df["Vendor Name"].tolist()
    competitors.remove(contractor)
    return competitors


@cache
def load_contract_size(contract_id: str) -> Callable[[str], float]:
    contract_ranking_df = pd.read_csv(
        f"contract_rankings/{contract_id}_rankings.csv"
    )
    contractors_bidsize_dict = dict(
        zip(
            contract_ranking_df["Vendor Name"],
            contract_ranking_df["Total Contract Amount"],
        )
    )

    def get_bidsize(contractor):
        return contractors_bidsize_dict.get(contractor, 0)

    return get_bidsize


def combine_csv_files(
    training: bool = True,
) -> Tuple[pd.DataFrame, Optional[List[str]]]:
    combined_df = pd.DataFrame()
    for file in Path("nc_csv").glob("*.csv"):
        df = pd.read_csv(file)
        combined_df = pd.concat([combined_df, df])
        combined_df["Item"] = combined_df["Item"].str.strip()
        combined_df["Proposal"] = combined_df["Proposal"].str.strip()
    if training:
        val_size = int(0.05 * len(list(Path("nc_csv").glob("*.csv"))))
        contract_date = combined_df.loc[
            :, ["Proposal", "Date"]
        ].drop_duplicates()
        contract_date["Date"] = pd.to_datetime(contract_date["Date"])
        contract_date = contract_date.sort_values(by="Date", ascending=True)
        val_contracts = contract_date.tail(val_size).astype(str)
        combined_df = combined_df[
            ~combined_df["Proposal"].astype(str).isin(val_contracts["Proposal"])
        ]
        return combined_df, val_contracts["Proposal"].tolist()
    return combined_df, None


def return_historic_bids_for_payitem(
    payitem: str, combined_df: pd.DataFrame
) -> List[PayItemBid]:
    historic_bids = []
    historic_payitem_bids = combined_df[combined_df["Item"] == payitem]
    contract_hash = {}
    for index_hash, (_, row) in enumerate(historic_payitem_bids.iterrows()):
        contract = str(row["Proposal"]).strip()
        hash_index = contract_hash.get(contract, None)
        if hash_index is None:
            contract_hash[contract] = str(index_hash)

        contractor = row["Vendor Name"]
        bid_size = load_contract_size(contract)(contractor)
        bid_date = row["Date"]
        quantity = row["Quantity"]
        county = row["County"]
        unit_price = row["Unit Price"]
        is_winning = load_winning_status(contract, contractor)
        competitors = load_competitors(contract, contractor)
        embedding = load_embedding(row["Proposal"])
        historic_bids.append(
            PayItemBid(
                contract=contract,
                county=county,
                payitem=payitem,
                embedding=embedding,
                distance=0,
                quantity=quantity,
                unit_price=unit_price,
                contractor_id=contractor,
                is_winning=is_winning,
                bid_date=bid_date,
                competitors=competitors,
                bid_hash=contract_hash[contract],
                bid_size=bid_size,
            )
        )
    return historic_bids


def loop_contractor_payitems_inference_contract(filepath: str):
    contract_df = pd.read_csv(filepath)
    contract_df["Item"] = contract_df["Item"].str.strip()
    contract_df["Proposal"] = contract_df["Proposal"].str.strip()
    contract_df["Vendor Name"] = contract_df["Vendor Name"].str.strip()
    for _, row in contract_df.iterrows():
        contractor = row["Vendor Name"]
        contract = row["Proposal"]
        payitem = row["Item"]
        bid_size = load_contract_size(contract)(contractor)
        bid_date = row["Date"]
        quantity = row["Quantity"]
        county = row["County"]
        unit_price = row["Unit Price"]
        is_winning = load_winning_status(contract, contractor)
        competitors = load_competitors(contract, contractor)
        embedding = load_embedding(row["Proposal"])
        yield PayItemBid(
            contract=contract,
            county=county,
            payitem=payitem,
            embedding=embedding,
            distance=0,
            quantity=quantity,
            unit_price=unit_price,
            contractor_id=contractor,
            is_winning=is_winning,
            bid_date=bid_date,
            competitors=competitors,
            bid_hash=contract,
            bid_size=bid_size,
        )


def estimate_inference_contract_size(
    total_df: pd.DataFrame,
    inference_contract_data: Tuple[List[str], List[float]],
    contract: str,
) -> float:
    total_size = 0
    payitems, quantities = inference_contract_data
    with open(f"global_weighted_avg_{contract}.csv", "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            [
                "Payitem",
                "Global Weighted Avg",
                "Proposal Quantity",
                "Global Unit Price Prediction",
            ]
        )
        for payitem, quantity in zip(payitems, quantities):
            payitem_df = total_df[total_df["Item"] == payitem]
            quantities = np.array(payitem_df["Quantity"])
            unit_prices = np.array(payitem_df["Unit Price"])
            if len(quantities) == 0 or np.all(quantities == 0):
                continue
            payitem_global_weighted_avg = np.round(
                np.average(unit_prices, weights=quantities), 2
            )

            """
            print(
                f"Payitem: {payitem} - Weighted Avg: {payitem_global_weighted_avg}"
            )
            """

            total_size += payitem_global_weighted_avg * quantity
            global_unit_price_prediction = round(
                payitem_global_weighted_avg * quantity, 2
            )
            csv_writer.writerow(
                [
                    payitem,
                    payitem_global_weighted_avg,
                    quantity,
                    global_unit_price_prediction,
                ]
            )
        csv_writer.writerow(["", "", "Total", round(total_size, 0)])
    return total_size


if __name__ == "__main__":
    combined_df, val_contracts = combine_csv_files(training=True)
    payitem = "1523000000-E"
    historic_bids = return_historic_bids_for_payitem(payitem, combined_df)
    print(
        f"Number of historic payitem records for {payitem}: {len(historic_bids)}"
    )

    for counter, bid in enumerate(historic_bids):
        if counter == 5:
            break
        counter += 1
        # Print all except embedding
        print(bid.contract)
        print(bid.payitem)
        print(bid.distance)
        print(bid.quantity)
        print(bid.unit_price)
        print(bid.contractor_id)
        print(bid.competitors)
        print(bid.is_winning)
        print(bid.bid_date)
        print(bid.bid_hash)
        print(bid.bid_size)
        print(50 * "-")
        print(1 * "\n")

import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def combine_csv_files_to_single_df(dir: Path) -> pd.DataFrame:
    files = dir.glob("*.csv")
    df = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    return df


def add_extension_total_column(csv_path: Path) -> pd.DataFrame:
    rankings_path = Path("contract_rankings")
    rankings_path.mkdir(exist_ok=True)
    contract_df = pd.read_csv(csv_path)
    contract_id = str(contract_df.at[0, "Proposal"]).strip()
    contractor_groups = contract_df.groupby("Vendor Name")
    contract_rankings = dict()
    for contractor_name, group in contractor_groups:
        contract_df.loc[group.index, "Total Contract Amount"] = int(
            group["Extension"].sum()
        )
        contract_rankings[contractor_name] = int(group["Extension"].sum())

    contract_df.to_csv(csv_path, index=False)

    # Save dict to csv
    rankings_df = pd.DataFrame(
        contract_rankings.items(),
        columns=["Vendor Name", "Total Contract Amount"],
    )
    rankings_df = rankings_df.sort_values(
        "Total Contract Amount", ascending=True
    )
    rankings_df.to_csv(
        rankings_path / f"{contract_id}_rankings.csv", index=False
    )

    return contract_df, rankings_df


def compute_payitem_weighted_averages(combined_df: pd.DataFrame) -> None:
    weighted_path = Path("payitem_weightedaverages")
    weighted_path.mkdir(exist_ok=True)
    combined_df["Item"] = combined_df["Item"].str.strip()
    payitem_groups = combined_df.groupby("Item")
    weighted_averages = dict()
    for payitem, group in payitem_groups:
        if len(group.index) == 1:
            weighted_averages[payitem] = group["Unit Price"].values[0]
        else:
            weighted_averages[payitem] = np.round(
                np.average(group["Unit Price"], weights=group["Quantity"]), 2
            )
    weighted_averages_df = pd.DataFrame(
        weighted_averages.items(),
        columns=["Item", "Unit Price Weighted Average"],
    )
    weighted_averages_df = weighted_averages_df.sort_values(
        "Item", ascending=True
    )
    weighted_averages_df.to_csv(
        weighted_path / "weighted_averages.csv", index=False
    )


def estimate_contract_extension(inference_df: pd.DataFrame) -> int:
    weighted_averages = pd.read_csv(
        "payitem_weightedaverages/weighted_averages.csv"
    )
    estimated_extension = 0
    for item, quantity in zip(
        inference_df["Item"].tolist(), inference_df["Quantity"].tolist()
    ):
        if item not in weighted_averages["Item"].values or pd.isna(
            weighted_averages[weighted_averages["Item"] == item][
                "Unit Price Weighted Average"
            ].values[0]
        ):
            print(f"Item {item} not found in weighted averages")
            continue
        weighted_average = weighted_averages[weighted_averages["Item"] == item][
            "Unit Price Weighted Average"
        ].values[0]
        estimated_extension += float(weighted_average) * float(quantity)
    return round(estimated_extension, 2)


if __name__ == "__main__":
    dir = Path("nc_csv")
    shutil.rmtree("contract_rankings", ignore_errors=True)
    for file in dir.glob("*.csv"):
        contract_df, rankings_df = add_extension_total_column(file)
    combined_df = combine_csv_files_to_single_df(dir)
    compute_payitem_weighted_averages(combined_df)
    inference_df = pd.read_csv(
        "~/Downloads/Proposal_NCDOT_L241217_C204988_003.csv"
    )
    estimated_extension = estimate_contract_extension(inference_df)
    print(f"Estimated extension: {estimated_extension}")

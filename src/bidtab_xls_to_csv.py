import csv
import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd


def read_excel(file_path: Path) -> pd.DataFrame:
    return pd.read_excel(file_path, header=None)


def desired_general_columns(
    general_contract_info: pd.DataFrame,
) -> pd.DataFrame:
    desired_columns = general_contract_info.iloc[
        :,
        [2, 5, 9, 12, 13, 14, 16, 18, 19, 20, 21, 22],
    ].copy()
    desired_columns.rename(
        columns={
            2: "Date",
            5: "Proposal",
            9: "County",
            12: "Proposal Type",
            13: "Proposal Description",
            14: "Item",
            16: "Line Item",
            18: "Item Type",
            19: "Item Description",
            20: "Item Description Subset",
            21: "Quantity",
            22: "Unit",
        },
        inplace=True,
    )
    desired_columns["Date"] = pd.to_datetime(desired_columns["Date"])
    desired_columns["Line Item"] = pd.to_numeric(
        desired_columns["Line Item"], errors="coerce"
    ).fillna(0)
    # Get mask of original empty values
    empty_mask = desired_columns["Line Item"] == 0
    # Forward fill values
    filled_values = desired_columns["Line Item"].ffill()
    # Where values were empty, add 1 to previous value
    desired_columns.loc[empty_mask, "Line Item"] = filled_values[empty_mask] + 1
    desired_columns["Line Item"] = desired_columns["Line Item"].astype(int)
    return desired_columns


def contractor_per_payitem(contract_df: pd.DataFrame) -> Dict:
    contract_df = contract_df.reset_index(drop=True)
    contractor_payitems = contract_df.iloc[:, 24:]
    contract_information = dict()
    proposal_id = None
    for row in range(0, contractor_payitems.shape[0]):
        for col in range(0, contractor_payitems.shape[1] - 1, 5):
            general_contract_row_info = desired_general_columns(
                contract_df.iloc[row, :24].to_frame().T
            )
            line_item = general_contract_row_info.at[row, "Line Item"]
            contractor = contractor_payitems.iloc[row, col]
            if pd.isna(contractor) or contractor == "" or contractor == " ":
                continue
            unit_price = contractor_payitems.iloc[row, col + 2]
            extension = contractor_payitems.iloc[row, col + 3]

            if line_item not in contract_information:
                contract_information[line_item] = dict()

            contract_information[line_item][contractor] = {
                "unit_price": unit_price,
                "extension": extension,
                "general_contract_row_info": general_contract_row_info.reset_index(
                    drop=True
                ),
            }
            if proposal_id is None:
                proposal_id = str(
                    general_contract_row_info.at[row, "Proposal"]
                ).strip()
    """
    for k, v in contract_information.items():
        print(contract_information[k].keys())
        break
    """
    return contract_information, proposal_id


def format_and_write_to_csv(
    contract_information: Dict, proposal: str, path=Path.cwd() / "nc_csv"
) -> None:
    HEADER = [
        "Proposal",
        "Proposal Type",
        "Proposal Description",
        "Line",
        "Item",
        "Item Type",
        "Item Description",
        "Item Description Subset",
        "Quantity",
        "Unit",
        "Vendor Name",
        "Unit Price",
        "Extension",
        "County",
        "Date",
    ]
    Path.mkdir(path, exist_ok=True)

    with open(path / f"{proposal}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(HEADER)
        for line_item, contractors in contract_information.items():
            for contractor, data in contractors.items():
                general_contract_row_info = data["general_contract_row_info"]
                row = [
                    general_contract_row_info.at[0, "Proposal"],
                    general_contract_row_info.at[0, "Proposal Type"],
                    general_contract_row_info.at[0, "Proposal Description"],
                    general_contract_row_info.at[0, "Line Item"],
                    general_contract_row_info.at[0, "Item"],
                    general_contract_row_info.at[0, "Item Type"],
                    general_contract_row_info.at[0, "Item Description"],
                    general_contract_row_info.at[0, "Item Description Subset"],
                    general_contract_row_info.at[0, "Quantity"],
                    general_contract_row_info.at[0, "Unit"],
                    contractor,
                    data["unit_price"],
                    data["extension"],
                    general_contract_row_info.at[0, "County"],
                    general_contract_row_info.at[0, "Date"].strftime(
                        "%m/%d/%Y"
                    ),
                ]
                writer.writerow(row)


def process_letting_group_excel(raw_df: pd.DataFrame) -> None:
    valid_rows = raw_df[16].notna()
    unique_contracts = raw_df[4].unique().tolist()
    for i in unique_contracts:
        contract_df = raw_df[valid_rows & (raw_df[4] == i)]
        assert contract_df.shape[0] > 0, "Contract dataframe is empty"
        contract_info, proposal = contractor_per_payitem(contract_df)
        format_and_write_to_csv(contract_info, proposal)


if __name__ == "__main__":
    excel_lettings_directory_path = Path.cwd() / "ExcelFiles"
    shutil.rmtree(Path.cwd() / "nc_csv", ignore_errors=True)
    for file_path in excel_lettings_directory_path.glob("*.xls"):
        raw_excel_df = read_excel(file_path)
        process_letting_group_excel(raw_excel_df)

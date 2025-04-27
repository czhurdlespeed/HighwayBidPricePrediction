import os

import pandas as pd


def excel_predictions(data, excel_file_path):
    os.makedirs(os.path.dirname(excel_file_path), exist_ok=True)
    rows = []
    """output[proposal][contractor][payitem_number][
                                f"{payitem}:{line_item}:{payitem_desc}"
                            ] = [
                                round(prediction, 2),
                                inference_quantity,
                                extension,
                                number_payitem_occurances,
                                round(confidence, 1),
                                list(zip(distance, nearest_neighbors)),
                            ]
    """
    for proposal, contractors in data.items():
        for contractor, payitems in contractors.items():
            for payitem_full, details in payitems.items():

                payitem, line_item, payitem_desc = payitem_full.split(":", 2)
                print(f"Payitem: {payitem}")
                print(f"Line Item: {line_item}")
                print(f"Description: {payitem_desc}")
                (
                    prediction,
                    excel_quantity,
                    extension,
                    occurences,
                    confidence,
                    neighbors,
                    unit,
                ) = details
                nearest_proposals = ", ".join(n[1] for n in neighbors)
                print("Prediction: ", prediction)
                print("Excel Quantity: ", excel_quantity)
                print("Extension: ", extension)
                print("Occurences: ", occurences)
                print("Confidence: ", confidence)
                print("Nearest Proposals: ", nearest_proposals)
                print("Unit: ", unit)
                rows.append(
                    {
                        "Proposal": proposal,
                        "Line": line_item,
                        "Item": payitem,
                        "Description": payitem_desc,
                        "Contractor": contractor,
                        "Unit Price": prediction,
                        "Quantity": excel_quantity,
                        "Extension": extension,
                        "Occurences": occurences,
                        "Nearest Proposals": nearest_proposals,
                        "Unit": unit,
                    }
                )
    df = pd.DataFrame(rows)
    print(df.head())
    pivot = pd.pivot_table(
        df,
        values=[
            "Unit Price",
            "Extension",
            "Occurences",
            "Nearest Proposals",
        ],
        index=["Proposal", "Line", "Item", "Quantity", "Unit", "Description"],
        columns=["Contractor"],
        aggfunc="first",
        sort=False,
    )

    # Sort by Line number
    pivot = pivot.reset_index()  # Make the index into columns
    pivot["Line"] = pd.to_numeric(pivot["Line"])  # Convert Line to numeric
    pivot = pivot.sort_values("Line")  # Sort by Line
    pivot = pivot.set_index(
        ["Proposal", "Line", "Item", "Quantity", "Unit", "Description"]
    )  # Reset the index

    # Reorder and sort columns
    pivot = pivot.reorder_levels([1, 0], axis=1)
    pivot = pivot.sort_index(axis=1, level=0)
    metric_order = [
        "Unit Price",
        "Extension",
        "Occurences",
        "Nearest Proposals",
    ]
    pivot = pivot.reindex(metric_order, axis=1, level=1)

    # Create empty total row with the same structure as pivot
    total_row = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            [("Total", "", "", "", "", "")], names=pivot.index.names
        ),
        columns=pivot.columns,
        data=[["" for _ in pivot.columns]],  # Initialize all cells as empty
    )

    # Calculate and set totals for each contractor's Extension column
    for contractor in df["Contractor"].unique():
        print(f"\nCalculating total for {contractor}")

        # Get all extensions for this contractor
        contractor_data = df[df["Contractor"] == contractor]
        contractor_extensions = contractor_data["Extension"]

        print("All extensions:", contractor_extensions.tolist())

        # Sum only numeric values
        total = 0
        for ext in contractor_extensions:
            if ext != "N/A" and isinstance(ext, (int, float)):
                total += ext

        # Set the total in the correct column
        col = (contractor, "Extension")
        if col in total_row.columns:
            total_row.loc[("Total", "", "", "", "", ""), col] = total

    # Combine pivot table with total row
    result = pd.concat([pivot, total_row])

    with pd.ExcelWriter(excel_file_path, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="Organized Data")
        df.to_excel(writer, sheet_name="Raw Data", index=False)
    print(f"Excel file saved to {excel_file_path}")

import matplotlib.pyplot as plt
import numpy as np

def money_left_on_table_viz():
    bid_amounts = [10_000_000, 10_750_000, 11_000_000]
    x = ["Contractor 1", "Contractor 2", "Contractor 3"]
    money_left = [10_749_999]

    width = 0.3
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    x_pos = [1, 1+width, 1+2*width]

    ax.set_title("Contract Bids and Money Left on Table", fontweight="bold")
    ax.bar(x_pos, bid_amounts, width=width, facecolor=["xkcd:light orange", "xkcd:light grey", "xkcd:gunmetal"], zorder=3, edgecolor="black", label=[f"${bid_amounts[0]:,.0f}", f"${bid_amounts[1]:,.0f}", f"${bid_amounts[2]:,.0f}"])
    ax.bar(x_pos[0], money_left, width=width, facecolor="xkcd:red", zorder=2, label=f"Money Left on Table = ${bid_amounts[1] - bid_amounts[0] -1:,.0f}")
    ax.set_xlabel("Contractor", fontweight="bold")
    ax.set_ylabel("Bid Amount", fontweight="bold")
    ax.grid(True, which="both", zorder=1)
    ax.legend(loc="upper right", ncol=2)
    ax.set_ylim(0, 13_000_000)
    # Format y-axis ticks with dollar signs and rotate them
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x)
    ax.set_xlim(0.7, 1.9)
    plt.savefig("./money_left_on_table.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    money_left_on_table_viz()
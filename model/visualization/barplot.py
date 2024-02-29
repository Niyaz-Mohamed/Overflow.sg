from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

DATASOURCENAME = "Model Barplot.csv"
IMAGENAME = "model-eval-barplot.png"

df = pd.read_csv(os.path.join(__file__, f"../datasources/{DATASOURCENAME}"))

fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # Adjust figsize as needed
metrics = ("R2", "MAE", "RMSE", "MAPE")
colors = {
    "XGB": "green",
    "CART": "turquoise",
    "RF": "mediumblue",
    "MLP": "orange",
    "SVM": "red",
}

for i, metric in enumerate(metrics):
    # Get current axes
    row = i // 2
    col = i % 2
    ax = axs[row, col]

    # Get lists of metric values for each model
    values = [
        df[df["Model"] == model][metric].values[0] for model in df["Model"].unique()
    ]

    # Draw plot
    bplots = ax.bar(
        df["Model"].unique(),
        values,
        color=[colors[model] for model in df["Model"].unique()],
    )

    # Add numbers above each bar
    # for bar, value in zip(bplots, values):
    #     ax.text(
    #         bar.get_x() + bar.get_width() / 2,
    #         bar.get_height(),
    #         f"{value:.2f}",
    #         ha="center",
    #         va="bottom",
    #     )

    # Marking, labels, grids, etc
    if metric == "R2":
        metric = "R²"
    ax.set_title(metric, fontsize=14)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="gray", linestyle="dashed")
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=10)

    # Scale and rotate xticks
    plt.sca(ax)
    plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.subplots_adjust(hspace=0.3)
plt.savefig(
    os.path.join(__file__, "../finalgraphs/", IMAGENAME),
    bbox_inches="tight",
    dpi=1200,
)
plt.show()

from matplotlib import pyplot as plt
import pandas as pd, numpy as np, os

DATASOURCENAME = "Distance Boxplot.csv"
IMAGENAME = "dist-boxplot.png"

# TODO: Provide a data csv file to construct the boxplot
df = pd.read_csv(os.path.join(__file__, f"../{DATASOURCENAME}"))

fig, axs = plt.subplots(2, 2)
metrics = ("R2", "MAE", "RMSE", "MAPE")
colors = {"XGB": "green", "CART": "turquoise", "RF": "mediumblue"}

for i, metric in enumerate(metrics):
    # Get current axes
    row = i // 2
    col = i % 2
    ax = axs[row, col]

    # Get lists of metric values for each model first)
    values = [df[df["Model"] == model][metric].values for model in df["Model"].unique()]
    # Draw plot
    bplots = ax.boxplot(
        values,
        notch=True,  # notch shape
        vert=True,  # vertical box alignment
        patch_artist=True,  # fill with color
        labels=df["Model"].unique(),
    )  # will be used to label x-ticks

    # Color in the plot
    colorList = np.repeat(list(colors.values()), 3)
    for bp, color in zip(bplots["boxes"], colorList):
        bp.set_facecolor(color)

    # Marking, labels, grids, etc
    if metric == "R2":
        metric = "R²"
    ax.set_title(metric)
    ax.yaxis.grid(True)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=7)

    # Scale and apply rotation
    plt.sca(ax)
    plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.subplots_adjust(hspace=0.5)
plt.savefig(os.path.join(__file__, "../", IMAGENAME), bbox_inches="tight")
plt.show()

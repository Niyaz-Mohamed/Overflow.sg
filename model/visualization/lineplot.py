from matplotlib import pyplot as plt
import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt

DATASOURCENAME = "XGB Convergence.csv"
IMAGENAME = "xgb-converge.png"

# TODO: Provide a csv file to generate the barplot from
df = pd.read_csv(os.path.join(__file__, f"../datasources/{DATASOURCENAME}"))

# Define colors for each line
colors = {"R2_times_10": "blue", "MAE": "orange", "RMSE": "green"}

# Adjust figure size if needed
plt.figure(figsize=(8, 4))

# Iterate through columns except 'Iterations' and plot them dynamically
for column in df.columns:
    if column != "Iterations":
        plt.plot(
            df["Iterations"],
            df[column],
            label=column,
            color=colors.get(column, "black"),
        )

# Customize plot settings
plt.xlabel("Iterations")
plt.xlim(left=0)
plt.legend(edgecolor="black", framealpha=1.0)
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.savefig(
    os.path.join(__file__, "../finalgraphs/", IMAGENAME),
    bbox_inches="tight",
    dpi=1200,
)
plt.show()

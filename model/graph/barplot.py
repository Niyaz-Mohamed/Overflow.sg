from matplotlib import pyplot as plt
import pandas as pd, numpy as np, os

DATASOURCENAME = "Model Barchart.csv"
IMAGENAME = "model-barchart.png"

# TODO: Provide a data csv file to construct the bar chart
df = pd.read_csv(os.path.join(__file__, f"../{DATASOURCENAME}"))

# Extracting data for plotting
models = df["Model"]
colors = ["blue", "orange", "green", "red", "purple"]  # Colors matched to each model
metrics = ["R2", "MAE", "RMSE", "MAPE"]
metric_values = df[metrics].values.T  # Transpose for easy plotting


# Plot the vertical bar chart
fig, ax = plt.subplots(figsize=(10, 6))
barWidth = 0.15
opacity = 0.8

for i, model in enumerate(models):
    plt.bar(
        np.arange(len(metrics)) + i * barWidth,
        metric_values[:, i],
        width=barWidth,
        alpha=opacity,
        label=model,
        color=colors[i],
    )

# Add zero line and minor ticks
plt.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
ax.yaxis.set_minor_locator(plt.MultipleLocator(1))

# Add graph labels
r2Index = metrics.index("R2")
metrics[r2Index] = "R²"
plt.xlabel("Evaluation Metrics")
plt.ylabel("Metric Values")
plt.title("Performance for each model after a single run")
plt.xticks(np.arange(len(metrics)) + (len(models) - 1) * barWidth / 2, metrics)
plt.legend()

# Save the figure
plt.savefig(os.path.join(__file__, "../", IMAGENAME), bbox_inches="tight")
plt.show()

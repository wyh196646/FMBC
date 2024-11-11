import pandas as pd
import matplotlib.pyplot as plt

# Let's read the data from the provided CSV file and re-create the plots with y-axis labels and save the image.
data_path = "/home/baizhiwang/pathology/xiaorong_test/huatu/plot.csv"

# Read the CSV file
df = pd.read_csv(data_path, index_col=0)

# Re-plotting with y-axis labels and saving the figure
fig, axes = plt.subplots(1, 5, figsize=(20, 6), sharey=True)

# Colors for each method with C2NO in a darker shade#A1A9D0#9AC9DB
colors = ["#A5B6C5", "#E7DAD2", "#E7EFFA", "#8A9A5B", "#556B2F"]


for i, gene in enumerate(df.index):
    ax = axes[i]
    df.loc[gene].plot(kind="bar", ax=ax, color=colors, edgecolor="black")
    ax.set_title(gene)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    # Annotating each bar with its value
    for j, value in enumerate(df.loc[gene]):
        ax.text(j, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=10)


plt.tight_layout()
plt.savefig(
    "/home/baizhiwang/pathology/xiaorong_test/huatu/gene_auc_plots.png"
)  # Saving the plot as an image

plt.show()
fig, ax = plt.subplots(figsize=(8, 6))
mean_values = df.mean()
mean_values.plot(kind="bar", color=colors, edgecolor="black", ax=ax)
ax.set_title("Average AUC for Each Method")
ax.set_ylabel("Average AUC")
ax.set_ylim(0, 1)

# Set x-axis labels horizontally
ax.set_xticklabels(mean_values.index, rotation=0)

# Annotating each bar with its average value
for i, value in enumerate(mean_values):
    ax.text(i, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig(
    "/home/baizhiwang/pathology/xiaorong_test/huatu/mean_auc_plots.png"
)  # Saving the plot as an image

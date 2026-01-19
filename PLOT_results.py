import numpy as np
import pandas as pd
import seaborn as sns
import os
from matplotlib import pyplot as plt

results = pd.read_csv('evaluation_results.csv')

def _plot_Result():
    # Create output folder
    output_dir = "metric_plots"
    os.makedirs(output_dir, exist_ok=True)
    # Professional color palette
    palette = sns.color_palette("Set2", n_colors=len(results['Model']))

    # --- Bar Plots for Metrics ---
    metrics = results.columns[1:]  # exclude 'Model'

    for metric in metrics:
        plt.figure( )
        bars = plt.bar(results['Model'], results[metric], width =0.4,color=palette, edgecolor='green', alpha=0.85)

        # Annotate bars
        for bar in bars:
            yval = bar.get_height()
            offset = max(0.01 * yval, 0.01)  # Ensure text above bar
            plt.text(
                bar.get_x() + bar.get_width()/2.0,
                yval + offset,
                f"{yval:.2f}",
                ha='center',
                # va='bottom',
                fontsize=9,
                fontweight='bold'
            )

        plt.title(f"{metric} Comparison Across Models", fontsize=12, weight='bold')
        plt.ylabel(metric, fontsize=12)
        # plt.xticks(rotation=30, ha='right', fontsize=11)
        # plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        png_path = os.path.join(output_dir, f"{metric}.png")
        plt.savefig(png_path, dpi=300)
        plt.close()
        print(f"Saved plot: {png_path}")



file_path = r"evaluation_results.csv"
df = pd.read_csv(file_path)

error_metrics = ["RMSE", "MAE", "MSE", "MAPE"]
performance_metrics = ["R2", "ExplainedVariance", "PearsonCorr", "Recall", "F1"]

# Normalize (invert error metrics so that higher = better)
for col in error_metrics:
    df[col + "_norm"] = 1 - (df[col] / df[col].max())

for col in performance_metrics:
    df[col + "_norm"] = df[col] / df[col].max()


df["Accuracy"] = df[["R2_norm", "PearsonCorr_norm", "Recall_norm", "F1_norm"]].mean(axis=1)
df["Diversity"] = 1 - df[["RMSE_norm", "MAE_norm", "MSE_norm"]].std(axis=1)  # consistency
df["Robustness"] = df[["Accuracy", "Diversity"]].mean(axis=1)

summary = df[["Model", "Accuracy", "Diversity", "Robustness"]].sort_values("Accuracy", ascending=False)
print("\n=== Recommendation Evaluation Summary ===\n")
# print(summary)
# summary.to_csv('_metrics.csv')
# print(summary)
summary = pd.read_csv('_metrics.csv')
print(summary)
df = summary

sns.set(style="whitegrid", context="talk")

plt.figure(figsize=(8, 6))
# plt.figure( )

x = np.arange(len(df["Model"]))
width = 0.25

# Custom colors (pleasant and distinct)
colors = {
    "Accuracy": "#1f77b4",    # blue
    "Diversity": "#2ca02c",   # green
    "Robustness": "#ff7f0e"   # orange
}

# Create bars with colors and alpha transparency
bar1 = plt.bar(x - width, df["Accuracy"], width, color=colors["Accuracy"], edgecolor='black', label='Accuracy', alpha=0.9)
bar2 = plt.bar(x, df["Diversity"], width, color=colors["Diversity"], edgecolor='black', label='Diversity', alpha=0.9)
bar3 = plt.bar(x + width, df["Robustness"], width, color=colors["Robustness"], edgecolor='black', label='Robustness', alpha=0.9)

# Add gridlines
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Axis labels and title
plt.xticks(x, df["Model"], fontsize=12)
plt.ylabel("Score", fontsize=12)
# plt.title("Recommendation Accuracy, Diversity, and Robustness Comparison", fontsize=15, fontweight='bold')

# Legend styling
plt.legend(frameon=True, shadow=True, fontsize=12, loc='lower left')

# Annotate bars with values
def annotate_bars(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, height + 0.02,
            f"{height:.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold'
        )

annotate_bars(bar1)
annotate_bars(bar2)
annotate_bars(bar3)

# Limit and layout
plt.ylim(0, 1.15)
plt.tight_layout()
plt.savefig('_plot.png', dpi = 300)
plt.show()
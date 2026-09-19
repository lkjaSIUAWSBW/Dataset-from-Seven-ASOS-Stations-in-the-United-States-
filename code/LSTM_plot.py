import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# 1. Read Excel file
# =========================
file_path = "LSTM画图.xlsx"   # Change this if needed
df = pd.read_excel(file_path, sheet_name=0)

# =========================
# 2. Rename columns to English
# =========================
df = df.rename(columns={
    "数据条件": "Condition",
    "站点": "Station",
    "平均绝对误差（m/s）": "MAE",
    "均方根误差（m/s）": "RMSE"
})

# =========================
# 3. Translate condition names
# =========================
condition_map = {
    "Raw 仅使用风速": "Raw (wind speed only)",
    "Processed 使用风速和质量特征": "Processed (wind speed + quality features)"
}
df["Condition"] = df["Condition"].replace(condition_map)

# =========================
# 4. Translate station names if needed
# =========================
station_map = {
    "全部站点汇总": "All stations"
}
df["Station"] = df["Station"].replace(station_map)

# =========================
# 5. Pivot tables
# =========================
pivot_mae = df.pivot(index="Station", columns="Condition", values="MAE")
pivot_rmse = df.pivot(index="Station", columns="Condition", values="RMSE")

# =========================
# 6. Define station order
# =========================
station_order = [
    "All stations",
    "ALBUQUERQUE INTL",
    "ALLENTOWN-BETHLEHEM",
    "Abilene",
    "BUFFALO",
    "DRYDEN",
    "SISSETON",
    "WICHITA (AAO)"
]

station_order = [s for s in station_order if s in pivot_mae.index]

pivot_mae = pivot_mae.loc[station_order]
pivot_rmse = pivot_rmse.loc[station_order]

# =========================
# 7. Plot settings
# =========================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.2

raw_label = "Raw (wind speed only)"
processed_label = "Processed (wind speed + quality features)"

raw_color = "#D55E00"        # orange-red
processed_color = "#009E73"  # green
line_color = "#A6A6A6"       # gray
grid_color = "#D9D9D9"

# =========================
# 8. Create figure (vertical layout)
# =========================
fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharey=True)
y = np.arange(len(station_order))

# =========================
# 9. Function to draw dumbbell plot
# =========================
def draw_dumbbell(ax, data, xlabel):
    raw_vals = data[raw_label].values
    processed_vals = data[processed_label].values

    # Connecting lines
    for i in range(len(y)):
        ax.plot(
            [raw_vals[i], processed_vals[i]],
            [y[i], y[i]],
            color=line_color,
            lw=2.0,
            zorder=1
        )

    # Scatter points
    ax.scatter(
        raw_vals, y,
        s=130,
        color=raw_color,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
        label=raw_label
    )
    ax.scatter(
        processed_vals, y,
        s=130,
        color=processed_color,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
        label=processed_label
    )

    # Value labels
    min_x = min(np.min(raw_vals), np.min(processed_vals))
    max_x = max(np.max(raw_vals), np.max(processed_vals))
    x_range = max_x - min_x
    offset = x_range * 0.02 if x_range > 0 else 0.01

    for i in range(len(y)):
        ax.text(
            raw_vals[i] + offset,
            y[i] + 0.14,
            f"{raw_vals[i]:.3f}",
            color=raw_color,
            fontsize=10,
            va="center"
        )
        ax.text(
            processed_vals[i] + offset,
            y[i] - 0.14,
            f"{processed_vals[i]:.3f}",
            color=processed_color,
            fontsize=10,
            va="center"
        )

    # Axis formatting
    ax.set_yticks(y)
    ax.set_yticklabels(station_order)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=13)
    ax.grid(axis="x", linestyle="--", color=grid_color, alpha=0.8)
    ax.set_axisbelow(True)

    # Add x-axis margins
    pad = x_range * 0.18 if x_range > 0 else 0.1
    ax.set_xlim(min_x - pad * 0.25, max_x + pad)

# =========================
# 10. Draw MAE and RMSE panels
# =========================
draw_dumbbell(axes[0], pivot_mae, "MAE (m s$^{-1}$)")
draw_dumbbell(axes[1], pivot_rmse, "RMSE (m s$^{-1}$)")

axes[0].set_ylabel("Station", fontsize=13)
axes[1].set_ylabel("Station", fontsize=13)

# =========================
# 11. Legend
# =========================
legend_elements = [
    Line2D([0], [0], marker='o', color='none',
           markerfacecolor=raw_color, markeredgecolor='white',
           markersize=10, label=raw_label),
    Line2D([0], [0], marker='o', color='none',
           markerfacecolor=processed_color, markeredgecolor='white',
           markersize=10, label=processed_label)
]

fig.legend(
    handles=legend_elements,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 0.99),
    fontsize=12
)

# =========================
# 12. Layout and save as SVG
# =========================
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("LSTM_raw_vs_processed_vertical.svg", format="svg", bbox_inches="tight")
plt.show()
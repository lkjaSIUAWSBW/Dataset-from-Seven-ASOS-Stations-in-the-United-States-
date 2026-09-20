"""
\u7ed8\u5236 LSTM \u5355\u6b65\u98ce\u901f\u9884\u6d4b\u7684 Raw \u4e0e Processed \u5bf9\u7167\u56fe。

\u8f93\u5165：LSTM\u5355\u6b65Raw\u4e0eProcessed\u8d28\u91cf\u7279\u5f81\u5bf9\u7167\u7ed3\u679c.csv
\u8f93\u51fa：
  1. LSTM\u5355\u6b65Raw\u4e0eProcessed\u5bf9\u7167\u56fe.png（600 dpi）
  2. LSTM\u5355\u6b65Raw\u4e0eProcessed\u5bf9\u7167\u56fe.pdf（\u77e2\u91cf）
  3. LSTM\u5355\u6b65Raw\u4e0eProcessed\u5bf9\u7167\u56fe.svg（\u77e2\u91cf）

\u56fe\u5f62\u8bbe\u8ba1：\u4e09\u9762\u677f\u6a2a\u5411 dumbbell/lollipop \u56fe，\u5206\u522b\u5c55\u793a MAE、RMSE \u548c R²。
\u6bcf\u4e00\u884c\u8fde\u63a5 Raw \u4e0e Processed \u4e24\u4e2a\u70b9，\u7a81\u51fa\u4e24\u79cd\u6570\u636e\u6761\u4ef6\u7684\u5dee\u5f02；“\u5168\u90e8\u7ad9\u70b9\u6c47\u603b”
\u5355\u72ec\u7f6e\u4e8e\u9876\u90e8\u5e76\u52a0\u7c97，\u907f\u514d\u53ea\u5c55\u793a\u6709\u5229\u7ad9\u70b9。
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "LSTM\u5355\u6b65Raw\u4e0eProcessed\u8d28\u91cf\u7279\u5f81\u5bf9\u7167\u7ed3\u679c.csv"
OUTPUT_STEM = BASE_DIR / "LSTM\u5355\u6b65Raw\u4e0eProcessed\u5bf9\u7167\u56fe"


def choose_font():
    """\u4f18\u5148\u4f7f\u7528\u5e38\u89c1\u4e2d\u6587\u5b57\u4f53；\u56fe\u4e2d\u4e3b\u8981\u4e3a\u82f1\u6587\u548c\u7ad9\u70b9\u540d，\u4e2d\u6587\u5b57\u4f53\u7528\u4e8e\u517c\u5bb9。"""
    candidates = ["Arial", "DejaVu Sans", "Microsoft YaHei", "SimHei"]
    installed = {f.name for f in mpl.font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            return name
    return "DejaVu Sans"


def load_data():
    data = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    data = data[data["\u9a8c\u8bc1\u72b6\u6001"] == "\u5b8c\u6210"].copy()
    data["\u5e73\u5747\u7edd\u5bf9\u8bef\u5dee（m/s）"] = pd.to_numeric(data["\u5e73\u5747\u7edd\u5bf9\u8bef\u5dee（m/s）"], errors="coerce")
    data["\u5747\u65b9\u6839\u8bef\u5dee（m/s）"] = pd.to_numeric(data["\u5747\u65b9\u6839\u8bef\u5dee（m/s）"], errors="coerce")
    data["\u51b3\u5b9a\u7cfb\u6570"] = pd.to_numeric(data["\u51b3\u5b9a\u7cfb\u6570"], errors="coerce")
    data = data.dropna(subset=["\u5e73\u5747\u7edd\u5bf9\u8bef\u5dee（m/s）", "\u5747\u65b9\u6839\u8bef\u5dee（m/s）", "\u51b3\u5b9a\u7cfb\u6570"])

    condition_map = {
        "Raw \u4ec5\u4f7f\u7528\u98ce\u901f": "Raw",
        "Processed \u4f7f\u7528\u98ce\u901f\u548c\u8d28\u91cf\u7279\u5f81": "Processed + quality features",
    }
    data["\u6761\u4ef6"] = data["\u6570\u636e\u6761\u4ef6"].map(condition_map)
    data = data.dropna(subset=["\u6761\u4ef6"])

    station_order = [
        "\u5168\u90e8\u7ad9\u70b9\u6c47\u603b",
        "ALBUQUERQUE INTL",
        "ALLENTOWN-BETHLEHEM",
        "Abilene",
        "BUFFALO",
        "DRYDEN",
        "SISSETON",
        "WICHITA (AAO)",
    ]
    data["\u7ad9\u70b9"] = pd.Categorical(data["\u7ad9\u70b9"], categories=station_order, ordered=True)
    return data.sort_values(["\u7ad9\u70b9", "\u6761\u4ef6"])


def plot_panel(ax, wide, metric, title, lower_is_better=True):
    raw_color = "#4C78A8"
    processed_color = "#F58518"
    line_color = "#B8B8B8"
    y = np.arange(len(wide))

    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)

    for i, (_, row) in enumerate(wide.iterrows()):
        raw_value = row.get((metric, "Raw"), np.nan)
        processed_value = row.get((metric, "Processed + quality features"), np.nan)
        if np.isfinite(raw_value) and np.isfinite(processed_value):
            ax.plot([raw_value, processed_value], [i, i], color=line_color, linewidth=2.2, zorder=1)
            ax.scatter(raw_value, i, s=64, color=raw_color, edgecolor="white", linewidth=0.9, zorder=3)
            ax.scatter(processed_value, i, s=64, color=processed_color, edgecolor="white", linewidth=0.9, zorder=3)

        
        if i == 0:
            ax.axhspan(i - 0.43, i + 0.43, color="#FFF2CC", alpha=0.75, zorder=0)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_yticks(y)
    ax.set_yticklabels(wide.index.astype(str), fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Error (m s$^{-1}$)" if metric != "\u51b3\u5b9a\u7cfb\u6570" else r"$R^2$", fontsize=10)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)

    if metric == "\u51b3\u5b9a\u7cfb\u6570":
        
        ax.set_xlim(0, max(1.0, float(wide[metric].max().max()) * 1.04))
    else:
        ax.set_xlim(0, float(wide[metric].max().max()) * 1.16)


def main():
    mpl.rcParams.update({
        "font.family": choose_font(),
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    data = load_data()
    metrics = [
        ("\u5e73\u5747\u7edd\u5bf9\u8bef\u5dee（m/s）", "MAE (m s$^{-1}$)"),
        ("\u5747\u65b9\u6839\u8bef\u5dee（m/s）", "RMSE (m s$^{-1}$)"),
        ("\u51b3\u5b9a\u7cfb\u6570", r"$R^2$"),
    ]
    wide = data.pivot_table(index="\u7ad9\u70b9", columns="\u6761\u4ef6", values=[m[0] for m in metrics], aggfunc="first")
    wide = wide.reindex([s for s in data["\u7ad9\u70b9"].cat.categories if s in wide.index])

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 5.8), sharey=True, constrained_layout=False)
    fig.subplots_adjust(left=0.21, right=0.98, bottom=0.16, top=0.84, wspace=0.24)

    for ax, (metric, title) in zip(axes, metrics):
        plot_panel(ax, wide, metric, title)

    handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="None", markersize=7,
                         markerfacecolor="#4C78A8", markeredgecolor="white", label="Raw"),
        mpl.lines.Line2D([], [], marker="o", linestyle="None", markersize=7,
                         markerfacecolor="#F58518", markeredgecolor="white", label="Processed + quality features"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.61, 0.925),
               ncol=2, frameon=False, fontsize=9.5, handletextpad=0.4, columnspacing=1.3)
    fig.text(0.02, 0.50, "Station", rotation=90, va="center", ha="center", fontsize=11)
    fig.text(0.50, 0.045,
             "Lower values indicate better performance for MAE and RMSE; higher values indicate better performance for $R^2$.",
             ha="center", fontsize=8.5, color="#444444")

    fig.savefig(f"{OUTPUT_STEM}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUTPUT_STEM}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUTPUT_STEM}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\u56fe\u5f62\u5df2\u4fdd\u5b58：{OUTPUT_STEM}.png/.pdf/.svg")


if __name__ == "__main__":
    main()

"""
绘制 LSTM 单步风速预测的 Raw 与 Processed 对照图。

输入：LSTM单步Raw与Processed质量特征对照结果.csv
输出：
  1. LSTM单步Raw与Processed对照图.png（600 dpi）
  2. LSTM单步Raw与Processed对照图.pdf（矢量）
  3. LSTM单步Raw与Processed对照图.svg（矢量）

图形设计：三面板横向 dumbbell/lollipop 图，分别展示 MAE、RMSE 和 R²。
每一行连接 Raw 与 Processed 两个点，突出两种数据条件的差异；“全部站点汇总”
单独置于顶部并加粗，避免只展示有利站点。
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "LSTM单步Raw与Processed质量特征对照结果.csv"
OUTPUT_STEM = BASE_DIR / "LSTM单步Raw与Processed对照图"


def choose_font():
    """优先使用常见中文字体；图中主要为英文和站点名，中文字体用于兼容。"""
    candidates = ["Arial", "DejaVu Sans", "Microsoft YaHei", "SimHei"]
    installed = {f.name for f in mpl.font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            return name
    return "DejaVu Sans"


def load_data():
    data = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    data = data[data["验证状态"] == "完成"].copy()
    data["平均绝对误差（m/s）"] = pd.to_numeric(data["平均绝对误差（m/s）"], errors="coerce")
    data["均方根误差（m/s）"] = pd.to_numeric(data["均方根误差（m/s）"], errors="coerce")
    data["决定系数"] = pd.to_numeric(data["决定系数"], errors="coerce")
    data = data.dropna(subset=["平均绝对误差（m/s）", "均方根误差（m/s）", "决定系数"])

    condition_map = {
        "Raw 仅使用风速": "Raw",
        "Processed 使用风速和质量特征": "Processed + quality features",
    }
    data["条件"] = data["数据条件"].map(condition_map)
    data = data.dropna(subset=["条件"])

    station_order = [
        "全部站点汇总",
        "ALBUQUERQUE INTL",
        "ALLENTOWN-BETHLEHEM",
        "Abilene",
        "BUFFALO",
        "DRYDEN",
        "SISSETON",
        "WICHITA (AAO)",
    ]
    data["站点"] = pd.Categorical(data["站点"], categories=station_order, ordered=True)
    return data.sort_values(["站点", "条件"])


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

        # 仅突出全部站点汇总行；不隐藏站点差异。
        if i == 0:
            ax.axhspan(i - 0.43, i + 0.43, color="#FFF2CC", alpha=0.75, zorder=0)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_yticks(y)
    ax.set_yticklabels(wide.index.astype(str), fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Error (m s$^{-1}$)" if metric != "决定系数" else r"$R^2$", fontsize=10)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)

    if metric == "决定系数":
        # R² 为越大越好，保持从 0 开始，避免夸大很小的差异。
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
        ("平均绝对误差（m/s）", "MAE (m s$^{-1}$)"),
        ("均方根误差（m/s）", "RMSE (m s$^{-1}$)"),
        ("决定系数", r"$R^2$"),
    ]
    wide = data.pivot_table(index="站点", columns="条件", values=[m[0] for m in metrics], aggfunc="first")
    wide = wide.reindex([s for s in data["站点"].cat.categories if s in wide.index])

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
    print(f"图形已保存：{OUTPUT_STEM}.png/.pdf/.svg")


if __name__ == "__main__":
    main()

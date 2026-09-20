from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.patches import Polygon





SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_FILE = SCRIPT_DIR / "location.xlsx"

OUTPUT_SVG = SCRIPT_DIR / "ASOS_station_locations_flag.svg"
OUTPUT_PNG = SCRIPT_DIR / "ASOS_station_locations_flag.png"





df = pd.read_excel(INPUT_FILE)


df.columns = df.columns.str.strip()

required_columns = {"Station name", "latitude", "longtitude"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"location.xlsx \u4e2d\u7f3a\u5c11\u4ee5\u4e0b\u5217: {sorted(missing_columns)}")


df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longtitude"] = pd.to_numeric(df["longtitude"], errors="coerce")


df = df.dropna(subset=["latitude", "longtitude"]).copy()


df["index"] = range(1, len(df) + 1)

print(df)





plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.unicode_minus": False,
    "svg.fonttype": "none"
})





projection = ccrs.PlateCarree()

fig = plt.figure(figsize=(10.5, 6.8), dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=projection)


ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())





ax.add_feature(
    cfeature.LAND.with_scale("50m"),
    facecolor="#F3E8C8",   
    edgecolor="none",
    zorder=0
)

ax.add_feature(
    cfeature.OCEAN.with_scale("50m"),
    facecolor="#DCEEFF",   
    edgecolor="none",
    zorder=0
)

ax.add_feature(
    cfeature.LAKES.with_scale("50m"),
    facecolor="#CFE8FF",   
    edgecolor="#7AA6C2",
    linewidth=0.5,
    zorder=1
)

ax.add_feature(
    cfeature.COASTLINE.with_scale("50m"),
    edgecolor="#3A3A3A",
    linewidth=0.8,
    zorder=2
)

ax.add_feature(
    cfeature.BORDERS.with_scale("50m"),
    edgecolor="#444444",
    linewidth=0.7,
    zorder=2
)

ax.add_feature(
    cfeature.STATES.with_scale("50m"),
    edgecolor="#8A8A8A",
    linewidth=0.45,
    zorder=2
)





xticks = [-120, -110, -100, -90, -80, -70]
yticks = [25, 30, 35, 40, 45, 50]

ax.set_xticks(xticks, crs=ccrs.PlateCarree())
ax.set_yticks(yticks, crs=ccrs.PlateCarree())

ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol="°"))
ax.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol="°"))

ax.tick_params(axis="both", which="major", labelsize=9, width=0.8, length=4)


ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.30, color="#808080")





flag_colors = [
    "#D84B4B",  
    "#F39C34",  
    "#4FA3D9",  
    "#3CB371",  
    "#9B59B6",  
    "#E91E63",  
    "#795548"   
]






def draw_flag(ax, lon, lat, color, label,
              pole_height=1.35, flag_length=1.15, flag_height=0.5):
    """
    \u5728\u7ecf\u7eac\u5ea6\u4f4d\u7f6e (lon, lat) \u4e0a\u7ed8\u5236\u4e00\u4e2a\u5f69\u8272\u5c0f\u65d7\u5b50。
    lon, lat \u4f7f\u7528 PlateCarree \u5750\u6807（\u7ecf\u7eac\u5ea6）。
    """

    
    top_lat = lat + pole_height

    
    ax.scatter(
        [lon], [lat],
        s=18,
        color="black",
        transform=ccrs.PlateCarree(),
        zorder=9
    )

    
    ax.plot(
        [lon, lon],
        [lat, top_lat],
        color="#333333",
        linewidth=1.2,
        transform=ccrs.PlateCarree(),
        zorder=10
    )

    
    triangle = Polygon(
        [
            (lon, top_lat),
            (lon + flag_length, top_lat - flag_height / 2),
            (lon, top_lat - flag_height)
        ],
        closed=True,
        facecolor=color,
        edgecolor="#333333",
        linewidth=0.8,
        transform=ccrs.PlateCarree(),
        zorder=11
    )
    ax.add_patch(triangle)

    
    ax.text(
        lon + flag_length + 0.18,
        top_lat - flag_height / 2,
        str(label),
        transform=ccrs.PlateCarree(),
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="center",
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="none",
            alpha=0.85
        ),
        zorder=12
    )



for i, (_, row) in enumerate(df.iterrows()):
    draw_flag(
        ax=ax,
        lon=row["longtitude"],
        lat=row["latitude"],
        color=flag_colors[i % len(flag_colors)],
        label=int(row["index"])
    )





legend_lines = []
for i, (_, row) in enumerate(df.iterrows()):
    legend_lines.append(f"{int(row['index'])}  {row['Station name']}")

legend_text = "\n".join(legend_lines)

ax.text(
    0.018,
    0.035,
    legend_text,
    transform=ax.transAxes,
    fontsize=9,
    ha="left",
    va="bottom",
    linespacing=1.45,
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="white",
        edgecolor="#666666",
        linewidth=0.7,
        alpha=0.95
    ),
    zorder=20
)





ax.set_title(
    "Geographical distribution of the seven ASOS stations",
    fontsize=13,
    pad=12
)





plt.tight_layout()

fig.savefig(
    OUTPUT_SVG,
    format="svg",
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_PNG,
    dpi=600,
    bbox_inches="tight"
)

print()
print("Map successfully generated.")
print(f"SVG saved to: {OUTPUT_SVG}")
print(f"PNG saved to: {OUTPUT_PNG}")

plt.show()